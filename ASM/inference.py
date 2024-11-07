from functools import partial
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import os
import torch
import numpy as np
import importlib
import random
from AToM.ddpAtoM import wrap
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import MotionDecoder

import torch.nn.functional as F
import pdb
import cv2

import glob
import argparse

from data_util.face3d_helper import Face3DHelper

# =========================================
HORIZON = 156
# =========================================
device = "cuda"
repr_dim = 204
feature_dim = 1024
seed = 2021
deterministic = True


def collate_2d(values, pad_value=0):
    """
    Convert a list of 2d tensors into a padded 3d tensor.
        values: list of Batch tensors with shape [T, C]
        return: [B, T, C]
    """
    max_len = values.size(0)
    hidden_dim = values.size(1)
    batch_size = len(values)
    ret = torch.ones([batch_size, max_len, hidden_dim], dtype=values[0].dtype) * pad_value
    for i, v in enumerate(values):
        ret[i, :v.shape[0], :].copy_(v)
    return ret


def prepare_models(args):
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MotionDecoder(
        nfeats=repr_dim,
        seq_len=HORIZON,
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=feature_dim,
        activation=F.gelu,
    )

    print("Model has {} parameters".format(sum(y.numel() for y in model.parameters())))

    EMA = False
    # model.load_state_dict(maybe_wrap(checkpoint["ema_state_dict" if EMA else "model_state_dict"], 1))

    # Determine which state_dict to use
    state_dict_key = "ema_state_dict" if EMA else "model_state_dict"
    state_dict = checkpoint[state_dict_key]

    # Debug: Print missing keys
    model_state_dict = model.state_dict()
    missing_keys = [key for key in model_state_dict.keys() if key not in state_dict]
    # print(f"Missing keys: {missing_keys}")
    # print(f"State dict keys: {list(state_dict.keys())}")

    # Load state_dict with strict=False to ignore missing keys
    model.load_state_dict(maybe_wrap(state_dict, 1), strict=False)

    diffusion = GaussianDiffusion(
        model,
        HORIZON,
        repr_dim,
        schedule="cosine",
        n_timestep=1000,
        predict_epsilon=False,
        loss_type="l2",
        use_p2=False,
        cond_drop_prob=0.25,
        guidance_weight=2,
    )
    diffusion = diffusion.to(device)
    diffusion.eval()

    face3d_helper = Face3DHelper(args.face3d_helper)

    return model, diffusion, face3d_helper


def save_lm_img(lm3D, out_path, WH=256, flip=True):
    if lm3D.shape[-1] == 3:
        lm3d = (lm3D * WH / 2 + WH / 2).astype(int)
        lm2d = lm3d[:, :2]
    else:
        lm2d = lm3D.astype(int)
    img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
    for i in range(len(lm2d)):
        x, y = lm2d[i]
        img = cv2.circle(img, center=(x, y), radius=3, color=(0, 0, 0), thickness=-1)

    if flip:
        img = cv2.flip(img, 0)
    else:
        pass
    cv2.imwrite(out_path, img)


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


cnt = 0


def add_noise_points(img, num_points, point_radius):
    height, width, _ = img.shape
    for _ in range(num_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        img = cv2.circle(img, center=(x, y), radius=point_radius, color=(0, 0, 0), thickness=-1)
    return img


def generate_images_with_noise(atom_out, save_dir, name, it, HORIZON, num_noise_points=100):
    # Create the directory if it doesn't exist
    frontalized_png_save_dir = os.path.join(save_dir, "frontalized_png", f"{name}")
    os.makedirs(frontalized_png_save_dir, exist_ok=True)

    atom_out = (atom_out * 256 / 2 + 256 / 2).astype(int)
    for i_img in range(156):
        vis_atom_out = atom_out[i_img, :, :2]
        img = np.ones([256, 256, 3], dtype=np.uint8) * 255
        for i in range(68):
            x, y = vis_atom_out[i]
            img = cv2.circle(img, center=(x, y), radius=3, color=(0, 0, 0), thickness=-1)
        img = cv2.flip(img, 0)
        out_name = f"{frontalized_png_save_dir}/{str(it * HORIZON + i_img).zfill(3)}.png"
        cv2.imwrite(out_name, img)

        # Add noise points to the image
        img_with_noise = add_noise_points(img, num_noise_points, point_radius=3)
        noisy_img_path = f"{frontalized_png_save_dir}/noisy_{str(it * HORIZON + i_img).zfill(3)}.png"
        cv2.imwrite(noisy_img_path, img_with_noise)

    print(f"Done")


def load_idlist(path):
    with open(path, "r") as f:
        lines = f.readlines()
        id_list = [line.replace("\n", "").replace(".mp4", "").strip() for line in lines]
    return id_list


def main(args):
    model, diffusion, face3d_helper = prepare_models(args)
    if args.id_list is None:
        id_list = os.listdir(args.data_root)
    else:
        id_list = load_idlist(args.id_list)

    for name in tqdm(id_list):
        for it in range(HORIZON // 156):
            STR = it * HORIZON
            END = (it + 1) * HORIZON

            if it == 0:
                file_path = os.path.join(args.data_root, name,
                                         "00000.npy")

                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"文件不存在，跳过：{file_path}")
                    continue  # 文件不存在时跳过当前name

                # 如果文件存在，加载文件
                if it == 0:
                    cond_keypoint = np.load(file_path)

                # cond_keypoint = np.load(os.path.join(args.data_root, "HDTF_videos_30_keypoints/face-centric/unposed/RD",name, "00000.npy"))
                cond_keypoint = torch.from_numpy(cond_keypoint)
                print(cond_keypoint.shape)
                ret = collate_2d(cond_keypoint)
                # cond_keypoint = np.load(f"../data/inference/init_kpt/{name}.npy")
                print(ret.shape)
                ret = ret[:, 0:1, :].to(device)
                # cond_keypoint = cond_keypoint[:, 0:1].to(device)
            else:  # TODO
                ret = np.load(f"")

            if ret.shape[-1] == 3:
                ret = ret.unsqueeze(0)
                ret = ret.view(1, 1, -1)

            cond_ldmk = torch.cat([ret for _ in range(HORIZON)], dim=1)
            # print("cond_ldmk:", cond_ldmk.shape)
            hubert_name = args.hubert_path
            cond = np.load(hubert_name)
            cond = torch.from_numpy(cond)
            num_padding = 312 - 60  # 需要填充102行
            cond = F.pad(cond, (0, 0, 0, num_padding), "constant", 0)
            cond = cond.unsqueeze(0)
            cond = cond[:, STR: (it + 2 * HORIZON), :].to(device)
            # print("cond:", cond.shape)
            shape = [1, HORIZON, repr_dim]

            pos = torch.zeros(1, HORIZON, 3, device=device)
            # print("pos:", pos.shape)
            _ = torch.rand(1, HORIZON, repr_dim, device=device)
            print(f"cond_ldmk:{cond_ldmk.shape}, _:{_.shape},pos:{pos.shape},cond:{cond.shape}")
            with torch.no_grad():
                atom_out = diffusion.render_sample(
                    face3d_helper,
                    shape,
                    cond_ldmk,
                    _,
                    pos,
                    _,
                    cond,
                    0,
                    "",
                    name=name,
                    sound=True,
                )

            frontalized_npy_save_dir = os.path.join(args.save_dir, "frontalized_npy", f"{name}")
            os.makedirs(frontalized_npy_save_dir, exist_ok=True)
            atom_out = atom_out[0]

            atom_out = atom_out.view(HORIZON, -1, 3).detach().cpu()
            cond_keypoint_ldmk = cond_ldmk.view(HORIZON, -1, 3).detach().cpu()
            atom_out += cond_keypoint_ldmk
            atom_out = atom_out.view(HORIZON, -1)

            atom_out = atom_out / 10 + face3d_helper.key_mean_shape.squeeze().reshape([1, -1]).cpu().numpy()
            atom_out = atom_out.view(HORIZON, 68, 3)
            atom_out = atom_out.cpu().numpy()

            np.save(f"{frontalized_npy_save_dir}/atom_{str(it)}.npy", atom_out)
            # ------------------------- visualization------------------------------------------ #
            frontalized_png_save_dir = os.path.join(args.save_dir, "frontalized_png", f"{name}")
            os.makedirs(frontalized_png_save_dir, exist_ok=True)

            scale_factor = 128 / 256  # 假设原始坐标是按256x256的图像比例

            atom_out = (atom_out * 256 / 2 + 256 / 2).astype(int)
            for i_img in range(156):
                vis_atom_out = atom_out[i_img, :, :2] * scale_factor
                img = np.zeros([128, 128, 3], dtype=np.uint8)  # 创建一个黑色背景的图像

                for i in range(68):
                    x, y = map(int, vis_atom_out[i])
                    img = cv2.circle(img, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                    # 定义面部特征的关键点
                jaw_points = vis_atom_out[0:17]  # 下颌
                right_eyebrow_points = vis_atom_out[17:22]  # 右眉毛
                left_eyebrow_points = vis_atom_out[22:27]  # 左眉毛
                nose_bridge_points = vis_atom_out[27:31]  # 鼻梁
                lower_nose_points = vis_atom_out[31:36]  # 鼻子底部
                right_eye_points = vis_atom_out[36:42]  # 右眼
                left_eye_points = vis_atom_out[42:48]  # 左眼
                outer_lip_points = vis_atom_out[48:60]  # 外嘴唇
                inner_lip_points = vis_atom_out[60:68]  # 内嘴唇

                # 依次绘制各个部位的连线
                def draw_lines(img, points, is_closed=False):
                    num_points = len(points)
                    for i in range(num_points - 1):
                        pt1 = tuple(map(int, points[i]))  # 确保坐标为整数类型的元组
                        pt2 = tuple(map(int, points[i + 1]))  # 同上
                        cv2.line(img, pt1, pt2, color=(255, 255, 255), thickness=1)
                    if is_closed:
                        pt1 = tuple(map(int, points[num_points - 1]))
                        pt2 = tuple(map(int, points[0]))
                        cv2.line(img, pt1, pt2, color=(255, 255, 255), thickness=1)

                def draw_smooth_lines(img, points, is_closed=False, color=(255, 255, 255), thickness=1):
                    # 将点扩大到更大的图像上绘制，然后再缩小回原尺寸，以模拟抗锯齿
                    scale = 2  # 缩放因子
                    big_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                    scaled_points = np.array(points, dtype=np.int32) * scale
                    scaled_points = scaled_points.reshape((-1, 1, 2))
                    cv2.polylines(big_img, [scaled_points], isClosed=is_closed, color=color,
                                  thickness=thickness * scale)
                    img = cv2.resize(big_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    return img

                img = draw_smooth_lines(img, jaw_points, is_closed=False)
                img = draw_smooth_lines(img, right_eyebrow_points, is_closed=False)
                img = draw_smooth_lines(img, left_eyebrow_points, is_closed=False)
                img = draw_smooth_lines(img, nose_bridge_points, is_closed=False)
                img = draw_smooth_lines(img, lower_nose_points, is_closed=False)
                img = draw_smooth_lines(img, right_eye_points, is_closed=True)
                img = draw_smooth_lines(img, left_eye_points, is_closed=True)
                img = draw_smooth_lines(img, outer_lip_points, is_closed=True)
                img = draw_smooth_lines(img, inner_lip_points, is_closed=True)

                img = cv2.flip(img, 0)
                out_name = f"{frontalized_png_save_dir}/{str(it * HORIZON + i_img).zfill(3)}.png"
                cv2.imwrite(out_name, img)
            # generate_images_with_noise(atom_out, args.save1_dir, name, it, HORIZON)
            # --------------------------------------------------------------------------------- #

            print(f"Done")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="")
    args.add_argument("--data_root", type=str,
                      default="/keypoints/face-centric/unposed",
                      help="data location of reference images")
    args.add_argument("--hubert_path", type=str,
                      default="results/hubert/16000/lrs212345_00020.npy",
                      help="path to the hubert extracted")
    args.add_argument("--face3d_helper", type=str,
                      default="data/data_utils/deep_3drecon/BFM",
                      help="path to the BFM folder")
    args.add_argument("--id_list", type=str,
                      default=None, help="if id_list is None, then the whole id in the data_root will be included")
    args.add_argument("--device", type=str,
                      default="cuda:0")
    args.add_argument("--checkpoint", type=str,
                      default=".pt",
                      help="path to the checkpoint of AToM")
    args.add_argument("--save_dir", type=str,
                      default="frontalized1",
                      help="path to the directory to save frontalized landmarks")
    args.add_argument("--save1_dir", type=str,
                      default="frontalized2",
                      help="path to the directory to save frontalized landmarks")
    args = args.parse_args()
    main(args)