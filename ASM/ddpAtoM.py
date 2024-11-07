import argparse
import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch import nn
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.atom_dataset import LRS3SeqDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import MotionDecoder
import cv2

from data_util.face3d_helper import Face3DHelper


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class AToM:
    def __init__(
            self,
            feature_type,
            checkpoint_path="",
            normalizer=None,
            EMA=True,
            learning_rate=4e-4,
            weight_decay=0.02,
    ):
        num_processes = torch.cuda.device_count()
        use_baseline_feats = feature_type == "baseline"
        self.checkpoint_num = checkpoint_path.split("/")[-1].split(".")[0].split("-")[
            -1] if checkpoint_path is not None else ""
        repr_dim = 204
        self.repr_dim = 204
        feature_dim = 1024
        self.horizon = horizon = 156

        model = MotionDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        self.face3d_helper = Face3DHelper("../data/data_utils/deep_3drecon/BFM")
        print("Model has {} parameters".format(sum(y.numel() for y in model.parameters())))
        self.model = model
        self.diffusion = diffusion

        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = optim


    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def train_loop(self, opt):
        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--cuda",
                                default=True)
            parser.add_argument('--device_ids', type=str, default='0')
            parser.add_argument("--model_dir",
                                type=str,
                                default=r"/media/lyz/3.6t/228dm/MoDiTalker-master/AToM")
            return parser.parse_args()

        config = parse_args()
        os.makedirs(config.model_dir, exist_ok=True)
        torch.backends.cudnn.benchmark = True

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            self.diffusion = nn.DataParallel(self.diffusion, device_ids=device_ids).cuda()


        print('Loading data...')
        train_data_loader = LRS3SeqDataset("train").get_dataloader(opt.batch_size)
        test_data_loader = LRS3SeqDataset("val").get_dataloader(opt.batch_size)
        print('Data loading completed.')



        # Loop setup if this is the main process
        if torch.cuda.current_device() == 0:
            save_dir = Path(opt.project) / opt.exp_name
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {wdir}")



        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            self.train()

            """
            item_id :        filename
            mel              [B, 312, 80]
            hubert           [B, 312, 1024]
            x_mask           [B, 312]
            exp              [B, 156, 64]
            pose             [B, 156, 7]
            y_mask           [B, 156]
            idexp_lm3d       [B, 156, 204]     
            ref_mean_lm3d    [B, 204]         -1~1?
            mouth_idexp_lm3d [B, 156, 60]
            f0               [B, 312]
            """
            for step, batch in enumerate(tqdm(train_data_loader)):
                if batch["hubert"].shape[1] == 304:
                    continue

                for k, v in batch.items():
                    if k != "item_id":
                        batch[k] = v.cuda()

                x = batch["idexp_lm3d"]
                x_pos = batch["pose"]
                batch_size = x.shape[0]
                # ------------------------------------------------------------------------------------------------ #
                cond_keypoint = x[:, 0:1, :]
                cond_keypoint = torch.cat([cond_keypoint for _ in range(self.horizon)], dim=1)
                cond = batch["hubert"]
                # ------------------------------------------------------------------------------------------------ #
                x_ldmk = x.view(batch_size, self.horizon, -1, 3)
                cond_keypoint_ldmk = cond_keypoint.view(batch_size, self.horizon, -1, 3)
                residual_ldmk = x_ldmk - cond_keypoint_ldmk
                residual = residual_ldmk.view(batch_size, self.horizon, -1)

                total_loss, (loss, v_loss) = self.diffusion(residual, x_pos, cond_keypoint, cond, t_override=None)
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                avg_loss += loss.item()
                avg_vloss += v_loss.item()


                if step % 100 == 0 or epoch % opt.save_interval == 0:
                    self.diffusion.eval()
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.diffusion.state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                    }
                    torch.save(ckpt,
                               wdir / f"train-{epoch + (int(self.checkpoint_num) if self.checkpoint_num else 0)}.pt")
                    self.diffusion.train()

                    print(f"Epoch {epoch}: Avg Loss {avg_loss}, V Loss {avg_vloss}")

                    batch = next(iter(test_data_loader))
                    if batch["hubert"].shape[1] == 304:
                        continue

                    batch = {k: v.to('cuda:1', dtype=torch.float32) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    render_count = batch["hubert"].shape[0]
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")

                    x = batch["idexp_lm3d"][:render_count]
                    x_pos = batch["pose"][:render_count]

                    cond_keypoint = x[:, 0:1, :].repeat(1, self.horizon, 1)
                    cond = batch["hubert"][:render_count]

                    x_ldmk = x.view(render_count, self.horizon, -1, 3)
                    cond_keypoint_ldmk = cond_keypoint.view(render_count, self.horizon, -1, 3)
                    residual_ldmk = x_ldmk - cond_keypoint_ldmk
                    residual = residual_ldmk.view(render_count, self.horizon, -1)

                    _ = self.diffusion.render_sample(
                        self.face3d_helper,
                        shape,
                        cond_keypoint,
                        residual,
                        x_pos,
                        residual,
                        cond,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=batch["item_id"][:render_count],
                        sound=True,
                    )

                    print(f"[MODEL SAVED at Epoch {epoch}]")

    def render_sample(self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render,
        )
