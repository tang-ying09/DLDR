# DLDR

Official PyTorch implementation of **["Adaptive Diffusion Landmark Dynamic Rendering for Realistic Talking Face Video Generation"]**.   
Ying Tang, 
Yazhi Liu, 
Xiong Li<sup>&dagger;</sup>, 
Wei Li<sup>&dagger;</sup>.
<br>
<sup>&dagger;</sup>Corresponding author
<br>


## 1. Environment setup

```bash
conda create -n DlDR python=3.8 -y
conda activate DlDR
python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install natsort tqdm gdown omegaconf einops lpips pyspng tensorboard imageio av moviepy numba p_tqdm soundfile face_alignemnt
```

## 2. Get ready to train models 

### 2.1. Dataset

Currently, we provide experiments for the following two datasets: [LRS2] and [LRW]. Please go to the [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) website to download the dataset. 


### Preprocess the audio

Extract the raw audio and Mel-spectrogram features from video files by running:

```bash
CUDA_VISIBLE_DEVICES=0 python preprocess_audio.py --data_root ....../lrs2_video/ --out_root ..../lrs2_audio
CUDA_VISIBLE_DEVICES=0 python preprocess_video.py --dataset_video_root ....../lrs2_video/ --output_sketch_root ..../lrs2_sketch --output_face_root ..../lrs2_face --output_landmark_root ..../lrs2_landmarks
```


### 2.2. Download auxiliary models
<!-- Download  [this link](https://drive.google.com/file/d/1d08qauPUH0Nu_yN2gcmreLSiOiweD5OE/view?usp=sharing) -->
Get the [`BFM_model_front.mat`](https://drive.google.com/file/d/1d08qauPUH0Nu_yN2gcmreLSiOiweD5OE/view?usp=sharing), [`similarity_Lm3D_all.mat`](https://drive.google.com/file/d/17zp_zuUYAuieCWXerQkbp8SRSU4KJ8Fx/view?usp=sharing) and [`Exp_Pca.bin`](https://drive.google.com/file/d/1SPeJ4jcJT9VS4IdA7opzyGHCYMKuCLRh/view?usp=sharing), and place them to the `MoDiTalker/data/data_utils/deep_3drecon/BFM` directory.
Obtain ['BaselFaceModel.tgz](https://drive.google.com/file/d/1Kogpizrcf2zTm1fX9uUUWZuMQqHM7DOc/view?usp=sharing) and extract a file named `01_MorphableModel.mat` and place it to the `MoDiTalker/data/data_utils/deep_3drecon/BFM` directory.



## 3. Training

### 3.1. ASM

```bash
cd ASM
bash scripts/train.sh
```
The checkpoints of AToM will be saved in `./runs`

### 3.2. DR
### train Video Renderer
train the video renderer network by running:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_video_renderer.py --sketch_root ..../lrs2_sketch --face_img_root ..../lrs2_face  --audio_root ..../lrs2_audio
```
Note that the translation module will only be trained  after 25 epochs, thus the fid and running_gen_loss will only decrease after epoch 25. 



### 4. Getting the Weights
We provide the corresponding checkpoints in the below:
Download and place them in the `./checkpoints/` directory. 

Full checkpoints will be released later.

### 5. Inference
#### 5.1. Generating Motions from Audio 
Before producing the motions from audio, there's need to preprocess the audio since we process audio in the type of HuBeRT. To produce hubert feature of audio you want, please follow the script below:

```bash
cd data
python data_utils/preprocess/process_audio.py \
--audio path to audio \
--ref_dir path to directory of reference images 
```

Then the processed audio hubert(npy) will be saved in `data/inference/hubert/{sampling rate}` 

Note that, you need to specify the path to (1) reference images (2) processed hubert and (3) checkpoint in the following bash script. 

```bash
cd AToM
bash scripts/inference.sh
```

The results of AToM will be saved in `AToM/results/frontalized_npy` and this path should be consistent with the `ldmk_path` of the following step.

#### 5.2. Align Motions
Note that, you need to specify the path to (1) reference images and (2) produced landmark. 

```bash 
cd data/data_utils
python motion_align/align_face_recon.py \
--ldmk_path path to directory of generated landmark \
--driv_video_path path to directory of reference images 
```
The final landmarks will be saved in `AToM/results/aligned_npy`.

#### 5.3. Generating Video from aligned Motions

later





### Reference
This code is mainly built upon [MoDiTalker](https://github.com/cvlab-kaist/MoDiTalker/tree/master) and [IP_LAP](https://github.com/Weizhi-Zhong/IP_LAP).\
We also used the code from following repository: [GeneFace](https://github.com/yerfor/GeneFace).
