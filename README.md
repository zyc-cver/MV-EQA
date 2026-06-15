# MV-EQA: Exercise Quality Assessment in Monocular Video Streaming

We provide PyTorch implementation for our paper as a part of supplementary materials.

- Paper page: https://www.sciencedirect.com/science/article/abs/pii/S0952197626011875
- Public data/code archive: https://drive.google.com/file/d/1NrhMDVL_DM8VPP_YGgKVVHCMyyoRfgDQ/view

## Prerequisites

- Linux
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3.8+
- PyTorch 2.0+


## Getting Started

### Install dependencies

- Install dependencies

  ```bash
  pip install -r requirements.txt
  ```


### Run demo examples

We provide pretrained models and several video demos.

- Run the model to combine motion, skeleton, view angle from teacher and student videos:

  ```bash
  python inference.py --model_path ./model/pretrained_model.pth -p1 ./demos/1_jump/input_npy_files/learner.npy -p2 ./demos/1_jump/input_npy_files/teacher.npy -h1 720 -w1 720 -h2 720 -w2 720
  ```

  Results will be saved in `./examples`:

<p align="center">

</p>

You will get three video results, a 2d keypoint visualization for the teacher, a 2d keypoint visualization for the learner, and a retargeting result. The redirection result is made up of the learner's movements, the teacher's skeleton, and perspective.

<div style="display: flex; justify-content: space-around; align-items: start;">
  <div>
    <img src="demos/gifs/learner.gif" width="300">
    <p style="text-align: center;">learner</p>
  </div>
  <div>
    <img src="demos/gifs/teacher.gif" width="300">
    <p style="text-align: center;">teacher</p>
  </div>
  <div>
    <img src="demos/gifs/retarget.gif" width="300">
    <p style="text-align: center;">retarget_result</p>
  </div>
</div>


- Simulating retargeting in real-time scenes on mixamo synthetic data, and evaluate the degree of shaking:

  ```bash
  python shake_evaluate.py --model_path ./model/pretrained_model.pth -g 0
  ```
In contrast to previous work on 2d retargeting([_View-invariant Skeleton Action Representation Learning via Motion Retargeting_](https://walker-a11y.github.io/ViA-project/)), our model has higher stabilization performance in real-time fitness scenarios.
<div style="display: flex; justify-content: space-around; align-items: start;">
  <div>
    <img src="demos/gifs/jitter_ViA2024.gif" width="300">
    <p style="text-align: center;">ViA2024</p>
  </div>
  <div>
    <img src="demos/gifs/jitter_ours.gif" width="300">
    <p style="text-align: center;">Ours</p>
  </div>
</div>

## Train from scratch


- Train the model on GPU:

  ```
  python train.py -n full -g 0
  ```
