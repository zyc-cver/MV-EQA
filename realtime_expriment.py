# import numpy as np
#
# # 读取原始的.npy文件
# original_data = np.load('original.npy')
#
# # 复制原始数据
# copy_data = np.copy(original_data)
#
# # 拼接两个数组
# concatenated_data = np.concatenate((original_data, copy_data), axis=2)
#
# # 保存拼接后的数组回到原来的.npy文件
# np.save('original.npy', concatenated_data)

import os
import sys

from scipy.ndimage import gaussian_filter1d
import torch
import argparse
import numpy as np
import time

from dataset import get_meanpose, get_dataloader, gen_meanpose
from model import get_autoencoder
from functional.visualization import motion2video, hex2rgb
from functional.motion import preprocess_motion2d, postprocess_motion2d, openpose2motion, process_test, \
    preprocess_mixamo_
from functional.utils import ensure_dir, pad_to_height
from common import config
import math
from itertools import combinations, permutations

VIEW_ANGLES = [(0, 0, -np.pi / 2),
               (0, 0, -np.pi / 3),
               (0, 0, -np.pi / 6),
               (0, 0, 0),
               (0, 0, np.pi / 6),
               (0, 0, np.pi / 3),
               (0, 0, np.pi / 2)]

MOTIONS = ['AirSquat', 'BicepCurl', 'JumpingJacks', 'WarmingUp']
#  58  178  34  191
CHARS = ['Pupmkinhulk', 'Remy', 'Ty']


def relocate(motion, fix_hip=True):
    if fix_hip:
        motion = motion - motion[8:9, :, :]
    else:
        # align hip joint in the first frame
        center = motion[8, :, 0]
        motion = motion - center[np.newaxis, :, np.newaxis]
    return motion


def test(config, args, m1=1, m2=1, window_size=64):
    pass
    mean_pose, std_pose = get_meanpose(config)
    dataloder = get_dataloader('test', config)
    char1 = CHARS[0]
    char2 = CHARS[2]
    mot1 = MOTIONS[3]
    mot2 = MOTIONS[3]
    idx1 = 1
    idx2 = 5
    view1 = VIEW_ANGLES[idx1]
    view2 = VIEW_ANGLES[idx2]
    path1 = f'./exp_data/{char1}/{mot1}/{mot1}.npy'
    input1 = np.load(path1)
    input1 = dataloder.dataset.preprocessing_exp(input1, view1).unsqueeze(0)
    input1 = postprocess_motion2d(input1, mean_pose, std_pose)
    input2 = np.load(path1)
    input2 = process_test(input2, view_angle=view1)
    input1 = relocate(input1)
    input2 = relocate(input2)
    print(1)


def realtime_exp(config, args, m1=1, m2=1, window_size=64):
    w1 = h1 = w2 = h2 = 512

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = get_meanpose(config)

    # get input
    dataloder = get_dataloader('test', config)
    char1 = CHARS[0]
    char2 = CHARS[2]
    mot1 = MOTIONS[2]
    mot2 = MOTIONS[3]
    idx1 = 3
    idx2 = 5
    view1 = VIEW_ANGLES[idx1]
    view2 = VIEW_ANGLES[idx2]
    path1 = f'./exp_data/{char1}/{mot1}/{mot1}.npy'
    path2 = f'./exp_data/{char2}/{mot2}/{mot2}.npy'
    input1 = input2 = np.array([])
    if m1 != 1:
        input1 = np.load(path1)
        copy_data = np.copy(input1)
        for i in range(m1 - 1):
            input1 = np.concatenate((input1, copy_data), axis=2)
    else:
        input1 = np.load(path1)
    if m2 != 1:
        input2 = np.load(path2)
        copy_data = np.copy(input2)
        for i in range(m2 - 1):
            input2 = np.concatenate((input2, copy_data), axis=2)
    else:
        input2 = np.load(path2)
    input1 = dataloder.dataset.preprocessing_exp(input1, view1).unsqueeze(0)
    input2 = dataloder.dataset.preprocessing_exp(input2, view2).unsqueeze(0)
    size = 0
    len1 = input1.shape[-1]
    len2 = input2.shape[-1]
    # 对齐长度
    if len1 > len2:
        input1 = input1[:, :, :len2]
        size = len2
    else:
        input2 = input2[:, :, :len1]
        size = len1
    input1 = input1.to(config.device)
    input2 = input2.to(config.device)
    out12 = np.zeros((15, 2, 1))
    out21 = np.zeros((15, 2, 1))
    for i in range(size - window_size + 1):
        cur1 = input1[:, :, i:i + window_size]
        cur2 = input2[:, :, i:i + window_size]
        # transfer by network
        cur12 = net.transfer_both(cur1, cur2)
        cur21 = net.transfer_both(cur2, cur1)
        cur12 = postprocess_motion2d(cur12, mean_pose, std_pose, w1 // 2, h1 // 2)
        cur21 = postprocess_motion2d(cur21, mean_pose, std_pose, w2 // 2, h2 // 2)
        out12 = np.concatenate((out12, cur12[:, :, -1:]), axis=2)
        out21 = np.concatenate((out21, cur21[:, :, -1:]), axis=2)
    input1 = postprocess_motion2d(input1[:, :, window_size - 1:], mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2[:, :, window_size - 1:], mean_pose, std_pose, w2 // 2, h2 // 2)
    # out12 = postprocess_motion2d(np.array(out12), mean_pose, std_pose, w2 // 2, h2 // 2)
    # out21 = postprocess_motion2d(np.array(out21), mean_pose, std_pose, w1 // 2, h1 // 2)
    print("Generating videos...")
    # out12 = relocate(out12) + 256
    # out21 = relocate(out21) + 256
    save_dir = f'./output_exp/{char1}_{mot1}_{idx1}_{char2}_{mot2}_{idx2}_{window_size}_{size - window_size}'
    if not os.path.exists(save_dir):
        # 使用os.makedirs()创建文件夹
        os.makedirs(save_dir)
    motion2video(input1, h2, w2, os.path.join(save_dir, f'input1.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(input2, h2, w2, os.path.join(save_dir, f'input2.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out12[:, :, 1:], h2, w2, os.path.join(save_dir, f'out12.mp4'),
                 hex2rgb('#4076e0#40a7e0#40d7e0'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out21[:, :, 1:], h2, w2, os.path.join(save_dir, f'out21.mp4'),
                 hex2rgb('#4076e0#40a7e0#40d7e0'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    print("Done.")


def gen_nor(config, args, m1=1, m2=1, window_size=120):
    w1 = h1 = w2 = h2 = 512

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = get_meanpose(config)

    # get input
    dataloder = get_dataloader('test', config)
    char1 = CHARS[0]
    char2 = CHARS[2]
    mot1 = MOTIONS[0]
    mot2 = MOTIONS[1]
    idx1 = 2
    idx2 = 5
    view1 = VIEW_ANGLES[idx1]
    view2 = VIEW_ANGLES[idx2]
    path1 = f'./exp_data/{char1}/{mot1}/{mot1}.npy'
    path2 = f'./exp_data/{char2}/{mot2}/{mot2}.npy'
    input1 = input2 = np.array([])
    if m1 != 1:
        input1 = np.load(path1)
        copy_data = np.copy(input1)
        for i in range(m1 - 1):
            input1 = np.concatenate((input1, copy_data), axis=2)
    else:
        input1 = np.load(path1)
    if m2 != 1:
        input2 = np.load(path2)
        copy_data = np.copy(input2)
        for i in range(m2 - 1):
            input2 = np.concatenate((input2, copy_data), axis=2)
    else:
        input2 = np.load(path2)
    input1 = dataloder.dataset.preprocessing_exp(input1, view1).unsqueeze(0)
    input2 = dataloder.dataset.preprocessing_exp(input2, view2).unsqueeze(0)
    size = 0
    len1 = input1.shape[-1]
    len2 = input2.shape[-1]
    # 对齐长度
    if len1 > len2:
        input1 = input1[:, :, :len2]
        size = len2
    else:
        input2 = input2[:, :, :len1]
        size = len1
    input1 = input1.to(config.device)
    input2 = input2.to(config.device)
    out12 = net.transfer_both(input1, input2)
    out21 = net.transfer_both(input2, input1)
    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)
    out12 = postprocess_motion2d(out12, mean_pose, std_pose, w2 // 2, h2 // 2)
    out21 = postprocess_motion2d(out21, mean_pose, std_pose, w1 // 2, h1 // 2)
    print("Generating videos...")
    save_dir = f'./output_exp/{char1}_{mot1}_{idx1}_{char2}_{mot2}_{idx2}_{window_size}_{size}/compare'
    if not os.path.exists(save_dir):
        # 使用os.makedirs()创建文件夹
        os.makedirs(save_dir)
    motion2video(input1, h2, w2, os.path.join(save_dir, f'input1.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(input2, h2, w2, os.path.join(save_dir, f'input2.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out12, h2, w2, os.path.join(save_dir, f'out12.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out21, h2, w2, os.path.join(save_dir, f'out21.mp4'),
                 hex2rgb('#4076e0#40a7e0#40d7e0'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    print("Done.")


#  student squat1 102  squat3  102   teacher  172
def real_data_nor_json(config, args, m1=1, m2=1, window_size=64):
    w1 = h1 = w2 = h2 = 512

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = gen_meanpose(config)

    path1 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/student_json/squat3'
    path2 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/teacher_json/sq'
    if m1 != 1:
        input1 = np.array(openpose2motion(path1, max_frame=102))
        copy_data = np.copy(input1)
        for i in range(m1 - 1):
            input1 = np.concatenate((input1, copy_data), axis=2)
    else:
        input1 = np.array(openpose2motion(path1, max_frame=102))
    if m2 != 1:
        input2 = np.array(openpose2motion(path2, max_frame=172))
        copy_data = np.copy(input2)
        for i in range(m2 - 1):
            input2 = np.concatenate((input2, copy_data), axis=2)
    else:
        input2 = np.array(openpose2motion(path2, max_frame=172))
    input1 = preprocess_motion2d(input1, mean_pose, std_pose)
    input2 = preprocess_motion2d(input2, mean_pose, std_pose)
    size = 0
    len1 = input1.shape[-1]
    len2 = input2.shape[-1]
    # 对齐长度
    if len1 > len2:
        input1 = input1[:, :, :len2]
        size = len2
    else:
        input2 = input2[:, :, :len1]
        size = len1

    input1 = input1.to(config.device)
    input2 = input2.to(config.device)
    out12 = net.transfer_both(input1, input2)
    out21 = net.transfer_both(input2, input1)
    out12 = postprocess_motion2d(out12, mean_pose, std_pose, w1 // 2, h1 // 2)
    out21 = postprocess_motion2d(out21, mean_pose, std_pose, w2 // 2, h2 // 2)
    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)
    print("Generating videos...")
    save_dir = f'./output_real_exp/{os.path.basename(path1)}_{os.path.basename(path2)}'
    if not os.path.exists(save_dir):
        # 使用os.makedirs()创建文件夹
        os.makedirs(save_dir)
    motion2video(input1, h2, w2, os.path.join(save_dir, f'input1.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(input2, h2, w2, os.path.join(save_dir, f'input2.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out12, h2, w2, os.path.join(save_dir, f'out12.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out21, h2, w2, os.path.join(save_dir, f'out21.mp4'),
                 hex2rgb('#4076e0#40a7e0#40d7e0'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    print("Done.")


def real_data_nor_npy(config, args, m1=1, m2=1, window_size=64):
    w1 = h1 = w2 = h2 = 512

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = gen_meanpose(config)

    # path1 = '//data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/keypoints_data_leg.npy'
    # path2 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/keypoints_data_teacher_leg.npy'
    # input1 = np.load(path1)
    # input1 = gaussian_filter1d(input1, sigma=3, axis=-1)
    # # input2 = np.array(openpose2motion(path2, max_frame=172))
    # input2 = np.load(path2)
    # input2 = gaussian_filter1d(input2, sigma=3, axis=-1)
    path1 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/keypoints_data_squat.npy'
    path2 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/teacher_json/sq'
    input1 = np.load(path1)
    input1 = gaussian_filter1d(input1, sigma=2, axis=-1)
    input2 = np.array(openpose2motion(path2, max_frame=172))
    # input2 = np.load(path2) / 1.6
    input2 = gaussian_filter1d(input2, sigma=2, axis=-1)
    input1 = preprocess_motion2d(input1, mean_pose, std_pose)
    input2 = preprocess_motion2d(input2, mean_pose, std_pose)
    size = 0
    len1 = input1.shape[-1]
    len2 = input2.shape[-1]
    # 对齐长度
    if len1 > len2:
        input1 = input1[:, :, :len2]
        size = len2
    else:
        input2 = input2[:, :, :len1]
        size = len1

    input1 = input1.to(config.device)
    input2 = input2.to(config.device)
    out12 = net.transfer_both(input1, input2)
    out21 = net.transfer_both(input2, input1)
    out12 = postprocess_motion2d(out12, mean_pose, std_pose, w1 // 2, h1 // 2)
    out21 = postprocess_motion2d(out21, mean_pose, std_pose, w2 // 2, h2 // 2)
    out12 = gaussian_filter1d(out12, sigma=2, axis=-1)
    out21 = gaussian_filter1d(out21, sigma=2, axis=-1)
    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)
    print("Generating videos...")
    save_dir = f'./output_real_exp/{os.path.basename(path1)}_{os.path.basename(path2)}'
    if not os.path.exists(save_dir):
        # 使用os.makedirs()创建文件夹
        os.makedirs(save_dir)
    motion2video(input1, h2, w2, os.path.join(save_dir, f'input1.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(input2, h2, w2, os.path.join(save_dir, f'input2.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out12, h2, w2, os.path.join(save_dir, f'out12.mp4'),
                 hex2rgb('#4076e0#40a7e0#40d7e0'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out21, h2, w2, os.path.join(save_dir, f'out21.mp4'),
                 hex2rgb('#4076e0#40a7e0#40d7e0'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    print("Done.")

#  student squat1 102  squat3  102   teacher  172
def real_data_exp(config, args, m1=1, m2=1, window_size=30):
    w1 = h1 = w2 = h2 = 512

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = gen_meanpose(config)
    # path1 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/student_json/squat1'
    # path2 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/teacher_json/sq'
    # if m1 != 1:
    #     input1 = np.array(openpose2motion(path1, max_frame=102))
    #     copy_data = np.copy(input1)
    #     for i in range(m1 - 1):
    #         input1 = np.concatenate((input1, copy_data), axis=2)
    # else:
    #     input1 = np.array(openpose2motion(path1, max_frame=102))
    # if m2 != 1:
    #     input2 = np.array(openpose2motion(path2, max_frame=172))
    #     copy_data = np.copy(input2)
    #     for i in range(m2 - 1):
    #         input2 = np.concatenate((input2, copy_data), axis=2)
    # else:
    #     input2 = np.array(openpose2motion(path2, max_frame=172))
    path1 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/keypoints_data_squat.npy'
    path2 = '/data/xbx/project/2D-Motion-Retargeting-original/exp_real_data/teacher_json/sq'
    input1 = np.load(path1)
    input1 = gaussian_filter1d(input1, sigma=2, axis=-1)
    input2 = np.array(openpose2motion(path2, max_frame=172))
    # input2 = np.load(path2) / 1.6
    input2 = gaussian_filter1d(input2, sigma=2, axis=-1)
    input1 = preprocess_motion2d(input1, mean_pose, std_pose)
    input2 = preprocess_motion2d(input2, mean_pose, std_pose)
    size = 0
    len1 = input1.shape[-1]
    len2 = input2.shape[-1]
    # 对齐长度
    if len1 > len2:
        input1 = input1[:, :, :len2]
        size = len2
    else:
        input2 = input2[:, :, :len1]
        size = len1

    input1 = input1.to(config.device)
    input2 = input2.to(config.device)
    out12 = np.zeros((15, 2, 1))
    out21 = np.zeros((15, 2, 1))
    for i in range(size - window_size):
        cur1 = input1[:, :, i:i + window_size]
        cur2 = input2[:, :, i:i + window_size]
        # transfer by network
        cur12 = net.transfer_both(cur1, cur2)
        cur21 = net.transfer_both(cur2, cur1)
        cur12 = postprocess_motion2d(cur12[:, :, 1:], mean_pose, std_pose, w1 // 2, h1 // 2)
        cur21 = postprocess_motion2d(cur21[:, :, 1:], mean_pose, std_pose, w2 // 2, h2 // 2)
        out12 = np.concatenate((out12, cur12[:, :, -1:]), axis=2)
        out21 = np.concatenate((out21, cur21[:, :, -1:]), axis=2)
    input1 = postprocess_motion2d(input1[:, :, 1:], mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2[:, :, 1:], mean_pose, std_pose, w2 // 2, h2 // 2)
    # out12 = postprocess_motion2d(np.array(out12), mean_pose, std_pose, w2 // 2, h2 // 2)
    # out21 = postprocess_motion2d(np.array(out21), mean_pose, std_pose, w1 // 2, h1 // 2)
    print("Generating videos...")
    save_dir = f'./output_real_exp/{os.path.basename(path1)}_{os.path.basename(path2)}'
    if not os.path.exists(save_dir):
        # 使用os.makedirs()创建文件夹
        os.makedirs(save_dir)
    motion2video(input1, h2, w2, os.path.join(save_dir, f'input1.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(input2, h2, w2, os.path.join(save_dir, f'input2.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out12, h2, w2, os.path.join(save_dir, f'out12.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out21, h2, w2, os.path.join(save_dir, f'out21.mp4'),
                 hex2rgb('#4076e0#40a7e0#40d7e0'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    print("Done.")


def transmomovideo(config, args):
    # save_dir = f'./output_transmomo'
    # if not os.path.exists(save_dir):
    #     # 使用os.makedirs()创建文件夹
    #     os.makedirs(save_dir)
    # input1 = np.load(
    #     '/data/xbx/project/transmomo/data/mixamo/36_800_24/test_random_rotate/PUMPKINHULK_L/Goalkeeper_Directing_(1)/Goalkeeper_Directing_(1).npy')
    # input2 = np.load(
    #     '/data/xbx/project/transmomo/transmomo_mixamo_36_800_24_results/motion_ANDROMEDA_0_body_PUMPKINHULK_L_0.npy')
    # # input = np.load('/data/xbx/project/transmomo/transmomo_mixamo_36_800_24_results/motion_ANDROMEDA_0_body_PUMPKINHULK_L_14.npy')
    # input1 = preprocess_mixamo_(input1[:, [0, 2], :], unit=1)
    # input1 = input1 * 128 + 256
    # input2 = input2 * 128 + 256
    # motion2video(input1, 512, 512, os.path.join(save_dir, f'inputab.mp4'),
    #              hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
    #              fps=args.fps, save_frame=args.save_frame)
    # motion2video(input2, 512, 512, os.path.join(save_dir, f'inputba.mp4'),
    #              hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
    #              fps=args.fps, save_frame=args.save_frame)
    mean_pose, std_pose = gen_meanpose(config)
    path = '/data/xbx/project/vitpose/output/keypoints_data_leg.npy'
    input = np.load(path)
    input = gaussian_filter1d(input, sigma=2, axis=-1)
    input = preprocess_motion2d(input, mean_pose, std_pose)
    input = postprocess_motion2d(input, mean_pose, std_pose, 512 // 2, 512 // 2)
    save_dir = './test-real-v'
    motion2video(input, 512, 512, os.path.join(save_dir, f'input_ce.mp4'),
                 hex2rgb('#a50b69#b73b87#db9dc3'), args.transparency,
                 fps=args.fps, save_frame=args.save_frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, choices=['skeleton', 'view', 'full'], required=True,
                        help='which structure to use.')
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('--render_video', type=bool, default=True, help="whether to save rendered video")
    parser.add_argument('--fps', type=float, default=30, help="fps of output video")
    parser.add_argument('--color1', type=str, default='#a50b69#b73b87#db9dc3', help='color1')
    parser.add_argument('--color2', type=str, default='#4076e0#40a7e0#40d7e0', help='color2')
    parser.add_argument('--color3', type=str, default='#ff8b06#ffb431#ffcd9d', help='color3')
    parser.add_argument('--save_frame', action='store_true', help="to save rendered video frames")
    parser.add_argument('--disable_smooth', action='store_true',
                        help="disable gaussian kernel smoothing")
    parser.add_argument('--transparency', action='store_true',
                        help="make background transparent in resulting frames")
    parser.add_argument('--max_length', type=int, default=120,
                        help='maximum input video length')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    config.initialize(args)

    # realtime_exp(config, args, m1=11, m2=2)
    # gen_nor(config, args, m1=3, m2=1)
    # real_data_nor(config, args, m1=1, m2=1)
    # test(config, args)
    # transmomovideo(config, args)
    real_data_nor_npy(config, args, m1=1, m2=1)
    # real_data_exp(config, args, m1=1, m2=1)


if __name__ == '__main__':
    main()
