import os
from scipy.ndimage import gaussian_filter1d
import torch
import argparse
import numpy as np
from dataset import get_meanpose, get_dataloader, gen_meanpose
from model import get_autoencoder
from functional.visualization import motion2video, hex2rgb
from functional.motion import postprocess_motion2d
from common import config

VIEW_ANGLES = [(0, 0, -np.pi / 2),
               (0, 0, -np.pi / 3),
               (0, 0, -np.pi / 6),
               (0, 0, 0),
               (0, 0, np.pi / 6),
               (0, 0, np.pi / 3),
               (0, 0, np.pi / 2)]

MOTIONS = ['AirSquat', 'BicepCurl', 'JumpingJacks', 'WarmingUp']
CHARS = ['Pupmkinhulk', 'Remy', 'Ty']

def relocate(motion, fix_hip=True):
    if fix_hip:
        motion = motion - motion[8:9, :, :]
    else:
        # align hip joint in the first frame
        center = motion[8, :, 0]
        motion = motion - center[np.newaxis, :, np.newaxis]
    return motion

def realtime_exp_on_mixamo(config, args, m1=5, m2=3, window_size=64):
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
    char2 = CHARS[1]
    mot1 = MOTIONS[2]
    mot2 = MOTIONS[3]
    idx1 = 3
    idx2 = 5
    view1 = VIEW_ANGLES[idx1]
    view2 = VIEW_ANGLES[idx2]
    path1 = f'./demos/Jitter_Comparison_demos/{char1}/{mot1}/{mot1}.npy'
    path2 = f'./demos/Jitter_Comparison_demos/{char2}/{mot2}/{mot2}.npy'
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
    print("Generating videos...")
    out12 = relocate(out12) + 256
    out21 = relocate(out21) + 256
    save_dir = args.out_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    color1 = hex2rgb(args.color1)
    color2 = hex2rgb(args.color2)
    color3 = hex2rgb(args.color3)
    motion2video(input1, h2, w2, os.path.join(save_dir, f'input1.mp4'),
                 color1, args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(input2, h2, w2, os.path.join(save_dir, f'input2.mp4'),
                 color2, args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out12[:, :, 1:], h2, w2, os.path.join(save_dir, f'out12.mp4'),
                 color3, args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(out21[:, :, 1:], h2, w2, os.path.join(save_dir, f'out21.mp4'),
                 color3, args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")

    parser.add_argument('--color1', type=str, default='#a50b69#b73b87#db9dc3', help='color1')
    parser.add_argument('--color2', type=str, default='#4076e0#40a7e0#40d7e0', help='color2')
    parser.add_argument('--color3', type=str, default='#ff8b06#ffb431#ffcd9d', help='color3')
    parser.add_argument('--fps', type=float, default=30, help="fps of output video")
    parser.add_argument('-o', '--out_dir', type=str, default='./examples/jitter', help="output saving directory")
    parser.add_argument('-g', '--gpu_ids', type=int, default=1, required=False)
    parser.add_argument('--save_frame', action='store_true', help="to save rendered video frames")
    parser.add_argument('--relocate', default=False, help="disable gaussian kernel smoothing")
    parser.add_argument('--disable_smooth', action='store_true',
                        help="disable relocate the motion")
    parser.add_argument('--transparency', action='store_true',
                        help="make background transparent in resulting frames")
    args = parser.parse_args()

    config.initialize(args)

    realtime_exp_on_mixamo(config, args, m1=5, m2=3, window_size=64)



if __name__ == '__main__':
    main()