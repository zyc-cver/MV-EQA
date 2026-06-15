import os
import sys

from scipy.ndimage import gaussian_filter1d
import torch
import argparse
import numpy as np

from dataset import get_meanpose, get_dataloader, gen_meanpose
from model import get_autoencoder
from functional.visualization import motion2video, hex2rgb
from functional.motion import preprocess_motion2d, postprocess_motion2d, openpose2motion, process_test
from functional.utils import ensure_dir, pad_to_height
from common import config


def relocate(motion, fix_hip=True):
    if fix_hip:
        motion = motion - motion[8:9, :, :]
    else:
        # align hip joint in the first frame
        center = motion[8, :, 0]
        motion = motion - center[np.newaxis, :, np.newaxis]
    return motion

def retarget_3x(config, args):
    h1, w1, scale1 = pad_to_height(config.img_size[0], args.img1_height, args.img1_width)
    h2, w2, scale2 = pad_to_height(config.img_size[0], args.img2_height, args.img2_width)

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = get_meanpose(config)

    input1 = np.load(args.path1)
    input1 = gaussian_filter1d(input1, sigma=2, axis=-1)
    input2 = np.load(args.path2)
    input2 = gaussian_filter1d(input2, sigma=2, axis=-1)
    # np.save(args.path1, input1)
    # np.save(args.path2, input2)
    if args.relocate:
        input1 = relocate(input1)
        input1[:, 0, :] += w1 / 2
        input1[:, 1, :] += h1 / 2
        input2 = relocate(input2)
        input2[:, 0, :] += w2 / 2
        input2[:, 1, :] += h2 / 2
    input1 = preprocess_motion2d(input1, mean_pose, std_pose)
    input2 = preprocess_motion2d(input2, mean_pose, std_pose)
    len1 = input1.shape[-1]
    len2 = input2.shape[-1]

    assert len1 == len2, "unequal in length."

    input1 = input1.to(config.device)
    input2 = input2.to(config.device)
    output = net.transfer_both(input1, input2)
    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)
    output = postprocess_motion2d(output, mean_pose, std_pose, w1 // 2, h1 // 2)
    if not args.disable_smooth:
        output = gaussian_filter1d(output, sigma=2, axis=-1)
    if args.relocate:
        output = relocate(output)
        output[:, 0, :] += w2 / 2
        output[:, 1, :] += h2 / 2
    print("Generating videos...")
    save_dir = args.out_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    color1 = hex2rgb(args.color1)
    color2 = hex2rgb(args.color2)
    color3 = hex2rgb(args.color3)
    motion2video(input1, h1, w2, os.path.join(save_dir, f'learner.mp4'),
                 color1, args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(input2, h2, w2, os.path.join(save_dir, f'teacher.mp4'),
                 color2, args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    motion2video(output, h2, w2, os.path.join(save_dir, f'retarget.mp4'),
                 color3, args.transparency,
                 fps=args.fps, save_frame=args.save_frame)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('-p1', '--path1', type=str)
    parser.add_argument('-p2', '--path2', type=str)
    parser.add_argument('-h1', '--img1_height', type=int, help="video1's height")
    parser.add_argument('-w1', '--img1_width', type=int, help="video1's width")
    parser.add_argument('-h2', '--img2_height', type=int, help="video2's height")
    parser.add_argument('-w2', '--img2_width', type=int, help="video2's width")
    parser.add_argument('-o', '--out_dir', type=str, default='./examples', help="output saving directory")
    parser.add_argument('--fps', type=float, default=30, help="fps of output video")
    parser.add_argument('--color1', type=str, default='#a50b69#b73b87#db9dc3', help='color1')
    parser.add_argument('--color2', type=str, default='#4076e0#40a7e0#40d7e0', help='color2')
    parser.add_argument('--color3', type=str, default='#ff8b06#ffb431#ffcd9d', help='color3')
    parser.add_argument('--save_frame', action='store_true', help="to save rendered video frames")
    parser.add_argument('--relocate', default=False, help="disable gaussian kernel smoothing")
    parser.add_argument('--disable_smooth', action='store_true',
                        help="disable relocate the motion")
    parser.add_argument('--transparency', action='store_true',
                        help="make background transparent in resulting frames")
    parser.add_argument('--max_length', type=int, default=120,
                        help='maximum input video length')
    parser.add_argument('-g', '--gpu_ids', type=int, default=1, required=False)
    args = parser.parse_args()

    config.initialize(args)

    retarget_3x(config, args)


if __name__ == '__main__':
    main()