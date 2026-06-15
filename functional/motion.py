from scipy.ndimage import gaussian_filter1d
import numpy as np
import json
import os
import torch


def process_test(motion3d, unit=128, view_angle=None):

    local3d = get_local3d(motion3d, view_angle)

    motion3d[1, :, :] = (motion3d[2, :, :] + motion3d[5, :, :]) / 2
    motion3d[8, :, :] = (motion3d[9, :, :] + motion3d[12, :, :]) / 2

    if local3d is not None:
        motion2d = local3d[[0, 2], :] @ motion3d  # (15, 2, 64)
    else:
        motion2d = motion3d[:, [0, 2], :]

    # rotate 180
    motion2d[:, 1, :] = - motion2d[:, 1, :]

    motion = motion2d * unit

    return motion

def preprocess_mixamo_(motion, unit=128):

    _, D, _ = motion.shape
    horizontal_dim = 0
    vertical_dim = D - 1

    motion[1, :, :] = (motion[2, :, :] + motion[5, :, :]) / 2
    motion[8, :, :] = (motion[9, :, :] + motion[12, :, :]) / 2

    # rotate 180
    motion[:, horizontal_dim, :] = - motion[:, horizontal_dim, :]
    motion[:, vertical_dim, :] = - motion[:, vertical_dim, :]

    motion = motion * unit

    return motion

def trans_motion3d(motion3d, local3d=None, unit=128):
    # orthonormal projection

    motion3d = motion3d * unit

    # neck and mid-hip
    motion3d[1, :, :] = (motion3d[2, :, :] + motion3d[5, :, :]) / 2
    motion3d[8, :, :] = (motion3d[9, :, :] + motion3d[12, :, :]) / 2

    if local3d is not None:
        motion_proj = local3d[[0, 2], :] @ motion3d  # (15, 2, 64)
    else:
        motion_proj = motion3d[:, [0, 2], :]  # (15, 2, 64)

    motion_proj[:, 1, :] = - motion_proj[:, 1, :]

    motion_proj = trans_motion2d(motion_proj)

    return motion_proj


def trans_motion2d(motion2d):
    # subtract centers to local coordinates
    centers = motion2d[8, :, :]
    motion_proj = motion2d - centers  # 得到局部坐标，不带速度信息

    # adding velocity
    velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)  # 计算速度信息
    motion_proj = np.r_[motion_proj[:8], motion_proj[9:], velocity]  # 把速度信息插入末尾

    return motion_proj


def calculate_angles(sequence):
    """
    Fully vectorized calculation of 8 specific angles for each frame in the sequence.
    Sequence is a numpy array of shape (15, 2, T).
    """
    _, _, T = sequence.shape
    zero_array = np.zeros((1, 2, T))

    # Concatenate the original array and the zero array
    sequence_ = np.concatenate((sequence, zero_array), axis=0)

    # Function to calculate angles using cross product and dot product
    def angle_between_vectors(v1, v2):
        dot = np.sum(v1 * v2, axis=1)
        det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return np.arctan2(det, dot)

    # Define key point indices for angle calculation
    indices = np.array([
        [1, 2, 3],  # Right shoulder (Head, Neck, RightArm)
        [1, 5, 6],  # Left shoulder (Head, Neck, LeftArm)
        [2, 3, 4],  # Right elbow (Neck, RightArm, RightForeArm)
        [5, 6, 7],  # Left elbow (Neck, LeftArm, LeftForeArm)
        [15, 8, 9],  # Right hip (Hips, RightUpLeg, RightLeg)
        [15, 11, 12],  # Left hip (Hips, LeftUpLeg, LeftLeg)
        [8, 9, 10],  # Right knee (RightUpLeg, RightLeg, RightFoot)
        [11, 12, 13]  # Left knee (LeftUpLeg, LeftLeg, LeftFoot)
    ])

    # Extract the coordinates for the key points
    p1, p2, p3 = sequence_[indices[:, 0]], sequence_[indices[:, 1]], sequence_[indices[:, 2]]

    # Compute vectors
    v1 = p2 - p1
    v2 = p3 - p2

    # Calculate angles for all key points across all frames
    angles = np.degrees(angle_between_vectors(v1, v2))

    return angles


def trans_motion_inv(motion, sx=256, sy=256, velocity=None):
    if velocity is None:
        velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    # 通过累加速度信息来计算中臀部随时间的位置。这实际上是在将速度积分，以恢复出中臀部的全局位置
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += np.array([[sx], [sy]])
    # 每个关节点位置都被调整回了全局坐标系统
    return motion_inv + centers.reshape((1, 2, -1))


def normalize_motion(motion, mean_pose, std_pose):
    """
    :param motion: (J, 2, T)
    :param mean_pose: (J, 2)
    :param std_pose: (J, 2)
    :return:
    """
    return (motion - mean_pose[:, :, np.newaxis]) / std_pose[:, :, np.newaxis]


def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]


def preprocess_motion2d(motion, mean_pose, std_pose):
    motion_trans = normalize_motion(trans_motion2d(motion), mean_pose, std_pose)
    motion_trans = motion_trans.reshape((-1, motion_trans.shape[-1]))
    return torch.Tensor(motion_trans).unsqueeze(0)


def postprocess_motion2d(motion, mean_pose, std_pose, sx=256, sy=256):
    motion = motion.detach().cpu().numpy()[0].reshape(-1, 2, motion.shape[-1])
    motion = trans_motion_inv(normalize_motion_inv(motion, mean_pose, std_pose), sx, sy)
    return motion


def get_local3d(motion3d, angles=None):
    """
    Get the unit vectors for local rectangular coordinates for given 3D motion
    :param motion3d: numpy array. 3D motion from 3D joints positions, shape (nr_joints, 3, nr_frames).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return: numpy array. unit vectors for local rectangular coordinates's , shape (3, 3).
    """
    # 2 RightArm 5 LeftArm 9 RightUpLeg 12 LeftUpLeg
    horizontal = (motion3d[2] - motion3d[5] + motion3d[9] - motion3d[12]) / 2
    horizontal = np.mean(horizontal, axis=1)
    horizontal = horizontal / np.linalg.norm(horizontal)
    local_z = np.array([0, 0, 1])
    local_y = np.cross(horizontal, local_z)  # bugs!!!, horizontal and local_Z may not be perpendicular
    local_y = local_y / np.linalg.norm(local_y)
    local_x = np.cross(local_y, local_z)
    local = np.stack([local_x, local_y, local_z], axis=0)

    if angles is not None:
        local = rotate_coordinates(local, angles)

    return local


def rotate_coordinates(local3d, angles):
    """
    Rotate local rectangular coordinates from given view_angles.

    :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return:
    """
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    x = local3d[0]
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x.T @ mat33_z
    return local3d


def rotation_matrix_along_axis(x, angle):
    cx = np.cos(angle)
    sx = np.sin(angle)
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)
    return mat33_x


def openpose2motion(json_dir, scale=1.0, smooth=True, max_frame=None):
    json_files = sorted(os.listdir(json_dir))
    length = max_frame if max_frame is not None else len(json_files) // 8 * 8
    json_files = json_files[:length]
    json_files = [os.path.join(json_dir, x) for x in json_files]

    motion = []
    for path in json_files:
        with open(path) as f:
            jointDict = json.load(f)
            joint = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
            if len(motion) > 0:
                joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
            motion.append(joint)

    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]

    motion = np.stack(motion, axis=2)
    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion


def get_foot_vel(batch_motion, foot_idx):
    return batch_motion[:, foot_idx, 1:] - batch_motion[:, foot_idx, :-1] + batch_motion[:, -2:, 1:].repeat(1, 2, 1)
