import transforms3d.quaternions

import common_prep
import pandas as pd
import argparse
import numpy as np
from util.spectral_sync_utils import calc_relative_poses, retrieve_abs_trans_and_rot_mat, gen_exp_rel_trans_mat, \
    spectral_sync_rot, spectral_sync_trans
from util.pose_utils import pose_err
import plotly.express as px


def add_noise(poses: np.array, trans_sigma: float, rot_sigma: float) -> np.array:
    noisy_poses = np.zeros(poses.shape) + poses
    num_imgs = poses.shape[0]
    noisy_poses[:, :3] += np.random.random(size=(num_imgs, 3)) * trans_sigma

    if rot_sigma != 0.0:
        for i in range(num_imgs):
            # quat_noise = np.random.random(size=3) * rot_sigma * np.pi / 180.0
            ai, aj, ak = np.random.random(size=3) * rot_sigma
            rot_mat = transforms3d.euler.euler2mat(ai, aj, ak)
            pose_rot_mat = transforms3d.quaternions.quat2mat(poses[i, 3:])
            noisy_rot_mat = np.dot(rot_mat, pose_rot_mat)
            noisy_poses[i, 3:] = transforms3d.quaternions.mat2quat(noisy_rot_mat)
    return noisy_poses


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file', help='Path to the dataset file')
    arg_parser.add_argument('num_of_rel_imgs',
                            help='The number of images to use in the Spectral Relative Pose estimator',
                            type=int, default=2)
    arg_parser.add_argument('--noise_trans_level', help='The level of translation uncertainty', type=float, default=0.0)
    arg_parser.add_argument('--noise_rot_level', help='The level of rotation uncertainty', type=float, default=0.0)
    arg_parser.add_argument('--noise_itr', help='The number of uncertainty tests to run', type=int, default=1)
    args = arg_parser.parse_args()

    # Extracting the ground-truth poses
    scene_data = pd.read_csv(args.input_file)

    position_errors = []
    orient_errors = []

    for itr in range(args.noise_itr):

        # Loading a random set of query image with its relative poses
        num_of_imgs = args.num_of_rel_imgs + 1
        start_idx = np.random.randint(scene_data.shape[0] - (args.num_of_rel_imgs + 1))
        gt_poses = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()[start_idx:(start_idx + num_of_imgs)]

        noisy_gt_poses = add_noise(gt_poses, args.noise_trans_level, args.noise_rot_level)

        # Calculating the relative translation and relative angular rotation
        rel_trans_mat, rel_rot_mat = calc_relative_poses(noisy_gt_poses)
        exp_rel_trans_mat = gen_exp_rel_trans_mat(rel_trans_mat)

        abs_trans_mat, abs_rot_mat = retrieve_abs_trans_and_rot_mat(gt_poses)
        exp_abs_trans_mat = np.exp(abs_trans_mat)

        # Run spectral sync
        est_abs_trans_mat = spectral_sync_trans(exp_rel_trans_mat, exp_abs_trans_mat[1:, :])
        est_abs_rot_mat = spectral_sync_rot(rel_rot_mat, abs_rot_mat[3:, :])

        trans_pos_est_err = np.mean(np.linalg.norm(est_abs_trans_mat - abs_trans_mat))
        print('Translation estimation err: {}'.format(trans_pos_est_err))
        rot_pos_est_err = np.mean(np.linalg.norm(est_abs_rot_mat - abs_rot_mat))
        print('Rotation estimation err: {}'.format(rot_pos_est_err))

        est_abs_quat_mat = np.zeros((num_of_imgs, 4))
        for i in range(num_of_imgs):
            est_abs_quat_mat[i, :] = transforms3d.quaternions.mat2quat(est_abs_rot_mat[(3 * i):(3 * (i + 1)), :])
        pos_err, orient_err = pose_err(np.hstack((est_abs_trans_mat, est_abs_quat_mat)), gt_poses)
        print("Camera pose estimation error: {:.2f}[m], {:.2f}[deg]".format(pos_err.mean().item(), orient_err.mean().item()))

        position_errors.append(pos_err.mean().item())
        orient_errors.append(orient_err.mean().item())

    if len(position_errors) > 1:

        fig = px.scatter(x=np.array(noise_levels)[idx], y=np.array(position_errors)[idx],
                      labels={'x': 'Noise Level', 'y': 'Translation Estimation Error'})
        fig.show()

        fig = px.scatter(x=np.array(orient_errors)[idx], y=np.array(position_errors)[idx],
                      labels={'x': 'Noise Level', 'y': 'Rotation Estimation Error'})
        fig.show()
