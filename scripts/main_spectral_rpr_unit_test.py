import transforms3d.quaternions

import common_prep
import pandas as pd
import argparse
import numpy as np
from util.spectral_sync_utils import calc_relative_poses, retrieve_abs_trans_and_rot_mat, gen_exp_rel_trans_mat, \
    spectral_sync_rot, spectral_sync_trans
from util.pose_utils import pose_err
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


def add_noise(poses: np.array, trans_sigma: float, rot_sigma: float) -> np.array:
    """
    Generates a noisy poses based on the given input poses and requested transitional and rotational noise levels
    :param poses: Input Nx7 poses to modify
    :param trans_sigma: Translational noise level [meters]
    :param rot_sigma: Rotational noise level [degrees]
    :return: Noisy Nx7 poses
    """
    noisy_poses = np.zeros(poses.shape) + poses
    num_imgs = poses.shape[0]

    # Adding translational noise by shifting the given XYZ coordinates
    noisy_poses[:, :3] += np.random.random(size=(num_imgs, 3)) * trans_sigma

    # Adding rotational noise by rotating the given quaternion
    if rot_sigma != 0.0:
        for i in range(num_imgs):
            ai, aj, ak = np.random.random(size=3) * rot_sigma
            rot_mat = transforms3d.euler.euler2mat(ai, aj, ak)
            pose_rot_mat = transforms3d.quaternions.quat2mat(poses[i, 3:])
            noisy_rot_mat = np.dot(rot_mat, pose_rot_mat)
            noisy_poses[i, 3:] = transforms3d.quaternions.mat2quat(noisy_rot_mat)
    return noisy_poses


def plot_pose_err_stats(noise_range: float, applied_noise: np.array, err_vect: list, num_bins: int, title: str,
                        xlabel: str, ylabel: str) -> None:
    """
    Plots the pose (translation or rotation) errors histogram, based on the applied noise level
    :param noise_range: The range of noise applied during the analysis - [0.0, noise_range]
    :param applied_noise: The actual random noise levels applied during the analysis
    :param err_vect: The pose error to analyze
    :param num_bins: Number of bins to use for the output histogram
    :param title: Output plot title
    :param xlabel: Output plot x-axis label
    :param ylabel: Output plot y-axis label
    :return: None
    """
    fig = go.Figure()
    count, bins = np.histogram(applied_noise, bins=num_bins)
    bin_noise_level = np.linspace(0.0,noise_range, num=num_bins)
    for i in range(num_bins - 1):
        # idx = np.where((bins[i] < noise_rot_level) & (noise_rot_level < bins[i + 1]))[0]
        idx = np.where(applied_noise < bins[i + 1])[0]
        idx = idx[np.where(idx < len(err_vect))[0]]
        fig.add_trace(go.Box(y=np.array(err_vect)[idx], name='{:.6f}'.format(bin_noise_level[i + 1]), boxpoints=False))
    fig.update_layout(title_text=title, xaxis_title=xlabel, yaxis_title=ylabel)
    fig.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file', help='Path to the dataset file')
    arg_parser.add_argument('num_of_rel_imgs',
                            help='The number of images to use in the Spectral Relative Pose estimator',
                            type=int, default=2)
    arg_parser.add_argument('--noise_trans_level', help='The level of translation uncertainty', type=float, default=0.0)
    arg_parser.add_argument('--noise_rot_level', help='The level of rotation uncertainty', type=float, default=0.0)
    arg_parser.add_argument('--num_itr', help='The number of uncertainty tests to run', type=int, default=1)
    args = arg_parser.parse_args()

    # Extracting the ground-truth poses
    scene_data = pd.read_csv(args.input_file)

    position_errors = []
    orient_errors = []

    if args.num_itr == 1:
        noise_trans_level = [args.noise_trans_level]
        noise_rot_level = [args.noise_rot_level]
        verbose = True
    else:
        noise_trans_level = np.linspace(0.0, args.noise_trans_level, num=args.num_itr)
        noise_rot_level = np.linspace(0.0, args.noise_rot_level, num=args.num_itr)
        verbose = False

    for itr in tqdm(range(args.num_itr)):

        # Loading a random set of query image with its relative poses
        num_of_imgs = args.num_of_rel_imgs + 1
        start_idx = np.random.randint(scene_data.shape[0] - (args.num_of_rel_imgs + 1))
        gt_poses = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()[start_idx:(start_idx + num_of_imgs)]

        noisy_gt_poses = add_noise(gt_poses, noise_trans_level[itr], noise_rot_level[itr])

        # Calculating the relative translation and relative angular rotation
        rel_trans_mat, rel_rot_mat = calc_relative_poses(noisy_gt_poses)
        exp_rel_trans_mat = gen_exp_rel_trans_mat(rel_trans_mat)

        abs_trans_mat, abs_rot_mat = retrieve_abs_trans_and_rot_mat(gt_poses)
        exp_abs_trans_mat = np.exp(abs_trans_mat)

        # Run spectral sync
        est_abs_trans_mat = spectral_sync_trans(exp_rel_trans_mat, exp_abs_trans_mat[1:, :])
        est_abs_rot_mat = spectral_sync_rot(rel_rot_mat, abs_rot_mat[3:, :])

        if est_abs_trans_mat is not None and est_abs_rot_mat is not None:
            trans_pos_est_err = np.mean(np.linalg.norm(est_abs_trans_mat - abs_trans_mat))
            rot_pos_est_err = np.mean(np.linalg.norm(est_abs_rot_mat - abs_rot_mat))

            est_abs_quat_mat = np.zeros((num_of_imgs, 4))
            for i in range(num_of_imgs):
                est_abs_quat_mat[i, :] = transforms3d.quaternions.mat2quat(est_abs_rot_mat[(3 * i):(3 * (i + 1)), :])
            pos_err, orient_err = pose_err(np.hstack((est_abs_trans_mat, est_abs_quat_mat)), gt_poses)

            if verbose:
                print('Translation estimation err: {}'.format(trans_pos_est_err))
                print('Rotation estimation err: {}'.format(rot_pos_est_err))
                print("Camera pose estimation error: {:.2f}[m], {:.2f}[deg]".format(pos_err.mean().item(),
                                                                                    orient_err.mean().item()))

            position_errors.append(pos_err.mean().item())
            orient_errors.append(orient_err.mean().item())

    if args.num_itr > 1:
        num_bins = 10

        plot_pose_err_stats(args.noise_trans_level, noise_trans_level, position_errors, num_bins,
                            title='Translation Estimation Error - {}-NN'.format(num_of_imgs - 1),
                            xlabel='Noise level [meters]', ylabel='Translation Estimation Error [meters]')

        plot_pose_err_stats(args.noise_rot_level, noise_rot_level, orient_errors, num_bins,
                            title='Rotation Estimation Error - {}-NN'.format(num_of_imgs - 1),
                            xlabel='Noise level [degrees]', ylabel='Rotation Estimation Error [degrees]')
