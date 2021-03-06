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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file', help='Path to the dataset file')
    args = arg_parser.parse_args()

    # Extracting the ground-truth poses
    scene_data = pd.read_csv(args.input_file)

    # Loading a random set of query image with its relative poses
    gt_poses = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()

    # Calculating the relative translation and relative angular rotation
    rel_trans_mat, rel_rot_mat = calc_relative_poses(gt_poses)
    exp_rel_trans_mat = gen_exp_rel_trans_mat(rel_trans_mat)

    abs_trans_mat, abs_rot_mat = retrieve_abs_trans_and_rot_mat(gt_poses)
    exp_abs_trans_mat = np.exp(abs_trans_mat)

    # Run spectral sync
    est_abs_trans_mat = spectral_sync_trans(exp_rel_trans_mat, exp_abs_trans_mat[1:, :])
    est_abs_rot_mat = spectral_sync_rot(rel_rot_mat, abs_rot_mat[3:, :])

    if est_abs_trans_mat is not None and est_abs_rot_mat is not None:
        trans_pos_est_err = np.mean(np.linalg.norm(est_abs_trans_mat - abs_trans_mat))
        rot_pos_est_err = np.mean(np.linalg.norm(est_abs_rot_mat - abs_rot_mat))

        est_abs_quat_mat = transforms3d.quaternions.mat2quat(est_abs_rot_mat)
        pos_err, orient_err = pose_err(np.hstack((est_abs_trans_mat, est_abs_quat_mat)), gt_poses)

        print('Translation estimation err: {}'.format(trans_pos_est_err))
        print('Rotation estimation err: {}'.format(rot_pos_est_err))
        print("Camera pose estimation error: {:.2f}[m], {:.2f}[deg]".format(pos_err.mean().item(),
                                                                            orient_err.mean().item()))
