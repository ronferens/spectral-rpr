import common_prep
import pandas as pd
import argparse
import numpy as np
from util.spectral_sync_utils import calc_relative_poses, retrieve_abs_trans_and_rot_mat, gen_exp_rel_trans_mat, \
    spectral_sync_rot, spectral_sync_trans

# init_notebook_mode(connected=False)
# cf.go_offline()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_file', help='Path to the dataset file')
    arg_parser.add_argument('num_of_rel_imgs',
                            help='The number of images to use in the Spectral Relative Pose estimator',
                            type=int, default=2)
    arg_parser.add_argument('--noise_level', help='The level of location uncertainty', type=float, default=0.0)
    args = arg_parser.parse_args()

    # Extracting the ground-truth poses
    scene_data = pd.read_csv(args.input_file)

    # Loading a random set of query image with its relative poses
    num_of_imgs = args.num_of_rel_imgs + 1
    start_idx = np.random.randint(scene_data.shape[0] - (args.num_of_rel_imgs + 1))
    gt_poses = scene_data[['t1', 't2', 't3', 'q1', 'q2', 'q3', 'q4']].to_numpy()[start_idx:(start_idx + num_of_imgs)]

    # Calculating the relative translation and relative angular rotation
    rel_trans_mat, rel_rot_mat = calc_relative_poses(gt_poses)
    exp_rel_trans_mat = gen_exp_rel_trans_mat(rel_trans_mat)

    abs_trans_mat, abs_rot_mat = retrieve_abs_trans_and_rot_mat(gt_poses)
    exp_abs_trans_mat = np.exp(abs_trans_mat)

    # Run spectral sync
    est_abs_trans_mat = spectral_sync_trans(exp_rel_trans_mat, exp_abs_trans_mat[1:, :])
    est_abs_rot_mat = spectral_sync_rot(rel_rot_mat, abs_rot_mat[3:, :])

    # Adding noise
    # mu, sigma = 0, 0.0001  # mean and standard deviation
    # noise_mat = np.random.normal(mu, sigma, size=rel_rot_mat.shape)
    # noise_mat_symm = (noise_mat + noise_mat.T) / 2.0
    # rel_rot_mat += noise_mat_symm

    trans_pos_est_err = np.mean(np.linalg.norm(est_abs_trans_mat - abs_trans_mat))
    print('Translation estimation err: {}'.format(trans_pos_est_err))
    rot_pos_est_err = np.mean(np.linalg.norm(est_abs_rot_mat - abs_rot_mat))
    print('Rotation estimation err: {}'.format(rot_pos_est_err))
