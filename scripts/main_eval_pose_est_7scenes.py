import common_prep
import argparse
import numpy as np
import logging
from util import utils
import typing
from tqdm import tqdm
from util.spectral_sync_utils import quaternion_to_mat, gen_exp_rel_trans_mat, spectral_sync_rot, spectral_sync_trans, \
    retrieve_abs_trans_and_rot_mat, calc_relative_poses, normalize_quaternion
from util.pose_utils import pose_err
import transforms3d
import pandas as pd

SEVEN_SCENE_NUM_ENTRIES = 85000
SEVEN_SCENE_NUM_REFS = 5


def process_rel_pose_est(rel_est_pose_filepath: str) -> typing.Tuple[np.array, np.array]:
    columns = ['rel_est_q1', 'rel_est_q2', 'rel_est_q3', 'rel_est_q4', 'rel_est_t1', 'rel_est_t2', 'rel_est_t3']
    rel_est_data = pd.read_csv(rel_est_pose_filepath, names=columns, delim_whitespace=True)
    rel_trans = np.float_(rel_est_data[['rel_est_t1', 'rel_est_t2', 'rel_est_t3']].values)
    rel_orientation = np.float_(rel_est_data[['rel_est_q1', 'rel_est_q2', 'rel_est_q3', 'rel_est_q4']].values)

    assert rel_trans.shape[0] == SEVEN_SCENE_NUM_ENTRIES, 'Wrong number of ground-truth entries'

    return rel_trans, rel_orientation


def process_gt_abs_poses(gt_pose_filepath: str) -> typing.Tuple[typing.List, typing.List, np.array, np.array]:
    columns = ['query_path', 'ref_path', 'idx0', 'idx1', 'idx2', 'q_t1', 'q_t2', 'q_t3', 'q_q1', 'q_q2', 'q_q3', 'q_q4',
               'ref_t1', 'ref_t2', 'ref_t3', 'ref_q1', 'ref_q2', 'ref_q3', 'ref_q4']
    gt_data = pd.read_csv(gt_pose_filepath, names=columns, delim_whitespace=True)
    return gt_data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("rel_pose_file", help="path to the relative poses for the 7-Scenes dataset")
    arg_parser.add_argument("seven_scenes_gt_file", help="path to the ground-truth poses")
    args = arg_parser.parse_args()

    utils.init_logger()
    logging.info("Processing 7-Scene relative estimations from - {}".format(args.rel_pose_file))

    # Retrieving the ground-truth poses
    gt_data = process_gt_abs_poses(args.seven_scenes_gt_file)

    # Retrieving the input relative pose estimation
    rel_est_trans, rel_est_orientation = process_rel_pose_est(args.rel_pose_file)

    position_errors = []
    orientation_errors = []
    gt_poses = np.zeros(((SEVEN_SCENE_NUM_REFS + 1), 7))

    for idx in tqdm(range(0, SEVEN_SCENE_NUM_ENTRIES, SEVEN_SCENE_NUM_REFS), desc='Calculating absolute poses'):
        # ====================================================
        # Retrieving Ground-Truth Poses
        # ====================================================
        # Retrieving the absolute pose of the query image
        gt_poses[0, :] = gt_data[['q_t1', 'q_t2', 'q_t3', 'q_q1', 'q_q2', 'q_q3', 'q_q4']].iloc[idx].values
        # Retrieving the absolute pose of the reference images
        gt_poses[1:, :] = gt_data[['ref_t1', 'ref_t2', 'ref_t3', 'ref_q1', 'ref_q2', 'ref_q3', 'ref_q4']].iloc[
                          idx:(idx + SEVEN_SCENE_NUM_REFS)].values

        # Creating the absolute poses matrices
        abs_trans_mat, abs_rot_mat = retrieve_abs_trans_and_rot_mat(gt_poses)
        exp_abs_trans_mat = np.exp(abs_trans_mat)

        # ====================================================
        # Assembling Relative Poses Matrices
        # ====================================================
        # Calculating the relative poses between the reference images
        rel_trans_mat, rel_rot_mat = calc_relative_poses(gt_poses[1:, :])

        # Adding the relative poses between the reference images and the query image (network's relative estimation)
        rel_trans_mat = np.hstack([np.ones((SEVEN_SCENE_NUM_REFS, 1, 3)), rel_trans_mat])
        rel_trans_mat = np.vstack([np.ones((1, SEVEN_SCENE_NUM_REFS + 1, 3)), rel_trans_mat])
        for i in range(3):
            rel_trans_mat[0, 1:, i] = rel_est_trans[idx:(idx + SEVEN_SCENE_NUM_REFS), :][:, i]
            rel_trans_mat[1:, 0, i] = -1 * rel_est_trans[idx:(idx + SEVEN_SCENE_NUM_REFS), :][:, i]

        rel_rot_mat = np.hstack([np.ones((SEVEN_SCENE_NUM_REFS * 3, 3)), rel_rot_mat])
        rel_rot_mat = np.vstack([np.ones((3, (3 * (SEVEN_SCENE_NUM_REFS + 1)))), rel_rot_mat])
        for i in range(1, (SEVEN_SCENE_NUM_REFS + 1)):
            quat = normalize_quaternion(rel_est_orientation[idx + (i - 1), :])
            rel_rot_mat[:3, (3 * i):(3 * (i + 1))] = quaternion_to_mat(quat)
            rel_rot_mat[(3 * i):(3 * (i + 1)), :3] = np.linalg.inv(quaternion_to_mat(quat))

        rel_rot_mat[:3, :3] = np.eye(3)

        exp_rel_trans_mat = gen_exp_rel_trans_mat(rel_trans_mat)

        # gt_rel_mat = np.zeros((SEVEN_SCENE_NUM_REFS, 7))
        # for i in range(1, SEVEN_SCENE_NUM_REFS + 1):
        #     gt_rel_mat[i - 1, :3] = calc_rel_translation(gt_poses[0, :3], gt_poses[i, :3])
        #     gt_rel_mat[i - 1, 3:] = calc_rel_rot_quat(gt_poses[0, 3:], gt_poses[i, 3:])
        # rel_pos_err, rel_orient_err = pose_err(np.hstack((rel_est_trans[idx:(idx + SEVEN_SCENE_NUM_REFS), :],
        #                                                   rel_est_orientation[idx:(idx + SEVEN_SCENE_NUM_REFS), :])),
        #                                       gt_rel_mat)
        # print("Relative Pose Error: {:.2f}[m], {:.2f}[deg]".format(rel_pos_err.mean().item(), rel_orient_err.mean().item()))

        # ====================================================
        # Run spectral sync
        # ====================================================
        est_abs_trans_mat = spectral_sync_trans(exp_rel_trans_mat, exp_abs_trans_mat)
        est_abs_rot_mat = spectral_sync_rot(rel_rot_mat, abs_rot_mat)

        if est_abs_trans_mat is not None and est_abs_rot_mat is not None:
            est_orientation = transforms3d.quaternions.mat2quat(est_abs_rot_mat[:3, :])
            pos_err, orient_err = pose_err(np.hstack((est_abs_trans_mat[0], est_orientation)), gt_poses[0, :])

            position_errors.append(pos_err.item())
            orientation_errors.append(orient_err.item())

    # ====================================================
    # Evaluation
    # ====================================================
    scene_ranges = {'chess': [6000, 8000],
                    'fire': [0, 2000],
                    'heads': [8000, 9000],
                    'office': [2000, 6000],
                    'pumpkin': [15000, 17000],
                    'redkitchen': [10000, 15000],
                    'stairs': [9000, 10000]
                    }
    scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    trans_pos_est_err = []
    rot_pos_est_err = []
    for s in scenes:
        idx_start = scene_ranges[s][0]
        idx_end = scene_ranges[s][1]
        trans_pos_est_err.append(np.median(position_errors[idx_start:idx_end]))
        rot_pos_est_err.append(np.median(orientation_errors[idx_start:idx_end]))
        print("Scene {}: {:.2f}[m], {:.2f}[deg]".format(s, trans_pos_est_err[-1], rot_pos_est_err[-1]))
    print("Average: {:.2f}[m], {:.2f}[deg]".format(np.mean(trans_pos_est_err), np.mean(rot_pos_est_err)))