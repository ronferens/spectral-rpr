import common_prep
import argparse
import numpy as np
import logging
from util import utils
import typing
from tqdm import tqdm
from util.spectral_sync_utils import quaternion_to_mat, gen_exp_rel_trans_mat, spectral_sync_rot, spectral_sync_trans, \
    assign_relative_poses, retrieve_abs_trans_and_rot_mat, calc_relative_poses
from util.pose_utils import pose_err
import transforms3d
import pandas as pd

SEVEN_SCENE_NUM_ENTRIES = 85000
SEVEN_SCENE_NUM_REFS = 5


def process_rel_pose_est(rel_est_pose_filepath: str) -> typing.Tuple[np.array, np.array]:
    columns = ['rel_est_t1', 'rel_est_t2', 'rel_est_t3', 'rel_est_q1', 'rel_est_q2', 'rel_est_q3', 'rel_est_q4']
    rel_est_data = pd.read_csv(rel_est_pose_filepath, names=columns, delim_whitespace=True)
    rel_trans = np.float_(rel_est_data[['rel_est_t1', 'rel_est_t2', 'rel_est_t3']].values)
    rel_orientation = np.float_(rel_est_data[['rel_est_q1', 'rel_est_q2', 'rel_est_q3', 'rel_est_q4']].values)

    assert rel_trans.shape[0] == SEVEN_SCENE_NUM_ENTRIES, 'Wrong number of ground-truth entries'

    return rel_trans, rel_orientation


def process_gt_abs_poses(gt_pose_filepath: str) -> typing.Tuple[typing.List, typing.List, np.array, np.array]:
    columns = ['query_path', 'ref_path', 'idx0', 'idx1', 'idx2', 'q_t1', 'q_t2', 'q_t3', 'q_q1', 'q_q2', 'q_q3', 'q_q4',
               'ref_t1', 'ref_t2', 'ref_t3', 'ref_q1', 'ref_q2', 'ref_q3', 'ref_q4']
    gt_data = pd.read_csv(gt_pose_filepath, names=columns, delim_whitespace=True)
    q_imgs = gt_data['query_path']
    r_imgs = gt_data['ref_path']
    q_abs_poses = np.float_(gt_data[['q_t1', 'q_t2', 'q_t3', 'q_q1', 'q_q2', 'q_q3', 'q_q4']].values)
    ref_abs_poses = np.float_(gt_data[['ref_t1', 'ref_t2', 'ref_t3', 'ref_q1', 'ref_q2', 'ref_q3', 'ref_q4']].values)

    assert q_abs_poses.shape[0] == SEVEN_SCENE_NUM_ENTRIES, 'Wrong number of ground-truth entries'

    return q_imgs, r_imgs, q_abs_poses, ref_abs_poses


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("rel_pose_file", help="path to the relative poses for the 7-Scenes dataset")
    arg_parser.add_argument("seven_scenes_gt_file", help="path to the ground-truth poses")
    args = arg_parser.parse_args()

    utils.init_logger()
    logging.info("Processing 7-Scene relative estimations from - {}".format(args.rel_pose_file))

    # Retrieving the ground-truth poses
    query_list, ref_list, query_abs_poses, ref_abs_poses = process_gt_abs_poses(args.seven_scenes_gt_file)

    # Retrieving the input relative pose estimation
    rel_est_trans, rel_est_orientation = process_rel_pose_est(args.rel_pose_file)

    position_errors = []
    orientation_errors = []
    gt_poses = np.zeros(((SEVEN_SCENE_NUM_REFS + 1), 7))

    for idx in range(1):#tqdm(range(0, SEVEN_SCENE_NUM_ENTRIES, SEVEN_SCENE_NUM_REFS), desc='Calculating absolute poses'):
        # ====================================================
        # Retrieving Ground-Truth Poses
        # ====================================================
        # Retrieving the absolute pose of the query image
        gt_poses[0, :] = query_abs_poses[idx, :]
        # Retrieving the absolute pose of the reference images
        gt_poses[1:, :] = ref_abs_poses[idx:(idx + SEVEN_SCENE_NUM_REFS), :]

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
        rel_trans_mat[0, 1:] = rel_est_trans[idx:(idx + SEVEN_SCENE_NUM_REFS), :]
        rel_trans_mat[1:, 0, :] = -1 * rel_est_trans[idx:(idx + SEVEN_SCENE_NUM_REFS), :]

        rel_rot_mat = np.hstack([np.ones((SEVEN_SCENE_NUM_REFS * 3, 3)), rel_rot_mat])
        rel_rot_mat = np.vstack([np.ones((3, (3 * (SEVEN_SCENE_NUM_REFS + 1)))), rel_rot_mat])
        for i in range(1, (SEVEN_SCENE_NUM_REFS + 1)):
            q_to_ref_rel_quaternion = rel_est_orientation[idx + (i - 1), :]
            q_to_ref_rel_quaternion = q_to_ref_rel_quaternion / np.linalg.norm(q_to_ref_rel_quaternion)
            rel_rot_mat[:3, (3 * i):(3 * (i + 1))] = quaternion_to_mat(q_to_ref_rel_quaternion)
            rel_rot_mat[(3 * i):(3 * (i + 1)), :3] = np.linalg.inv(quaternion_to_mat(q_to_ref_rel_quaternion))

        exp_rel_trans_mat = gen_exp_rel_trans_mat(rel_trans_mat)

        # ====================================================
        # Run spectral sync
        # ====================================================
        est_abs_trans_mat = spectral_sync_trans(exp_rel_trans_mat, exp_abs_trans_mat[1:, :])
        est_abs_rot_mat = spectral_sync_rot(rel_rot_mat, abs_rot_mat[3:, :])

        est_orientation = transforms3d.quaternions.mat2quat(est_abs_rot_mat[:3, :])
        pos_err, orient_err = pose_err(np.hstack((est_abs_trans_mat[0], est_orientation)), gt_poses[0, :])

        position_errors.append(pos_err.item())
        orientation_errors.append(orient_err.item())

    # chess = median(err_trans(6001:8000));
    # fire = median(err_trans(1:2000));
    # heads = median(err_trans(8001:9000));
    # office = median(err_trans(2001:6000));
    # pumpkin = median(err_trans(15001:17000));
    # redkitchen = median(err_trans(10001:15000));
    # stairs = median(err_trans(9001:10000))
    scene_ranges = [0, 2000, 6000, 8000, 9000, 10000, 15000, 17000]
    scenes = ['fire', 'office', 'chess', 'heads', 'stairs', 'redkitchen', 'pumpkin']
    for idx, scene in enumerate(scenes):
        if scene_ranges[idx] < len(position_errors):
            trans_pos_est_err = np.mean(position_errors[scene_ranges[idx]:scene_ranges[idx+1]])
            rot_pos_est_err = np.mean(orientation_errors[scene_ranges[idx]:scene_ranges[idx + 1]])
            print("Scene {}: {:.2f}[m], {:.2f}[deg]".format(scene, trans_pos_est_err, rot_pos_est_err))