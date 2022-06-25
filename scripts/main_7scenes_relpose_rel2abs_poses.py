from os.path import join, exists
import re
import pandas as pd
import typing
from util.spectral_sync_utils import calc_relative_poses, retrieve_abs_trans_and_rot_mat, gen_exp_rel_trans_mat, \
    spectral_sync_rot, spectral_sync_trans
import numpy as np
from tqdm import tqdm

SEVEN_SCENES_NUM_ENTRIES = 85000
NUM_NN = 5


def str_list_to_floats(strlist: typing.List) -> typing.List:
    return [float(f) for f in strlist]


def merge_relpose_gt_and_est_results(gt_filepath: str, est_filepath: str) -> pd.DataFrame:
    data = []
    gt_file = open(gt_filepath, 'r')
    est_file = open(est_filepath, 'r')

    for i in tqdm(range(SEVEN_SCENES_NUM_ENTRIES), desc='Preprocessing 7-Scene GT and estimation data'):
        # Get next line from file
        gt_line = gt_file.readline()
        est_line = est_file.readline()

        # if line is empty - end of file is reached
        if not gt_line or not est_line:
            break
        gt_m = re.match('(.+\.png)\s+(.+\.png)\s\d\s\d\s\d\s(.+)', gt_line)
        if gt_m is not None:
            # Parsing GT data
            file_q = gt_m.group(1)
            file_db = gt_m.group(2)
            gt_poses = gt_m.group(3).split(' ')
            translation_gt_q = str_list_to_floats(gt_poses[:3])
            orientation_gt_q = str_list_to_floats(gt_poses[3:7])
            translation_gt_db = str_list_to_floats(gt_poses[7:10])
            orientation_gt_db = str_list_to_floats(gt_poses[10:])

            # Parsing RelPose estimation
            est_rel_pose = est_line.strip().split(' ')
            rel_translation = str_list_to_floats(est_rel_pose[:3])
            rel_orientation = str_list_to_floats(est_rel_pose[3:])

            data.append([file_q, file_db,
                         translation_gt_q[0], translation_gt_q[1], translation_gt_q[2],
                         orientation_gt_q[0], orientation_gt_q[1], orientation_gt_q[2], orientation_gt_q[3],
                         translation_gt_db[0], translation_gt_db[1], translation_gt_db[2],
                         orientation_gt_db[0], orientation_gt_db[1], orientation_gt_db[2], orientation_gt_db[3],
                         rel_translation[0], rel_translation[1], rel_translation[2],
                         rel_orientation[0], rel_orientation[1], rel_orientation[2], rel_orientation[3]])

    gt_file.close()
    est_file.close()

    df = pd.DataFrame(data, columns=['file_q', 'file_db',
                                     'q_t1', 'q_t2', 'q_t3', 'q_q1', 'q_q2', 'q_q3', 'q_q4',
                                     'db_t1', 'db_t2', 'db_t3', 'db_q1', 'db_q2', 'db_q3', 'db_q4',
                                     'rel_t1', 'rel_t2', 'rel_t3', 'rel_q1', 'rel_q2', 'rel_q3', 'rel_q4'])
    return df


path_dir = '/home/ronf/Documents/PhD/Research/[NNNet]RelPoseNet-main/assets/data/'
gt_file = 'NN_7scenes.txt'
est_file = 'est_rel_poses.txt'
relpose_data_file = 'relpose_data.csv'

# Merging 7Scenes GT data and RelPose relative poses estimation
if not exists(join(path_dir, relpose_data_file)):
    relpose_data = merge_relpose_gt_and_est_results(join(path_dir, gt_file), join(path_dir, est_file))
else:
    relpose_data = pd.read_csv(join(path_dir, relpose_data_file))
assert relpose_data.shape[0] == SEVEN_SCENES_NUM_ENTRIES

for i in range(0, SEVEN_SCENES_NUM_ENTRIES, NUM_NN):
    poses_data = relpose_data.loc[i:i+NUM_NN, :]

    abs_poses = poses_data['translation_gt_q']

    # Calculating the relative translation and relative angular rotation
    rel_trans_mat, rel_rot_mat = calc_relative_poses(poses_data)
    exp_rel_trans_mat = gen_exp_rel_trans_mat(rel_trans_mat)

    abs_trans_mat, abs_rot_mat = retrieve_abs_trans_and_rot_mat(poses_data)
    exp_abs_trans_mat = np.exp(abs_trans_mat)

    # Run spectral sync
    est_abs_trans_mat = spectral_sync_trans(exp_rel_trans_mat, exp_abs_trans_mat[1:, :])
    est_abs_rot_mat = spectral_sync_rot(rel_rot_mat, abs_rot_mat[3:, :])


