import numpy as np
import transforms3d as t3d
import torch
import torch.nn.functional as F

def calc_rel_rot_mat(rot_mat1, rot_mat2):
    """
    Calculates the relative rotation matrix
    """
    rot_mat2_inv = np.linalg.inv(rot_mat2)
    rel_rot_mat_12 = np.dot(rot_mat1, rot_mat2_inv)
    return rel_rot_mat_12


def calc_rel_trans(trans1, trans2):
    """
    Calculate the relative translation
    """
    rel_trans_12 = trans1 - trans2
    return rel_trans_12


def quat_to_mat(q):
    return t3d.quaternions.quat2mat(q / np.linalg.norm(q))


def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    est_pose = torch.Tensor(est_pose)
    gt_pose = torch.Tensor(gt_pose)

    if len(est_pose.shape) == 1:
        est_pose = est_pose.reshape(1, -1)
    if len(gt_pose.shape) == 1:
        gt_pose = gt_pose.reshape(1, -1)

    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.nan_to_num(torch.acos(torch.abs(inner_prod))) * 180 / np.pi
    return posit_err, orient_err

