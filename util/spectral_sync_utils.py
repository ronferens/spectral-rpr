import math

import numpy as np
import typing
import transforms3d as t3d


def normalize_quaternion(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if mag2 > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)


def quaternion_to_mat(q: t3d.quaternions) -> np.array:
    """
    Calculate rotation matrix corresponding to quaternion
    :param q: A 4-element quaternion array
    :return: A 3x3 rotation matrix
    """
    return t3d.quaternions.quat2mat(normalize_quaternion(q))


def calc_rel_rot_mat(rot_mat_i: np.array, rot_mat_j: np.array) -> np.array:
    """
    Calculates the relative rotation matrix between the 1st and 2nd rotation matrices
    :param rot_mat_i: 1st 3x3 rotation matrix
    :param rot_mat_j: 2nd 3x3 rotation matrix
    :return: A 3x3 matrix relative rotation matrix (rotating from j to i)
    """
    rot_mat_j_inv = np.linalg.inv(rot_mat_j)
    rel_rot_mat_ij = np.dot(rot_mat_i, rot_mat_j_inv)
    return rel_rot_mat_ij


def calc_rel_rot_quat(rot_quat_i: np.array, rot_quat_j: np.array) -> np.array:
    """
    Calculates the relative rotation matrix between the 1st and 2nd rotation matrices
    :param rot_quat_i: 1st quaternion
    :param rot_quat_j: 2nd quaternion
    :return: A relative rotation quaternion (rotating from j to i)
    """
    rot_mat_i = quaternion_to_mat(rot_quat_i)
    rot_mat_j = quaternion_to_mat(rot_quat_j)
    rel_rot_mat_ij = calc_rel_rot_mat(rot_mat_i, rot_mat_j)
    rel_rot_quat_ij = t3d.quaternions.mat2quat(rel_rot_mat_ij)
    return rel_rot_quat_ij


def calc_rel_translation(trans_i: np.array, trans_j: np.array):
    """
    Calculate the relative translation between the 1st and 2nd positions
    :param trans_i: 1st absolute location, 3-elements array pose
    :param trans_j: 2nd absolute location, 3-elements array pose
    :return: A 3-elements array relative pose (displacement from j to i)
    """
    rel_trans_ij = trans_i - trans_j
    return rel_trans_ij


def assign_relative_poses(rel_est_trans: np.array, rel_est_orientation: np.array) -> typing.Tuple[np.array, np.array]:
    """
    Assign the given relative translation and angular rotation
    :param rel_est_trans: An Nx3 matrix containing relative translation estimation
    :param rel_est_orientation: An Nx3 matrix containing relative angular estimation
    :return: An Nx3 relative translation matrix and (NX3)X3 relative angular rotation matrix
    """
    num_poses = rel_est_trans.shape[0] + 1
    rel_trans_mat = np.zeros((num_poses, num_poses, 3))
    rel_rot_mat = np.zeros((3 * num_poses, 3 * num_poses))

    # Generating the relative translation and relative angular rotation
    for i in range(num_poses):
        for j in range(i, num_poses):
            if i == j:
                rel_trans_mat[i, j, :] = 1
                rel_rot_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = np.eye(3)
            else:
                # Calculating the relative translation
                rel_trans_mat[i, j, :] = rel_est_trans[(j - 1), :]
                rel_trans_mat[j, i, :] = -1 * rel_est_trans[(j - 1), :]

                # Calculating the relative rotation matrices
                quat_query_to_j = normalize_quaternion(rel_est_orientation[(j - 1), :])
                rel_rot_mat_query_to_j = quaternion_to_mat(quat_query_to_j)
                rel_rot_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = rel_rot_mat_query_to_j
                rel_rot_mat[(3 * j):(3 * (j + 1)), (3 * i):(3 * (i + 1))] = np.linalg.inv(rel_rot_mat_query_to_j)

    return rel_trans_mat, rel_rot_mat


def calc_relative_poses(abs_poses: np.array) -> typing.Tuple[np.array, np.array]:
    """
    Calculate the relative translation and angular rotation given absolute poses
    :param abs_poses: An Nx7 matrix containing absolute poses: 1:3=Translation, 4:7=Quaternion
    :return: An Nx3 relative translation matrix and (NX3)X3 relative angular rotation matrix
    """
    rel_trans_mat = np.zeros((abs_poses.shape[0], abs_poses.shape[0], 3))
    rel_rot_mat = np.zeros((3 * abs_poses.shape[0], 3 * abs_poses.shape[0]))

    # Generating the relative translation and relative angular rotation
    for i, pi in enumerate(abs_poses):
        for j, pj in enumerate(abs_poses):
            if i == j:
                rel_trans_mat[i, j, :] = 1
                rel_rot_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = np.eye(3)
            else:
                # Calculating the relative translation
                rel_trans_mat[i, j, :] = calc_rel_translation(pi[:3], pj[:3])

                # Calculating the relative rotation matrices
                rot_mat_i = quaternion_to_mat(normalize_quaternion(pi[3:]))
                rot_mat_j = quaternion_to_mat(normalize_quaternion(pj[3:]))
                rel_rot_mat_ij = calc_rel_rot_mat(rot_mat_i, rot_mat_j)
                rel_rot_mat[(3 * i):(3 * (i + 1)), (3 * j):(3 * (j + 1))] = rel_rot_mat_ij

    return rel_trans_mat, rel_rot_mat


def retrieve_abs_trans_and_rot_mat(poses: np.array) -> typing.Tuple[np.array, np.array]:
    """
    Retrieves the absolute translation and rotation matrices of the given poses
    :param poses: An Nx7 matrix containing absolute poses: 1:3=Translation, 4:7=Quaternion
    :return: An Nx3 absolute translation matrix and (NX3)X3 absolute rotation matrix
    """
    num_of_poses = poses.shape[0]
    abs_trans_mat = np.zeros((num_of_poses, 3))
    abs_rot_mat = np.zeros((num_of_poses * 3, 3))
    for i, p in enumerate(poses):
        abs_trans_mat[i, :] = p[:3]
        abs_rot_mat[(3 * i):(3 * (i + 1)), :] = quaternion_to_mat(normalize_quaternion(p[3:]))
    return abs_trans_mat, abs_rot_mat


def gen_exp_rel_trans_mat(rel_trans_mat: np.array) -> np.array:
    """
    Generate the exponential relative translation matrix
    :param rel_trans_mat: An NxN relative translation matrix
    :return: An NxN exponential relative translation matrix
    """
    num_dims = rel_trans_mat.shape[2]
    exp_rel_trans_mat = np.exp(rel_trans_mat)

    # Setting the diagonal for each dimension to be `1`
    for d in range(num_dims):
        np.fill_diagonal(exp_rel_trans_mat[:, :, d], 1)

    return exp_rel_trans_mat


def spectral_sync_trans(exp_rel_trans_mat: np.array, exp_abs_trans_mat_ref: np.array) -> np.array:
    """
    Calculate the absolute translation using spectral synchronization
    :param exp_rel_trans_mat: An NxN exponential relative translation matrix
    :param exp_abs_trans_mat_ref: An NX3 exponential absolute translation matrix of the reference images
    :return: An NxN matrix estimated absolute translation
    """
    num_imgs = exp_rel_trans_mat.shape[0]
    num_dims = exp_rel_trans_mat.shape[2]
    est_abs_trans = np.zeros((num_imgs, num_dims))

    for d in range(num_dims):
        # (1) Extract the eigenvectors and eigenvalues of the relative translation matrix
        eig_vals, eig_vects = np.linalg.eig(exp_rel_trans_mat[:, :, d])
        eig_vals = np.real(eig_vals)
        v = np.real(eig_vects[:, np.argmin(np.abs(eig_vals - num_imgs))])
        v = np.abs(v)

        # (2) Finding the eigenvector scale based on the know poses of the reference images
        v *= np.mean(exp_abs_trans_mat_ref[1:, d] / v[1:])

        # (3) Calculate the estimated absolute translation based on the extracted eigenvalue
        est_abs_trans[:, d] = np.log(v)

    return est_abs_trans


def spectral_sync_rot(rel_rot_mat: np.array, abs_rot_mat_ref: np.array) -> np.array:
    """
    Calculate the absolute orientation using spectral synchronization
    :param rel_rot_mat: An 3Nx3N relative angular rotation matrix
    :param abs_rot_mat_ref: An 3NX3 absolute angular rotation matrices of the reference images
    :return: An 3NxN absolute angular rotation matrices (of both the query and reference images)
    """
    num_imgs = rel_rot_mat.shape[0] // 3

    # (1) Extract the eigenvectors and eigenvalues of the relative rotation matrix
    eig_vals, eig_vects = np.linalg.eig(rel_rot_mat)
    eig_vals = np.real(eig_vals)
    v = np.real(eig_vects[:, np.argsort(np.abs(eig_vals - num_imgs))])[:, :3]

    # (2) Finding the linear combination of the calculated ev using the known ground-truth
    est_abs_rot_mat = np.zeros(abs_rot_mat_ref.shape)
    for i in range(3):
        res = np.linalg.lstsq(v[(3 * (i + 1)):, :],
                              abs_rot_mat_ref[(3 * (i + 1)):, i], rcond=None)[0]
        est_abs_rot_mat[:, i] = np.dot(v, res)

    u, s, vh = np.linalg.svd(est_abs_rot_mat[:3, :3], full_matrices=True)
    smat = np.diag(s)
    est_abs_rot_mat[:3, :3] = np.dot(u, np.dot(smat, vh))

    return est_abs_rot_mat
