import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def get_camera_mat(fov=49.13, invert=True):
    """
        求projection矩阵，透视投影
    """
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    # 计算得出r,l,t,b
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    # projection矩阵
    mat = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)
    # 是否求逆
    if invert:
        mat = torch.inverse(mat)
    return mat


def get_random_pose(range_u, range_v, range_radius, batch_size=32,
                    invert=False):
    """
        在范围内任取u,v 生成world_matrix（LookAt^-1）
    """
    loc = sample_on_sphere(range_u, range_v, size=(batch_size))
    radius = range_radius[0] + \
        torch.rand(batch_size) * (range_radius[1] - range_radius[0])
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_middle_pose(range_u, range_v, range_radius, batch_size=32,
                    invert=False):
    """
        取u，v范围中点计算world_matrix（LookAt^-1）
    """
    u_m, u_v, r_v = sum(range_u) * 0.5, sum(range_v) * \
        0.5, sum(range_radius) * 0.5
    loc = sample_on_sphere((u_m, u_m), (u_v, u_v), size=(batch_size))
    radius = torch.ones(batch_size) * r_v
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_camera_pose(range_u, range_v, range_r, val_u=0.5, val_v=0.5, val_r=0.5,
                    batch_size=32, invert=False):
    """
        指定u,v,计算world_matrix（LookAt^-1）
    """
    """
        LookAt矩阵的逆矩阵：

        Xe_x Xe_y Xe_z X_l
        Ye_x Ye_y Ye_z Y_l
        Ze_x Ze_y Ze_z Z_l
           0    0    0   1

    """
    
    """
        u0|-----------*--------------------|u0+ur
    """
    u0, ur = range_u[0], range_u[1] - range_u[0]
    v0, vr = range_v[0], range_v[1] - range_v[0]
    r0, rr = range_r[0], range_r[1] - range_r[0]
    u = u0 + val_u * ur
    v = v0 + val_v * vr
    r = r0 + val_r * rr
    # 获取指定的u,v对应的单位球体上的坐标
    loc = sample_on_sphere((u, u), (v, v), size=(batch_size))
    # 坐标缩放到指定r的球体上
    radius = torch.ones(batch_size) * r
    loc = loc * radius.unsqueeze(-1)
    # 相机去观察loc点（注意这只是LookAt矩阵的一部分）
    R = look_at(loc)
    # 建立4x4单位阵
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    # LookAt矩阵的左上3x3是
    RT[:, :3, :3] = R
    # LookAt矩阵的第4列是观察点的坐标
    RT[:, :3, -1] = loc
    # 是否求逆矩阵
    if invert:
        RT = torch.inverse(RT)
    return RT


def to_sphere(u, v):
    """
        用u,v控制经纬角来获取单位球体上的指定坐标
    """
    # 这里u，v和theta,phi都是正相关的，可以根据u,v的大小控制经纬角
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    # 球体的参数方程
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,),
                     to_pytorch=True):
    """
        根据u，v的范围随机采样球体上的坐标
    """
    # 在u，v的范围内随机取
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)
    # 转换为球面上的坐标
    sample = to_sphere(u, v)
    if to_pytorch:
        sample = torch.tensor(sample).float()

    return sample


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5,
            to_pytorch=True):
    """
        生成LookAt逆矩阵左上角3x3矩阵
    """
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)
    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)
    # 从相机位置指向观察点作为z轴
    z_axis = eye - at
    # 单位化
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis,
                                              axis=1, keepdims=True), eps]))
    # z轴和上向量叉积得到x轴方向
    x_axis = np.cross(up, z_axis)
    # 单位化
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis,
                                              axis=1, keepdims=True), eps]))
    # z，x叉积得到y轴方向
    y_axis = np.cross(z_axis, x_axis)
    # 单位化
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis,
                                              axis=1, keepdims=True), eps]))
    # x，y，z轴的单位向量作为3x3矩阵的每一行
    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(
            -1, 3, 1)), axis=2)

    if to_pytorch:
        r_mat = torch.tensor(r_mat).float()

    return r_mat


def get_rotation_matrix(axis='z', value=0., batch_size=32):
    r = Rot.from_euler(axis, value * 2 * np.pi).as_dcm()
    r = torch.from_numpy(r).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return r
