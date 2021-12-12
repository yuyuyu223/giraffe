import torch
import numpy as np
import logging
logger_py = logging.getLogger(__name__)


def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None, invert_y_axis=False):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    # 分辨率拆分h，w
    h, w = resolution
    # 计算像素点个数
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    # 建立h*w的网格
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    # 在最里层新建维度上进行拼接，然后reshape再repeat
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    # [0-w],[0-h]-->[-0.5scale,0.5scale],[-0.5scale,0.5scale]
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    # 子采样
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        # 随机选取subsample_to个数
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]
    # 是否反转y轴
    if invert_y_axis:
        assert(image_range == (-1, 1))
        pixel_scaled[..., -1] *= -1.
        pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]

    return pixel_locations, pixel_scaled


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.
        将numpy数组转换为torch张量
    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    # 如果是numpy数组
    if type(tensor) == np.ndarray:
        # 转换为torch
        tensor = torch.from_numpy(tensor)
        is_numpy = True
    # 克隆一份
    tensor = tensor.clone()
    # 是否返回类型
    if return_type:
        return tensor, is_numpy
    return tensor


def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat=None,
                       invert=True, use_absolute_depth=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)
    # 如果scale_mat为空则创建
    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(0).repeat(
            camera_mat.shape[0], 1, 1).to(camera_mat.device)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    # 矩阵求逆
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    # 转换到齐次坐标
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    # 假设近平面深度为1，相机位于坐标原点，投影屏幕上的x,y,1就是相机空间中的dx,dy,d，所以要把前三个分量乘深度
    if use_absolute_depth:
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    # openGL中：相机坐标左乘view矩阵的逆，左乘model矩阵(包括缩放，平移，旋转)的逆
    # 这里与scale_mat @ world_mat @ camera_mat @ pixels对应
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    # 恢复世界坐标3D表式
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def transform_to_camera_space(p_world, camera_mat, world_mat, scale_mat):
    ''' Transforms world points to camera space.
        Args:
        p_world (tensor): world points tensor of size B x N x 3
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
    '''
    batch_size, n_p, _ = p_world.shape
    device = p_world.device

    # Transform world points to homogen coordinates
    # 每个坐标都添加一个齐次坐标
    p_world = torch.cat([p_world, torch.ones(
        batch_size, n_p, 1).to(device)], dim=-1).permute(0, 2, 1)

    # Apply matrices to transform p_world to camera space
    # 视角变换
    p_cam = camera_mat @ world_mat @ scale_mat @ p_world

    # Transform points back to 3D coordinates
    # 还原3D坐标
    p_cam = p_cam[:, :3].permute(0, 2, 1)
    return p_cam


def origin_to_world(n_points, camera_mat, world_mat, scale_mat=None,
                    invert=False):
    ''' 
    获取相机在世界坐标下的坐标
    Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: false)
    '''
    batch_size = camera_mat.shape[0]
    device = camera_mat.device
    # Create origin in homogen coordinates
    # 建立其次坐标，第四分量填1
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(
            0).repeat(batch_size, 1, 1).to(device)

    # Invert matrices
    # 矩阵求逆
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    # 视角变换
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    # 还原3D表示
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world


def image_points_to_world(image_points, camera_mat, world_mat, scale_mat=None,
                          invert=False, negative_depth=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: False)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    # 深度全是1
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    if negative_depth:
        d_image *= -1.
    return transform_to_world(image_points, d_image, camera_mat, world_mat,
                              scale_mat, invert=invert)


def interpolate_sphere(z1, z2, t):
    """
    球面线性插值(SLERP)
    是四元数的一种线性插值运算，主要用于在两个表示旋转的四元数之间平滑差值。
    """
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2
    return z
