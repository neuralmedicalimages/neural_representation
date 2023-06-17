import os
import shutil

import numpy as np
from scipy.ndimage import zoom


def meshgrid_any_dim(*arrs):
    arrs = tuple(arrs)  # edit
    lens = list(map(len, arrs))
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)


def clean_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(folder)


def shape_check_and_rescale(voxel, des_dim_y, des_dim_x, des_dim_z):
    vox_dim_y, vox_dim_x, vox_dim_z = voxel.shape
    if des_dim_y != vox_dim_y or des_dim_x != vox_dim_x or des_dim_z != vox_dim_z:
        vox_data_type = voxel.dtype
        zoom_par = [des_dim_y / vox_dim_y, des_dim_x / vox_dim_x, des_dim_z / vox_dim_z]
        new_voxel = zoom(voxel, zoom_par, output=vox_data_type)
        return new_voxel
    else:
        return voxel


def to_abs_path(path):
    if not os.path.isabs(path):
        abs_path = os.path.join(os.environ['ROOT_DIR'], path)
    else:
        abs_path = path
    return abs_path


def update_dataset_cfg(task_cfg, dataset_cfg):
    if task_cfg['shape_rescale']:
        img_shapes = [task_cfg['des_dim0'], task_cfg['des_dim1'], task_cfg['des_dim2']]
    else:
        img_shapes = task_cfg['image_sizes'][task_cfg['case_id']]
        img_shapes = [img_shapes[1], img_shapes[2], img_shapes[0]]

    dataset_cfg['img_shapes'] = img_shapes
    if dataset_cfg['pad']:
        img_shapes_np = np.array(img_shapes)
        pads = np.array((dataset_cfg['pad_size'], dataset_cfg['pad_size'], dataset_cfg['pad_size']), dtype=int)
        dataset_shapes = img_shapes_np + 2 * pads
        dataset_cfg['dataset_shapes'] = dataset_shapes.tolist()
        resolutions = (np.array(dataset_cfg['end_locations']) - np.array(dataset_cfg['start_locations'])) / \
                      (np.array(dataset_cfg['dataset_shapes']) - 1)
        dataset_cfg['resolutions'] = resolutions.tolist()
        img_start_locations = dataset_cfg['start_locations'] + pads * resolutions
        img_end_locations = dataset_cfg['end_locations'] - pads * resolutions
        dataset_cfg['img_start_locations'] = img_start_locations.tolist()
        dataset_cfg['img_end_locations'] = img_end_locations.tolist()
    else:
        dataset_cfg['dataset_shapes'] = img_shapes
        resolutions = (np.array(dataset_cfg['end_locations']) - np.array(dataset_cfg['start_locations'])) / \
                      (np.array(dataset_cfg['dataset_shapes']) - 1)
        dataset_cfg['resolutions'] = resolutions.tolist()
        dataset_cfg['img_start_locations'] = dataset_cfg['start_locations']
        dataset_cfg['img_end_locations'] = dataset_cfg['end_locations']
    return dataset_cfg
