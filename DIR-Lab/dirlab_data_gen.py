import os

import SimpleITK as sitk
import h5py
import numpy as np
from einops import rearrange
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_fill_holes

from assist_script.utilities import shape_check_and_rescale
from assist_script.utilities import to_abs_path


def generate_one_hdf5_data(task_cfg, case_id, data_id):
    case_id_str = str(case_id)
    hdf5_path = to_abs_path(task_cfg['hdf5_data_folder'])

    dirlab_path = to_abs_path(task_cfg['dirlab_path']) + os.sep + "Case" + case_id_str + "Pack"
    image_path = dirlab_path + os.sep + "Images"
    mask_path = dirlab_path + os.sep + "Masks"

    ori_image_sizes = task_cfg['image_sizes'][case_id]
    ori_voxel_sizes = task_cfg['voxel_sizes'][case_id]
    ori_voxel_sizes = np.array([ori_voxel_sizes[1], ori_voxel_sizes[2], ori_voxel_sizes[0]])

    image_file = data_id + ".img"
    with open(image_path + os.sep + image_file, "rb") as f:
        slice_img = np.fromfile(f, np.int16) - 1024
    slice_img = slice_img.reshape(ori_image_sizes)
    slice_img = rearrange(slice_img, "z x y -> x y z")

    mask_file = data_id + ".mhd"
    mask_itk = sitk.ReadImage(mask_path + os.sep + mask_file)
    slice_mask = np.clip(sitk.GetArrayFromImage(mask_itk), 0, 1)
    slice_mask = rearrange(slice_mask, "z x y -> x y z")

    print('[' + data_id + '] hdf5 dataset process start!')

    # normalization setting
    if task_cfg['norm_method'] == 'fix':
        slice_img = (slice_img - task_cfg['min_value']) / (task_cfg['max_value'] - task_cfg['min_value'])

    slice_img = np.clip(slice_img, 0, 1)

    if slice_img is None:
        print('slice_img error for [' + data_id + ']')
        return None

    # scale image
    if task_cfg['shape_rescale']:
        dim_0 = task_cfg['des_dim0']
        dim_1 = task_cfg['des_dim1']
        dim_2 = task_cfg['des_dim2']
        slice_img_rescale = shape_check_and_rescale(slice_img, des_dim_y=dim_0, des_dim_x=dim_1, des_dim_z=dim_2)
    else:
        slice_img_rescale = slice_img

    # expand mask and fill hole
    if task_cfg['dilation']:
        struct = np.ones((5, 5, 5), dtype=bool)
        print('[' + data_id + '] dilation!')
        slice_mask = binary_dilation(slice_mask, structure=struct)

    if task_cfg['fill_hole']:
        print('[' + data_id + '] fill hole!')
        for i in range(slice_img.shape[0]):
            slice_mask[i, :, :] = binary_fill_holes(slice_mask[i, :, :])
        for i in range(slice_img.shape[1]):
            slice_mask[:, i, :] = binary_fill_holes(slice_mask[:, i, :])
        for i in range(slice_img.shape[2]):
            slice_mask[:, :, i] = binary_fill_holes(slice_mask[:, :, i])

    # scale mask
    if task_cfg['shape_rescale']:
        dim_0 = task_cfg['des_dim0']
        dim_1 = task_cfg['des_dim1']
        dim_2 = task_cfg['des_dim2']
        slice_mask_rescale = shape_check_and_rescale(slice_mask.astype(float),
                                                     des_dim_y=dim_0, des_dim_x=dim_1, des_dim_z=dim_2) > 0.5
    else:
        slice_mask_rescale = slice_mask

    # change data type
    slice_img_rescale = slice_img_rescale.astype(np.float32)
    slice_img = slice_img.astype(np.float32)
    slice_mask_rescale = slice_mask_rescale.astype(bool)
    slice_mask = slice_mask.astype(bool)

    # save
    with h5py.File(hdf5_path + os.sep + data_id + ".h5", "w") as hf:
        hf.create_dataset("slice_img", data=slice_img_rescale[:, :, :], compression='lzf')
        hf.create_dataset("slice_mask", data=slice_mask_rescale[:, :, :], compression='lzf')
        hf.create_dataset("original_slice_img", data=slice_img[:, :, :], compression='lzf')
        hf.create_dataset("original_slice_mask", data=slice_mask[:, :, :], compression='lzf')
        hf.create_dataset("ori_voxel_sizes", data=ori_voxel_sizes[:], compression='lzf')

    print('[' + data_id + '] hdf5 dataset done!')