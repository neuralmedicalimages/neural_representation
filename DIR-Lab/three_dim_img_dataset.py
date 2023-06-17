import h5py
import numpy as np
from torch.utils.data import Dataset

from assist_script.utilities import meshgrid_any_dim


def flat_mesh_and_img(loc_list, voxel_loc_list, img):
    loc_mesh = meshgrid_any_dim(*loc_list)
    voxel_loc_mesh = meshgrid_any_dim(*voxel_loc_list)
    return np.stack((loc_mesh[0].flatten(),
                     loc_mesh[1].flatten(),
                     loc_mesh[2].flatten(),
                     voxel_loc_mesh[0].flatten(),
                     voxel_loc_mesh[1].flatten(),
                     voxel_loc_mesh[2].flatten(),
                     img.flatten(),), axis=1)


def flat_mesh_and_img_with_mask(loc_list, voxel_loc_list, img, mask):
    loc_mesh = meshgrid_any_dim(*loc_list)
    voxel_loc_mesh = meshgrid_any_dim(*voxel_loc_list)
    return np.stack((loc_mesh[0][mask],
                     loc_mesh[1][mask],
                     loc_mesh[2][mask],
                     voxel_loc_mesh[0][mask],
                     voxel_loc_mesh[1][mask],
                     voxel_loc_mesh[2][mask],
                     img[mask]), axis=1)


# 3D image dataset
class ThreeDimImageDataset(Dataset):
    def __init__(self, dataset_cfg, filepath):
        Dataset.__init__(self)

        self.filepath = filepath

        # read data
        print('load [' + filepath + ']')
        hdf5_file = h5py.File(filepath, 'r')
        self.img = hdf5_file["slice_img"][:, :]
        if "original_slice_img" in hdf5_file:
            self.ori_img = hdf5_file["original_slice_img"][:, :]
        else:
            self.ori_img = self.img
        if dataset_cfg['image_modeling_task_type'] == 'with_mask':
            self.mask = hdf5_file["slice_mask"][:, :]
            if "original_slice_mask" in hdf5_file:
                self.ori_mask = hdf5_file["original_slice_mask"][:, :]
            else:
                self.ori_mask = self.mask
        if "ori_voxel_sizes" in hdf5_file:
            self.ori_voxel_sizes = hdf5_file["ori_voxel_sizes"][:]
        else:
            self.ori_voxel_sizes = np.array([1.0, 1.0, 1.0])
        hdf5_file.close()

        self.img_shapes = dataset_cfg['img_shapes']
        self.size = np.prod(self.img_shapes)
        self.iter_num = 0

        # get start and end point
        self.img_start_locations = dataset_cfg['img_start_locations']
        self.img_end_locations = dataset_cfg['img_end_locations']

        # set up array
        arrs = []
        voxel_ind_arrs = []
        for dim_i in range(3):
            arrs.append(np.linspace(self.img_start_locations[dim_i],
                                    self.img_end_locations[dim_i],
                                    self.img_shapes[dim_i]))
            voxel_ind_arrs.append(np.arange(self.img_shapes[dim_i]))

        # set up array
        self.pad_img = np.pad(self.img, (dataset_cfg['pad_size'], dataset_cfg['pad_size']))
        if dataset_cfg['image_modeling_task_type'] == 'with_mask':
            self.pad_mask = np.pad(self.mask, (dataset_cfg['pad_size'], dataset_cfg['pad_size']))
            self.pts_value = flat_mesh_and_img_with_mask(arrs, voxel_ind_arrs, self.img, self.mask)
            self.size = np.sum(self.mask)
        else:
            self.pts_value = flat_mesh_and_img(arrs, voxel_ind_arrs, self.img)
            self.size = np.prod(self.img_shapes)

        # random
        if dataset_cfg['shuffle']:
            np.random.seed(None)
            np.random.shuffle(self.pts_value)

        print('image shape:          ' + str(self.img.shape))
        print('pad image shape:      ' + str(self.pad_img.shape))
        print('point number:         ' + str(len(self)))
        print('dataset initialed!')

    def __getitem__(self, idx):

        idx = idx % self.size

        # get the location of the pixel
        pix_loc = self.pts_value[idx, 0:3]

        # get the pixel value
        pix_value = self.pts_value[idx, 6:7]

        # get output
        return pix_loc.astype(np.float32), pix_value.astype(np.float32)

    def get_img_arr(self):
        return self.img

    def get_ori_img_array(self):
        return self.ori_img

    def get_mask_array(self):
        return self.mask

    def get_ori_mask_array(self):
        return self.ori_mask

    def __len__(self):
        if self.iter_num == 0:
            return self.size
        else:
            return self.iter_num

    def set_iter_num(self, num):
        self.iter_num = int(num)

    def get_check_img_dataset(self):
        return CoordDataset(self)


# a dataset which output the coordinate
class CoordDataset(Dataset):
    def __init__(self, ds: ThreeDimImageDataset):
        Dataset.__init__(self)

        self.start_locations = ds.img_start_locations
        self.end_locations = ds.img_end_locations
        self.img_shapes = ds.img_shapes
        self.pts_value = ds.pts_value
        self.size = ds.size

        self.pix_loc = None
        self.rel_pix_loc = None

        self.cal_pix_loc()
        self.cal_rel_pix_loc()
        print('coordinate dataset initialed!')

    def cal_pix_loc(self):
        self.pix_loc = self.pts_value[:, 0:3]

    def cal_rel_pix_loc(self):
        self.rel_pix_loc = self.pts_value[:, 3:6]

    def __getitem__(self, idx):
        return self.pix_loc[idx].astype(np.float32), self.rel_pix_loc[idx].astype(np.int)

    def __len__(self):
        return self.size

    def get_2d_mesh_grid(self, dim_i, dim_j):
        mesh_grid_i, mesh_grid_j = np.meshgrid(np.arange(self.img_shapes[dim_i]),
                                               np.arange(self.img_shapes[dim_j]),
                                               indexing='ij')
        return mesh_grid_j, mesh_grid_i