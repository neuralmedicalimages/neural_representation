import os
import pprint
import sys
import json

from dirlab_data_gen import generate_one_hdf5_data
from assist_script.utilities import to_abs_path
from assist_script.utilities import update_dataset_cfg
from train_df_model import train_deform_model
from train_3d_img import train_moving_image_model
from evaluation import eval_df

# set some globe running variables
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '../..')))
os.environ['ROOT_DIR'] = os.path.abspath(os.path.dirname(__file__))

default_task_cfg = {
    'running_folder': './output',
    'dirlab_path': './data',

    'case_id': 1,
    'fixed_data_id': 'case1_T00_s',
    'moving_data_id': 'case1_T50_s',

    'shape_rescale': False,
    'des_dim0': 512,
    'des_dim1': 512,
    'des_dim2': 256,

    'norm_method': "fix",
    'min_value': -1024,
    'max_value': 2048,

    'fill_hole': True,
    'dilation': False,

    # Size of data, per image pair
    'image_sizes': [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ],

    # Scale of data, per image pair
    'voxel_sizes': [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ],

    # train image model config

    # moving image training debug setting
    'moving_img_training_debug_config': {
        'train_one_batch': False,
        'train_one_epoch': False,
    },

    # device
    'device': 'cuda:0',

    # moving image modeling type with_mask/without_mask
    'moving_img_modeling_task_type': 'without_mask',

    # moving image dataset config
    'moving_img_dataset_config': {
        # start and end point
        'start_locations': [-1.0, -1.0, -1.0],
        'end_locations': [1.0, 1.0, 1.0],

        # a parameter for padding
        'pad': True,
        'pad_size': 10,

        # this config is only used for training
        'shuffle': True,
    },

    # moving image model embed config
    'moving_img_embed': {
        'flag': True,
        'L': 10,
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': 10 - 1,
        'num_freqs': 10,
        'log_sampling': True,
    },

    # moving image network config
    'moving_img_net_conf': {
        # width
        'net_width': 1024,
        # depth
        'net_depth': 16,
        # have skip or not
        'skip': True,
        # which layer to have skip connection
        'skip_layers': [2, 4, 6, 8, 10, 12, 14],
    },

    # moving image modeling training config
    'moving_img_train_config': {
        # epoch
        'epoch_num': 21,

        # batch size
        'batch_size': 128 * 128,

        'train_loader_works': 4,
        'dp': False,
        'model_save_interval': 1,

        # learning rate
        'lrate': 5e-4,
        'end_lrate': 5e-7,

        # loss
        'loss_setting': "TopKMSELoss",
        'start_k': 0,
        'end_k': -5,

        # iterations each epoch
        # -1 is not use this config
        'iterations_num': 2e4,
    },

    # deform model setting

    # deform training debug setting
    'df_training_debug_config': {
        'train_one_batch': False,
        'train_one_epoch': False,
    },

    # deform modeling type with_mask/without_mask
    'df_task_type': 'with_mask',

    # fixed image dataset config for deform target
    'fixed_img_dataset_config': {
        # start and end point
        'start_locations': [-1.0, -1.0, -1.0],
        'end_locations': [1.0, 1.0, 1.0],

        # a parameter for padding
        'pad': True,
        'pad_size': 10,

        # this config is only used for training
        'shuffle': True,
    },

    # deformable embedding setting
    'df_embed': {
        'flag': True,
        'L': 10,
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': 10 - 1,
        'num_freqs': 10,
        'log_sampling': True,
    },

    # deformable network setting
    'df_net_conf': {
        # width
        'net_width': 512,
        # depth
        'net_depth': 16,
        # have skip or not
        'skip': True,
        # which layer to have skip connection
        'skip_layers': [2, 4, 6, 8, 10, 12, 14],
    },

    # deform training setting
    'df_train_conf': {
        # elastic beta setting
        'use_decay_elastic': False,
        'elastic_beta': 1e-3,

        # epoch
        'epoch_num': 21,

        'batch_size': 512,
        'train_loader_works': 4,
        'dp': False,
        'model_save_interval': 1,

        # learning rate
        'lrate': 5e-5,
        'end_lrate': 5e-7,

        # loss
        'loss_setting': "MSELoss",

        # iterations each epoch
        # -1 is not use this config
        'iterations_num': 2e4,
    }

}


def main(task_cfg):
    task_cfg['running_folder'] = './output/case1'
    task_cfg['fixed_data_id'] = 'case1_T00_s'
    task_cfg['moving_data_id'] = 'case1_T50_s'

    # case id and data id
    task_cfg['case_id'] = 1

    run_one_case(task_cfg)


def run_one_case(task_cfg):
    # two path need input
    task_cfg['dirlab_path'] = './data'

    # clean running folder
    running_folder = task_cfg['running_folder']
    if os.path.exists(running_folder):
        pass
        # clean_folder(running_folder)
    else:
        os.mkdir(running_folder)

    # some folders get from running_folder
    hdf5_data_folder = task_cfg['running_folder'] + os.sep + 'hdf5'
    task_cfg['hdf5_data_folder'] = hdf5_data_folder
    task_cfg['model_path'] = task_cfg['running_folder'] + os.sep + 'model'

    # some parameter decided by data id
    task_cfg['moving_dataset_file'] = task_cfg['hdf5_data_folder'] + os.sep + task_cfg['moving_data_id'] + '.h5'
    task_cfg['fixed_dataset_file'] = task_cfg['hdf5_data_folder'] + os.sep + task_cfg['fixed_data_id'] + '.h5'
    task_cfg['moving_image_model_file_path'] = task_cfg['model_path'] + os.sep + task_cfg['moving_data_id'] + '.pth'
    task_cfg['df_model_file_path'] = task_cfg['model_path'] + os.sep + \
        task_cfg['moving_data_id'] + '_to_' + task_cfg['fixed_data_id'] + '.pth'

    print('DIRLAB_path: ' + task_cfg['dirlab_path'])
    print('hdf5_path: ' + hdf5_data_folder)

    print('prepare the hdf5 folder.')
    hdf5_path = to_abs_path(hdf5_data_folder)
    if os.path.exists(hdf5_path):
        pass
        # clean_folder(running_folder)
    else:
        os.mkdir(hdf5_path)

    # update dataset config
    task_cfg['moving_img_dataset_config'] = update_dataset_cfg(task_cfg, task_cfg['moving_img_dataset_config'])
    task_cfg['fixed_img_dataset_config'] = update_dataset_cfg(task_cfg, task_cfg['fixed_img_dataset_config'])

    # print the cfg
    pprint.pprint(task_cfg)

    # save current config
    with open(task_cfg['running_folder'] + os.sep + 'config.json', 'w') as fp:
        json.dump(task_cfg, fp, indent=4)

    # generate hdf5 files
    print('start create hdf5 dataset.')
    # generate_one_hdf5_data(task_cfg, task_cfg['case_id'], task_cfg['moving_data_id'])
    # generate_one_hdf5_data(task_cfg, task_cfg['case_id'], task_cfg['fixed_data_id'])

    # train moving image model
    # train_moving_image_model(task_cfg)

    # train deform model
    # train_deform_model(task_cfg)

    # evaluation
    mean_difference, std_difference, landmark_details = eval_df(task_cfg)

    print(mean_difference, std_difference)


if __name__ == "__main__":
    main(default_task_cfg)