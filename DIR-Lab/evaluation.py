import os
import shutil
import numpy as np
import torch

from network.embedder import Embedder
from network.three_dim_network import ThreeDimNetwork


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def load_case_point_data(dirlab_path, case_i: int):
    # Landmarks
    landmark_path = dirlab_path + os.sep + "Case" + str(case_i) + "Pack" + os.sep + "ExtremePhases"
    with open(landmark_path + os.sep + "Case" + str(case_i) + "_300_T00_xyz.txt") as f:
        landmarks_fixed = np.array([list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()])

    with open(landmark_path + os.sep + "Case" + str(case_i) + "_300_T50_xyz.txt") as f:
        landmarks_moving = np.array([list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()])

    return landmarks_fixed, landmarks_moving


def voxel_loc_to_embed_loc(voxel_loc, img_shape, img_start_locations, img_end_locations):
    embed_loc = img_start_locations + (voxel_loc - 1) * (img_end_locations - img_start_locations) / (img_shape - 1)
    return embed_loc


def embed_loc_to_voxel_loc(embed_loc, img_shape, img_start_locations, img_end_locations):
    voxel_loc = (img_shape - 1) * (embed_loc - img_start_locations) / (img_end_locations - img_start_locations) + 1
    return voxel_loc


def run_dirlab_quantitative_evaluation(df_model, df_embed_fn, cfg):
    dirlab_path = cfg['dirlab_path']
    case_i = cfg['case_id']
    img_shape = cfg['image_sizes'][case_i]
    img_shape = np.array([img_shape[1], img_shape[2], img_shape[0]])

    img_start_locations = np.array(cfg['moving_img_dataset_config']['img_start_locations'])
    img_end_locations = np.array(cfg['moving_img_dataset_config']['img_end_locations'])

    landmarks_fixed, landmarks_moving = load_case_point_data(dirlab_path, case_i)
    landmarks_fixed = landmarks_fixed[:, [1, 0, 2]]
    landmarks_moving = landmarks_moving[:, [1, 0, 2]]
    fixed_loc_input_landmarks = voxel_loc_to_embed_loc(landmarks_fixed, img_shape, img_start_locations,
                                                       img_end_locations)

    device = cfg['device']

    df_model.eval()

    with torch.no_grad():

        # point part
        fixed_loc_input_landmarks = fixed_loc_input_landmarks.astype(np.float32)
        fixed_loc_input_landmarks = torch.from_numpy(fixed_loc_input_landmarks).to(device)
        fixed_loc_input_embedded = df_embed_fn(fixed_loc_input_landmarks)
        delta_loc_landmarks = df_model(fixed_loc_input_embedded)

        moving_loc_input_landmarks = fixed_loc_input_landmarks + delta_loc_landmarks
        # point stop

    moving_loc_input_landmarks = moving_loc_input_landmarks.cpu().numpy()
    landmarks_moving_pred = embed_loc_to_voxel_loc(moving_loc_input_landmarks, img_shape, img_start_locations,
                                                   img_end_locations)

    delta_landmarks_moving = landmarks_moving_pred - landmarks_moving

    # delta_landmarks_moving = landmarks_fixed - landmarks_moving
    """ check the code
    CASE1: 3.89 +/- 2.78
    CASE2: 4.34 +/- 3.90 
    """

    voxel_sizes = cfg['voxel_sizes'][case_i]
    voxel_sizes = np.array([voxel_sizes[1], voxel_sizes[2], voxel_sizes[0]])

    delta_landmarks_moving = delta_landmarks_moving * voxel_sizes

    difference = np.square(delta_landmarks_moving)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    mean_difference = np.mean(difference)
    std_difference = np.std(difference)

    return mean_difference, std_difference, [landmarks_moving_pred, landmarks_moving, landmarks_fixed, difference]


def eval_df(task_cfg):

    running_folder = task_cfg['running_folder']
    if not os.path.exists(running_folder):
        os.mkdir(running_folder)

    # training hyperparameters
    device = torch.device(task_cfg['device'])

    # set up deformable embed to get input channel
    df_embed_fn = Embedder(task_cfg['df_embed'])
    df_input_ch = df_embed_fn.get_out_dim()

    # set up the deformable network
    df_model = ThreeDimNetwork(task_cfg['df_net_conf'], input_ch=df_input_ch, output_ch=3).to(device)
    df_model_file_path = task_cfg['df_model_file_path']
    if task_cfg['moving_img_train_config']['dp']:
        # original saved file with DataParallel
        state_dict = torch.load(df_model_file_path, map_location=device)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        # load params
        df_model.load_state_dict(new_state_dict)
    else:
        df_model.load_state_dict(torch.load(df_model_file_path, map_location=device))

    df_model.eval()
    mean_difference, std_difference, landmark_details = run_dirlab_quantitative_evaluation(
        df_model, df_embed_fn, task_cfg)

    return mean_difference, std_difference, landmark_details