import os

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from three_dim_img_dataset import ThreeDimImageDataset
from network.embedder import Embedder
from network.jacobian import JacobianReg
from network.loss_module import build_loss
from network.three_dim_network import ThreeDimNetwork


def train_deform_model(task_cfg):
    running_folder = task_cfg['running_folder']

    if not os.path.exists(running_folder):
        os.mkdir(running_folder)

    # read training dataset
    task_cfg['fixed_img_dataset_config']['image_modeling_task_type'] = task_cfg['df_task_type']
    train_ds = ThreeDimImageDataset(task_cfg['fixed_img_dataset_config'], task_cfg['fixed_dataset_file'])

    # get train config
    train_cfg = task_cfg['df_train_conf']

    # set network path
    if not os.path.exists(task_cfg['model_path']):
        os.mkdir(task_cfg['model_path'])
    df_model_file_path = task_cfg['df_model_file_path']

    # training hyperparameters
    device = torch.device(task_cfg['device'])
    lrate = train_cfg['lrate']
    end_lrate = train_cfg['end_lrate']
    epoch_num = train_cfg['epoch_num']
    batch_size = train_cfg['batch_size']
    train_loader_works = train_cfg['train_loader_works']
    iterations_num = train_cfg['iterations_num']
    decay_gamma = (end_lrate / lrate) ** (1 / (epoch_num - 1))

    elastic_beta = None
    epoch_elastic_beta = None
    if train_cfg['use_decay_elastic']:
        epoch_elastic_beta = np.logspace(train_cfg['start_elastic_beta'], train_cfg['end_elastic_beta'], num=epoch_num)
    else:
        elastic_beta = train_cfg['elastic_beta']

    # loss function
    loss_fun = build_loss(train_cfg['loss_setting'])
    regularization_loss = JacobianReg(device=device)

    # set the location embed function
    moving_img_embed_fn = Embedder(task_cfg['moving_img_embed'])
    input_ch = moving_img_embed_fn.get_out_dim()

    # load image and loss network
    moving_img_model = ThreeDimNetwork(task_cfg['moving_img_net_conf'], input_ch=input_ch, output_ch=1).to(device)

    if task_cfg['moving_img_train_config']['dp']:
        # original saved file with DataParallel
        state_dict = torch.load(task_cfg['image_model_file_path'], map_location=device)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        # load params
        moving_img_model.load_state_dict(new_state_dict)
    else:
        moving_img_model.load_state_dict(torch.load(task_cfg['moving_image_model_file_path'], map_location=device))

    moving_img_model.eval()

    # set up deformable embed to get input channel
    df_embed_fn = Embedder(task_cfg['df_embed'])
    df_input_ch = df_embed_fn.get_out_dim()

    # set up the deformable network
    df_model = ThreeDimNetwork(task_cfg['df_net_conf'], input_ch=df_input_ch, output_ch=3).to(device)

    if train_cfg['dp']:
        moving_img_model = nn.DataParallel(moving_img_model)
        df_model = nn.DataParallel(df_model)

    optimizer = torch.optim.Adam(params=df_model.parameters(), lr=lrate, betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=decay_gamma)

    # set the loaders
    if iterations_num != -1:
        train_ds.set_iter_num(iterations_num * batch_size)
    tr_loader = DataLoader(train_ds, batch_size=batch_size,
                           pin_memory=True, num_workers=train_loader_works)

    # for debug
    if task_cfg['df_training_debug_config']['train_one_epoch']:
        epoch_num = 1

    print('training settings:')
    print('loss_setting:               ' + str(train_cfg['loss_setting']))
    print('use_decay_elastic:          ' + str(train_cfg['use_decay_elastic']))
    if train_cfg['use_decay_elastic']:
        print('    start_elastic_beta:     ' + str(train_cfg['start_elastic_beta']))
        print('    end_elastic_beta:       ' + str(train_cfg['end_elastic_beta']))
    else:
        print('    elastic_beta:           ' + str(train_cfg['elastic_beta']))
    print('start learning rate:        ' + str(lrate))
    print('end learning rate:          ' + str(end_lrate))
    print('epoch_num:                  ' + str(epoch_num))
    print('batch_size:                 ' + str(batch_size))
    print('train_loader_works:         ' + str(train_loader_works))
    if iterations_num != -1:
        print('iterations_num:             ' + str(iterations_num))
    print('device:                     ' + task_cfg['device'])

    # loss curver
    with open(running_folder + os.sep + "df_modeling_loss.csv", 'w') as csv_file:
        csv_file.write('epoch,learning rate,total loss,elastic loss,similarity loss,' +
                       'elastic beta, quantitative evaluation mean, quantitative evaluation std')
        csv_file.write('\n')

    # start train
    for epoch in range(epoch_num):

        # if use elastic decay
        if train_cfg['use_decay_elastic']:
            elastic_beta = epoch_elastic_beta[epoch]
            print("current elastic_beta: " + str(elastic_beta))

        df_model.train()

        # initial training loss
        total_sample_size = 0
        total_loss = 0
        total_elastic_loss = 0
        total_similarity_loss = 0

        for batch_i, batch_data in enumerate(tr_loader):
            fixed_loc_input, true_value = batch_data

            sample_size = true_value.size(0)
            total_sample_size = total_sample_size + sample_size

            fixed_loc_input = fixed_loc_input.to(device)
            fixed_loc_input.requires_grad = True
            fixed_loc_input_embedded = df_embed_fn(fixed_loc_input)
            delta_loc = df_model(fixed_loc_input_embedded)

            # regulation loss
            elastic_loss = regularization_loss(fixed_loc_input, delta_loc)

            # new location
            moving_loc_input = fixed_loc_input + delta_loc

            # only deal with location belong to (-1, 1)
            loc_inside_boundary_index = torch.logical_and(moving_loc_input > -1, moving_loc_input < 1)
            loc_inside_boundary_index = (torch.sum(loc_inside_boundary_index, -1) == 3)

            # only deal with location in the moving image's mask
            moving_loc_input_embedded = moving_img_embed_fn(moving_loc_input)

            pred_value = moving_img_model(moving_loc_input_embedded[loc_inside_boundary_index, :])

            true_value = true_value.to(device)
            similarity_loss = loss_fun(pred_value, true_value[loc_inside_boundary_index])
            loss = similarity_loss + elastic_beta * elastic_loss

            total_loss = total_loss + loss.item() * sample_size
            epoch_loss = total_loss / total_sample_size
            total_elastic_loss = total_elastic_loss + elastic_loss.item() * sample_size
            epoch_elastic_loss = total_elastic_loss / total_sample_size
            total_similarity_loss = total_similarity_loss + similarity_loss.item() * sample_size
            epoch_similarity_loss = total_similarity_loss / total_sample_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch: ' + "{:03d}".format(epoch)
                  + ' batch: ' + "{:06d}".format(batch_i)
                  + ' percent: ' + "{:0.2f} %".format(total_sample_size / len(train_ds) * 100)
                  + ' elastic loss: ' + "{:.4e}".format(epoch_elastic_loss)
                  + ' similarity loss: ' + "{:.4e}".format(epoch_similarity_loss)
                  + ' loss: ' + "{:.4e}".format(epoch_loss))

            if task_cfg['df_training_debug_config']['train_one_batch']:
                break

        torch.save(df_model.state_dict(), df_model_file_path)
        print('network saved!')