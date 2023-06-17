import os

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from three_dim_img_dataset import ThreeDimImageDataset
from network.embedder import Embedder
from network.loss_module import build_loss
from network.three_dim_network import ThreeDimNetwork


def train_moving_image_model(task_cfg):
    running_folder = task_cfg['running_folder']

    if not os.path.exists(running_folder):
        os.mkdir(running_folder)

    # read training data
    task_cfg['moving_img_dataset_config']['image_modeling_task_type'] = task_cfg['moving_img_modeling_task_type']
    train_ds = ThreeDimImageDataset(task_cfg['moving_img_dataset_config'], task_cfg['moving_dataset_file'])

    # get train config
    train_cfg = task_cfg['moving_img_train_config']

    # set network path
    if not os.path.exists(task_cfg['model_path']):
        os.mkdir(task_cfg['model_path'])
    model_file_path = task_cfg['moving_image_model_file_path']

    # training hyperparameters
    device = torch.device(task_cfg['device'])
    lrate = train_cfg['lrate']
    end_lrate = train_cfg['end_lrate']
    epoch_num = train_cfg['epoch_num']
    batch_size = train_cfg['batch_size']
    train_loader_works = train_cfg['train_loader_works']
    decay_gamma = (end_lrate / lrate) ** (1 / (epoch_num - 1))

    # loss function
    loss_fun = build_loss(train_cfg['loss_setting'])

    if "TopK" in train_cfg['loss_setting']:
        epoch_k = np.logspace(train_cfg['start_k'], train_cfg['end_k'], num=epoch_num)
    else:
        epoch_k = [1] * epoch_num

    # set the location embed function
    embed_fn = Embedder(task_cfg['moving_img_embed'])
    input_ch = embed_fn.get_out_dim()

    # load image and loss network
    moving_img_model = ThreeDimNetwork(task_cfg['moving_img_net_conf'], input_ch=input_ch, output_ch=1).to(device)

    moving_img_model.eval()

    if train_cfg['dp']:
        moving_img_model = nn.DataParallel(moving_img_model)

    optimizer = torch.optim.Adam(params=list(moving_img_model.parameters()), lr=lrate, betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=decay_gamma)

    # set the loaders
    tr_loader = DataLoader(train_ds, batch_size=batch_size,
                           pin_memory=True, num_workers=train_loader_works)

    # for debug
    if task_cfg['moving_img_training_debug_config']['train_one_epoch']:
        epoch_num = 1

    print('training settings:')
    print('loss_setting:               ' + str(train_cfg['loss_setting']))
    if "topk" in train_cfg['loss_setting']:
        print('    start_k:                ' + str(train_cfg['start_k']))
        print('    end_k:                  ' + str(train_cfg['end_k']))
    print('start learning rate:        ' + str(lrate))
    print('end learning rate:          ' + str(end_lrate))
    print('epoch_num:                  ' + str(epoch_num))
    print('batch_size:                 ' + str(batch_size))
    print('train_loader_works:         ' + str(train_loader_works))
    print('device:                     ' + task_cfg['device'])

    # loss curve
    with open(running_folder + os.sep + "image_modeling_loss.csv", 'w') as csv_file:
        csv_file.write('epoch,learning rate,k,tr_loss')
        csv_file.write('\n')

    # start train
    for epoch in range(epoch_num):

        moving_img_model.train()

        # if loss is topk series
        if "topk" in train_cfg['loss_setting']:
            loss_fun = build_loss(train_cfg['loss_setting'], epoch_k[epoch])
            print("epoch_k: " + str(epoch_k[epoch]))

        # initial training loss
        total_sample_size = 0
        epoch_loss = 0
        total_loss = 0

        for batch_i, batch_data in enumerate(tr_loader):
            loc_in, true_value = batch_data

            sample_size = true_value.size(0)
            total_sample_size = total_sample_size + sample_size

            loc_in = loc_in.to(device)
            loc_embedded = embed_fn(loc_in)
            pred_value = moving_img_model(loc_embedded)

            true_value = true_value.to(device)
            loss = loss_fun(pred_value, true_value)

            total_loss = total_loss + loss.item() * sample_size
            epoch_loss = total_loss / total_sample_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch: ' + "{:03d}".format(epoch)
                  + ' batch: ' + "{:06d}".format(batch_i)
                  + ' percent: ' + "{:0.2f} %".format(total_sample_size / len(train_ds) * 100)
                  + ' loss: ' + "{:.4e}".format(epoch_loss))

            if task_cfg['moving_img_training_debug_config']['train_one_batch']:
                break

        if epoch % train_cfg['model_save_interval'] == 0:
            torch.save(moving_img_model.state_dict(), model_file_path)
            print('network saved!')

        with open(running_folder + os.sep + "image_modeling_loss.csv", 'a+') as csv_file:
            csv_file.write('{},{},{},{}'.format(epoch, scheduler.get_last_lr()[0], epoch_k[epoch], epoch_loss))
            csv_file.write('\n')

        scheduler.step()
