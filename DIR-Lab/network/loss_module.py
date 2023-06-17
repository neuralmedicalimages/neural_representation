import torch
import torch.nn.functional as F


def topk_mse_loss(k):
    def loss_fun(input, target):
        pred = input.view(1, -1)
        true = target.view(1, -1)
        full_loss = F.mse_loss(pred, true, reduction='none')
        k_num = int(full_loss.size()[1] * k)
        k_num = min(max(k_num, 10), full_loss.size()[1])
        top_k, _ = torch.topk(full_loss, k_num)
        top_k_loss = torch.mean(top_k)
        return top_k_loss
    return loss_fun


def build_loss(loss_name, k=1):
    if loss_name == 'L1Loss':
        return torch.nn.L1Loss()
    if loss_name == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    if loss_name == 'MSELoss':
        return torch.nn.MSELoss()
    if loss_name == 'TopKMSELoss':
        return topk_mse_loss(k)
