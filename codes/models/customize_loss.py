import torch
import torch.nn as nn
import cv2
import numpy as np
import math
import os



def get_entropy(img_):
    x, y = img_.shape[0:2]
    img_ = cv2.resize(img_, (100, 100))  # 缩小的目的是加快计算速度
    tmp = []
    for i in range(65536):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res


class tanh_L1Loss_Entropy(nn.Module):
    def __init__(self):
        super(tanh_L1Loss_Entropy, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        x = x.cpu().detach().numpy()
        # print(x)
        x = x.transpose(3, 2, 1, 0)
        # print(x.shape)
        ent = 0
        # print(x.shape[3])
        for i in range(x.shape[3]):
            # print(x[:,:,:,i].shape)
            x_image = cv2.cvtColor(x[:, :, :, i], cv2.COLOR_RGB2BGR)
            x_image = cv2.cvtColor(x_image, cv2.COLOR_BGR2GRAY)
            # print(x_image)
            x_image = x_image * 65536
            # print(x_image)
            x_image = x_image.astype(np.int16)
            ent = get_entropy(x_image) + ent
        # print("TanLoss")
        # print(loss)
        # print("EntLoss")
        # print(ent)

        # fector
        ent = ent / 256
        loss = loss / ent

        return loss

class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss


class tanh_L2Loss(nn.Module):
    def __init__(self):
        super(tanh_L2Loss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.pow((torch.tanh(x) - torch.tanh(y)), 2))
        return loss