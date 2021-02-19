# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/2/18
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image


LABEL = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']


class MetricLogger(object):
    def __init__(self, max_label_len=5):
        self.correct = 0
        self.char_correct = 0
        self.num = 0
        self.max_label_len = max_label_len
        self.loss = 0
        self.batch_num = 0

    def update(self, res, loss):
        num = res.shape[0]
        self.num += num
        b_correct = res.eq(self.max_label_len).sum().item()
        # print(b_correct)
        self.correct += b_correct
        bc_correct = res.sum().item()
        # print(bc_correct)
        self.char_correct += bc_correct
        self.loss += loss
        self.batch_num += 1
        return b_correct/num, bc_correct/(num*self.max_label_len)

    def __str__(self):
        log_info = "[evaluate] [images {}] avg loss: {:.6f}, acc: {:.4f}, char acc: {:.4f}".format(
            self.num, self.loss / self.batch_num, self.correct / self.num,
            self.char_correct / self.num / self.max_label_len
        )
        return log_info


def check_image(image, trans=None):
    if isinstance(image, str):
        if os.path.exists(image) and trans is not None:
            img = cv2.imread(image)
            img = Image.fromarray(img)
            img = trans(img)
            img = torch.unsqueeze(img, 0)
        else:
            raise ValueError
    elif isinstance(image, np.ndarray) and trans is not None:
        try:
            img = Image.fromarray(image)
            img = trans(img)
            img = torch.unsqueeze(img, 0)
        except:
            raise ValueError
    elif isinstance(image, Image.Image) and trans is not None:
        img = trans(image)
        img = torch.unsqueeze(img, 0)
    elif isinstance(image, torch.Tensor):
        img = image
    else:
        raise ValueError
    return img


def decode(output):
    outs = []
    output = torch.squeeze(output, 1)
    for i in range(output.shape[0]):
        out = ""
        for j in range(output.shape[1]):
            index = output[i, j]
            c = LABEL[index]
            out += c
        outs.append(out)
    return outs


def check_or_make_dir(root, dir_name, mkdir=False):
    dir_path = os.path.join(root, dir_name)
    if not os.path.exists(dir_path):
        if mkdir:
            os.mkdir(dir_path)
        else:
            raise ValueError("Error path: " + dir_path)
    return dir_path


def random_noise(image, noise_num=500):
    '''
    添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
    :param image: 需要加噪的图片
    :param noise_num: 添加的噪音点数目，一般是上千级别的
    :return: img_noise
    '''
    img_noise = image
    rows, cols, chn = img_noise.shape
    # 加噪声
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        img_noise[x, y, :] = [r, g, b]
    return img_noise


def gasuss_noise(image, mean=0, var=0.00001):
    '''
        添加高斯噪声
        image:原始图像
        mean : 均值
        var : 方差,越大，噪声越大
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

