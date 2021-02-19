# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/2/18
"""
import os
import argparse
import torch
from tqdm import tqdm
from dataset import get_transform
from utils import check_image, decode


FILE_SUFFIX = ["png", "jpg", "jpeg"]


def predict(model, image, trans):
    """
    预测一张图片
    :param model: 训练好的模型
    :param image: 图片路径、图片数组(opencv读取的图片)、PIL读取的图片或者torch Tensor
    :param trans: 图片输入到模型前的处理，如ToTensor、Normalize等
    :return: 预测结果
    """
    img = check_image(image, trans)
    output = model(img)
    output = output.argmax(2)
    out = decode(output)
    return out[0]


def check_is_dir_or_file(path):
    if os.path.isdir(path):
        return True
    elif os.path.isfile(path):
        if path.split(".")[-1].lower() in FILE_SUFFIX:
            return False
        else:
            raise ValueError("The path {} is not (png, jpg, or jpeg) picture, please check it!".format(path))
    else:
        raise ValueError("The path {} is not correct, please check it!".format(path))


def main(args):
    data_path = args.data_path
    output_dir = args.output_dir
    reload_checkpoint = './checkpoints/model.pt'
    device = args.device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    model = torch.load(reload_checkpoint, map_location=device)
    model.to(device)
    correct = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "result.txt")
    fp = open(file_path, "w")
    if check_is_dir_or_file(data_path):
        file_list = [f for f in os.listdir(data_path) if f.split(".")[-1] in FILE_SUFFIX]
        files_path = [os.path.join(data_path, f) for f in file_list]
        for i, file in enumerate(tqdm(files_path)):
            outs = predict(model, file, get_transform())
            if outs == file_list[i].split(".")[0]:
                correct += 1
            fp.write(file_list[i] + " " + outs + "\n")
        print(correct/len(file_list)*100)
        fp.write(str(correct/len(file_list)*100) + "\n")
    else:
        outs = predict(model, data_path, get_transform())
        fp.write(data_path + " " + outs + "\n")
        print(data_path, outs)
    fp.close()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Verification Code Recognize')
    parser.add_argument('--data_path', default='../ocr_train', help='dataset')
    parser.add_argument('--device', default='', help='device cpu or gpu')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
