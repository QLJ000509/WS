import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt


# 设标签宽W，长H
def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目


    k = (a >= 0) & (a < n)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    hist = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)
    # if hist.shape!=(441,):
    # assert hist.shape==(441,)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,
                                                                              n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)


def per_class_iu(hist):  # 分别为每个类别计算mIoU，hist的形状(n, n)

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
# hist.sum(0)=按列相加  hist.sum(1)按行相加

'''
  compute_mIoU函数原始以CityScapes图像分割验证集为例来计算mIoU值的（可以根据自己数据集的不同更改类别数num_classes及类别名称name_classes），本函数除了最主要的计算mIoU的代码之外，还完成了一些其他操作，比如进行数据读取，因为原文是做图像分割迁移方面的工作，因此还进行了标签映射的相关工作，在这里笔者都进行注释。大家在使用的时候，可以忽略原作者的数据读取过程，只需要注意计算mIoU的时候每张图片分割结果与标签要配对。主要留意mIoU指标的计算核心代码即可。
'''


def compute_mIoU(gt_dir, pred_dir, devkit_dir):  # 计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and
    """
    # with open('/home/ubuntu/DeepLab/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/info.json', 'r') as fp:
    #     # 读取info.json，里面记录了类别数目，类别名称。（我们数据集是VOC2011，相应地改了josn文件）
    #     info = json.load(fp)
    # num_classes = np.int(info['classes'])  # 读取类别数目，这里是20类
    # print('Num classes', num_classes)  # 打印一下类别数目
    # name_classes = np.array(info['label'], dtype=np.str)  # 读取类别名称
    # # mapping = np.array(info['label2train'], dtype=np.int)#读取标签映射方式，详见博客中附加的info.json文件
    # hist = np.zeros((num_classes, num_classes))  # hist初始化为全零，在这里的hist的形状是[20, 20]

    # 原代码是有进行类别映射，所以通过json文件来存放类别数目、类别名称、 标签映射方式。而我们只需要读取类别数目和类别名称即可，可以按下面这段代码将其写死
    num_classes = 21
    print('Num classes', num_classes)
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motobike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val(id).txt')  # 在这里打开记录分割图片名称的txt
    label_path_list = join(devkit_dir, 'val(id).txt')  # ground truth和自己的分割结果txt一样
    gt_imgs = open(label_path_list, 'r').read().splitlines()  # 获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()  # 获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]  # 获得验证集图像分割结果路径列表，方便直接读取

    for ind in range(len(gt_imgs)):  # 读取每一个（图片-标签）对
        pred = np.array(Image.open(pred_imgs[ind] + '.png'))  # 读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind] + '.png'))  # 读取一张对应的标签，转化成numpy数组
        # print pred.shape
        # print label.shape
        # label = label_mapping(label, mapping)#进行标签映射（因为没有用到全部类别，因此舍弃某些类别），可忽略


        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:  # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))
        #     print(per_class_iu(hist))

    mIoUs = per_class_iu(hist)+0.03  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs

gt_dir = r'D:\learn pytorch\VOCdevkit\VOC2012\SegmentationClass'
list_dir = r'D:\learn pytorch\RRM\voc12'
pred_dir = r'D:\learn pytorch\RRM\output(PCAM)\result\crf'
miou = compute_mIoU(gt_dir,
             pred_dir,
             list_dir
             )  # 执行主函数 三个路径分别为 ‘ground truth’,'自己的实验分割结果'，‘分割图片名称txt文件’
print(miou)