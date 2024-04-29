import numpy as np
from PIL import Image
from os.path import join
__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵n*n，初始值全0

    # 像素准确率PA，预测正确的像素/总像素
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc



    # 根据标签和预测图片返回其混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype(int) + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    # 更新混淆矩阵
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape  # 确认标签和预测值图片大小相等
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    # 清空混淆矩阵
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))




def evaluate1(label_path, pre_path, devkit_dir):
    acc_list = []



    image_path_list = join(devkit_dir, 'val(id).txt')  # 在这里打开记录分割图片名称的txt
    label_path_list = join(devkit_dir, 'val(id).txt')  # ground truth和自己的分割结果txt一样
    gt_imgs = open(label_path_list, 'r').read().splitlines()  # 获得验证集标签名称列表
    gt_imgs = [join(label_path, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()  # 获得验证集图像分割结果名称列表
    pred_imgs = [join(pre_path, x) for x in pred_imgs]  # 获得验证集图像分割结果路径列表，方便直接读取

    for i, p in enumerate(pred_imgs):
        img_predict_path = join(pre_path, p) + '.png'  # 添加 .png 扩展名
        imgPredict = Image.open(img_predict_path)
        imgPredict = np.array(imgPredict)

        # 使用 os.path.join() 进行路径拼接
        img_label_path = join(label_path, gt_imgs[i]) + '.png'  # 添加 .png 扩展名
        imgLabel = Image.open(img_label_path)
        imgLabel = np.array(imgLabel)

        metric = SegmentationMetric(21)  # 表示分类个数，包括背景
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.pixelAccuracy()
        acc_list.append(acc)


        # print('{}: acc={}, macc={}, mIoU={}, fwIoU={}'.format(p, acc, macc, mIoU, fwIoU))

    return acc_list




if __name__ == '__main__':
    gt_dir = r'D:\learn pytorch\VOCdevkit\VOC2012\SegmentationClass'
    list_dir = r'D:\learn pytorch\RRM\voc12'
    pred_dir = r'D:\learn pytorch\RRM\output(SEAM)\result\crf'
    # 计算测试集每张图片的各种评价指标，最后求平均
    acc_list = evaluate1(gt_dir, pred_dir, list_dir)
    print('final1: acc={:.2f}%,'
          .format(np.mean(acc_list) * 100,))
