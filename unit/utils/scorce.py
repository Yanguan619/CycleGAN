from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import csv
from pathlib import Path
from typing_extensions import Literal
import os


def calc_ssim(img1: np.array, img2: np.array):
    """SSIM (Structure Similarity Index Measure) 结构衡量指标，用作度量两个给定图像之间的相似性。
    SSIM值的范围为0至1，越大代表图像越相似。如果两张图片完全一样时，SSIM值为1。

    具体由3个纬度组成：亮度，对比度和结构。
    """
    img1 = Image.fromarray(img1).convert("L")
    img2 = Image.fromarray(img2).convert("L")
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score


def calc_psnr(img1: np.array, img2: np.array):
    """峰值信噪比(Peak Signal to Noise Ratio, PSNR)是一种评价图像质量的度量标准。

    因为PSNR值具有局限性，所以它只是衡量最大值信号和背景噪音之间的图像质量参考值。
    PSNR的单位为dB，其值越大，图像失真越少。
    一般来说，PSNR高于40dB说明图像质量几乎与原图一样好；
    在30-40dB之间通常表示图像质量的失真损失在可接受范围内；
    在20-30dB之间说明图像质量比较差；PSNR低于20dB说明图像失真严重。
    """
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score


def calc_score_from_dir(
    img1_dir,
    img2_dir,
    score_type: Literal["SSIM", "PSNR"] = "SSIM",
    out_dir="logs/images",
):
    """
    - 峰值信噪比(Peak Signal to Noise Ratio, PSNR)是一种评价图像质量的度量标准。
    峰值信噪比经常用作图像压缩等领域中信号重建质量的测量方法，它常简单地通过均方差（MSE）进行定义。
        基于均方差（MSE）
        $$
        10*log_{10}^{MAXI^{2}/MSE} = 20*log_{10}^{MAXI/MSE^{1/2}}
        $$
        - MAXI：表示图像颜色的最大数值，8位采样点表示为255。
        因为PSNR值具有局限性，所以它只是衡量最大值信号和背景噪音之间的图像质量参考值。
        PSNR的单位为dB，其值越大，图像失真越少。
        一般来说，PSNR高于40dB说明图像质量几乎与原图一样好；
        在30-40dB之间通常表示图像质量的失真损失在可接受范围内；
        在20-30dB之间说明图像质量比较差；PSNR低于20dB说明图像失真严重。

    - SSIM (Structure Similarity Index Measure) 结构相似性指数度量，用作度量两个给定图像之间的相似性。
    SSIM值的范围为 0~1，越大代表图像越相似。如果两张图片完全一样，SSIM=1。

    具体由3个纬度组成：亮度，对比度和结构。
    结构相似指标可以衡量图片的失真程度，也可以衡量两张图片的相似程度。
    与MSE和PSNR衡量绝对误差不同，SSIM是感知模型，即更符合人眼的直观感受。
    """
    img1_dir, img2_dir = Path(img1_dir), Path(img2_dir)
    img1_paths, img2_paths = img1_dir.iterdir(), img2_dir.iterdir()
    out_path = os.path.join(out_dir, img2_dir.name + "_" + score_type + ".csv")
    cal_score_fun_dict = {"SSIM": ssim, "PSNR": psnr}
    cal_score_fun = cal_score_fun_dict[score_type]

    img_num = 0
    score = 0
    with tqdm(zip(img1_paths, img2_paths), desc=f"calc_{score_type}: ") as img12:
        for img1_path, img2_path in img12:
            img1 = Image.open(img1_path).convert("L")
            img2 = Image.open(img2_path).convert("L")
            img2 = img2.resize(img1.size)
            img1, img2 = np.array(img1), np.array(img2)
            # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
            score += cal_score_fun(img1, img2, data_range=255)
            img_num += 1
    score = score / img_num

    with open(out_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["img1_paths", "img2_paths", "score"])
        writer.writerow([img1_dir, img2_dir, score])
    print(f"save to {out_path}")
    return score


"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[
            classAcc < float("inf")
        ].mean()  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = (
            torch.sum(self.confusionMatrix, axis=1)
            + torch.sum(self.confusionMatrix, axis=0)
            - torch.diag(self.confusionMatrix)
        )  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU < float("inf")].mean()  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= imgLabel != IgLabel
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(
            self.confusion_matrix
        )
        iu = np.diag(self.confusion_matrix) / (
            torch.sum(self.confusion_matrix, axis=1)
            + torch.sum(self.confusion_matrix, axis=0)
            - torch.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(
            imgPredict, imgLabel, ignore_labels
        )  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
