# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import torch.optim as optim
from typing import List
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import random
# from torch.utils.serialization import torchfile.load
import torchfile
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from matplotlib import pyplot as plt
#  import higher

from torch.distributions import Normal, Independent, kl

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)
        init.xavier_uniform(m.bias)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        init.constant(m.bias, 0)

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0
        sum2 = 0.0
        num = 0

        flat_mask = torch.flatten(mask)
        assert (len(flat_mask) == len(diff2))
        for i in range(len(diff2)):
            if flat_mask[i] == 1:
                sum2 += diff2[i]
                num += 1

        return sum2 / num


class compute_MMD(object):
    def __init__(self, domain_num, batch_size):
        self.domain_num = domain_num
        self.batch_size = batch_size
        self.kernel_type = "gaussian"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[1]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):

        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def forward(self, inputs, **kwargs):
        # minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = self.domain_num
        features = [inputs[self.batch_size * (xi):self.batch_size * (xi + 1)] for xi in range(nmb)]


        for i in range(nmb):
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])


        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        if torch.is_tensor(penalty):
            loss = penalty.item()

        return penalty


def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = -(0.5) * np.log(2 * np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term).mean()
    return out


class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        self.det_sig_term = -np.log(self.sigma)

    def loglike(self, x, do_sum=True):

        dist_term = -(0.5) * ((x - self.mu) / self.sigma) ** 2
        if do_sum:
            return (self.cte_term + self.det_sig_term + dist_term).sum()
        else:
            return (self.cte_term + self.det_sig_term + dist_term).mean()


class isotropic_mixture_gauss_prior(object):
    def __init__(self, mu1=0, mu2=0, sigma1=0.1, sigma2=1.5, pi=0.5):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.pi1 = pi
        self.pi2 = 1 - pi

        self.cte_term = -(0.5) * np.log(2 * np.pi)

        self.det_sig_term1 = -np.log(self.sigma1)

        self.det_sig_term2 = -np.log(self.sigma2)

    def loglike(self, x, do_sum=True):

        dist_term1 = -(0.5) * ((x - self.mu1) / self.sigma1) ** 2
        dist_term2 = -(0.5) * ((x - self.mu2) / self.sigma2) ** 2

        if do_sum:
            return (torch.log(
                self.pi1 * torch.exp(self.cte_term + self.det_sig_term1 + dist_term1) + self.pi2 * torch.exp(
                    self.cte_term + self.det_sig_term2 + dist_term2))).sum()
        else:
            return (torch.log(
                self.pi1 * torch.exp(self.cte_term + self.det_sig_term1 + dist_term1) + self.pi2 * torch.exp(
                    self.cte_term + self.det_sig_term2 + dist_term2))).mean()





class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.loss_func = FocalLoss(class_num=7, gamma=2.)

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        # loss = F.cross_entropy(self.predict(all_x), all_y)
        loss = self.loss_func(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)




class DSU(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DSU, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.DSU(uncertainty=0.5)
        self.classifier = nn.Linear(512, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        # self.loss_func = FocalLoss(class_num=7, gamma=2.)

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        # loss = self.loss_func(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)










class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvWT(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)





class encoder4(nn.Module):
    def __init__(self, vgg):
        super(encoder4, self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(0).weight).float())
        self.conv1.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(0).bias).float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(2).weight).float())
        self.conv2.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(2).bias).float())
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(5).weight).float())
        self.conv3.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(5).bias).float())
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(9).weight).float())
        self.conv4.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(9).bias).float())
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(12).weight).float())
        self.conv5.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(12).bias).float())
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(16).weight).float())
        self.conv6.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(16).bias).float())
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(19).weight).float())
        self.conv7.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(19).bias).float())
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv8.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(22).weight).float())
        self.conv8.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(22).bias).float())
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv9.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(25).weight).float())
        self.conv9.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(25).bias).float())
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv10.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(29).weight).float())
        self.conv10.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(29).bias).float())
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out, pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out, pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        pool3 = self.relu9(out)
        out, pool_idx3 = self.maxPool3(pool3)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        return out


class encoder1(nn.Module):
    def __init__(self, vgg1):
        super(encoder1, self).__init__()
        # dissemble vgg2 and decoder2 layer by layer
        # then resemble a new encoder-decoder network
        # 224 x 224
        self.requires_grad = False
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        # self.conv1.weight = torch.nn.Parameter(vgg1.get(0).weight.float())
        self.conv1.weight = torch.nn.Parameter(torch.from_numpy(vgg1.get(0).weight).float())
        self.conv1.bias = torch.nn.Parameter(torch.from_numpy(vgg1.get(0).bias).float())
        # 224 x 224
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(torch.from_numpy(vgg1.get(2).weight).float())
        self.conv2.bias = torch.nn.Parameter(torch.from_numpy(vgg1.get(2).bias).float())

        self.relu = nn.ReLU(inplace=True)
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        # 224 x 224

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class decoder1(nn.Module):
    def __init__(self, d1):
        super(decoder1, self).__init__()
        self.requires_grad = False
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(torch.from_numpy(d1.get(1).weight).float())
        self.conv3.bias = torch.nn.Parameter(torch.from_numpy(d1.get(1).bias).float())

        # 224 x 224

        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.reflecPad2(x)
        out = self.conv3(out)
        return out


class decoder4(nn.Module):
    def __init__(self, d):
        super(decoder4, self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv11.weight = torch.nn.Parameter(torch.from_numpy(d.get(1).weight).float())
        self.conv11.bias = torch.nn.Parameter(torch.from_numpy(d.get(1).bias).float())
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv12.weight = torch.nn.Parameter(torch.from_numpy(d.get(5).weight).float())
        self.conv12.bias = torch.nn.Parameter(torch.from_numpy(d.get(5).bias).float())
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv13.weight = torch.nn.Parameter(torch.from_numpy(d.get(8).weight).float())
        self.conv13.bias = torch.nn.Parameter(torch.from_numpy(d.get(8).bias).float())
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv14.weight = torch.nn.Parameter(torch.from_numpy(d.get(11).weight).float())
        self.conv14.bias = torch.nn.Parameter(torch.from_numpy(d.get(11).bias).float())
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv15.weight = torch.nn.Parameter(torch.from_numpy(d.get(14).weight).float())
        self.conv15.bias = torch.nn.Parameter(torch.from_numpy(d.get(14).bias).float())
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv16.weight = torch.nn.Parameter(torch.from_numpy(d.get(18).weight).float())
        self.conv16.bias = torch.nn.Parameter(torch.from_numpy(d.get(18).bias).float())
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv17.weight = torch.nn.Parameter(torch.from_numpy(d.get(21).weight).float())
        self.conv17.bias = torch.nn.Parameter(torch.from_numpy(d.get(21).bias).float())
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv18.weight = torch.nn.Parameter(torch.from_numpy(d.get(25).weight).float())
        self.conv18.bias = torch.nn.Parameter(torch.from_numpy(d.get(25).bias).float())
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv19.weight = torch.nn.Parameter(torch.from_numpy(d.get(28).weight).float())
        self.conv19.bias = torch.nn.Parameter(torch.from_numpy(d.get(28).bias).float())

    def forward(self, x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out


class encoder2(nn.Module):
    def __init__(self, vgg):
        super(encoder2, self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(0).weight).float())
        self.conv1.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(0).bias).float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(2).weight).float())
        self.conv2.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(2).bias).float())
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(5).weight).float())
        self.conv3.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(5).bias).float())
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(torch.from_numpy(vgg.get(9).weight).float())
        self.conv4.bias = torch.nn.Parameter(torch.from_numpy(vgg.get(9).bias).float())
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool = self.relu3(out)
        out, pool_idx = self.maxPool(pool)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        return out


class decoder2(nn.Module):
    def __init__(self, d):
        super(decoder2, self).__init__()
        # decoder
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(torch.from_numpy(d.get(1).weight).float())
        self.conv5.bias = torch.nn.Parameter(torch.from_numpy(d.get(1).bias).float())
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(torch.from_numpy(d.get(5).weight).float())
        self.conv6.bias = torch.nn.Parameter(torch.from_numpy(d.get(5).bias).float())
        self.relu6 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(torch.from_numpy(d.get(8).weight).float())
        self.conv7.bias = torch.nn.Parameter(torch.from_numpy(d.get(8).bias).float())

    def forward(self, x):
        out = self.reflecPad5(x)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.unpool(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        return out


class pytorch_lua_wrapper:
    def __init__(self, lua_path):
        self.lua_model = torchfile.load(lua_path)

    def get(self, idx):
        return self.lua_model._obj.modules[idx]._obj
def normalization(planes, norm='gn', num_domains=None):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(1, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    elif norm == 'dsbn':
        m = DomainSpecificBatchNorm2d(planes, num_domains=num_domains)
    else:
        raise ValueError('Normalization type {} is not supporter'.format(norm))
    return m


#### Note: All are functional units except the norms, which are sequential
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='bn', first=False, activation='relu'):
        super(ConvD, self).__init__()

        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):


        if not self.first:
            x = self.maxpool2D(x)

        #layer 1 conv, bn
        x = self.conv1(x)
        x = self.bn1(x)

        #layer 2 conv, bn, relu
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.activation(y)

        #layer 3 conv, bn
        z = self.conv3(y)
        z = self.bn3(z)
        z = self.activation(z)

        return z


class ConvU(nn.Module):
    def __init__(self, planes, norm='bn', first=False, activation='relu'):
        super(ConvU, self).__init__()

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2*planes, planes, 3, 1, 1, bias=True)
            self.bn1   = normalization(planes, norm)

        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(planes, planes//2, 1, 1, 0, bias=True)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = normalization(planes, norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x, prev):
        #layer 1 conv, bn, relu
        if not self.first:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)

        #upsample, layer 2 conv, bn, relu
        y = self.pool(x)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)

        #concatenation of two layers
        y = torch.cat([prev, y], 1)

        #layer 3 conv, bn
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.activation(y)

        return y


class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):
        x = self.instance_standardization(x)
        w = x

        return x, w



class ShapeVariationalDist_y_x(nn.Module):
    def __init__(self, hparams, device, n_channels, bilinear, n_classes, wt=True, prior=True, number_source_domain=3):
        super(ShapeVariationalDist_y_x, self).__init__()
        self.device = device
        self.prior = prior
        self.wt = hparams['whitening']
        self.momentum = 0.99
        self.number_source_domain = number_source_domain
        # here
        n = 16
        if self.wt:
            self.inc = DoubleConv(n_channels, n)
            self.fusion = nn.Sequential(nn.Conv2d(n*2, n, kernel_size=1), nn.ReLU())
        else:

            self.inc = DoubleConv(n_channels + 1, n)
        norm = 'bn'
        activation = 'relu'
        self.down1 = ConvD(n, 2 * n, norm, activation=activation)
        self.down2 = ConvD(2 * n, 4 * n, norm, activation=activation)
        self.down3 = ConvD(4 * n, 8 * n, norm, activation=activation)
        self.down4 = ConvD(8 * n, 16 * n, norm, activation=activation)
        self.up1 = ConvU(16 * n, norm, first=True, activation=activation)
        self.up2 = ConvU(8 * n, norm, activation=activation)
        self.up3 = ConvU(4 * n, norm, activation=activation)
        self.up4 = ConvU(2 * n, norm, activation=activation)
        # TODO:UPDATA the structure of the variaional network which is the same as the main network
        self.mu_prior = nn.Sequential(nn.Conv2d(2 * n, 2 * n, kernel_size=1), nn.ReLU(),
                                      nn.Conv2d(2 * n, 8, kernel_size=1), nn.ReLU(),
                                      nn.Conv2d(8, n_classes, kernel_size=1))

        self.logvar_prior = nn.Sequential(nn.Conv2d(2 * n, 2 * n, kernel_size=1), nn.ReLU(),
                                          nn.Conv2d(2 * n, 8, kernel_size=1), nn.ReLU(),
                                          nn.Conv2d(8, n_classes, kernel_size=1))

    def unet_extractor(self, inputs, mask):
        if self.wt:

            mask_x1 = self.inc(mask)
            x1 = torch.cat([mask_x1, inputs], dim=1)
            x1 = self.fusion(x1) #mask_x1#
        else:

            x1 = torch.cat([mask, inputs], dim=1)
            x1 = self.inc(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def distrubution_forward(self, inputs, mask=None):

        b, c, w, h = inputs.shape
        feature_map = self.unet_extractor(inputs, mask)
        mu = self.mu_prior(feature_map)
        mu = torch.sigmoid(mu)
        logvar = self.logvar_prior(feature_map)
        mu = mu.view(b, -1)
        log_sigma = logvar.view(b, -1)
        if torch.isnan(mu).any() == True or torch.isinf(mu).any() == True:
            mu = torch.nan_to_num(mu)
            mu[mu == float("Inf")] = 0
        if torch.isnan(log_sigma).any():
            log_sigma = torch.nan_to_num(log_sigma)
            log_sigma[log_sigma == float("Inf")] = 0
        log_sigma = torch.exp(log_sigma)
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)

        return dist

    def sample_forward(self, inputs, mask=None, training=True):
        b, c, w, h = inputs.shape

        feature_map = self.unet_extractor(inputs, mask)
        mu = self.mu_prior(feature_map)
        logvar = self.logvar_prior(feature_map)
        if training:
            feature = self.reparameterization(mu, logvar)  # b*c*w*h
            return feature, mu
        else:
            feature = mu
            return feature

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        # sampled_z = torch.tensor(np.random.normal(0, 1, (mu.size(0), mu.size(1))))
        # sampled_z = torch.normal(mu, std)
        epsilon = torch.randn_like(std).to(self.device)  # sampling epsilon
        z = mu + std * epsilon
        # z = sampled_z * std + mu
        return z




class DeepWT(nn.Module):
    def __init__(self, input_channel, out_channel, whitening):
        super(DeepWT, self).__init__()
        self.whitening = whitening
        if self.whitening:
            self.whitening = whitening
            self.DoubleConv = DoubleConvWT(in_channels=input_channel, out_channels=out_channel)
            self.IN = nn.Sequential(InstanceWhitening(out_channel), nn.ReLU())
            self.DoubleConv2 = DoubleConvWT(in_channels=out_channel, out_channels=out_channel)
            self.IN2 = nn.Sequential(InstanceWhitening(out_channel), nn.ReLU())

    def forward(self, x):
        '''
        :param x: input tensor
        :return:  embedding list
        '''
        output = []
        if self.whitening:

            z = self.DoubleConv(x)
            if self.whitening:
                z_instance = z#self.IN(z)
            else:
                z_instance = F.relu(z)
            output.append(z_instance)
            z_instance = F.relu(z_instance)
            z_instance = self.DoubleConv2(z_instance)
            if self.whitening:
                z_instance = z_instance #self.IN(z_instance)
            else:
                z_instance = F.relu(z_instance)
            output.append(z_instance)
            z_instance = F.relu(z_instance)
            output.append(z_instance)

        else:
            output.append(x)
        return output


class attention_layer(nn.Module):
    def __init__(self, channel1, channel2):
        super(attention_layer, self).__init__()
        self.layer1 = nn.Conv2d(channel1, channel2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.layer1(x)
        out = self.sigmoid(x1)  # F.softmax(x1.reshape(x1.size(0), x1.size(1), -1), 2).view_as(x1)#self.softmax(x1)
        return out, x1




class WT_PSE(Algorithm):
    def __init__(self, n_channels, n_classes, hparams, device, two_step, per_domain_batch=8, source_domain_num=3,
                 feature_dim=8, bilinear=True):
        super(WT_PSE, self).__init__(n_channels, n_classes, hparams, device)
        self.n_channels = n_channels
        self.device = device
        self.eps = 1e-5
        self.n_classes = n_classes
        self.num_domains = 3
        self.mmd_operator = compute_MMD(domain_num=source_domain_num, batch_size=per_domain_batch)
        self.bilinear = bilinear

        self.feature_dim = feature_dim
        self.hparams = hparams
        self.two_step = two_step
        self.per_domain_batch = per_domain_batch
        self.number_source_domain = source_domain_num

        self.whitening = hparams['whitening']
        self.start_shape_step = hparams['shape_start']
        self.cat_shape = hparams['cat_shape']
        self.margin = hparams['margin']
        # dimension of intermedicated features of WT
        self.dim = 16

        n = 16
        # here we difine a WT model and related parameters
        self.wt_model = DeepWT(3, n, whitening=self.whitening)
        self.i = torch.eye(self.dim, self.dim).cuda()
        self.reversal_i = torch.ones(self.dim, self.dim).triu(diagonal=1).cuda()
        self.diagonal = torch.eye(self.dim).cuda()

        self.num_diagonal = torch.sum(self.diagonal)

        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='none')
        self.mask_mse = MaskedMSELoss()

        # Unet model
        norm = 'bn'
        activation = 'relu'
        self.inc = ConvD(self.n_channels, n, norm, first=True, activation=activation)
        self.down1 = ConvD(n,   2*n, norm, activation=activation)
        self.down2 = ConvD(2*n, 4*n, norm, activation=activation)
        self.down3 = ConvD(4*n, 8*n, norm, activation=activation)
        self.down4 = ConvD(8*n,16*n, norm, activation=activation)
        self.up1 = ConvU(16*n, norm, first=True, activation=activation)
        self.up2 = ConvU(8*n, norm, activation=activation)
        self.up3 = ConvU(4*n, norm, activation=activation)
        self.up4 = ConvU(2*n, norm, activation=activation)


        if hparams['shape_prior']:
            # shape network
            self.prior_dist = ShapeVariationalDist_y_x(hparams, self.device, 1, bilinear, n_classes=1,
                                                       wt=self.whitening, prior=True,
                                                       number_source_domain=self.number_source_domain)
            if self.cat_shape:
                feature_dim_fuse = feature_dim + 1
            else:
                feature_dim_fuse = feature_dim
        else:
            feature_dim_fuse = feature_dim

        self.mu = nn.Sequential(nn.Conv2d(2*n, 2*n, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(2*n, feature_dim, kernel_size=1))
        self.outc = nn.Sequential(nn.Conv2d(feature_dim_fuse, n_classes, kernel_size=1))


        self.attention_layer = attention_layer(1, 1)
        self.global_step = 0

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = torch.normal(mu, std)
        z = sampled_z * std + mu
        return z

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def update(self, inputs, mask, step=0, plot_show=0, two_stage_inputs=None, sp_mask=None, two_step=False):
        b, c, w, h = inputs.shape
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        embedding = self.mu(x)

        # insert shape regularization
        if self.hparams['shape_prior']:
            # two step denotes the second phase of a coarse-to-fine segmentation strategy
            if two_step:
                whiting_outputs1 = self.wt_model.forward(two_stage_inputs)
            else:
                whiting_outputs1 = self.wt_model.forward(inputs)

            # whiting_outputs is a list that save the intermedicated features and the final features of wt process
            z_posterior, z_posterior_mu = self.prior_dist.sample_forward(whiting_outputs1[-1], mask, training=True)


            if self.hparams['shape_attention']:
                #
                z_posterior_attention, _ = self.attention_layer.forward(z_posterior)
                z_posterior_attention_mask = (z_posterior_attention > 0.75)
                z_posterior_attention_mask = z_posterior_attention_mask.float()

                # fuse the shape regularization and the embedding
                fuse_embedding = self.hparams['shape_attention_coeffient']  * embedding +  (
                            z_posterior_attention * embedding)
            else:
                fuse_embedding = embedding

            embedding = torch.cat([fuse_embedding, z_posterior], 1) if self.cat_shape else fuse_embedding

        # compute wt loss
        if self.hparams['whitening']:
            instance_wt_loss = 0
            domain_wt_loss = 0
            num_embeddings = len(whiting_outputs1)
            # compute the WT loss for each intermediated features
            for embedding_wt in range(num_embeddings-1):
                instance_wt_loss1, domain_wt_loss1 = self.compute_whitening_loss(whiting_outputs1[embedding_wt])
                instance_wt_loss += instance_wt_loss1
                domain_wt_loss += domain_wt_loss1

            instance_wt_loss /= num_embeddings
            domain_wt_loss /= num_embeddings
        # final output
        output = self.outc(embedding)
        if self.hparams['shape_prior'] == True and self.hparams['whitening'] == True:
            return output, z_posterior_attention_mask, z_posterior_attention_mask, instance_wt_loss, domain_wt_loss,
        elif self.hparams['shape_prior'] == True and self.hparams['whitening'] == False:
            return output, z_posterior_attention_mask, z_posterior_attention_mask, 0, 0
        else:
            return output, 0, 0, 0, 0

    def compute_whitening_loss(self, z):
        B, C, H, W = z.shape  # i-th feature size (B X C X H X W)
        HW = H * W
        f_map = z.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        eye, reverse_eye = self.get_eye_matrix()
        # compute the covariance for each feature map
        f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # B X C X C / HW
        f_cor_masked = f_cor * reverse_eye

        f_cor_masked_diag = f_cor * self.diagonal

        # instance whitening loss
        off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1, 2), keepdim=True) - self.margin  # B X 1 X 1
        instance_loss = torch.clamp(torch.div(off_diag_sum, self.num_off_diagonal), min=0)  # B X 1 X 1
        instance_loss = torch.sum(instance_loss) / B

        def eye_like(tensor):
            return torch.eye(tensor.shape[1]).repeat(tensor.shape[0],1,1)

        diagonal_matrix = eye_like(f_cor_masked_diag).cuda()
        diag_sum = torch.sum(torch.abs(f_cor_masked_diag-diagonal_matrix), dim=(1, 2), keepdim=True) - self.margin  # B X 1 X 1
        instance_loss_diag = torch.clamp(torch.div(diag_sum, self.num_diagonal), min=0)  # B X 1 X 1
        instance_loss_diag = torch.sum(instance_loss_diag) / B

        instance_loss += instance_loss_diag

        # domian whitening loss
        # make the off diagonal elements of each covariance matrix as vectors
        index_upper_triangle = torch.triu_indices(self.dim, self.dim, 1).cuda()
        vector_ut_domains = f_cor_masked[:, index_upper_triangle[0], index_upper_triangle[1]]
        domain_loss = self.mmd_operator.forward(vector_ut_domains)

        return instance_loss, domain_loss

    def predict(self, learn_x_network, inputs_all):
        if self.two_step:
            inputs = inputs_all[0]
            two_stage_inputs = inputs_all[1]
        else:
            inputs = inputs_all
        b, c, w, h = inputs.shape
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        embedding = self.mu(x)

        # insert shape regularization
        if self.hparams['shape_prior']:
            # two step denotes the second phase of a coarse-to-fine segmentation strategy
            if self.two_step:
                whiting_outputs1 = learn_x_network.wt_model.forward(two_stage_inputs)
                z_posterior = learn_x_network.sample_forward(whiting_outputs1[-1], training=False)

            else:
                whiting_outputs1 = learn_x_network.wt_model.forward(inputs)
                z_posterior = learn_x_network.sample_forward(whiting_outputs1[-1], training=False)

            if self.hparams['shape_attention']:
                # if step is not up tp the shredshold step, attention is not added.
                z_posterior_attention, no_sigmoid_embeddings = self.attention_layer.forward(z_posterior)
                fuse_embedding = self.hparams['shape_attention_coeffient'] * embedding + (
                        z_posterior_attention * embedding)
            else:
                fuse_embedding = embedding

            embedding = torch.cat([fuse_embedding, z_posterior], 1) if self.cat_shape else fuse_embedding

        else:
            no_sigmoid_embeddings = None
        output = self.outc(embedding)
        return output, no_sigmoid_embeddings



class Unet_nips2023_joint_shape_regularization(Algorithm):
    def __init__(self, n_channels, n_classes, hparams, device, two_step, per_domain_batch=8, source_domain_num=3,
                 feature_dim=8, bilinear=True):
        super(Unet_nips2023_joint_shape_regularization, self).__init__(n_channels, n_classes, hparams, device)
        self.n_channels = n_channels
        self.device = device
        self.eps = 1e-5
        self.n_classes = n_classes
        self.num_domains = 3
        self.mmd_operator = compute_MMD(domain_num=source_domain_num, batch_size=per_domain_batch)
        self.bilinear = bilinear
        # self.learn_parameter = nn.Parameter(torch.tensor([0.5], device=device))
        self.feature_dim = feature_dim
        self.hparams = hparams
        self.two_step = two_step
        self.per_domain_batch = per_domain_batch
        self.number_source_domain = source_domain_num
        # print(self.hparams['local_loss'])
        # exit()
        self.whitening = hparams['whitening']
        self.start_shape_step = hparams['shape_start']
        self.cat_shape = hparams['cat_shape']
        self.margin = hparams['margin']
        self.dim = 16

        n = 16
        # here we difine a
        self.wt_model = DeepWT(3, n, whitening=self.whitening)
        self.i = torch.eye(self.dim, self.dim).cuda()

        # print(torch.ones(16, 16).triu(diagonal=1))
        self.reversal_i = torch.ones(self.dim, self.dim).triu(diagonal=1).cuda()
        self.diagonal = torch.eye(self.dim).cuda()
        # num_off_diagonal = ((dim * dim - dim) // 2)  # number of off-diagonal

        self.num_diagonal = torch.sum(self.diagonal)


        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='none')
        self.mask_mse = MaskedMSELoss()

        # Unet model

        norm = 'bn'
        activation = 'relu'
        self.inc = ConvD(self.n_channels, n, norm, first=True, activation=activation)
        self.down1 = ConvD(n,   2*n, norm, activation=activation)
        self.down2 = ConvD(2*n, 4*n, norm, activation=activation)
        self.down3 = ConvD(4*n, 8*n, norm, activation=activation)
        self.down4 = ConvD(8*n,16*n, norm, activation=activation)
        self.up1 = ConvU(16*n, norm, first=True, activation=activation)
        self.up2 = ConvU(8*n, norm, activation=activation)
        self.up3 = ConvU(4*n, norm, activation=activation)
        self.up4 = ConvU(2*n, norm, activation=activation)

        #TODO: the number of shape class is 1
        if hparams['shape_prior']:
            self.prior_dist = ShapeVariationalDist_x(hparams, self.device, 3, bilinear, n_classes=1,
                                                       wt=self.whitening, prior=True,
                                                       number_source_domain=self.number_source_domain)
            # self.posterior_dist = ShapeVariationalDist_x(hparams,self.device,n_channels,bilinear,n_classes,wt=self.whitening,prior=False,number_source_domain=self.number_source_domain)
            if self.cat_shape:
                feature_dim_fuse = feature_dim + 1
            else:
                feature_dim_fuse = feature_dim
        else:
            feature_dim_fuse = feature_dim
        self.mu = nn.Sequential(nn.Conv2d(2*n, 2*n, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(2*n, feature_dim, kernel_size=1))
        self.outc = nn.Sequential(nn.Conv2d(feature_dim_fuse, n_classes, kernel_size=1))
        # if self.start_shape_step >0:
        #    self.outc_warning = nn.Sequential(nn.Conv2d(feature_dim_fuse-1, n_classes, kernel_size=1))

        self.attention_layer = attention_layer(1, 1)
        self.global_step = 0

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        # sampled_z = torch.tensor(np.random.normal(0, 1, (mu.size(0), mu.size(1))))
        sampled_z = torch.normal(mu, std)
        z = sampled_z * std + mu
        return z

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def update(self, inputs, mask, step=0, plot_show=0, two_stage_inputs=None, sp_mask=None, two_step=False):
        b, c, w, h = inputs.shape
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        embedding = self.mu(x)
        if self.hparams['shape_prior']:
            if two_step:
                whiting_outputs1 = self.wt_model.forward(two_stage_inputs)
                # self.prior_space = self.prior_dist.distrubution_forward(whiting_outputs1[-1], mask)
                # whiting_outputs2 = self.wt_model.forward(two_stage_inputs)
                # self.posterior_space = self.posterior_dist.distrubution_forward(whiting_outputs2[-1])
            else:
                whiting_outputs1 = self.wt_model.forward(inputs)
                # self.prior_space = self.prior_dist.distrubution_forward(whiting_outputs1[-1],mask)
                # whiting_outputs2 = self.wt_model.forward(inputs)
                # self.posterior_space = self.posterior_dist.distrubution_forward(whiting_outputs2[-1])

            # self.kl = torch.mean(
            #    self.kl_divergence(self.prior_space,self.posterior_space,analytic=True, calculate_posterior=False, z_posterior=None))

            if two_step:
                z_posterior, z_posterior_mu = self.prior_dist.sample_forward(whiting_outputs1[-1], None, training=True)
                # z_pre,z_pre_mu = self.posterior_dist.sample_forward(whiting_outputs2[-1],training=True)

            else:
                z_posterior, z_posterior_mu = self.prior_dist.sample_forward(whiting_outputs1[-1], None, training=True)
                # z_pre, z_pre_mu = self.posterior_dist.sample_forward(whiting_outputs2[-1],training=True)

            # self.kl = self.wasser_distance(z_posterior_mu, z_pre_mu,sp_mask,two_stage=two_step)
            if self.hparams['shape_attention']:

                z_posterior_attention, _ = self.attention_layer.forward(z_posterior)

                # _, z_pre_attention = self.attention_layer.forward(z_pre)

                z_posterior_attention_mask = (z_posterior_attention > 0.75)
                z_posterior_attention_mask = z_posterior_attention_mask.float()

                '''
                if plot_show == 0:
                    if step % 2 == 0 and step != 0 and two_step == True:
                        plt.imshow(torch.squeeze(two_stage_inputs[1]).cpu(), cmap='gray')
                        plt.show()

                    #   plt.imshow(torch.squeeze(mask[1]).cpu(), cmap='gray')

                        #inter_mask = (z_posterior_attention).detach().float()
                        #plt.imshow(torch.squeeze(inter_mask[1]).cpu(), cmap='gray')
                        #plt.show()
                        plt.imshow(torch.squeeze(torch.sigmoid(z_posterior_mu)[1]).detach().cpu(), cmap='gray')
                        plt.show()

                        plt.imshow(torch.squeeze(torch.sigmoid(z_pre_mu)[1]).detach().cpu(), cmap='gray')
                        plt.show()
                '''

                fuse_embedding = self.hparams['shape_attention_coeffient']  * embedding +  (
                            z_posterior_attention * embedding)
            # else:
            #    fuse_embedding = embedding
            else:
                fuse_embedding = embedding
            if self.cat_shape:
                embedding = torch.cat([fuse_embedding, z_posterior], 1)
            else:
                embedding = fuse_embedding

        if self.hparams['global_loss']:
            pass

        if self.hparams['whitening']:
            instance_wt_loss = 0
            domain_wt_loss = 0
            num_embeddings = len(whiting_outputs1)
            for embedding_wt in range(num_embeddings-1):
                instance_wt_loss1, domain_wt_loss1 = self.compute_whitening_loss(whiting_outputs1[embedding_wt])
                instance_wt_loss += instance_wt_loss1
                domain_wt_loss += domain_wt_loss1

            instance_wt_loss /= num_embeddings
            domain_wt_loss /= num_embeddings
        output = self.outc(embedding)
        logits = output
        if self.hparams['shape_prior'] == True and self.hparams['whitening'] == True:
            return logits, z_posterior_attention_mask, z_posterior_attention_mask, instance_wt_loss, domain_wt_loss,
        elif self.hparams['shape_prior'] == True and self.hparams['whitening'] == False:
            return logits, z_posterior_attention_mask, z_posterior_attention_mask, 0, 0
        else:
            return logits, 0, 0, 0, 0

    def compute_whitening_loss(self, z):
        B, C, H, W = z.shape  # i-th feature size (B X C X H X W)
        HW = H * W
        f_map = z.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        eye, reverse_eye = self.get_eye_matrix()
        # compute the covariance for each feature map
        f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # B X C X C / HW
        f_cor_masked = f_cor * reverse_eye

        f_cor_masked_diag = f_cor * self.diagonal

        # instance whitening loss
        off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1, 2), keepdim=True) - self.margin  # B X 1 X 1
        instance_loss = torch.clamp(torch.div(off_diag_sum, self.num_off_diagonal), min=0)  # B X 1 X 1
        instance_loss = torch.sum(instance_loss) / B

        def eye_like(tensor):
            return torch.eye(tensor.shape[1]).repeat(tensor.shape[0],1,1)

        diagonal_matrix = eye_like(f_cor_masked_diag).cuda()
        diag_sum = torch.sum(torch.abs(f_cor_masked_diag-diagonal_matrix), dim=(1, 2), keepdim=True) - self.margin  # B X 1 X 1
        instance_loss_diag = torch.clamp(torch.div(diag_sum, self.num_diagonal), min=0)  # B X 1 X 1
        instance_loss_diag = torch.sum(instance_loss_diag) / B

        instance_loss += instance_loss_diag



        # domian whitening loss

        # make the off diagonal elements of each covariance matrix as vectors
        index_upper_triangle = torch.triu_indices(self.dim, self.dim, 1).cuda()
        vector_ut_domains = f_cor_masked[:, index_upper_triangle[0], index_upper_triangle[1]]
        domain_loss = self.mmd_operator.forward(vector_ut_domains)

        return instance_loss, domain_loss

    def wasser_distance(self, prior_space_mu, posterior_space_mu, mask, two_stage):
        if two_stage:
            # loss = self.mse(prior_space_mu,posterior_space_mu)
            # binary mask, mask plus 1 is equal to highlight the ROI
            # weight_mask = mask
            # loss = (loss * weight_mask)
            # loss = torch.mean(loss)
            # return loss
            # return self.mask_mse(prior_space_mu,posterior_space_mu,mask)
            return self.mse_mean(prior_space_mu, posterior_space_mu)

        else:
            return self.mse_mean(prior_space_mu, posterior_space_mu)

    def kl_divergence(self, prior_space, posterior_space, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(prior_space, posterior_space)
        else:
            if calculate_posterior:
                z_posterior = prior_space.rsample()
            log_posterior_prob = prior_space.log_prob(z_posterior)
            log_prior_prob = posterior_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def predict(self, learn_x_network, inputs_all):
        if self.two_step:
            inputs = inputs_all[0]
            two_stage_inputs = inputs_all[1]
        else:
            inputs = inputs_all
        b, c, w, h = inputs.shape
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        embedding = self.mu(x)

        if self.hparams['shape_prior']:
            if self.two_step:
                whiting_outputs1 = self.wt_model.forward(two_stage_inputs)
                z_posterior = self.prior_dist.sample_forward(whiting_outputs1[-1], training=False)

            else:
                whiting_outputs1 = self.wt_model.forward(inputs)
                z_posterior = self.prior_dist.sample_forward(whiting_outputs1[-1], training=False)

            # z_posterior = self.posterior_space.rsample()
            # z_posterior = z_posterior.view(b,c,w,h)
            if self.hparams['shape_attention']:
                # if step is not up tp the shredshold step, attention is not added.

                z_posterior_attention, no_sigmoid_embeddings = self.attention_layer.forward(z_posterior)
                fuse_embedding = self.hparams['shape_attention_coeffient'] * embedding + (
                        z_posterior_attention * embedding)
            # else:
            #    fuse_embedding = embedding
            else:
                fuse_embedding = embedding
            if self.cat_shape:
                embedding = torch.cat([fuse_embedding, z_posterior], 1)
            else:
                embedding = fuse_embedding
        else:
            no_sigmoid_embeddings = None
        output = self.outc(embedding)
        logits = output
        return logits, no_sigmoid_embeddings



