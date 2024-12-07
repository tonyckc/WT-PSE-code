# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision import models
import math

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from matplotlib import pyplot as plt
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


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """
    A module which squeezes the last two dimensions,
    ordinary squeeze can be a problem for batch size 1
    """

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class UNetOurs(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, feature_dim=8):
        super(UNetOurs, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_dim = feature_dim

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        #self.mu = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        #                        nn.Conv2d(64, feature_dim, kernel_size=1))
        #self.logvar = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        #                            nn.Conv2d(64, feature_dim, kernel_size=1))
        #self.outc = OutConv(feature_dim, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #mu = self.mu(x)
        #logvar = self.logvar(x)
        #if self.training:
        #    feature = self.reparameterization(mu, logvar)  # b*c*w*h
        #    logits = self.outc(feature)
        #    return logits, torch.stack([mu, logvar, feature], dim=1)
        #else:
        feature = x
        #logits = self.outc(feature)
        return feature #logits, mu

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
class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):
        x = self.instance_standardization(x)
        w = x

        return w
class DeepWT(nn.Module):
    def __init__(self,input_channel,out_channel,whitening):
        super(DeepWT,self).__init__()
        self.whitening = whitening
        if self.whitening:
            self.DoubleConv = DoubleConvWT(in_channels=input_channel,out_channels=out_channel)
            self.IN = nn.Sequential(InstanceWhitening(out_channel), nn.ReLU())
            self.DoubleConv2 = DoubleConvWT(in_channels=out_channel, out_channels=out_channel)
            self.IN2 = nn.Sequential(InstanceWhitening(out_channel), nn.ReLU())
    def forward(self,x):
        '''
        :param x: input tensor
        :return:  embedding list
        '''
        output = []
        if self.whitening:
            z = self.DoubleConv(x)
            if self.whitening:
                z_instance = z
            else:
                z_instance = F.relu(z)
            output.append(z_instance)
            z_instance = F.relu(z_instance)
            z_instance = self.DoubleConv2(z_instance)
            if self.whitening:
                z_instance = z_instance#self.IN(z_instance)
            else:
                z_instance = F.relu(z_instance)
            output.append(z_instance)
            z_instance = F.relu(z_instance)
            output.append(z_instance)
        else:
            output.append(x)
        return output
class compute_MMD(object):
    def __init__(self,domain_num,batch_size):
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
        #minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = self.domain_num

        features = [inputs[self.batch_size*(xi):self.batch_size*(xi+1)] for xi in range(nmb)]
        #classifs = [self.classifier(fi) for fi in features]
        #targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            #objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        #objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        #self.optimizer.zero_grad()
        #(objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        #self.optimizer.step()

        if torch.is_tensor(penalty):
            loss = penalty.item()

        return penalty


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


class ShapeVariationalDist_x(nn.Module):
    def __init__(self,hparams,device,n_classes,number_source_domain=3,batch_size=3):
        super(ShapeVariationalDist_x, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hparams = hparams
        self.wt = hparams['whitening']
        self.momentum = 0.99
        self.number_source_domain = number_source_domain
        self.whitening = hparams['whitening']

        self.eps = 1e-5
        self.margin = hparams['margin']
        n = 16
        self.dim = n
        self.wt_model = DeepWT(3, n, whitening=self.whitening)
        if not self.wt:
            self.inc = DoubleConv(3, n)

        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='none')

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

        self.mmd_operator = compute_MMD(domain_num=3, batch_size=self.batch_size)
        self.i = torch.eye(self.dim, self.dim).cuda()

        self.diagonal = torch.eye(self.dim).cuda()

        self.num_diagonal = torch.sum(self.diagonal)

        self.reversal_i = torch.ones(self.dim, self.dim).triu(diagonal=1).cuda()
        self.num_off_diagonal = torch.sum(self.reversal_i)

        # distribution parameters
        self.mu_prior = nn.Sequential(nn.Conv2d(2 * n, 2 * n, kernel_size=1), nn.ReLU(),
                                      nn.Conv2d(2 * n, 8, kernel_size=1), nn.ReLU(),
                                      nn.Conv2d(8, n_classes, kernel_size=1))

        self.logvar_prior = nn.Sequential(nn.Conv2d(2 * n, 2 * n, kernel_size=1), nn.ReLU(),
                                          nn.Conv2d(2 * n, 8, kernel_size=1), nn.ReLU(),
                                          nn.Conv2d(8, n_classes, kernel_size=1))


    def unet_extractor(self,inputs):
        if self.wt:
            x1 = inputs
        else:
            x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def sample_forward(self,inputs,training):
        b, c, w, h = inputs.shape
        feature_map = self.unet_extractor(inputs)
        mu = self.mu_prior(feature_map)
        logvar = self.logvar_prior(feature_map)

        if training:
            if torch.isnan(mu).any():
                mu = torch.nan_to_num(mu)
                mu[mu == float("Inf")] = 0
            feature = self.reparameterization(mu, logvar)  # b*c*w*h
            return feature,mu
        else:
            if torch.isnan(mu).any():
                mu = torch.nan_to_num(mu)
                mu[mu == float("Inf")] = 0
            feature = mu
            return feature

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        if torch.isnan(std).any():
            std = torch.nan_to_num(std)
            std[std == float("Inf")] = 0
        sampled_z = torch.normal(mu, std)

        z = sampled_z * std + mu
        return z

    def update(self, main_network,inputs, mask, step=0, plot_show=0, two_stage_inputs=None, two_step=False):

        if self.hparams['shape_prior']:
            if two_step:
                whiting_outputs1 = main_network.wt_model.forward(two_stage_inputs)
                whiting_outputs2 = self.wt_model.forward(two_stage_inputs)
            else:
                whiting_outputs1 = main_network.wt_model.forward(inputs)
                whiting_outputs2 = self.wt_model.forward(inputs)


            # teacher model inputs: WT features and mask
            z_posterior, z_posterior_mu = main_network.prior_dist.sample_forward(whiting_outputs1[-1], mask, training=True)
            # student model inputs: only single WT features
            z_pre,z_pre_mu = self.sample_forward(whiting_outputs2[-1],training=True)

            # knowledge distillation loss, we actually ignore the logvar to compute the KD loss for numerical stabilization; logvar is still for feature reparameterization
            kd_loss = self.wasser_distance(z_posterior_mu, z_pre_mu)

            if self.hparams['shape_attention']:

                z_posterior_attention, _ = main_network.attention_layer.forward(z_posterior)

                _, z_pre_attention = main_network.attention_layer.forward(z_pre)

            if self.hparams['whitening']:
                # instance_loss_NonDiag
                instance_wt_loss = 0
                # instance_loss_diag
                instance_wt_loss2 = 0
                # domain loss
                domain_wt_loss = 0
                num_embeddings = len(whiting_outputs2)
                for embedding_wt in range(num_embeddings-1):
                    instance_wt_loss1, instance_wt_loss2,domain_wt_loss1 = self.compute_whitening_loss(whiting_outputs2[embedding_wt])
                    instance_wt_loss += instance_wt_loss1
                    instance_wt_loss2 += instance_wt_loss2
                    domain_wt_loss += domain_wt_loss1

                instance_wt_loss /= num_embeddings
                instance_wt_loss2 /= num_embeddings
                instance_wt_loss_total = instance_wt_loss + instance_wt_loss2
                domain_wt_loss /= num_embeddings
        if self.hparams['whitening']:
            return kd_loss,instance_wt_loss_total,instance_wt_loss,instance_wt_loss2,domain_wt_loss
        else:
            return 0, 0, 0,0,0
    def get_eye_matrix(self):
        return self.i, self.reversal_i
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
            return torch.eye(tensor.shape[1]).repeat(tensor.shape[0], 1, 1)

        diagonal_matrix = eye_like(f_cor_masked_diag).cuda()

        diag_sum = torch.sum(torch.abs(f_cor_masked_diag-diagonal_matrix), dim=(1, 2), keepdim=True) - self.margin  # B X 1 X 1
        instance_loss_diag = torch.clamp(torch.div(diag_sum, self.num_diagonal), min=0)  # B X 1 X 1
        instance_loss_diag = torch.sum(instance_loss_diag) / B

        # domian whitening loss

        # make the off diagonal elements of each covariance matrix as vectors
        index_upper_triangle = torch.triu_indices(self.dim, self.dim, 1).cuda()
        vector_ut_domains = f_cor_masked[:, index_upper_triangle[0], index_upper_triangle[1]]
        domain_loss = self.mmd_operator.forward(vector_ut_domains)

        return instance_loss, instance_loss_diag,domain_loss

    def wasser_distance(self, prior_space_mu, posterior_space_mu,):
            return self.mse_mean(prior_space_mu, posterior_space_mu)


    def predict(self, inputs_all):
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
                z_posterior = self.posterior_dist.sample_forward(whiting_outputs1[-1], training=False)


            else:
                whiting_outputs1 = self.wt_model.forward(inputs)
                z_posterior = self.posterior_dist.sample_forward(whiting_outputs1[-1], training=False)


            if self.hparams['shape_attention']:

                z_posterior_attention, no_sigmoid_embeddings = self.attention_layer.forward(z_posterior)
                fuse_embedding = self.learn_parameter * embedding + (z_posterior_attention * embedding)

            else:
                fuse_embedding = embedding
            if self.cat_shape:
                embedding = torch.cat([fuse_embedding, z_posterior], 1)
            else:
                embedding = fuse_embedding

        output = self.outc(embedding)
        return output, no_sigmoid_embeddings

