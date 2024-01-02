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
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
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
        # classifs = [self.classifier(fi) for fi in features]
        # targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            # objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        # objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        # self.optimizer.zero_grad()
        # (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        # self.optimizer.step()

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


class BayesLinear_Normalq(nn.Module):
    """Linear Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
     of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
     to estimate the KL term in the ELBO.
    """

    def __init__(self, n_in, n_out, prior_class, with_bias=True):
        super(BayesLinear_Normalq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class
        self.with_bias = with_bias

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        # self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).normal_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))
        # self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).normal_(-2, 0.5))

        # if self.with_bias:
        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        # self.b_mu = nn.Parameter(torch.Tensor(self.n_out).normal_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))
        # self.b_p = nn.Parameter(torch.Tensor(self.n_out).normal_(-2, 0.5))
        # pdb.set_trace()

    def forward(self, X, sample=0, local_rep=False, ifsample=True):
        # # local_rep = True
        # # pdb.set_trace()
        if not ifsample:  # When training return MLE of w for quick validation
            # pdb.set_trace()
            if self.with_bias:
                output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            else:
                output = torch.mm(X, self.W_mu)
            return output, torch.Tensor([0]).cuda()

        else:
            if not local_rep:
                # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
                # the same random sample is used for every element in the minibatch
                # pdb.set_trace()
                W_mu = self.W_mu.unsqueeze(1).repeat(1, sample, 1)
                W_p = self.W_p.unsqueeze(1).repeat(1, sample, 1)

                b_mu = self.b_mu.unsqueeze(0).repeat(sample, 1)
                b_p = self.b_p.unsqueeze(0).repeat(sample, 1)
                # pdb.set_trace()

                eps_W = W_mu.data.new(W_mu.size()).normal_()
                eps_b = b_mu.data.new(b_mu.size()).normal_()

                if not ifsample:
                    eps_W = eps_W * 0
                    eps_b = eps_b * 0

                # sample parameters
                std_w = 1e-6 + f.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + f.softplus(b_p, beta=1, threshold=20)

                W = W_mu + 1 * std_w * eps_W
                b = b_mu + 1 * std_b * eps_b

                if self.with_bias:
                    lqw = isotropic_gauss_loglike(W, W_mu, std_w) + isotropic_gauss_loglike(b, b_mu, std_b)
                    lpw = self.prior.loglike(W) + self.prior.loglike(b)
                else:
                    lqw = isotropic_gauss_loglike(W, W_mu, std_w)
                    lpw = self.prior.loglike(W)

                W = W.view(W.size()[0], -1)
                b = b.view(-1)
                # pdb.set_trace()

                if self.with_bias:
                    output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)
                else:
                    output = torch.mm(X, W)

            else:
                W_mu = self.W_mu.unsqueeze(0).repeat(X.size()[0], 1, 1)
                W_p = self.W_p.unsqueeze(0).repeat(X.size()[0], 1, 1)

                b_mu = self.b_mu.unsqueeze(0).repeat(X.size()[0], 1)
                b_p = self.b_p.unsqueeze(0).repeat(X.size()[0], 1)
                # pdb.set_trace()
                eps_W = W_mu.data.new(W_mu.size()).normal_()
                eps_b = b_mu.data.new(b_mu.size()).normal_()

                # sample parameters
                std_w = 1e-6 + f.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + f.softplus(b_p, beta=1, threshold=20)

                W = W_mu + 1 * std_w * eps_W
                b = b_mu + 1 * std_b * eps_b

                # W = W.view(W.size()[0], -1)
                # b = b.view(-1)
                # pdb.set_trace()

                if self.with_bias:
                    output = torch.bmm(X.view(X.size()[0], 1, X.size()[1]), W).squeeze() + b  # (batch_size, n_output)
                    lqw = isotropic_gauss_loglike(W, W_mu, std_w) + isotropic_gauss_loglike(b, b_mu, std_b)
                    lpw = self.prior.loglike(W) + self.prior.loglike(b)
                else:
                    output = torch.bmm(X.view(X.size()[0], 1, X.size()[1]), W).squeeze()
                    lqw = isotropic_gauss_loglike(W, W_mu, std_w)
                    lpw = self.prior.loglike(W)

            return output, lqw - lpw

    def extra_repr(self):
        return 'in_channel={n_in}, out_channel={n_out}, with_bias={with_bias}, prior={prior}'.format(**self.__dict__)


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


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


class net0(Algorithm):
    def __init__(self, num_class, mctimes, prior_type, with_bias=True, local_rep=True, norm=False, fe='bayes',
                 num_bfe=1):
        super(net0, self).__init__()
        self.MCtimes = mctimes
        self.subpixel_scale = 2
        self.num_class = num_class
        self.prior_type = prior_type
        self.with_bias = with_bias
        self.local_rep = local_rep
        self.feature_extractor = fe
        self.norm = norm
        self.num_bfe = num_bfe

        self.resnet = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool
        )

        if self.feature_extractor == 'linear':
            self.mu_layer = nn.Linear(512, 512)
            self.sigma_layer = nn.Linear(512, 512)
        elif self.feature_extractor == 'bayes':
            if self.num_bfe == 2:
                self.bayesian_layer0 = BayesLinear_Normalq(512, 512, isotropic_mixture_gauss_prior(), with_bias=True)
            self.bayesian_layer = BayesLinear_Normalq(512, 512, isotropic_mixture_gauss_prior(), with_bias=True)
        elif self.feature_extractor == 'line':
            self.fe_layer = nn.Linear(512, 512)

        if self.norm:
            self.normalization = nn.LayerNorm(512)
        # self.normalization = nn.InstanceNorm1d(512)
        # self.normalization = nn.BatchNorm1d(512)

        if self.prior_type == 'NO':
            self.classifier = nn.Linear(512, self.num_class)

        elif self.prior_type == 'SGP':
            # self.bayesian_classfier = BayesLinear_Normalq(512, self.num_class, isotropic_gauss_prior(0, 1), with_bias=self.with_bias)
            # self.bayesian_classfier = BayesLinear_Normalq(512, self.num_class, isotropic_mixture_gauss_prior(0,0,0.1,2), with_bias=self.with_bias)
            self.bayesian_classfier = BayesLinear_Normalq(512, self.num_class, isotropic_mixture_gauss_prior(),
                                                          with_bias=self.with_bias)

    def forward(self, x, label, meta_im, meta_l, meta_classes, num_domains, num_samples_perclass, withnoise=True,
                hierar=False, sampling=True):
        # pdb.set_trace()
        z = self.resnet(x)
        z = z.flatten(1)

        meta_fea = self.resnet(meta_im)
        meta_fea = meta_fea.flatten(1)
        feature_samples = 1
        # meta_fea0 = meta_fea.view(meta_classes, num_domains*num_samples_perclass, meta_fea.size()[-1])

        # z = self.another_layer(z)
        # meta_fea = self.another_layer(meta_fea)

        if self.feature_extractor == 'bayes' and self.num_bfe == 2 and sampling:
            z_mu = torch.mm(z, self.bayesian_layer0.W_mu) + self.bayesian_layer0.b_mu.expand(z.size()[0], 512)
            mf_mu = torch.mm(meta_fea, self.bayesian_layer0.W_mu) + self.bayesian_layer0.b_mu.expand(meta_fea.size()[0],
                                                                                                     512)
            std_z = torch.mm(z ** 2, (f.softplus(self.bayesian_layer0.W_p, beta=1, threshold=20)) ** 2) + (
                f.softplus(self.bayesian_layer0.b_p, beta=1, threshold=20).expand(z.size()[0], 512)) ** 2
            std_mf = torch.mm(meta_fea ** 2, (f.softplus(self.bayesian_layer0.W_p, beta=1, threshold=20)) ** 2) + (
                f.softplus(self.bayesian_layer0.b_p, beta=1, threshold=20).expand(meta_fea.size()[0], 512)) ** 2
            # pdb.set_trace()
            if self.training:
                # kl20 = self.domain_invariance_kl(z_mu, std_z, label,
                #     mf_mu.view(meta_classes, num_domains*num_samples_perclass,-1),
                #     std_mf.view(meta_classes, num_domains*num_samples_perclass,-1))
                kl20 = 0
            else:
                kl20 = 0
            all_fe, phi_entropy0 = self.bayesian_layer0(torch.cat([z, meta_fea], 0), self.MCtimes, self.local_rep,
                                                        ifsample=sampling)

            all_fe = all_fe.view(all_fe.size()[0], self.MCtimes, 512)

            if self.norm:
                all_fe = self.normalization(all_fe)

            all_fe = f.relu(all_fe)

            feature_samples = self.MCtimes

            # whether 2 or 1 layers
            meta_fea = all_fe[z.size()[0]:].view(meta_fea.size()[0], feature_samples, 512).view(
                meta_fea.size()[0] * feature_samples, 512)
            z = all_fe[:z.size()[0]].view(z.size()[0], feature_samples, 512).view(z.size()[0] * feature_samples, 512)
            # meta_fea = all_fe[z.size()[0]:].view(meta_fea.size()[0], feature_samples, 512).mean(1).view(meta_fea.size()[0], 512)
            # z = all_fe[:z.size()[0]].view(z.size()[0], feature_samples, 512).mean(1).view(z.size()[0], 512)

        elif self.feature_extractor == 'bayes' and self.num_bfe == 2 and not sampling:
            feature_samples = 1
            all_fe, phi_entropy0 = self.bayesian_layer0(torch.cat([z, meta_fea], 0), self.MCtimes, self.local_rep,
                                                        sampling)

            if self.norm:
                all_fe = self.normalization(all_fe)

            all_fe = f.relu(all_fe)

            z = all_fe[:z.size()[0]]
            meta_fea = all_fe[z.size()[0]:]
            kl20 = 0

        else:
            feature_samples = 1
            kl20 = 0
            phi_entropy0 = 0

        if self.feature_extractor == 'linear' and self.training:
            # pdb.set_trace()
            feature_samples = self.MCtimes

            z_mu = self.mu_layer(z)
            z_log = self.sigma_layer(z)

            z_mu = z_mu.unsqueeze(1).repeat(1, feature_samples, 1)
            z_log = z_log.unsqueeze(1).repeat(1, feature_samples, 1)

            eps_z = z_mu.data.new(z_mu.size()).normal_()
            # sample parameters
            std_z = 1e-6 + f.softplus(z_log, beta=1, threshold=20)
            z = z_mu + 1 * std_z * eps_z
            # logz = isotropic_gauss_loglike(z, z_mu, std_z)
            z = z.view(z.size()[0] * feature_samples, -1)

            mf_mu = self.mu_layer(meta_fea)
            mf_log = self.sigma_layer(meta_fea)

            mf_mu = mf_mu.unsqueeze(1).repeat(1, feature_samples, 1)
            mf_log = mf_log.unsqueeze(1).repeat(1, feature_samples, 1)

            eps_mf = mf_mu.data.new(mf_mu.size()).normal_()
            # sample parameters
            std_mf = 1e-6 + f.softplus(mf_log, beta=1, threshold=20)
            meta_fea = mf_mu + 1 * std_mf * eps_mf
            # logmf = isotropic_gauss_loglike(mf_mu + 1 * std_mf * eps_mf, mf_mu, std_mf)
            meta_fea = meta_fea.view(meta_fea.size()[0] * feature_samples, -1)
            # pdb.set_trace()
            # KLD
            kl2 = self.domain_invariance_kl(z_mu[:, 0, :], std_z[:, 0, :],
                                            label,
                                            mf_mu[:, 0, :].view(self.num_class, num_domains * num_samples_perclass, -1),
                                            std_mf[:, 0, :].view(self.num_class, num_domains * num_samples_perclass,
                                                                 -1))

            phi_entropy = torch.Tensor([0]).cuda()

        elif self.feature_extractor == 'linear' and not self.training:
            z = self.mu_layer(z)
            meta_fea = self.mu_layer(meta_fea)
            feature_samples = 1
            kl2 = 0
            phi_entropy = 0

        elif self.feature_extractor == 'bayes' and sampling:
            z_mu = torch.mm(z, self.bayesian_layer.W_mu) + self.bayesian_layer.b_mu.expand(z.size()[0], 512)
            mf_mu = torch.mm(meta_fea, self.bayesian_layer.W_mu) + self.bayesian_layer.b_mu.expand(meta_fea.size()[0],
                                                                                                   512)
            std_z = torch.mm(z ** 2, (f.softplus(self.bayesian_layer.W_p, beta=1, threshold=20)) ** 2) + (
                f.softplus(self.bayesian_layer.b_p, beta=1, threshold=20).expand(z.size()[0], 512)) ** 2
            std_mf = torch.mm(meta_fea ** 2, (f.softplus(self.bayesian_layer.W_p, beta=1, threshold=20)) ** 2) + (
                f.softplus(self.bayesian_layer.b_p, beta=1, threshold=20).expand(meta_fea.size()[0], 512)) ** 2
            if self.training:
                # whether 2 or 1 layers
                kl2 = kl20 + self.domain_invariance_kl(z_mu.view(-1, self.MCtimes ** (self.num_bfe - 1), 512),
                                                       std_z.view(-1, self.MCtimes ** (self.num_bfe - 1), 512), label,
                                                       mf_mu.view(meta_classes, num_domains * num_samples_perclass,
                                                                  self.MCtimes ** (self.num_bfe - 1), -1),
                                                       std_mf.view(meta_classes, num_domains * num_samples_perclass,
                                                                   self.MCtimes ** (self.num_bfe - 1), -1))

            else:
                kl2 = kl20 + 0
            all_fe, phi_entropy = self.bayesian_layer(torch.cat([z, meta_fea], 0), self.MCtimes, self.local_rep,
                                                      ifsample=sampling)

            all_fe = all_fe.view(all_fe.size()[0], self.MCtimes, 512)

            if self.norm:
                all_fe = self.normalization(all_fe)

            all_fe = f.relu(all_fe)

            if self.local_rep:
                feature_samples = 1
            else:
                feature_samples = self.MCtimes  # ** self.num_bfe

            # pdb.set_trace()
            meta_fea = all_fe[z.size()[0]:].view(meta_fea.size()[0], feature_samples, 512).view(
                meta_fea.size()[0] * feature_samples, 512)
            z = all_fe[:z.size()[0]].view(z.size()[0], feature_samples, 512).view(z.size()[0] * feature_samples, 512)

            phi_entropy += phi_entropy0


        elif self.feature_extractor == 'bayes' and not sampling:
            feature_samples = 1
            all_fe, phi_entropy = self.bayesian_layer(torch.cat([z, meta_fea], 0), self.MCtimes, self.local_rep,
                                                      sampling)
            if self.norm:
                all_fe = self.normalization(all_fe)

            all_fe = f.relu(all_fe)
            # all_fe = all_fe.view(all_fe.size()[0], -1)

            z = all_fe[:z.size()[0]]
            meta_fea = all_fe[z.size()[0]:]
            kl2 = kl20 + 0

            phi_entropy += phi_entropy0

        elif self.feature_extractor == 'line':
            feature_samples = 1
            all_fe = self.fe_layer(torch.cat([z, meta_fea], 0))

            if self.norm:
                all_fe = self.normalization(all_fe)

            all_fe = f.relu(all_fe)

            z = all_fe[:z.size()[0]]
            meta_fea = all_fe[z.size()[0]:]
            if self.training:
                kl2 = self.domain_invariance_l2(z, label,
                                                meta_fea.view(meta_classes, num_domains * num_samples_perclass, -1))
            else:
                kl2 = 0
            phi_entropy = torch.Tensor([0]).cuda()

        else:
            feature_samples = 1
            kl2 = kl20 + torch.Tensor([0]).cuda()
            phi_entropy = phi_entropy0 + torch.Tensor([0]).cuda()

        if self.prior_type == 'SGP':
            meta_fea0 = meta_fea.view(-1, meta_fea.size()[-1])
            # pdb.set_trace()
            all_f, theta_entropy = self.bayesian_classfier(torch.cat([z, meta_fea0], 0), self.MCtimes, self.local_rep,
                                                           sampling)

            y00 = all_f[:z.size()[0]]
            y_meta = all_f[z.size()[0]:]

            if not self.local_rep and sampling:

                # whether 2 or 1 layers
                y0 = y00.view(x.size()[0], feature_samples ** self.num_bfe, self.MCtimes, self.num_class)
                # y0 = y00.view(x.size()[0], feature_samples, self.MCtimes, self.num_class)
                # y = y0.mean(2)
                # y = y0.mean(1)

                if self.training:
                    # whether 2 or 1 layers
                    y_meta = y_meta.view(meta_classes, num_domains * num_samples_perclass,
                                         feature_samples ** self.num_bfe, self.MCtimes, self.num_class)
                    # y_meta = y_meta.view(meta_classes, num_domains*num_samples_perclass, feature_samples, self.MCtimes, self.num_class)
                else:
                    y_meta = 0
                # y_meta = y_meta.mean(3)

            else:
                y0 = y00.view(x.size()[0], feature_samples ** self.num_bfe, 1, self.num_class)
                # y = y0.mean(2)
                # y = y0.mean(1)

                y_meta = 0
                # y_meta = y_meta.mean(3)


        elif self.prior_type == 'NO':
            theta_entropy = torch.zeros(1).cuda()
            y = self.classifier(z)
            y0 = y.view(x.size()[0], feature_samples ** self.num_bfe, 1, self.num_class)

            meta_fea0 = meta_fea.view(-1, meta_fea.size()[-1])
            y_meta = self.classifier(meta_fea0)
            y_meta = y_meta.view(meta_classes, num_domains * num_samples_perclass, feature_samples ** self.num_bfe, 1,
                                 self.num_class)

        return y0, y_meta, theta_entropy, phi_entropy, kl2

    def domain_invariance_kl(self, x_m, x_s, label, mx_m, mx_s):
        # pdb.set_trace()
        # pdb.set_trace()
        mx_m0 = mx_m[label]
        mx_s0 = mx_s[label]
        x_m = x_m.unsqueeze(1)
        x_s = x_s.unsqueeze(1)

        # pdb.set_trace()
        # kld = 0.5*(torch.log(1e-6 + mx_s0**2) - torch.log(1e-6 + x_s**2)-1+(1e-6 + x_s**2+(mx_m0 - x_m)**2)/(1e-6 + mx_s0**2))
        kld = 0.5 * (torch.log(1e-6 + mx_s0) - torch.log(1e-6 + x_s) - 1 + (1e-6 + x_s + (mx_m0 - x_m) ** 2) / (
                1e-6 + mx_s0))

        return kld.mean()

    def domain_invariance_l2(self, x_m, label, mx_m):
        # pdb.set_trace()

        mx_m0 = mx_m[label]

        x_m = x_m.unsqueeze(1)

        kld = torch.sqrt(torch.sum((mx_m0 - x_m) ** 2, -1))

        return kld.mean()

    def reparameterize(self, mu, logvar, withnoise=True):
        dim = len(mu.size())
        # pdb.set_trace()
        if withnoise:
            if logvar is not None:
                sigma = torch.exp(logvar)
            else:
                sigma = torch.ones(mu.size()).cuda()
            # each instance different dim share one random sample
            if dim == 2:
                eps = torch.cuda.FloatTensor(sigma.size()[0], 1).normal_(0, 1)
            elif dim == 3:
                eps = torch.cuda.FloatTensor(sigma.size()[0], sigma.size()[1], 1).normal_(0, 1)
            else:
                print('the dim of input vector is invalid')
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu


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


class DNA(Algorithm):
    """
    Diversified Neural Averaging(DNA)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DNA, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.MCdropClassifier(
            in_features=self.featurizer.n_outputs,
            num_classes=num_classes,
            bottleneck_dim=self.hparams["bottleneck_dim"],
            dropout_rate=self.hparams["dropout_rate"],
            dropout_type=self.hparams["dropout_type"]
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.train_sample_num = 5
        self.lambda_v = self.hparams["lambda_v"]

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        all_f = self.featurizer(all_x)
        loss_pjs = 0.0
        row_index = torch.arange(0, all_x.size(0))

        probs_y = []
        for i in range(self.train_sample_num):
            pred = self.classifier(all_f)
            prob = F.softmax(pred, dim=1)
            prob_y = prob[row_index, all_y]
            probs_y.append(prob_y.unsqueeze(0))
            loss_pjs += PJS_loss(prob, all_y)

        probs_y = torch.cat(probs_y, dim=0)
        X = torch.sqrt(torch.log(2 / (1 + probs_y)) + probs_y * torch.log(2 * probs_y / (1 + probs_y)) + 1e-6)
        loss_v = (X.pow(2).mean(dim=0) - X.mean(dim=0).pow(2)).mean()
        loss_pjs /= self.train_sample_num
        loss = loss_pjs - self.lambda_v * loss_v

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "loss_c": loss_pjs.item(), "loss_v": loss_v.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class SAM(ERM):
    """Sharpness-Aware Minimization
    """

    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, x, y, **kwargs):
        self.update_count += 1
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        minibatches = to_minibatch(x, y)
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx: all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx: all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """

    def update(self, x, y, **kwargs):
        x = torch.cat(x)
        y = torch.cat(y)

        indices = torch.randperm(x.size(0))
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

        x = lam * x + (1 - lam) * x2
        predictions = self.predict(x)

        objective = lam * F.cross_entropy(predictions, y)
        objective += (1 - lam) * F.cross_entropy(predictions, y2)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class CutMix(ERM):
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update(self, x, y, **kwargs):
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        x = torch.cat(x)
        y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            output = self.predict(x)
            objective = F.cross_entropy(output, target_a) * lam + F.cross_entropy(
                output, target_b
            ) * (1.0 - lam)
        else:
            output = self.predict(x)
            objective = F.cross_entropy(output, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        minibatches = to_minibatch(x, y)
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(
                self.hparams["optimizer"],
                #  "SGD",
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # 1. Compute supervised loss for meta-train set
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # 2. Compute meta loss for meta-val set
            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}


#  class SOMLDG(MLDG):
#      """Second-order MLDG"""
#      # This commented "update" method back-propagates through the gradients of
#      # the inner update, as suggested in the original MAML paper.  However, this
#      # is twice as expensive as the uncommented "update" method, which does not
#      # compute second-order derivatives, implementing the First-Order MAML
#      # method (FOMAML) described in the original MAML paper.

#      def update(self, x, y, **kwargs):
#          minibatches = to_minibatch(x, y)
#          objective = 0
#          beta = self.hparams["mldg_beta"]
#          inner_iterations = self.hparams.get("inner_iterations", 1)

#          self.optimizer.zero_grad()

#          with higher.innerloop_ctx(
#              self.network, self.optimizer, copy_initial_weights=False
#          ) as (inner_network, inner_optimizer):
#              for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
#                  for inner_iteration in range(inner_iterations):
#                      li = F.cross_entropy(inner_network(xi), yi)
#                      inner_optimizer.step(li)

#                  objective += F.cross_entropy(self.network(xi), yi)
#                  objective += beta * F.cross_entropy(inner_network(xj), yj)

#              objective /= len(minibatches)
#              objective.backward()

#          self.optimizer.step()

#          return {"loss": objective.item()}


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


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


'''
class ICML22_MMD_Unet(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,device,skin=False,optim=False):
        super(ICML22_MMD_Unet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.num_domains = num_domains
        #print(num_domains)
        #exit()
        self.featurizer__spinal_cord = networks.Featurizer(input_shape, self.hparams)
        self.featurizer__gm = networks.Featurizer(input_shape, self.hparams)
        #print(self.featurizer__spinal_cord.shape)
        #exit()
        # the embeding network using bayesian network
        #self.network_embedding_spinal_cord = torch.nn.Sequential(
        #    torch.nn.Conv2d(64, 64, kernel_size=1), torch.nn.ReLU())
        self.network_embedding_spinal_cord = torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        nn.Conv2d(64, 32, kernel_size=1))
        #self.network_embedding_gm = torch.nn.Sequential(
        #    torch.nn.Conv2d(64, 64, kernel_size=1), torch.nn.ReLU())
        self.network_embedding_gm = torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                            nn.Conv2d(64, 32, kernel_size=1))

        if (self.hparams['local_loss'] or self.hparams['global_loss']):
            self.metric_net_spinal = torch.nn.Sequential(torch.nn.Conv2d(32, 16, kernel_size=1),
            torch.nn.LeakyReLU(),torch.nn.Conv2d(16, 8, kernel_size=1),torch.nn.LeakyReLU())
            self.metric_net_gm = torch.nn.Sequential(torch.nn.Conv2d(32, 16, kernel_size=1),
                                                         torch.nn.LeakyReLU(), torch.nn.Conv2d(16, 8, kernel_size=1),
                                                         torch.nn.LeakyReLU())

        self.classifer_spinal_cord =  torch.nn.Sequential(torch.nn.Conv2d(32, 1, kernel_size=1))
        self.classifer_gm = torch.nn.Sequential(torch.nn.Conv2d(32, 1, kernel_size=1))
        #nn.Sequential(self.extractor_bayesian, self.relu, self.classifier_bayesian)
        # the parametes about the bnn


        # TODO there is no pretrained network for moped enable
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": self.hparams['bnn_rho_init'],
            "type": "Reparameterization",#"Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": self.hparams['moped_delta_factor'], #0.5=>0.2
        }
        #dnn_to_bnn(self.network_embedding_gm, const_bnn_prior_parameters)
        #dnn_to_bnn(self.classifer_gm, const_bnn_prior_parameters)

        #dnn_to_bnn(self.network_embedding_spinal_cord, const_bnn_prior_parameters)
        #dnn_to_bnn(self.classifer_spinal_cord, const_bnn_prior_parameters)

        #self.optimizer = get_optimizer(
        #    hparams["optimizer"],
        #    self.network.parameters(),
        #    lr=self.hparams["lr"],
        #    weight_decay=self.hparams["weight_decay"],
        #)
        self.network_spinal = nn.Sequential(self.featurizer__spinal_cord, self.network_embedding_spinal_cord,self.metric_net_spinal,self.classifer_spinal_cord)
        self.network_gm= nn.Sequential(self.featurizer__gm, self.network_embedding_gm,
                                            self.metric_net_gm, self.classifer_gm)

        #self.optimizer_spinal = optim.Adam(self.network_spinal.parameters(), lr=self.hparams['lr'])
        #self.optimizer_gm = optim.Adam(self.network_gm.parameters(), lr=self.hparams['lr'])

        self.optimizer_spinal = get_optimizer(
            hparams["optimizer"],
            self.network_spinal.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.optimizer_gm = get_optimizer(
            hparams["optimizer"],
            self.network_gm.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.local_loss = LocalLoss(device,self.hparams['margin'])
        self.global_loss = GlobalLoss(device, self.hparams['margin'])
        self.skin = skin

    def update(self, optimizer, input, spinal_cord_mask,gm_mask, **kwargs):
        #minibatch = to_minibatch(x,y)
        #num = len(minibatch)
        all_x = torch.cat(input)
        all_y = torch.cat(spinal_cord_mask)
        # all_y 24,1,144,144
        #print(all_y.shape)
        #exit()
        all_z = torch.cat(gm_mask)

        embeddings_spinal_cord =[]
        output_spinal_cord = []
        kl_spinal_cord = []
        z_spinal_cord = self.featurizer__spinal_cord(all_x)

        for mc_run in range(self.hparams['num_mc']):
            input = z_spinal_cord
            #if mc_run == 0:
            embedding = self.network_embedding_spinal_cord(input)
            #embedding = input
            #embedding_metric = self.metric_net(embedding)
            #embeddings.append(embedding_metric)
            embeddings_spinal_cord.append(embedding)
            output = self.classifer_spinal_cord(embedding)
            kl = 0#get_kl_loss(self.network_embedding_spinal_cord) + get_kl_loss(self.classifer_spinal_cord)
            output_spinal_cord.append(output)
            kl_spinal_cord.append(kl)

        if self.hparams['local_loss']:
            input_embeddings_spinal_cord = torch.stack(embeddings_spinal_cord)
            # 10, 24, 64, 144, 144
            # 10,48,512
            #print(input_embeddings_spinal_cord.shape)
            #exit()
            embeddings_out = []
            for index in range(self.hparams['num_mc']):
                embeddings_out.append(self.metric_net_spinal(input_embeddings_spinal_cord[index]))
            embeddings_out = torch.stack(embeddings_out) #[10, 24, 8, 144, 144])

            #print(embeddings_out.shape)
            #exit()
            embeddings_out = torch.transpose(embeddings_out,0,1)
             #torch.reshape(embeddings_out, [-1,self.hparams['num_mc'],8]
            local_loss_spinal_cord,loss_pos_spinal_cord,loss_neg_spinal_cord,equal_sample_index1_spinal_cord,nequal_sample_index1_spinal_cord = self.local_loss.forward(embeddings_out,all_y,self.hparams['contrastive_type'],self.hparams['num_mc'],self.hparams['batch_size'], self.hparams['pairs_number'],self.num_domains) # contrastive
            #print('local',local_loss)

        if self.hparams['global_loss']:
            input_embeddings_spinal_cord = torch.stack(embeddings_spinal_cord)
            # 10, 24, 64, 144, 144
            # 10,48,512
            embeddings_out = []
            for index in range(self.hparams['num_mc']):
                embeddings_out.append(self.metric_net_spinal(input_embeddings_spinal_cord[index]))
            embeddings_out = torch.stack(embeddings_out)
            embeddings_out = torch.transpose(embeddings_out, 0, 1)
            embedding = embeddings_out  # torch.reshape(embeddings_out, [-1,self.hparams['num_mc'],8]
            #input_embeddings = torch.stack(embeddings)

            #embeddings_out = self.metric_net(input_embeddings)
            #print(embeddings_out.shape)
            #embedding = torch.transpose(embeddings_out,0,1)
            #print(embedding.shape) # totoal_batch, num_mc, 128
            #features = [self.featurizer(xi) for xi, _ in minibatches]

            global_loss_spinal_cord = self.global_loss.forward(embedding,all_y,self.hparams['contrastive_type_global'],self.hparams['num_mc'],self.hparams['batch_size'], self.hparams['pairs_number'],self.hparams['level2_gamma'],self.hparams['level1_gamma_global'],self.num_domains,equal_sample_index1_spinal_cord,nequal_sample_index1_spinal_cord) # contrastive
            #print('global',global_loss)

        output_sc = torch.mean(torch.stack(output_spinal_cord), dim=0)
        kl_sc = 0#torch.mean(torch.stack(kl_spinal_cord), dim=0)

        spinal_pos_weight = torch.tensor(1.) / torch.mean(all_y.detach()) * self.hparams['p_weight1']
        if torch.isinf(spinal_pos_weight) or torch.isnan(spinal_pos_weight):
            spinal_pos_weight = torch.tensor(1.).cuda()

        loss_spinal_cord = F.binary_cross_entropy_with_logits(output_sc, all_y,
                                                              pos_weight=spinal_pos_weight)
        spinal_cord_pred = output_sc
        #print('cross',cross_entropy_loss)
        scaled_kl_sc = kl_sc / self.hparams['batch_size']

        loss_total_sp = loss_spinal_cord + self.hparams['kl_weight'] * scaled_kl_sc + self.hparams[
            'global_weight'] * global_loss_spinal_cord + self.hparams['local_weight'] * local_loss_spinal_cord
        self.optimizer_spinal.zero_grad()
        loss_total_sp.backward()
        self.optimizer_spinal.step()

        spinal_mask_pred = (torch.sigmoid(spinal_cord_pred) > 0.5).detach().float()  # N*C*W*H
        local_max = (spinal_mask_pred * all_x).max(dim=2)[0].max(dim=2)[0]
        local_min = ((1 - spinal_mask_pred) * 9999 + spinal_mask_pred * all_x).min(dim=2)[0].min(dim=2)[0]
        local_min *= (local_min < 9000).float()
        local_max = local_max.view(-1, 1, 1, 1)
        local_min = local_min.view(-1, 1, 1, 1)
        all_x = torch.clamp((all_x - local_min) / ((local_max - local_min) + ((local_max - local_min) == 0).float()), 0, 1)

        embeddings_gm = []
        output_gm = []
        kl_gm = []
        z_gm = self.featurizer__gm(all_x)

        for mc_run in range(self.hparams['num_mc']):
            input = z_gm
            embedding = self.network_embedding_gm(input)
            #
            # embedding_metric = self.metric_net(embedding)
            # embeddings.append(embedding_metric)
            embeddings_gm.append(embedding)
            output = self.classifer_gm(embedding)
            kl = 0#get_kl_loss(self.network_embedding_gm) + get_kl_loss(self.classifer_gm)
            output_gm.append(output)
            kl_gm.append(kl)

        if self.hparams['local_loss']:
            input_embeddings_gm = torch.stack(embeddings_gm)
            # 10, 24, 64, 144, 144
            # 10,48,512
            embeddings_out = []
            for index in range(self.hparams['num_mc']):
                embeddings_out.append(self.metric_net_gm(input_embeddings_gm[index]))
            embeddings_out = torch.stack(embeddings_out)
            embeddings_out = torch.transpose(embeddings_out, 0, 1)
            # torch.reshape(embeddings_out, [-1,self.hparams['num_mc'],8]
            local_loss_gm, loss_pos_gm, loss_neg_gm, equal_sample_index1_gm, nequal_sample_index1_gm = self.local_loss.forward(
                embeddings_out, all_z, self.hparams['contrastive_type'], self.hparams['num_mc'],
                self.hparams['batch_size'], self.hparams['pairs_number'], self.num_domains)  # contrastive
            # print('local',local_loss)

        if self.hparams['global_loss']:
            input_embeddings_gm = torch.stack(embeddings_gm)
            # 10, 24, 64, 144, 144
            # 10,48,512
            embeddings_out = []
            for index in range(self.hparams['num_mc']):
                embeddings_out.append(self.metric_net_gm(input_embeddings_gm[index]))
            embeddings_out = torch.stack(embeddings_out)
            embeddings_out = torch.transpose(embeddings_out, 0, 1)
            embedding = embeddings_out  # torch.reshape(embeddings_out, [-1,self.hparams['num_mc'],8]
            # input_embeddings = torch.stack(embeddings)

            # embeddings_out = self.metric_net(input_embeddings)
            # print(embeddings_out.shape)
            # embedding = torch.transpose(embeddings_out,0,1)
            # print(embedding.shape) # totoal_batch, num_mc, 128
            # features = [self.featurizer(xi) for xi, _ in minibatches]

            global_loss_gm = self.global_loss.forward(embedding, all_z,
                                                               self.hparams['contrastive_type_global'],
                                                               self.hparams['num_mc'], self.hparams['batch_size'],
                                                               self.hparams['pairs_number'],
                                                               self.hparams['level2_gamma'],
                                                               self.hparams['level1_gamma_global'], self.num_domains,
                                                               equal_sample_index1_gm,
                                                               nequal_sample_index1_gm)  # contrastive
            # print('global',global_loss)

        output_gm = torch.mean(torch.stack(output_gm), dim=0)
        kl_gm = 0#torch.mean(torch.stack(kl_gm), dim=0)

        gm_pos_weight = torch.tensor(1.) / torch.sum(spinal_mask_pred * all_z)
        if torch.isinf(gm_pos_weight) or torch.isnan(gm_pos_weight):
            gm_pos_weight = torch.tensor(1.).cuda()

        loss_gm = F.binary_cross_entropy_with_logits(output_gm * spinal_mask_pred, all_z,
                                                              pos_weight=gm_pos_weight)

        # print('cross',cross_entropy_loss)
        scaled_kl_gm = kl_gm / self.hparams['batch_size']

        loss_total_gm = loss_gm + self.hparams['kl_weight'] * scaled_kl_gm + self.hparams[
            'global_weight'] * global_loss_gm + self.hparams['local_weight'] * local_loss_gm
        self.optimizer_gm.zero_grad()
        loss_total_gm.backward()
        self.optimizer_gm.step()

        cross_entropy_loss = loss_gm + loss_spinal_cord
        global_loss = global_loss_gm +  global_loss_spinal_cord
        local_loss = local_loss_gm + local_loss_spinal_cord
        loss_pos = loss_pos_gm + loss_pos_spinal_cord
        loss_neg = loss_neg_gm + loss_neg_spinal_cord
        loss = (loss_total_sp + loss_total_gm)/2





        if (self.hparams['local_loss'] == True and self.hparams['global_loss'] ==False):
            return {'loss_total': loss.item(),'contrastive_loss_total':local_loss.item(),'pos_ccas':loss_pos.item(),'neg_ccas':loss_neg.item()}
        elif (self.hparams['global_loss'] ==True and self.hparams['local_loss'] ==False):
            return {'loss_total': loss.item(),'global_loss':global_loss.item()}
        elif (self.hparams['global_loss'] and self.hparams['local_loss']):
            return {'loss_total': loss.item(),'global_loss':global_loss.item(),'contrastive_loss_total':local_loss.item(),'pos_ccas':loss_pos.item(),'neg_ccas':loss_neg.item()}
        else:
            return {'loss_total': loss.item()}

    def predict(self, x):

        file = open('/home/inputs_x3.txt','a')
        file.write(str(x))
        file.close()
        #print(x.shape)
        #print(torch.max(x))
        #print(torch.min(x))

        z_spinal = self.featurizer__spinal_cord(x)
        embedding_spinal = self.network_embedding_spinal_cord(z_spinal)
        #print(embedding_spinal)
        spinal_cord_pred  = self.classifer_spinal_cord(embedding_spinal)
        #print(spinal_cord_pred)
        print(spinal_cord_pred)
        spinal_mask_pred = (torch.sigmoid(spinal_cord_pred) > 0.5).detach().float()
        #spinal_mask_pred = (torch.sigmoid(spinal_cord_pred) > 0.5).detach().float()
        #for i in range(spinal_mask_pred.shape[0]):
        #    index1, index2 = torch.where(torch.squeeze(spinal_mask_pred[i]) > 0.1)
        #    print('ckc')
        #    print(index1)
            #print(index2)
        #exit()
        local_max = (spinal_mask_pred * x).max(dim=2)[0].max(dim=2)[0]
        local_min = ((1 - spinal_mask_pred) * 9999 + spinal_mask_pred * x).min(dim=2)[0].min(dim=2)[0]

        local_max = local_max.view(-1, 1, 1, 1)
        local_min = local_min.view(-1, 1, 1, 1)
        local_min *= (local_min < 9000).float()
        x = torch.clamp((x - local_min) / ((local_max - local_min) + ((local_max - local_min) == 0).float()), 0, 1)
        z_gm = self.featurizer__gm(x)
        embedding_gm = self.network_embedding_gm(z_gm)
        gm_pred = self.classifer_gm(embedding_gm)  # * spinal_mask_pred

        #z = self.featurizer(x)
        #embedding = self.network_embedding(z)
        return gm_pred,spinal_cord_pred
    def predict_spinal_cord(self, x):
        z = self.featurizer__spinal_cord(x)
        embedding = self.network_embedding_spinal_cord(z)
        return self.classifer_spinal_cord(embedding)
    def predict_gm(self, x):
        z = self.featurizer__gm(x)
        embedding = self.network_embedding_gm(z)
        return self.classifer_gm(embedding)
'''


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


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x


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


class ICML22_MMD_Unet(Algorithm):
    def __init__(self, n_channels, n_classes, hparams, device, feature_dim=8, bilinear=True):
        super(ICML22_MMD_Unet, self).__init__(n_channels, n_classes, hparams, device)
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
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3,
            "type": "Reparameterization",  # "Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.1,  # 0.5=>0.2
        }
        self.mu = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(64, feature_dim, kernel_size=1))
        dnn_to_bnn(self.mu, const_bnn_prior_parameters)
        # self.logvar = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        #                            nn.Conv2d(64, feature_dim, kernel_size=1))
        self.outc = nn.Sequential(nn.Conv2d(feature_dim, n_classes, kernel_size=1))
        dnn_to_bnn(self.outc, const_bnn_prior_parameters)

    def update(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # mu = self.mu(x)
        # logvar = self.logvar(x)

        embeddings_lists = []
        out_lists = []
        kl_lists = []
        # print(self.hparams['num_mc'])
        # exit()
        for mc_run in range(10):
            input = x
            # if mc_run == 0:
            embedding = self.mu(input)
            # embedding = input
            # embedding_metric = self.metric_net(embedding)
            # embeddings.append(embedding_metric)
            embeddings_lists.append(embedding)
            output = self.outc(embedding)
            kl = get_kl_loss(self.mu) + get_kl_loss(self.outc)
            out_lists.append(output)
            kl_lists.append(kl)
        logits = torch.mean(torch.stack(out_lists), dim=0)
        kl_loss = torch.mean(torch.stack(kl_lists), dim=0)
        # feature = mu#self.reparameterization(mu, logvar)  # b*c*w*h
        # logits = self.outc(feature)
        return logits, kl_loss  # torch.stack([mu, logvar, feature], dim=1)

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature = self.mu(x)
        logits = self.outc(feature)
        return logits


class ICML22_MMD_Unet_DSU(Algorithm):
    def __init__(self, n_channels, n_classes, hparams, device, feature_dim=8, bilinear=True):
        super(ICML22_MMD_Unet_DSU, self).__init__(n_channels, n_classes, hparams, device)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_dim = feature_dim

        self.inc = DoubleConv(n_channels, 64)
        self.uncertainty1 = DistributionUncertainty()
        self.down1 = Down(64, 128)
        self.uncertainty2 = DistributionUncertainty()
        self.down2 = Down(128, 256)
        self.uncertainty3 = DistributionUncertainty()
        self.down3 = Down(256, 512)
        self.uncertainty4 = DistributionUncertainty()
        self.down4 = Down(512, 512)
        self.uncertainty5 = DistributionUncertainty()
        self.up1 = Up(1024, 256, bilinear)
        self.uncertainty6 = DistributionUncertainty()
        self.up2 = Up(512, 128, bilinear)
        self.uncertainty7 = DistributionUncertainty()
        self.up3 = Up(256, 64, bilinear)
        self.uncertainty8 = DistributionUncertainty()
        self.up4 = Up(128, 64, bilinear)
        self.uncertainty9 = DistributionUncertainty()

        self.mu = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(64, feature_dim, kernel_size=1))

        # self.logvar = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        #                            nn.Conv2d(64, feature_dim, kernel_size=1))
        self.outc = nn.Sequential(nn.Conv2d(feature_dim, n_classes, kernel_size=1))
        # dnn_to_bnn(self.outc, const_bnn_prior_parameters)

    def update(self, x):
        x1 = self.inc(x)
        x1 = self.uncertainty1(x1)
        x2 = self.down1(x1)
        x2 = self.uncertainty2(x2)
        x3 = self.down2(x2)
        x3 = self.uncertainty3(x3)
        x4 = self.down3(x3)
        x4 = self.uncertainty4(x4)
        x5 = self.down4(x4)
        x5 = self.uncertainty5(x5)
        x = self.up1(x5, x4)
        x = self.uncertainty6(x)
        x = self.up2(x, x3)
        x = self.uncertainty7(x)
        x = self.up3(x, x2)
        x = self.uncertainty8(x)
        x = self.up4(x, x1)
        x = self.uncertainty9(x)
        mu = self.mu(x)
        logits = self.outc(mu)
        # logvar = self.logvar(x)

        # feature = mu#self.reparameterization(mu, logvar)  # b*c*w*h
        # logits = self.outc(feature)
        return logits, logits, logits, logits, logits, logits  # torch.stack([mu, logvar, feature], dim=1)

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature = self.mu(x)
        logits = self.outc(feature)
        return logits


class ICML22_MMD_Unet_Ablation(Algorithm):
    def __init__(self, n_channels, n_classes, hparams, device, feature_dim=8, bilinear=True):
        super(ICML22_MMD_Unet_Ablation, self).__init__(n_channels, n_classes, hparams, device)
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
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3,
            "type": "Reparameterization",  # "Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.1,  # 0.5=>0.2
        }
        self.mu = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(64, feature_dim, kernel_size=1))
        dnn_to_bnn(self.mu, const_bnn_prior_parameters)
        # self.logvar = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        #                            nn.Conv2d(64, feature_dim, kernel_size=1))
        self.outc = nn.Sequential(nn.Conv2d(feature_dim, n_classes, kernel_size=1))
        # dnn_to_bnn(self.outc, const_bnn_prior_parameters)

    def update(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # mu = self.mu(x)
        # logvar = self.logvar(x)

        embeddings_lists = []
        out_lists = []
        kl_lists = []
        # print(self.hparams['num_mc'])
        # exit()
        for mc_run in range(10):
            input = x
            # if mc_run == 0:
            embedding = self.mu(input)
            # embedding = input
            # embedding_metric = self.metric_net(embedding)
            # embeddings.append(embedding_metric)
            embeddings_lists.append(embedding)
            output = self.outc(embedding)
            kl = get_kl_loss(self.mu)  # + get_kl_loss(self.outc)
            out_lists.append(output)
            kl_lists.append(kl)
        logits = torch.mean(torch.stack(out_lists), dim=0)
        kl_loss = torch.mean(torch.stack(kl_lists), dim=0)
        # feature = mu#self.reparameterization(mu, logvar)  # b*c*w*h
        # logits = self.outc(feature)
        return logits, kl_loss, kl_loss, kl_loss, kl_loss, kl_loss  # torch.stack([mu, logvar, feature], dim=1)

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature = self.mu(x)
        logits = self.outc(feature)
        return logits


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


class ShapeVariationalDist(nn.Module):
    def __init__(self, hparams, device, n_channels, bilinear, n_classes, wt=True, prior=True, number_source_domain=3):
        super(ShapeVariationalDist, self).__init__()
        self.device = device
        self.prior = prior
        self.wt = wt
        self.InstanceWhitening = InstanceWhitening(64)
        self.wt_type = hparams['whitening_type']
        self.wt_type_inference = hparams['wt_type_inference']
        self.cca_type = hparams['CCA_type']
        self.cca_transform_type = hparams['CCA_transform_type']
        self.posterior_transform_follow_prior = hparams['posterior_transform_follow_prior']
        self.Sigma_exp = torch.eye(64).to(self.device).float()
        self.mu_exp = torch.ones([64]).to(self.device).float()
        self.momentum = 0.99
        if self.prior:
            n_channels += 1
        self.number_source_domain = number_source_domain
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.mu_prior = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                      nn.Conv2d(64, n_classes, kernel_size=1))

        self.logvar_prior = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                          nn.Conv2d(64, n_classes, kernel_size=1))
        if self.wt:
            vgg1 = pytorch_lua_wrapper(hparams['vgg1'])
            decoder1_torch = pytorch_lua_wrapper(hparams['decoder1'])
            self.e1 = encoder1(vgg1)
            self.d1 = decoder1(decoder1_torch)

    def unet_extractor(self, inputs):
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

    def moving_average_updata_mu(self, inputs):
        mu = self.momentum * inputs + (1 - self.momentum) * self.mu_exp
        self.mu_exp = mu

    def moving_average_updata_sigma(self, inputs):
        sigma = self.momentum * inputs + (1 - self.momentum) * self.Sigma_exp
        self.Sigma_exp = sigma

    def forward(self, inputs, mask=None, training=True):
        if training:
            if self.wt:
                if self.prior == True:
                    if self.wt_type == 'instance_wt':
                        phi_x = self.wct(inputs)
                        phi_x = phi_x.unsqueeze(1)
                    elif self.wt_type == 'cca':
                        phi_x = self.cca_wct(inputs, training=True)
                        phi_x = phi_x.unsqueeze(1)
                    else:
                        print('-' * 10, 'The type of whitening is not found', '-' * 10)
                        exit()
                else:
                    if self.posterior_transform_follow_prior:
                        if self.wt_type == 'instance_wt':
                            phi_x = self.wct(inputs)
                            phi_x = phi_x.unsqueeze(1)
                        elif self.wt_type == 'cca':
                            phi_x = self.cca_wct(inputs)
                            phi_x = phi_x.unsqueeze(1)
                        else:
                            print('-' * 10, 'The type of whitening is not found', '-' * 10)
                            exit()
                    else:
                        phi_x = self.wct(inputs)
                        phi_x = phi_x.unsqueeze(1)

            else:
                phi_x = inputs

            b, c, w, h = inputs.shape
            if self.prior:
                x_y = torch.cat([mask, phi_x], dim=1)
                feature_map = self.unet_extractor(x_y)
                mu = self.mu_prior(feature_map)
                logvar = self.logvar_prior(feature_map)
                mu = mu.view(b, -1)
                log_sigma = logvar.view(b, -1)
                dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
                return dist
            else:
                feature_map = self.unet_extractor(phi_x)
                mu = self.mu_prior(feature_map)
                logvar = self.logvar_prior(feature_map)
                mu = mu.view(b, -1)
                log_sigma = logvar.view(b, -1)
                dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
                return dist
        else:
            if self.wt:
                if self.wt_type_inference == 'instance_wt':
                    phi_x = self.wct(inputs)
                    phi_x = phi_x.unsqueeze(1)
                elif self.wt_type_inference == 'cca':
                    phi_x = self.cca_wct(inputs, training=False)
                    phi_x = phi_x.unsqueeze(1)
                else:
                    print('-' * 10, 'The type of whitening is not found', '-' * 10)
                    exit()
            else:
                phi_x = inputs

            b, c, w, h = inputs.shape
            if self.prior:
                x_y = torch.cat([mask, phi_x], dim=1)
                feature_map = self.unet_extractor(x_y)
                mu = self.mu_prior(feature_map)
                logvar = self.logvar_prior(feature_map)
                mu = mu.view(b, -1)
                log_sigma = logvar.view(b, -1)
                dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
                return dist
            else:
                feature_map = self.unet_extractor(phi_x)
                mu = self.mu_prior(feature_map)
                logvar = self.logvar_prior(feature_map)
                mu = mu.view(b, -1)
                log_sigma = logvar.view(b, -1)
                dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
                return dist

    def wct(self, x):
        b, c, w, h = x.shape

        x = x.repeat(1, 3, 1, 1)
        wt_features = []
        cF1 = self.e1(x)
        for index in range(b):
            # cF1 = cF1.data.cpu().squeeze(0)
            # avoid nan for svd
            # if torch.count_nonzero(torch.isnan(cF1[index])) > 0:
            #    cF1[index] = torch.nan_to_num(cF1[index])
            #    print('has nan!!!')
            cF1_input = torch.squeeze(cF1[index])
            cF1_input = self.transform(cF1_input)
            cF1_input_copy = cF1_input.float()
            # csF1_ = csF1_.to(device=self.device).float()
            # cF1_input_copy = cF1_input_copy.unsqueeze(0)
            wt_features.append(cF1_input_copy)
        wt_features = torch.stack(wt_features, 0)
        # print(wt_features.shape)
        # exit()
        output = self.d1(wt_features)
        # output is 3 channles, conducting mean operation
        output = torch.mean(output, 1)
        return output

    def whiten_and_color(self, cF):
        cFSize = cF.size()
        cF = cF.contiguous().view(1, 64, 144, 144)
        # c_mean = torch.mean(cF, 1)  # c x (h x w)
        # c_mean = c_mean.unsqueeze(1).expand_as(cF)
        # cF = cF - c_mean
        #
        _, cF = self.InstanceWhitening(cF)
        cF = cF.contiguous().view(64, -1)
        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double().to(self.device)
        c_u, c_e, c_v = torch.svd(contentConv, some=False)

        k_c = cFSize[0]
        # for i in range(cFSize[0]):
        #    if c_e[i] < 0.00001:
        #        k_c = i
        #        break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cF)
        return whiten_cF

    def transform(self, cF):
        cF = cF.double()
        C, W, H = cF.size(0), cF.size(1), cF.size(2)
        cFView = cF.view(C, -1)
        targetFeature = self.whiten_and_color(cFView)
        targetFeature = targetFeature.view_as(cF)
        # with torch.no_grad():
        # csF.data.resize_(targetFeature.size()).copy_(targetFeature)
        #    csF.set_(targetFeature)
        return targetFeature

    def cca_wct(self, x, training=True):

        if training:
            if self.cca_type == 'cca_all':
                b, c, w, h = x.shape

                batch_size = int(b / self.number_source_domain)
                x = x.repeat(1, 3, 1, 1)
                wt_features = []
                cF1 = self.e1(x)

                result = torch.zeros_like(cF1).to(self.device)

                # conduct cca is always a pair
                reference_domain = self.number_source_domain // 2
                # the number of conducing CCA operation
                CCA_number = self.number_source_domain - 1
                # if this domain conduct CCA, the label turns from 0 to 1
                CCA_label_list = [0] * self.number_source_domain
                domain_id = list(np.arange(self.number_source_domain))

                mu_list = 0
                sigma_list = 0
                for index_d in domain_id:
                    # reference domain has been processed!
                    if index_d == reference_domain:
                        continue
                    Sigma_two_domain = torch.zeros_like(self.Sigma_exp).to(self.device)
                    Mu_two_domain = torch.zeros_like(self.mu_exp).to(self.device)
                    for index_i in range(batch_size):
                        input1 = cF1[int((index_d * batch_size) + index_i)]
                        input2 = cF1[int(reference_domain * batch_size + index_i)]

                        cF1_input = torch.squeeze(input1)
                        cF2_input = torch.squeeze(input2)

                        cF1_input, cF2_input, sigma, mu = self.cca_transform(cF1_input, cF2_input)
                        Sigma_two_domain += sigma
                        Mu_two_domain += mu
                        # csF1_ = csF1_.to(device=self.device).float()
                        # cF1_input_copy = cF1_input_copy.unsqueeze(0)
                        result[int((index_d * batch_size) + index_i)] = cF1_input
                        # if reference domain has been processed！
                        if CCA_label_list[reference_domain] == 0:
                            result[int((reference_domain * batch_size) + index_i)] = cF2_input
                    CCA_label_list[index_d] = 1
                    CCA_label_list[reference_domain] = 1

                    # summarize two domain mu and sigma
                    mean_sigma = Sigma_two_domain / batch_size
                    mean_mu = Mu_two_domain / batch_size
                    sigma_list += mean_sigma
                    mu_list += mean_mu

                batch_mu = mu_list / 2
                batch_sigma = sigma_list / 2
                if not self.prior:
                    self.moving_average_updata_mu(batch_mu)
                    self.moving_average_updata_sigma(batch_sigma)

                wt_features = result
                # print(wt_features.shape)
                # exit()
                output = self.d1(wt_features)
                # output is 3 channles, conducting mean operation
                output = torch.mean(output, 1)
                return output
            elif self.cca_type == 'caa_random_2':
                b, c, w, h = x.shape
                batch_size = int(b / self.number_source_domain)
                x = x.repeat(1, 3, 1, 1)

                cF1 = self.e1(x)

                result = torch.zeros_like(cF1).to(self.device)

                # conduct cca is always a pair
                pratical_domain = self.number_source_domain // 2

                for index_d in range(pratical_domain):
                    for index_i in range(batch_size):
                        input1 = cF1[int((index_d * batch_size) + index_i)]
                        input2 = cF1[int(((index_d + 1) * batch_size) + index_i)]

                        cF1_input = torch.squeeze(input1)
                        cF2_input = torch.squeeze(input2)

                        cF1_input, cF2_input = self.cca_transform(cF1_input, cF2_input)

                        # csF1_ = csF1_.to(device=self.device).float()
                        # cF1_input_copy = cF1_input_copy.unsqueeze(0)
                        result[int((index_d * batch_size) + index_i)] = cF1_input
                        result[int(((index_d + 1) * batch_size) + index_i)] = cF2_input

                wt_features = result
                # print(wt_features.shape)
                # exit()
                output = self.d1(wt_features)
                # output is 3 channles, conducting mean operation
                output = torch.mean(output, 1)
                return output

        else:
            b, c, w, h = x.shape
            batch_size = int(b / 2)
            x = x.repeat(1, 3, 1, 1)

            cF1 = self.e1(x)

            result = torch.zeros_like(cF1[:batch_size]).to(self.device)

            # conduct cca is always a pair
            pratical_domain = 1

            for index_d in range(pratical_domain):
                for index_i in range(batch_size):
                    input1 = cF1[int((index_d * batch_size) + index_i)]
                    input2 = cF1[int(((index_d + 1) * batch_size) + index_i)]

                    cF1_input = torch.squeeze(input1)
                    cF2_input = torch.squeeze(input2)

                    cF1_input, cF2_input, _, __ = self.cca_transform(cF1_input, cF2_input, training=False)

                    # csF1_ = csF1_.to(device=self.device).float()
                    # cF1_input_copy = cF1_input_copy.unsqueeze(0)
                    result[int((index_d * batch_size) + index_i)] = cF1_input

            wt_features = result
            # print(wt_features.shape)
            # exit()
            output = self.d1(wt_features)
            # output is 3 channles, conducting mean operation
            output = torch.mean(output, 1)
            return output

    def cca_transform(self, cF, cF2, training=True):
        cF = cF.double()
        C, W, H = cF.size(0), cF.size(1), cF.size(2)
        cFView = cF.view(C, -1)
        cF2 = cF2.double()
        cFView2 = cF2.view(C, -1)
        if self.cca_transform_type == 'CCA':
            targetFeature1, targetFeature2, sigma, mu = self.CCA(cFView, cFView2, training=training)
        elif self.cca_transform_type == 'ZCA':
            targetFeature1, targetFeature2, sigma, mu = self.CCA_ZCA(cFView, cFView2, training=training)
        targetFeature1 = targetFeature1.view_as(cF)
        targetFeature2 = targetFeature2.view_as(cF)
        # with torch.no_grad():
        # csF.data.resize_(targetFeature.size()).copy_(targetFeature)
        #    csF.set_(targetFeature)
        return targetFeature1, targetFeature2, sigma, mu

    def CCA(self, cF, cF2, training=True):
        if training:
            cFSize = cF.size()
            cFSize2 = cFSize
            size = cFSize2
            all_inputs = torch.cat((cF, cF2), 1)
            c_mean = torch.mean(all_inputs, 1)  # c x (h x w)
            c_mean_all = c_mean.unsqueeze(1).expand_as(all_inputs)
            all_inputs_mean = all_inputs - c_mean_all

            cF = all_inputs_mean[:, :size[1]]
            cF2 = all_inputs_mean[:, size[1]:]

            xxConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + 0.001 * torch.eye(cFSize[0]).double().to(self.device)
            ux, ux_sigma, vx = torch.svd(xxConv, some=False)
            ux_sigma = ux_sigma.pow(-0.5)
            step1 = torch.mm(vx, torch.diag(ux_sigma))
            xxConv_12 = torch.mm(step1, (vx.t()))

            yyConv = torch.mm(cF2, cF2.t()).div(cFSize2[1] - 1) + 0.001 * torch.eye(cFSize[0]).double().to(self.device)
            uy, uy_sigma, vy = torch.svd(yyConv, some=False)
            uy_sigma = uy_sigma.pow(-0.5)
            step1 = torch.mm(vy, torch.diag(uy_sigma))
            yyConv_12 = torch.mm(step1, (vy.t()))

            xyConv = torch.mm(cF, cF2.t()).div(cFSize2[1] - 1) + 0.001 * torch.eye(cFSize[0]).double().to(self.device)

            K = torch.mm(torch.mm(xxConv_12, xyConv), yyConv_12)

            c_x, c_e, c_y = torch.svd(K, some=False)

            whiten_cF_step1 = torch.mm(xxConv_12, c_x)
            whiten_cF = torch.mm(whiten_cF_step1.t(), cF)
            whiten_cF2_step1 = torch.mm(yyConv_12, c_y)
            whiten_cF2 = torch.mm(whiten_cF2_step1.t(), cF2)

            return whiten_cF, whiten_cF2, 0, 0
        else:
            cFSize = cF.size()
            cFSize2 = cFSize
            size = cFSize2
            all_inputs = torch.cat((cF, cF2), 1)
            c_mean = torch.mean(all_inputs, 1)  # c x (h x w)
            c_mean_all = c_mean.unsqueeze(1).expand_as(all_inputs)
            all_inputs_mean = all_inputs - c_mean_all

            cF = all_inputs_mean[:, :size[1]]
            cF2 = all_inputs_mean[:, size[1]:]

            xxConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double().to(self.device)
            ux, ux_sigma, vx = torch.svd(xxConv, some=False)
            ux_sigma = ux_sigma.pow(-0.5)
            step1 = torch.mm(vx, torch.diag(ux_sigma))
            xxConv_12 = torch.mm(step1, (vx.t()))

            yyConv = torch.mm(cF2, cF2.t()).div(cFSize2[1] - 1) + torch.eye(cFSize[0]).double().to(self.device)
            uy, uy_sigma, vy = torch.svd(yyConv, some=False)
            uy_sigma = uy_sigma.pow(-0.5)
            step1 = torch.mm(vy, torch.diag(uy_sigma))
            yyConv_12 = torch.mm(step1, (vy.t()))

            xyConv = torch.mm(cF, cF2.t()).div(cFSize2[1] - 1) + torch.eye(cFSize[0]).double().to(self.device)

            K = torch.mm(torch.mm(xxConv_12, xyConv), yyConv_12)

            c_x, c_e, c_y = torch.svd(K, some=False)

            whiten_cF_step1 = torch.mm(xxConv_12, c_x)
            whiten_cF = torch.mm(whiten_cF_step1.t(), cF)
            whiten_cF2_step1 = torch.mm(yyConv_12, c_y)
            whiten_cF2 = torch.mm(whiten_cF2_step1.t(), cF2)

            return whiten_cF, whiten_cF2, 0, 0

    def CCA_ZCA(self, cF, cF2, training=True):
        '''
        cFSize = cF.size()
        c_mean = torch.mean(cF, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        cFSize2 = cF2.size()
        c_mean2 = torch.mean(cF2, 1)  # c x (h x w)
        c_mean2 = c_mean2.unsqueeze(1).expand_as(cF2)
        cF2 = cF2 - c_mean2

        xxConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double().to(self.device)
        ux, ux_sigma, vx = torch.svd(xxConv, some=False)
        ux_sigma = ux_sigma.pow(-0.5)
        step1 = torch.mm(vx, torch.diag(ux_sigma))
        xxConv_12 = torch.mm(step1, (vx.t()))

        yyConv = torch.mm(cF2, cF2.t()).div(cFSize2[1] - 1) + torch.eye(cFSize[0]).double().to(self.device)
        uy, uy_sigma, vy = torch.svd(yyConv, some=False)
        uy_sigma = uy_sigma.pow(-0.5)
        step1 = torch.mm(vy, torch.diag(uy_sigma))
        yyConv_12 = torch.mm(step1, (vy.t()))


        xyConv = torch.mm(cF, cF2.t()).div(cFSize2[1] - 1) + torch.eye(cFSize[0]).double().to(self.device)

        K = torch.mm(torch.mm(xxConv_12,xyConv),yyConv_12)


        c_x, c_e, c_y = torch.svd(K, some=False)

        c_e = c_e.pow(-0.5)
        whiten_cF_step1 = torch.mm(c_x, torch.diag(c_e))
        whiten_cF_step2 = torch.mm(whiten_cF_step1, c_x.t())
        whiten_cF = torch.mm(whiten_cF_step2, cF)

        whiten_cF2_step1 = torch.mm(c_y, torch.diag(c_e))
        whiten_cF2_step2 = torch.mm(whiten_cF2_step1, c_y.t())
        whiten_cF2 = torch.mm(whiten_cF2_step2, cF2)
        '''
        if training:

            cFSize = cF.size()
            cFSize2 = cFSize
            size = cFSize2
            all_inputs = torch.cat((cF, cF2), 1)
            c_mean = torch.mean(all_inputs, 1)  # c x (h x w)
            c_mean_all = c_mean.unsqueeze(1).expand_as(all_inputs)
            all_inputs_mean = all_inputs - c_mean_all

            cF = all_inputs_mean[:, :size[1]]
            cF2 = all_inputs_mean[:, size[1]:]

            # cFSize = cF.size()
            # c_mean = torch.mean(cF, 1)  # c x (h x w)
            # c_mean1 = c_mean.unsqueeze(1).expand_as(cF)
            # cF = cF - c_mean1

            # cFSize2 = cF2.size()
            # c_mean2 = torch.mean(cF2, 1)  # c x (h x w)
            # c_mean2 = c_mean2.unsqueeze(1).expand_as(cF2)
            # cF2 = cF2 - c_mean2

            xyConv = torch.mm(cF, cF2.t()).div(cFSize2[1] - 1) + 0.1 * torch.eye(cFSize[0]).double().to(self.device)
            c_u, c_e, c_v = torch.svd(xyConv, some=False)
            c_e = c_e.pow(-0.5)
            step1 = torch.mm(c_v, torch.diag(c_e))
            step2 = torch.mm(step1, c_v.t())

            whiten_cF = torch.mm(step2, cF)
            whiten_cF2 = torch.mm(step2, cF2)

            return whiten_cF, whiten_cF2, step2, c_mean
        else:
            cFSize = cF.size()
            cFSize2 = cFSize
            size = cFSize2
            all_inputs = torch.cat((cF, cF2), 1)
            c_mean_all = self.mu_exp.unsqueeze(1).expand_as(all_inputs)
            all_inputs_mean = all_inputs - c_mean_all

            cF = all_inputs_mean[:, :size[1]].float()
            cF2 = all_inputs_mean[:, size[1]:].float()

            whiten_cF = torch.mm(self.Sigma_exp, cF)
            whiten_cF2 = torch.mm(self.Sigma_exp, cF2)

            return whiten_cF, whiten_cF2, 0, 0


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
            x1 = self.fusion(x1)
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


class ShapeVariationalDist_x(nn.Module):
    def __init__(self, hparams, device, n_channels, bilinear, n_classes, wt=True, prior=True, number_source_domain=3):
        super(ShapeVariationalDist_x, self).__init__()
        self.device = device
        self.prior = prior
        self.wt = wt
        self.momentum = 0.99
        self.number_source_domain = number_source_domain
        # here

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.mu_prior = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                      nn.Conv2d(64, 8, kernel_size=1), nn.ReLU(),
                                      nn.Conv2d(8, n_classes, kernel_size=1))

        self.logvar_prior = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                          nn.Conv2d(64, 8, kernel_size=1), nn.ReLU(),
                                          nn.Conv2d(8, n_classes, kernel_size=1))

    def unet_extractor(self, inputs):

        x1 = inputs
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def distrubution_forward(self, inputs):

        b, c, w, h = inputs.shape
        feature_map = self.unet_extractor(inputs)
        mu = self.mu_prior(feature_map)
        mu = torch.sigmoid(mu)
        logvar = self.logvar_prior(feature_map)
        mu = mu.view(b, -1)
        log_sigma = logvar.view(b, -1)
        if torch.isnan(mu).any():
            mu = torch.nan_to_num(mu)
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist

    def sample_forward(self, inputs, training):
        b, c, w, h = inputs.shape
        feature_map = self.unet_extractor(inputs)
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
        sampled_z = torch.normal(mu, std)
        # epsilon = torch.randn_like(std).to(self.device)  # sampling epsilon
        # z = mu + std * epsilon
        z = sampled_z * std + mu
        return z


class InstanceWhitening(nn.Module):

    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):
        x = self.instance_standardization(x)
        w = x

        return w


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


class Unet_nips2023(Algorithm):
    def __init__(self, n_channels, n_classes, hparams, device, two_step, per_domain_batch=8, source_domain_num=3,
                 feature_dim=8, bilinear=True):
        super(Unet_nips2023, self).__init__(n_channels, n_classes, hparams, device)
        self.n_channels = n_channels
        self.device = device
        self.eps = 1e-5
        self.n_classes = n_classes
        self.num_domains = 3
        self.hparams = hparams

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
            self.prior_dist = ShapeVariationalDist_y_x(hparams, self.device, 1, bilinear, n_classes=1,
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
                z_posterior, z_posterior_mu = self.prior_dist.sample_forward(whiting_outputs1[-1], mask, training=True)
                # z_pre,z_pre_mu = self.posterior_dist.sample_forward(whiting_outputs2[-1],training=True)

            else:
                z_posterior, z_posterior_mu = self.prior_dist.sample_forward(whiting_outputs1[-1], mask, training=True)
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
                whiting_outputs1 = learn_x_network.wt_model.forward(two_stage_inputs)
                z_posterior = learn_x_network.sample_forward(whiting_outputs1[-1], training=False)

            else:
                whiting_outputs1 = learn_x_network.wt_model.forward(inputs)
                z_posterior = learn_x_network.sample_forward(whiting_outputs1[-1], training=False)

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




class Unet_nips2023_onlyWTPSE(Algorithm):
    def __init__(self, n_channels, n_classes, hparams, device, two_step, per_domain_batch=8, source_domain_num=3,
                 feature_dim=8, bilinear=True):
        super(Unet_nips2023_onlyWTPSE, self).__init__(n_channels, n_classes, hparams, device)
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
            self.prior_dist = ShapeVariationalDist_x(hparams, self.device, 1, bilinear, n_classes=1,
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
                z_posterior, z_posterior_mu = self.prior_dist.sample_forward(whiting_outputs1[-1], training=True)
                # z_pre,z_pre_mu = self.posterior_dist.sample_forward(whiting_outputs2[-1],training=True)

            else:
                z_posterior, z_posterior_mu = self.prior_dist.sample_forward(whiting_outputs1[-1], training=True)
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
            for embedding_wt in whiting_outputs1:
                instance_wt_loss1, domain_wt_loss1 = self.compute_whitening_loss(embedding_wt)
                instance_wt_loss += instance_wt_loss1
                domain_wt_loss += domain_wt_loss1

            instance_wt_loss /= num_embeddings
            domain_wt_loss /= num_embeddings
        if self.droptout:
            embedding = F.dropout(embedding,p=0.5)
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

        # instance whitening loss
        off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1, 2), keepdim=True) - self.margin  # B X 1 X 1
        instance_loss = torch.clamp(torch.div(off_diag_sum, self.num_off_diagonal), min=0)  # B X 1 X 1
        instance_loss = torch.sum(instance_loss) / B
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
                whiting_outputs1 = learn_x_network.wt_model.forward(two_stage_inputs)
                z_posterior = learn_x_network.sample_forward(whiting_outputs1[-1], training=False)

            else:
                whiting_outputs1 = learn_x_network.wt_model.forward(inputs)
                z_posterior = learn_x_network.sample_forward(whiting_outputs1[-1], training=False)

            # z_posterior = self.posterior_space.rsample()
            # z_posterior = z_posterior.view(b,c,w,h)
            if self.hparams['shape_attention']:
                # if step is not up tp the shredshold step, attention is not added.

                z_posterior_attention, no_sigmoid_embeddings = self.attention_layer.forward(z_posterior)
                if self.hparams['fusing_mode'] == 'P':
                    fuse_embedding = self.hparams['shape_attention_coeffient'] * embedding + (
                            z_posterior_attention * embedding)
                else:
                    fuse_embedding = self.hparams['shape_attention_coeffient'] * embedding + (
                                1 - self.hparams['shape_attention_coeffient'])*(
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
        if self.droptout:
            embedding = F.dropout(embedding)
        output = self.outc(embedding)
        logits = output
        return logits, no_sigmoid_embeddings


class ICML21_Unet_Alignment(Algorithm):
    def __init__(self, n_channels, n_classes, hparams, device, feature_dim=8, bilinear=True):
        super(ICML21_Unet_Alignment, self).__init__(n_channels, n_classes, hparams, device)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_domains = 3
        self.bilinear = bilinear
        self.feature_dim = feature_dim
        self.hparams = hparams
        # print(self.hparams['local_loss'])
        # exit()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3,
            "type": "Reparameterization",  # "Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.1,  # 0.5=>0.2
        }
        self.mu = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(64, feature_dim, kernel_size=1))
        dnn_to_bnn(self.mu, const_bnn_prior_parameters)
        # self.logvar = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        #                            nn.Conv2d(64, feature_dim, kernel_size=1))
        self.outc = nn.Sequential(nn.Conv2d(feature_dim, n_classes, kernel_size=1))
        dnn_to_bnn(self.outc, const_bnn_prior_parameters)

        # if (hparams['local_loss'] or hparams['global_loss']):
        #    self.metric_net = torch.nn.Sequential(torch.nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        #                                          torch.nn.LeakyReLU(), torch.nn.Conv2d(feature_dim, self.hparams['metric_dimension'], kernel_size=1),
        #                                          torch.nn.LeakyReLU())
        self.local_loss = LocalLoss(device, self.hparams['margin'])
        self.global_loss = GlobalLoss(device, self.hparams['margin'])

    def update(self, x, y):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # mu = self.mu(x)
        # logvar = self.logvar(x)

        embeddings_lists = []
        out_lists = []
        kl_lists = []
        # print(self.hparams['num_mc'])
        # exit()
        for mc_run in range(10):
            input = x
            # if mc_run == 0:
            embedding = self.mu(input)
            # embedding = input
            # embedding_metric = self.metric_net(embedding)
            # embeddings.append(embedding_metric)
            embeddings_lists.append(embedding)
            output = self.outc(embedding)
            kl = get_kl_loss(self.mu) + get_kl_loss(self.outc)
            out_lists.append(output)
            kl_lists.append(kl)
        logits = torch.mean(torch.stack(out_lists), dim=0)
        kl_loss = torch.mean(torch.stack(kl_lists), dim=0)
        # print(self.hparams['batch_size'])
        # exit()
        scaled_kl = kl_loss / self.hparams['batch_size']

        if self.hparams['local_loss']:
            input_embeddings = torch.stack(embeddings_lists)
            input_labels = torch.stack(out_lists)
            # 10, 24, 64, 144, 144
            # 10,48,512
            # embeddings_out = []
            # for index in range(self.hparams['num_mc']):
            #    embeddings_out.append(self.metric_net(input_embeddings[index]))
            # embeddings_out = torch.stack(embeddings_out)
            embeddings_out = torch.transpose(input_embeddings, 0, 1)
            labels_out = torch.transpose(input_labels, 0, 1)
            # torch.reshape(embeddings_out, [-1,self.hparams['num_mc'],8]
            local_loss, loss_pos, loss_pos_label, loss_neg, equal_sample_index1, nequal_sample_index1 = self.local_loss.forward(
                embeddings_out, labels_out, y, self.hparams['contrastive_type'], self.hparams['num_mc'],
                self.hparams['batch_size'], self.hparams['pairs_number'], self.num_domains)  # contrastive
            # print('local',local_loss)

        if self.hparams['global_loss']:
            # input_embeddings_spinal_cord = torch.stack(embeddings_spinal_cord)
            # 10, 24, 64, 144, 144
            # 10,48,512
            # embeddings_out = []
            # for index in range(self.hparams['num_mc']):
            #    embeddings_out.append(self.metric_net(input_embeddings_spinal_cord[index]))
            # embeddings_out = torch.stack(embeddings_out)
            # embeddings_out = torch.transpose(embeddings_out, 0, 1)
            # input_embeddings = torch.stack(embeddings_lists)
            # 10, 24, 64, 144, 144
            # 10,48,512
            # embeddings_out = []
            # for index in range(self.hparams['num_mc']):
            #    embeddings_out.append(self.metric_net(input_embeddings[index]))
            # embeddings_out = torch.stack(embeddings_out)
            # embeddings_out = torch.transpose(embeddings_out, 0, 1)
            # embedding = embeddings_out  # torch.reshape(embeddings_out, [-1,self.hparams['num_mc'],8]
            # input_embeddings = torch.stack(embeddings)

            # embeddings_out = self.metric_net(input_embeddings)
            # print(embeddings_out.shape)
            # embedding = torch.transpose(embeddings_out,0,1)
            # print(embedding.shape) # totoal_batch, num_mc, 128
            # features = [self.featurizer(xi) for xi, _ in minibatches]

            global_loss = loss_pos_label
            # self.global_loss.forward(embedding, y, self.hparams['contrastive_type_global'],
            #                                       self.hparams['num_mc'], self.hparams['batch_size'],
            #                                       self.hparams['pairs_number'], self.hparams['level2_gamma'],
            #                                       self.hparams['level1_gamma_global'], self.num_domains,
            #                                       equal_sample_index1, nequal_sample_index1)  # contrastive

        # feature = mu#self.reparameterization(mu, logvar)  # b*c*w*h
        # logits = self.outc(feature)
        return logits, scaled_kl, local_loss, global_loss, loss_pos, loss_neg  # torch.stack([mu, logvar, feature], dim=1)

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature = self.mu(x)
        logits = self.outc(feature)
        return logits


class ICML22_MMD(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device, skin=False, optim=False):
        super(ICML22_MMD, self).__init__(input_shape, num_classes, num_domains,
                                         hparams)
        self.num_domains = num_domains
        self.num_classes = num_classes
        # print(num_domains)
        # exit()
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.network_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
            torch.nn.ReLU())

        if (self.hparams['local_loss'] or self.hparams['global_loss']):
            self.metric_net = torch.nn.Sequential(torch.nn.Linear(self.featurizer.n_outputs, 256),
                                                  torch.nn.LeakyReLU(), torch.nn.Linear(256, 128), torch.nn.LeakyReLU())

        self.classifer = torch.nn.Sequential(torch.nn.Linear(self.featurizer.n_outputs, num_classes))
        # nn.Sequential(self.extractor_bayesian, self.relu, self.classifier_bayesian)
        # the parametes about the bnn

        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": self.hparams['bnn_rho_init'],
            "type": "Reparameterization",  # "Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": self.hparams['moped_delta_factor'],  # 0.5=>0.2
        }
        dnn_to_bnn(self.network_embedding, const_bnn_prior_parameters)
        dnn_to_bnn(self.classifer, const_bnn_prior_parameters)

        # self.optimizer = get_optimizer(
        #    hparams["optimizer"],
        #    self.network.parameters(),
        #    lr=self.hparams["lr"],
        #    weight_decay=self.hparams["weight_decay"],
        # )
        self.local_loss = LocalLoss(device, self.hparams['margin'])
        self.global_loss = GlobalLoss(device, self.hparams['margin'])
        self.skin = skin
        if self.skin:
            self.loss_func = FocalLoss(class_num=self.num_classes, gamma=2.)

    def update(self, optimizer, x, y, **kwargs):
        minibatch = to_minibatch(x, y)
        num = len(minibatch)
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        embeddings = []
        output_ = []
        kl_ = []
        z = self.featurizer(all_x)

        for mc_run in range(self.hparams['num_mc']):
            input = z
            embedding = self.network_embedding(input)
            #
            # embedding_metric = self.metric_net(embedding)
            # embeddings.append(embedding_metric)
            embeddings.append(embedding)
            output = self.classifer(embedding)
            kl = get_kl_loss(self.network_embedding) + get_kl_loss(self.classifer)
            output_.append(output)
            kl_.append(kl)
        # print('ckc')
        # print(self.hparams['contrastive_type'])
        if self.hparams['local_loss']:
            input_embeddings = torch.stack(embeddings)
            # print(input_embeddings.shape)
            embeddings_out = self.metric_net(input_embeddings)
            # print(embeddings_out.shape)
            # exit()
            embedding = torch.transpose(embeddings_out, 0, 1)
            # test_emb = torch.squeeze(embedding[:,0,])
            # end = time.time()
            # print(end - start)
            # print(self.hparams['contrastive_type'])
            local_loss, loss_pos, loss_neg = self.local_loss.forward(embedding, 0, all_y,
                                                                     self.hparams['contrastive_type'],
                                                                     self.hparams['num_mc'], self.hparams['batch_size'],
                                                                     self.hparams['pairs_number'],
                                                                     self.num_domains)  # contrastive
            # print('local',local_loss)
        # print('ckc2')
        if self.hparams['global_loss']:
            input_embeddings = torch.stack(embeddings)

            embeddings_out = self.metric_net(input_embeddings)
            # print(embeddings_out.shape)
            embedding = torch.transpose(embeddings_out, 0, 1)
            # print(embedding.shape) # totoal_batch, num_mc, 128
            # features = [self.featurizer(xi) for xi, _ in minibatches]
            # print(self.hparams['contrastive_type'])
            # print('ckckckc')
            global_loss = self.global_loss.forward(embedding, all_y, self.hparams['contrastive_type'],
                                                   self.hparams['num_mc'], self.hparams['batch_size'],
                                                   self.hparams['pairs_number'], self.hparams['level2_gamma'],
                                                   self.hparams['level1_gamma_global'], self.num_domains)  # contrastive
            # print('global',global_loss)

        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        if self.skin:
            cross_entropy_loss = self.loss_func(output, all_y)
            # print('focal',cross_entropy_loss)
        else:
            cross_entropy_loss = F.cross_entropy(output, all_y)
            # print('cross',cross_entropy_loss)
        scaled_kl = kl / self.hparams['batch_size']
        # ELBO loss
        if (self.hparams['local_loss'] == True and self.hparams['global_loss'] == False):
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'local_weight'] * local_loss
        elif (self.hparams['global_loss'] == True and self.hparams['local_loss'] == False):
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'global_weight'] * global_loss
        elif (self.hparams['local_loss'] and self.hparams['global_loss']):

            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'global_weight'] * global_loss + self.hparams['local_weight'] * local_loss
        else:
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl
        # print(local_loss)
        # print(cross_entropy_loss)
        # print(scaled_kl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # exit()
        if (self.hparams['local_loss'] == True and self.hparams['global_loss'] == False):
            return {'loss_total': loss.item(), 'contrastive_loss_total': local_loss.item(), 'pos_ccas': loss_pos.item(),
                    'neg_ccas': loss_neg.item()}
        elif (self.hparams['global_loss'] == True and self.hparams['local_loss'] == False):
            return {'loss_total': loss.item(), 'global_loss': global_loss.item()}
        elif (self.hparams['global_loss'] and self.hparams['local_loss']):
            return {'loss_total': loss.item(), 'global_loss': global_loss.item(),
                    'contrastive_loss_total': local_loss.item(), 'pos_ccas': loss_pos.item(),
                    'neg_ccas': loss_neg.item()}
        else:
            return {'loss_total': loss.item()}

    def predict(self, x):
        z = self.featurizer(x)
        embedding = self.network_embedding(z)
        return self.classifer(embedding)


class ICML22_MMD_mean(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device, skin=False, optim=False):
        super(ICML22_MMD_mean, self).__init__(input_shape, num_classes, num_domains,
                                              hparams)
        self.num_domains = num_domains
        self.num_classes = num_classes
        # print(num_domains)
        # exit()
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.network_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
            torch.nn.ReLU())

        if (self.hparams['local_loss'] or self.hparams['global_loss']):
            self.metric_net = torch.nn.Sequential(torch.nn.Linear(self.featurizer.n_outputs, 256),
                                                  torch.nn.LeakyReLU(), torch.nn.Linear(256, 128), torch.nn.LeakyReLU())

        self.classifer = torch.nn.Sequential(torch.nn.Linear(self.featurizer.n_outputs, num_classes))
        # nn.Sequential(self.extractor_bayesian, self.relu, self.classifier_bayesian)
        # the parametes about the bnn

        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": self.hparams['bnn_rho_init'],
            "type": "Reparameterization",  # "Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": self.hparams['moped_delta_factor'],  # 0.5=>0.2
        }
        dnn_to_bnn(self.network_embedding, const_bnn_prior_parameters)
        dnn_to_bnn(self.classifer, const_bnn_prior_parameters)

        # self.optimizer = get_optimizer(
        #    hparams["optimizer"],
        #    self.network.parameters(),
        #    lr=self.hparams["lr"],
        #    weight_decay=self.hparams["weight_decay"],
        # )
        self.local_loss = LocalLoss(device, self.hparams['margin'])
        self.global_loss = GlobalLoss(device, self.hparams['margin'])
        self.skin = skin
        if self.skin:
            self.loss_func = FocalLoss(class_num=self.num_classes, gamma=2.)
        self.mmd = compute_MMD(self.num_domains, self.hparams['batch_size'])

    def update(self, optimizer, x, y, **kwargs):
        minibatch = to_minibatch(x, y)
        num = len(minibatch)
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        embeddings = []
        output_ = []
        kl_ = []
        z = self.featurizer(all_x)

        for mc_run in range(self.hparams['num_mc']):
            input = z
            embedding = self.network_embedding(input)
            #
            # embedding_metric = self.metric_net(embedding)
            # embeddings.append(embedding_metric)
            embeddings.append(embedding)
            output = self.classifer(embedding)
            kl = get_kl_loss(self.network_embedding) + get_kl_loss(self.classifer)
            output_.append(output)
            kl_.append(kl)
        # print('ckc')
        if self.hparams['local_loss']:
            input_embeddings = torch.stack(embeddings)
            # print(input_embeddings.shape)
            embeddings_out = self.metric_net(input_embeddings)
            # print(embeddings_out.shape)
            # exit()
            embedding = torch.transpose(embeddings_out, 0, 1)
            embedding_mean = torch.mean(embedding, dim=1)
            # test_emb = torch.squeeze(embedding[:,0,])
            # end = time.time()
            # print(end - start)
            # print(self.hparams['contrastive_type'])
            local_loss, loss_pos, loss_neg = self.local_loss.forward(embedding_mean, 0, all_y,
                                                                     self.hparams['contrastive_type'],
                                                                     self.hparams['num_mc'], self.hparams['batch_size'],
                                                                     self.hparams['pairs_number'],
                                                                     self.num_domains)  # contrastive
            # print('local',local_loss)
        # print('ckc2')
        if self.hparams['global_loss']:
            input_embeddings = torch.stack(embeddings)

            embeddings_out = self.metric_net(input_embeddings)
            # print(embeddings_out.shape)
            embedding = torch.transpose(embeddings_out, 0, 1)
            # print(embedding.shape) # totoal_batch, num_mc, 128
            # features = [self.featurizer(xi) for xi, _ in minibatches]
            # get the mean of sampling distribution
            embedding_mean = torch.mean(embedding, dim=1)

            global_loss = self.mmd.forward(
                embedding_mean)  # self.global_loss.forward(embedding,all_y,self.hparams['contrastive_type'],self.hparams['num_mc'],self.hparams['batch_size'], self.hparams['pairs_number'],self.hparams['level2_gamma'],self.hparams['level1_gamma_global'],self.num_domains) # contrastive
            # print('global',global_loss)

        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        if self.skin:
            cross_entropy_loss = self.loss_func(output, all_y)
            # print('focal',cross_entropy_loss)
        else:
            cross_entropy_loss = F.cross_entropy(output, all_y)
            # print('cross',cross_entropy_loss)
        scaled_kl = kl / self.hparams['batch_size']
        # ELBO loss
        if (self.hparams['local_loss'] == True and self.hparams['global_loss'] == False):
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'local_weight'] * local_loss
        elif (self.hparams['global_loss'] == True and self.hparams['local_loss'] == False):
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'global_weight'] * global_loss
        elif (self.hparams['local_loss'] and self.hparams['global_loss']):

            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'global_weight'] * global_loss + self.hparams['local_weight'] * local_loss
        else:
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl
        # print(local_loss)
        # print(cross_entropy_loss)
        # print(scaled_kl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # exit()
        if (self.hparams['local_loss'] == True and self.hparams['global_loss'] == False):
            return {'loss_total': loss.item(), 'contrastive_loss_total': local_loss.item(), 'pos_ccas': loss_pos.item(),
                    'neg_ccas': loss_neg.item()}
        elif (self.hparams['global_loss'] == True and self.hparams['local_loss'] == False):
            return {'loss_total': loss.item(), 'global_loss': global_loss.item()}
        elif (self.hparams['global_loss'] and self.hparams['local_loss']):
            return {'loss_total': loss.item(), 'global_loss': global_loss.item(),
                    'contrastive_loss_total': local_loss.item(), 'pos_ccas': loss_pos.item(),
                    'neg_ccas': loss_neg.item()}
        else:
            return {'loss_total': loss.item()}

    def predict(self, x):
        z = self.featurizer(x)
        embedding = self.network_embedding(z)
        return self.classifer(embedding)


class ICML21_bayesian(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device, skin=False, optim=False):
        super(ICML21_bayesian, self).__init__(input_shape, num_classes, num_domains,
                                              hparams)

        self.batch_size = self.hparams['batch_size']
        self.num_domains = num_domains
        # print(num_domains)
        # exit()
        self.device = device
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.network_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
            torch.nn.ReLU())

        # if (self.hparams['local_loss'] or self.hparams['global_loss']):
        #    self.metric_net = torch.nn.Sequential(torch.nn.Linear(self.featurizer.n_outputs,self.featurizer.n_outputs//2),
        #    torch.nn.LeakyReLU(),torch.nn.Linear(self.featurizer.n_outputs//2, self.featurizer.n_outputs//4),torch.nn.LeakyReLU())

        self.classifer = torch.nn.Sequential(torch.nn.Linear(self.featurizer.n_outputs, num_classes))
        # nn.Sequential(self.extractor_bayesian, self.relu, self.classifier_bayesian)
        # the parametes about the bnn
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": self.hparams['bnn_rho_init'],
            "type": "Reparameterization",  # "Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": self.hparams['moped_delta_factor'],  # 0.5=>0.2
        }
        dnn_to_bnn(self.network_embedding, const_bnn_prior_parameters)
        dnn_to_bnn(self.classifer, const_bnn_prior_parameters)

        # self.optimizer = get_optimizer(
        #    hparams["optimizer"],
        #    self.network.parameters(),
        #    lr=self.hparams["lr"],
        #    weight_decay=self.hparams["weight_decay"],
        # )
        self.local_loss = LocalLoss(device, self.hparams['margin'])
        self.global_loss = GlobalLoss(device, self.hparams['margin'])
        self.skin = skin
        if self.skin:
            self.loss_func = FocalLoss(class_num=num_classes, gamma=2.)

    def update(self, optimizer, x, y, **kwargs):
        minibatch = to_minibatch(x, y)
        num = len(minibatch)
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        embeddings = []
        output_ = []
        kl_ = []
        z = self.featurizer(all_x)

        for mc_run in range(self.hparams['num_mc']):
            input = z
            embedding = self.network_embedding(input)
            #
            # embedding_metric = self.metric_net(embedding)
            # embeddings.append(embedding_metric)
            embeddings.append(embedding)
            output = self.classifer(embedding)
            kl = get_kl_loss(self.network_embedding) + get_kl_loss(self.classifer)
            output_.append(output)
            kl_.append(kl)

        if self.hparams['local_loss']:
            input_embeddings = torch.stack(embeddings)
            # print(input_embeddings.shape)
            embeddings_out = input_embeddings  # self.metric_net(input_embeddings)
            # print(embeddings_out.shape)
            # exit()
            embedding = torch.transpose(embeddings_out, 0, 1)
            # test_emb = torch.squeeze(embedding[:,0,])
            # end = time.time()
            # print(end - start)
            local_loss, loss_pos, loss_neg, equal_sample_pairs_1, equal_sample_pairs_2 = self.local_loss.forward(
                embedding, 0, all_y, self.hparams['contrastive_type'], self.hparams['num_mc'],
                self.hparams['batch_size'], self.hparams['pairs_number'], self.num_domains)  # contrastive
            # print('local',local_loss)

        if self.hparams['global_loss']:
            input_labels = torch.stack(output_)

            # embeddings_out = self.metric_net(input_embeddings)
            # print(embeddings_out.shape)
            input_labels = torch.transpose(input_labels, 0, 1)
            # print(embedding.shape) # totoal_batch, num_mc, 128
            # features = [self.featurizer(xi) for xi, _ in minibatches]
            pairs_number = len(equal_sample_pairs_1)
            global_loss = self.kl_loss(input_labels, equal_sample_pairs_1, equal_sample_pairs_2)
            global_loss /= pairs_number
            global_loss = global_loss.to(dtype=embedding.dtype)
            global_loss = global_loss.to(self.device)
            # print('global',global_loss)

        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        if self.skin:
            cross_entropy_loss = self.loss_func(output, all_y)
            # print('focal',cross_entropy_loss)
        else:
            cross_entropy_loss = F.cross_entropy(output, all_y)
            # print('cross',cross_entropy_loss)
        scaled_kl = kl / self.hparams['batch_size']
        # ELBO loss
        if (self.hparams['local_loss'] == True and self.hparams['global_loss'] == False):
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'local_weight'] * local_loss
        elif (self.hparams['global_loss'] == True and self.hparams['local_loss'] == False):
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'global_weight'] * global_loss
        elif (self.hparams['local_loss'] and self.hparams['global_loss']):

            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl + self.hparams[
                'global_weight'] * global_loss + self.hparams['local_weight'] * local_loss
        else:
            loss = cross_entropy_loss + self.hparams['kl_weight'] * scaled_kl
        # print(local_loss)
        # print(cross_entropy_loss)
        # print(scaled_kl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # exit()
        if (self.hparams['local_loss'] == True and self.hparams['global_loss'] == False):
            return {'loss_total': loss.item(), 'contrastive_loss_total': local_loss.item(), 'pos_ccas': loss_pos.item(),
                    'neg_ccas': loss_neg.item()}
        elif (self.hparams['global_loss'] == True and self.hparams['local_loss'] == False):
            return {'loss_total': loss.item(), 'global_loss': global_loss.item()}
        elif (self.hparams['global_loss'] and self.hparams['local_loss']):
            return {'loss_total': loss.item(), 'global_loss': global_loss.item(),
                    'contrastive_loss_total': local_loss.item(), 'pos_ccas': loss_pos.item(),
                    'neg_ccas': loss_neg.item()}
        else:
            return {'loss_total': loss.item()}

    def predict(self, x):
        z = self.featurizer(x)
        embedding = self.network_embedding(z)
        return self.classifer(embedding)

    def kl_loss(self, embeddings, row, col):  # x, y, class_eq
        loss = 0
        length_ = len(row)
        for i in range(length_):
            # print(get_single_mmd(kall, row[i], col[i], sample_size))
            # print((0.5 * get_single_mmd(kall, row[i], col[i], sample_size)))
            input_ = embeddings[row[i]]
            target = embeddings[col[i]]

            loss += F.kl_div(input_.softmax(dim=-1).log(), target.softmax(dim=-1), reduction='sum')
            # print('ckc')
            # print(loss)
        return loss


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return get_optimizer(
                hparams["optimizer"], p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, x, y, **kwargs):
        # inputs
        all_x = torch.cat([xi for xi in x])
        # labels
        all_y = torch.cat([yi for yi in y])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
