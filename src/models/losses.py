import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common.util import L2Norm

import sys


class softmax(nn.Module):
    def __init__(self, class_num, args, bias=False):
        super(softmax, self).__init__()
        self.in_features = args.feature_dim
        self.out_features = class_num
        self.w_norm = args.use_w_norm
        self.f_norm = args.use_f_norm
        self.s = args.s
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input, target):
        if self.w_norm:
            weight = L2Norm()(self.weight)
        else:
            weight = self.weight
        if self.f_norm:
            x = L2Norm()(input)
        else:
            x = input
        scores = F.linear(x, weight, self.bias)  # x @ weight.t() + bias(if any)
        if self.w_norm and self.f_norm:
            assert self.s > 1.0, 'scaling factor s should > 1.0'
            scores_new = self.s * scores
        else:
            scores_new = scores
        return scores_new, self.CELoss(scores_new, target.view(-1))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class asoftmax(nn.Module):
    def __init__(self, class_num, args):
        super(asoftmax, self).__init__()
        self.in_features = args.feature_dim
        self.out_features = class_num
        self.w_norm = args.use_w_norm
        self.f_norm = args.use_f_norm
        self.m = args.m_1
        self.it = 0
        self.LambdaMin = 3.0
        self.LambdaMax = 1500.0
        self.Lambda = 1500.0
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.register_parameter('bias', None)
        self.reset_parameters()  # weight initialization
        assert (self.w_norm == True and self.f_norm == False), 'Wrong implementation of A-Softmax loss.'
        assert self.m >= 1., 'margin m of asoftmax should >= 1.0'

    def forward(self, input, target):
        if self.w_norm:
            weight = L2Norm()(self.weight)
        else:
            weight = self.weight
        scores = F.linear(input, weight, self.bias)  # x @ weight.t() + bias(if any)
        index = torch.zeros_like(scores).scatter_(1, target, 1)  # the index of the target is 1, others are 0

        x_len = input.norm(dim=1)
        cos_theta = scores / (x_len.view(-1, 1).clamp(min=1e-12))  # cos_theta = a.b / |a|x|b|
        cos_theta = cos_theta.clamp(-1, 1)
        m_theta = self.m * torch.acos(cos_theta)  # acos returns arc cosine in radians
        k = (m_theta / 3.141592653589793).floor().detach()  # floor(), return the largest integer smaller or equal to
        cos_m_theta = torch.cos(m_theta)
        psi_theta = ((-1) ** k) * cos_m_theta - 2 * k
        psi_theta = psi_theta * x_len.view(-1, 1)  # ||x|| * psi_theta

        self.Lambda = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        self.it += 1
        scores_new = scores - scores * index / (1 + self.Lambda) + psi_theta * index / (1 + self.Lambda)
        return scores, self.CELoss(scores_new, target.view(-1))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class amsoftmax(nn.Module):
    def __init__(self, class_num, args):
        super(amsoftmax, self).__init__()
        self.in_features = args.feature_dim
        self.out_features = class_num
        self.w_norm = args.use_w_norm
        self.f_norm = args.use_f_norm
        self.s = args.s
        self.m = args.m_3
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.register_parameter('bias', None)
        self.reset_parameters()  # weight initialization
        assert (self.w_norm and self.f_norm), 'Wrong implementation of AMSoftmax loss.'
        assert self.s > 1.0, 'scaling factor s should > 1.0'
        assert self.m > 0., 'scaling factor s should > 1.0'

    def forward(self, input, target):
        if self.w_norm:
            weight = L2Norm()(self.weight)
        else:
            weight = self.weight
        if self.f_norm:
            x = L2Norm()(input)
        else:
            x = input
        scores = F.linear(x, weight, self.bias)  # x @ weight.t() + bias(if any)
        index = torch.zeros_like(scores).scatter_(1, target, 1)
        scores_new = self.s * (scores - scores * index + (scores - self.m) * index)
        # scores_new = input_norm*(scores - scores*index + (scores - self.m)*index)
        return scores, self.CELoss(scores_new, target.view(-1))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class centerloss(nn.Module):
    def __init__(self, class_num, args, bias=False):
        super(centerloss, self).__init__()
        self.device = args.device
        self.lamb = args.lamb  # weight of center loss
        self.alpha = 0.5  # weight of updating centers
        self.in_features = args.feature_dim
        self.class_num = class_num
        self.f_norm = args.use_f_norm
        self.centers = torch.nn.Parameter(torch.Tensor(self.class_num, self.in_features))
        self.centers.requires_grad = False
        self.delta_centers = torch.zeros_like(self.centers)
        self.softmaxloss = softmax(class_num, args)
        self.reset_parameters()

    def forward(self, input, target):
        scores, loss = self.softmaxloss(input, target)  # Softmax loss
        '''
            Center loss: follow the paper's implementation.
            Inspired by https://github.com/louis-she/center-loss.pytorch/blob/5be899d1f622d24d7de0039dc50b54ce5a6b1151/loss.py
        '''
        ## Center loss
        if self.f_norm:
            x = L2Norm()(input)
        else:
            x = input
        self.update_center(x, target)

        target_centers = self.centers[target].squeeze()
        center_loss = ((x - target_centers) ** 2).sum(dim=1).mean()
        return scores, loss + self.lamb * 0.5 * center_loss

    def update_center(self, features, targets):
        # implementation equation (4) in the center-loss paper
        targets, indices = torch.sort(targets.view(-1))
        target_centers = self.centers[targets]
        features = features.detach()[indices]
        delta_centers = target_centers - features
        uni_targets, indices = torch.unique(targets.cpu(), sorted=True, return_inverse=True)
        uni_targets = uni_targets.to(self.device)
        indices = indices.to(self.device)
        delta_centers = torch.zeros(uni_targets.size(0), delta_centers.size(1)).to(self.device).index_add_(0, indices, delta_centers)
        targets_repeat_num = uni_targets.size()[0]
        uni_targets_repeat_num = targets.size()[0]
        targets_repeat = targets.repeat(targets_repeat_num).view(targets_repeat_num, -1)
        uni_targets_repeat = uni_targets.unsqueeze(1).repeat(1, uni_targets_repeat_num)
        same_class_feature_count = torch.sum(targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)
        delta_centers = delta_centers / (same_class_feature_count + 1.0) * self.alpha
        result = torch.zeros_like(self.centers)
        result[uni_targets, :] = delta_centers
        self.centers -= result

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))
