import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from models.resnet import resnet18
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class Tripletnet(nn.Module):
    def __init__(self, margin):
        super(Tripletnet, self).__init__()
        self.embeddingnet = resnet18().cuda(1)
        self.triple_loss = torch.nn.MarginRankingLoss(margin = margin)

    def forward(self, x, y, z):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        general_x = self.embeddingnet(x)
        general_y = self.embeddingnet(y)
        general_z = self.embeddingnet(z)
        # l2-normalize embeddings
        norm = torch.norm(general_x, p=2, dim=1) + 1e-10
        norm = norm.view(norm.size(0), 1)
        general_x = general_x / norm
        norm = torch.norm(general_y, p=2, dim=1) + 1e-10
        norm = norm.view(norm.size(0), 1)
        general_y = general_y / norm
        norm = torch.norm(general_z, p=2, dim=1) + 1e-10
        norm = norm.view(norm.size(0), 1)
        general_z = general_z / norm

        dist_a = F.pairwise_distance(general_x, general_y, 2)
        dist_b = F.pairwise_distance(general_x, general_z, 2)
        dist_b = 0.3 - dist_b
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        target = target.cuda(1)
        target = Variable(target)
        loss = self.triple_loss(dist_a, dist_b, target)
        return loss