import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from network.pointnet import PointNetEncoder, feature_transform_reguliarzer


class pointnet_reg(nn.Module):
    def __init__(self, num_class=1, with_rgb=True):
        super(pointnet_reg, self).__init__()
        if with_rgb:
            channel = 6
        else:
            channel = 3
        self.k = num_class
        self.feat_o = PointNetEncoder(global_feat=False, feature_transform=False, channel=channel)  # feature trans True
        self.feat_h = PointNetEncoder(global_feat=False, feature_transform=False, channel=channel)  # feature trans True
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.convfuse = torch.nn.Conv1d(3778, 3000, 1)
        self.bnfuse = nn.BatchNorm1d(3000)

    def forward(self, x, hand):
        '''
        :param x: obj pc [B, D, N]
        :param hand: hand pc [B, D, 778]
        :return: regressed cmap
        '''
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # for obj
        x, trans, trans_feat = self.feat_o(x)  # x: [B, 1088, N] global+point feature of object
        # for hand
        hand, trans2, trans_feat2 = self.feat_h(hand)  # hand: [B, 1088, 778] global+point feature of hand
        # fuse feature of object and hand
        x = torch.cat((x, hand), dim=2).permute(0,2,1).contiguous()  # [B, N+778, 1088]
        x = F.relu(self.bnfuse(self.convfuse(x)))  # [B, N, 1088]
        x = x.permute(0,2,1).contiguous()  # [B, 1088, N]
        # inference cmap
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N]
        x = self.conv4(x)  # [B, 1, N]
        x = x.transpose(2,1).contiguous()
        x = torch.sigmoid(x)
        x = x.view(batchsize, n_pts)  # n_pts  [B, N]
        return x

