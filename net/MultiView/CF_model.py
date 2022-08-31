import torch
import torch.nn as nn
from net.utils.st_gcn.graph import Graph
from net.backbone.st_gcn import st_gcn
import torch.nn.functional as F

"""
    多视角融合方式1：直接拼接特征（Concatenate Feature）
"""


class ConcatenateFeature(nn.Module):
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()
        self.view_num = 3

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A,dtype=torch.float32,requires_grad=False)
        As = torch.stack([A,A,A],dim=0)
        self.register_buffer('A', A)
        self.register_buffer('As', As)

        # build networks
        spatial_kernel_size_stage1 = A.size(0)
        temporal_kernel_size_stage1 = 9
        kernel_size_stage1 = (temporal_kernel_size_stage1, spatial_kernel_size_stage1)
        self.data_bn = nn.BatchNorm1d(in_channels*A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # 第一阶段网络
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size_stage1, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size_stage1, 1, **kwargs),
            st_gcn(64, 64, kernel_size_stage1, 1, **kwargs),  # delete
            st_gcn(64, 64, kernel_size_stage1, 1, **kwargs),
            st_gcn(64, 128, kernel_size_stage1, 2, **kwargs),
            st_gcn(128, 128, kernel_size_stage1, 1, **kwargs),
            st_gcn(128, 128, kernel_size_stage1, 1, **kwargs),  # delete
            st_gcn(128, 256, kernel_size_stage1, 2, **kwargs),
            st_gcn(256, 256, kernel_size_stage1, 1, **kwargs),
            st_gcn(256, 256, kernel_size_stage1, 1, **kwargs),
        ))
        self.predict_stage1 = nn.ModuleList([nn.Conv2d(256, num_class, kernel_size=1)
                                                for _ in range(self.view_num)])

        # 第二阶段融合网络
        self.predict_stage2 = nn.Conv2d(256*self.view_num, num_class,kernel_size=1)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_stage1 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])

    def data_process(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        return x

    def forward(self,x1,x2,x3):
        N, C, T, V, M = x1.size()
        # 预处理阶段
        x1 = self.data_process(x1)
        x2 = self.data_process(x2)
        x3 = self.data_process(x3)
        x = [x1,x2,x3]
        # 第一阶段学习
        for view_index in range(self.view_num):
            for gcn, importance in zip(self.st_gcn_networks, self.edge_importance_stage1):
                x[view_index], _ = gcn(x[view_index], self.A * importance)
        output1 = x.copy()
        for view_index in range(self.view_num):
            output1[view_index] = F.avg_pool2d(output1[view_index], output1[view_index].size()[2:])
            output1[view_index] = output1[view_index].view(N, M, -1, 1, 1).mean(dim=1)
            output1[view_index] = self.predict_stage1[view_index](output1[view_index])
            output1[view_index] = output1[view_index].view(output1[view_index].size(0), -1)
        # 第二阶段融合特征
        output2 = torch.cat(x, dim=1)   # 直接在通道上拼接特征
        output2 = F.avg_pool2d(output2, output2.size()[2:])
        output2 = output2.view(N, M, -1, 1, 1).mean(dim=1)
        output2 = self.predict_stage2(output2)
        output2 = output2.view(output2.size(0),-1)

        return output1, output2








