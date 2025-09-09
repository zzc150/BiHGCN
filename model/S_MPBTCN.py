import torch
import torch.nn as nn
import torch.nn.functional as F
from . import hypergraph_utils as hgu

import math


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x): # (B, T, N, D)
        # x = x.transpose(1, 3)
        x = self.linear(x)
        # x = x.transpose(1, 3)

        return x # (B, T, N, D)

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        x = x[:, :, :-self.chomp_size]

        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, groups, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, groups = 1)
        self.chomp1 = Chomp(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.net_u = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

    def forward(self, x):
        x = self.net_u(x)
        return x

class BTCN(nn.Module):
    def __init__(self, nodes, hidden_dim, groups, dropout):
        super(BTCN, self).__init__()

        kernel_size = 2
        in_channels = nodes
        out_channels = nodes
        stride = 1
        layer1 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=1, padding= 1, dropout=dropout) # padding=(kernel_size - 1) * 1
        layer2 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=2, padding= 2, dropout=dropout)
        layer3 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=4, padding= 4, dropout=dropout)
        layer4 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=4, padding= 4, dropout=dropout)
        self.GRU_p = nn.GRU(in_channels, hidden_dim, num_layers=2, batch_first=True)
        self.GRU_b = nn.GRU(out_channels, hidden_dim, num_layers=2, batch_first=True)
        self.network_p = nn.Sequential(layer1, layer2, layer3, layer4)
        self.network_b = nn.Sequential(layer1, layer2, layer3, layer4)
        self.linear = nn.Linear(hidden_dim*2, nodes)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x): # (B, T, N, D)
        B, T, N, D = x.shape
        x = x.reshape(B, T, N * D)
        res = x

        # bitcn
        x = x.transpose(1, 2)  # (B, N*D, T)
        x_p = self.network_p(x)
        x_re = torch.flip(x, dims=[2])  # (B, N*D, T)
        x_b = self.network_b(x_re)
        x_b = torch.flip(x_b, dims=[2])
        x = x_p + x_b
        x = x.transpose(1, 2)  # (B, T, N*D)
        x = x + res

        # bigru
        x_p, hidden = self.GRU_p(x)
        x_p = self.dropout1(x_p)
        x_re = torch.flip(x, dims=[1])
        x_b, hidden = self.GRU_b(x_re)
        x_b = torch.flip(x_b, dims=[1])
        x_b = self.dropout1(x_b)
        x = x_p + x_b
        x = torch.cat([x_p, x_b], dim=2)

        x = self.linear(x)
        x = x + res
        x = x.reshape(B, T, N, D)
        return x

class HypergraphLearning(nn.Module):
    def __init__(self, args, num_edges):
        super(HypergraphLearning, self).__init__()
        self.args = args
        self.fc = FC(args.feature, args.hidden_dim)
        self.fcout = FC(args.hidden_dim, args.feature)
        self.num_edges = num_edges
        self.edge_clf = torch.randn(args.hidden_dim, self.num_edges) / math.sqrt(self.num_edges)  # 64 32
        self.edge_clf = nn.Parameter(self.edge_clf, requires_grad=True)
        self.edge_map = torch.randn(self.num_edges, self.num_edges) / math.sqrt(self.num_edges) # 32 32
        self.edge_map = nn.Parameter(self.edge_map, requires_grad=True)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(args.hidden_dim)

    def forward(self, x):  # 32 12 170 64  B x T x N x D
        x = self.fc(x)
        feat = x.reshape(x.size(0), -1, x.size(3))  # 32 2040 64
        hyper_assignment = torch.softmax(feat @ self.edge_clf, dim=-1) # 32 2040 32 feat相当于论文里的H，edge_clf相当于W
        hyper_feat = hyper_assignment.transpose(1, 2) @ feat # 32 32 64 U AT H
        hyper_feat_mapped = self.activation(self.edge_map @ hyper_feat)
        hyper_out = hyper_feat_mapped + hyper_feat # 32 32 64
        y = self.activation(hyper_assignment @ hyper_out)  # 32 2040 64
        y = y.reshape(x.size(0), x.size(1), x.size(2), x.size(3))  # 32 12 170 64
        y_final = self.norm(y + x)
        y_final = self.fcout(y_final)
        return y_final

#
# class ST_Block(nn.Module):
#     def __init__(self, args, adj_data, f_nodes, s_nodes, feature, input_dim, hidden_dim, output_dim, dropout, G):
#         super(ST_Block, self).__init__()
#
#         self.graph = adj_data
#         self.f_nodes = f_nodes
#         self.s_nodes = s_nodes
#         self.nodes = f_nodes + s_nodes
#
#         self.spatial_f = Spatial(feature, input_dim)
#         self.spatial_s = Spatial(feature, input_dim)
#
#         self.spatial = Spatial(feature, input_dim)
#
#         self.btcn_f = BTCN(f_nodes * feature, groups=self.f_nodes * feature, dropout=dropout)
#         self.btcn_s = BTCN(s_nodes * feature, groups=self.s_nodes * feature, dropout=dropout)
#
#         self.btcn = BTCN(self.nodes * feature, groups=self.nodes * feature, dropout=dropout)
#         self.hyper = HypergraphLearning(args, num_edges=32)
#         self.HGNN = HGNN_conv(args, G)
#
#
#
#     def forward(self, x): # (B, T, N, D)
#         res = x
#
#         # s_cr = self.spatial(x, self.graph)
#         s_cr = self.HGNN(x)
#         xs = s_cr + res
#         # t1 = self.btcn(xs)
#         t1 = self.hyper(self.btcn(x))
#
#         t = t1 + xs
#
#         t_f = t[:, :, :self.f_nodes, :]
#         t_s = t[:, :, self.f_nodes:, :]
#         t_f = self.btcn_f(t_f)
#         t_s = self.btcn_s(t_s)
#         t2 = torch.cat([t_f, t_s], dim=2)
#
#         xt = t2
#
#         return xt


class ST_Block(nn.Module):
    def __init__(self, args, nodes, feature, input_dim, hidden_dim, output_dim, dropout, Gs):
        super(ST_Block, self).__init__()


        self.nodes = nodes
        self.Gs = Gs
        # self.Gtaxi = Gtaxi
        # self.Gbike = Gbike
        # self.spatial = Spatial(feature, input_dim)
        self.winter = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(4)])
        self.wintra = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(2)])
        self.weights = nn.Parameter(torch.ones(1))
        self.btcn = BTCN(self.nodes * feature, hidden_dim, groups=self.nodes * feature, dropout=dropout)

        # self.backbone = GNNLayer(args)
        self.hyper = HypergraphLearning(args, num_edges=32)

        self.HGNN1 = HGNN_conv(args)
        self.HGNN2 = HGNN_conv(args)
        self.mlp = MLP(args.input_dim, args.hidden_dim, args.output_dim, args.dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




        # 定义两个可学习的权重参数
        self.weight_f = nn.Parameter(torch.tensor(0.5))
        self.weight_s = nn.Parameter(torch.tensor(0.5))
        num = 3
        kernel_size = 1
        new_dilation = 1
        self.start_conv = nn.Conv2d(in_channels=feature,
                                    out_channels=args.hidden_dim,
                                    kernel_size=(1, 1))

        self.filter_convs = (nn.Conv2d(in_channels=num * args.hidden_dim,
                                           out_channels=feature,
                                           kernel_size=(1, kernel_size), dilation=new_dilation))

        self.gate_convs = (nn.Conv2d(in_channels=num * args.hidden_dim,
                                         out_channels=feature,
                                         kernel_size=(1, kernel_size), dilation=new_dilation))
    # 相同区域数
    def update_hypergraph(self, x):
        # try:
            edge_dict = hgu.construct_edge_list_from_knn(x, k_neighs=[5, 5])
            H = hgu._edge_dict_to_H(edge_dict)
            Gt = hgu._generate_G_from_H(H)
        # except Exception as e:
        #     print(f"Error during hypergraph update: {e}")
        #     return None

            Gt = torch.Tensor(Gt).to(self.device)  # 动态决定数据存放设备
            return Gt, H, edge_dict

    # 不同区域数
    # def update_hypergraph(self, x):
    #     # try:
    #         H = hgu.construct_H_with_KNN(x, K_neigs=[12], split_diff_scale=False, is_probH=False, m_prob=1)
    #         # H = hgu._edge_dict_to_H(edge_dict)
    #         Gt = hgu._generate_G_from_H(H)
    #     # except Exception as e:
    #     #     print(f"Error during hypergraph update: {e}")
    #     #     return None
    #
    #         Gt = torch.Tensor(Gt).to(self.device)  # 动态决定数据存放设备
    #         return Gt, H

    def forward(self, x):
        #taxi和bike具有相同的区域数曼哈顿
        res = x     #64,12,130,2 =(B, T, N, D)

        # 分开输入数据
        Xall = x   #64,12,65,2
        # 将上下车数据分开
        pick = Xall[:, :, :, 0]  #64,12,65
        drop = Xall[:, :, :, 1] #64,12,65
        X1 = [pick, drop]
        Gt, Ht, Edict = self.update_hypergraph(X1)
        S1 = self.weights * self.HGNN2(Gt, x)
        S2 = self.HGNN1(self.Gs, x)
        x = S1 + S2 + res
        t = self.btcn(x)
        # t2 = self.hyper(t2)
        xt = self.mlp(t)
        return xt


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))

    # def forward(self, x): # (B, T', N, D)
    #
    #     x = x.transpose(1,3)
    #     x = self.linear(x)
    #     x = x.transpose(1, 3)
    def forward(self, x):
        if len(x.shape) == 4:  # 输入张量是四维 (B, T', N, D)
            x = x.transpose(1, 3)  # 将 T' 维度交换到最后一个维度，形状变为 (B, D, N, T')
            x = self.linear(x)  # 线性层应用于 T' 维度
            x = x.transpose(1, 3)  # 将维度交换回来，形状变为 (B, T', N, D)
        elif len(x.shape) == 3:  # 输入张量是三维 (B, T', N)
            x = x.transpose(1, 2)  # 将 T' 维度交换到最后一个维度，形状变为 (B, N, T')
            x = self.linear(x)  # 线性层应用于 T' 维度，需要增加一个维度
            x = x.transpose(1, 2)  # 将维度交换回来，形状变为 (B, T', N)

        return x # (B, T, D, N)




class HGNN_conv(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, args ):
        super(HGNN_conv, self).__init__()

        self.dim_in = args.input_dim
        self.dim_out = args.output_dim
        self.fc = MLP(args.input_dim, args.hidden_dim, args.output_dim, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        # self.G = G


    def forward(self, G, x):
        if x.dim() == 4:
            x = nn.ReLU()(self.fc(x))
            x = G.matmul(x)
        elif x.dim() == 3:
            B, T, N = x.shape
            # 增加一个特征维度 D=1
            x = x.unsqueeze(-1)  # (B, T, N) -> (B, T, N, 1)
            x = nn.ReLU()(self.fc(x))
            x = G.matmul(x)
            _, _, _, D = x.shape
            x = x.view(B, T, N * D)
        x = self.dropout(x)
        return x


# class HGNN_conv(nn.Module):
#     """
#     A HGNN layer
#     """
#
#     def __init__(self, args):
#         super(HGNN_conv, self).__init__()
#
#         self.dim_in = args.input_dim
#         self.dim_out = args.output_dim
#         self.fc = MLP(args.input_dim, args.hidden_dim, args.output_dim, args.dropout)
#         self.dropout = nn.Dropout(args.dropout)
#         self.batch_normalization1 = nn.BatchNorm2d(args.feature, momentum=0.1)
#         self.batch_normalization2 = nn.BatchNorm2d(1, momentum=0.1)
#         # 添加一个线性层来设置可学习的权重和偏置
#         self.weight1 = nn.Parameter(torch.Tensor(args.feature, args.feature))
#         self.bias1 = nn.Parameter(torch.Tensor(args.feature))
#         self.weight2 = nn.Parameter(torch.Tensor(1, 1))
#         self.bias2 = nn.Parameter(torch.Tensor(1))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv1 = 1. / math.sqrt(self.weight1.size(1))
#         stdv2 = 1. / math.sqrt(self.weight2.size(1))
#
#         self.weight1.data.uniform_(-stdv1, stdv1)
#         self.bias1.data.uniform_(-stdv1, stdv1)
#         self.weight2.data.uniform_(-stdv2, stdv2)
#         self.bias2.data.uniform_(-stdv2, stdv2)
#
#     def forward(self, G, x):
#         if x.dim() == 4:
#             x = nn.ReLU()(self.fc(x))
#             x = x.permute(0, 3, 1, 2)
#             x = self.batch_normalization1(x)
#             x = x.permute(0, 2, 3, 1)  # 恢复形状: (B, T, N, D)
#             x = x.matmul(self.weight1)  # 应用权重
#             x = G.matmul(x)
#             x = x + self.bias1  # 应用偏置
#         elif x.dim() == 3:
#             B, T, N = x.shape
#             # 增加一个特征维度 D=1
#             x = x.unsqueeze(-1)  # (B, T, N) -> (B, T, N, 1)
#             x = nn.ReLU()(self.fc(x))
#             x = x.permute(0, 3, 1, 2)
#             x = self.batch_normalization2(x)
#             x = x.permute(0, 2, 3, 1)  # 恢复形状: (B, T, N, D)
#             x = x.matmul(self.weight2)  # 应用权重
#             x = G.matmul(x)
#             _, _, _, D = x.shape
#             x = x.view(B, T, N * D)
#             x = x + self.bias2 # 应用偏置
#         # x = self.dropout(x)
#         return x


class HGNN(nn.Module):
    def __init__(self, args, momentum=0.1):
        super(HGNN, self).__init__()
        self.fc = MLP(args.input_dim, args.hidden_dim, args.output_dim, args.dropout)
        self.hgc1 = HGNN_conv(args.input_dim, args.hidden_dim)
        self.hgc2 = HGNN_conv(args.hidden_dim, args.output_dim)
        self.batch_normalization1 = nn.BatchNorm2d(args.feature, momentum=momentum)
        self.batch_normalization2 = nn.BatchNorm2d(1, momentum=momentum)
        self.dropout = nn.Dropout(args.dropout)
    def forward(self, G, x):
        if x.dim() == 4:
            B, T, N, D = x.shape
            x = nn.ReLU()(self.fc(x))
            x = x.permute(0, 3, 1, 2)
            x = self.batch_normalization1(x)
            x = x.permute(0, 2, 3, 1)  # 恢复形状: (B, T, N, D)
            x = self.hgc1(x, G)
            x = x.permute(0, 3, 2, 1)
            x = nn.ReLU()(x)
            x = self.dropout(x)
            x = self.hgc2(x, G)
            x = x.permute(0, 3, 2, 1)

        elif x.dim() == 3:
            B, T, N = x.shape
            # 增加一个特征维度 D=1
            x = x.unsqueeze(-1)  # (B, T, N) -> (B, T, N, 1)
            x = nn.ReLU()(self.fc(x))
            x = x.permute(0, 3, 1, 2)
            x = self.batch_normalization2(x)
            x = x.permute(0, 2, 3, 1)  # 恢复形状: (B, T, N, D)
            x = self.hgc1(x, G)
            x = x.permute(0, 3, 2, 1)
            x = nn.ReLU()(x)
            x = self.dropout(x)
            x = self.hgc2(x, G)
            _, D, _, _ = x.shape
            x = x.view(B, T, N * D)


        return x

class Network(nn.Module):
    def __init__(self, args, nodes, blocks, feature, input_dim, hidden_dim, output_dim, dropout, G):
        super(Network, self).__init__()
        self.nodes = nodes
        self.time_embedding = nn.Linear(args.day_len, args.hidden_dim)
            # nn.Embedding(48, args.hidden_dim)
        self.date_embedding = nn.Linear(args.week_len, args.hidden_dim)
        self.node_embedding = nn.Embedding(self.nodes, args.hidden_dim)
        self.input_embedding = nn.Sequential(nn.Linear(args.input_dim, args.hidden_dim), nn.ReLU())
        self.day_time_embed_conv = nn.Conv2d(in_channels=args.hidden_dim, # hard code to avoid errors
                                                out_channels=2,
                                                kernel_size=(1, 1))
        self.week_time_embed_conv = nn.Conv2d(in_channels=args.hidden_dim,  # hard code to avoid errors
                                                 out_channels=2,
                                                 kernel_size=(1, 1))
        self.input_embed_conv = nn.Conv2d(in_channels=args.hidden_dim,  # hard code to avoid errors
                                       out_channels=input_dim,
                                       kernel_size=(1, 1))
        self.node_embed_conv= nn.Conv2d(in_channels=args.hidden_dim,  # hard code to avoid errors
                                       out_channels=2,
                                       kernel_size=(1, 1))


        self.blocks = blocks
        st_blocks = []
        for i in range(self.blocks):
            st_blocks.append(ST_Block(args, nodes, feature, input_dim, hidden_dim, output_dim, dropout, G))
        self.st = nn.ModuleList(st_blocks)

        self.mlp = MLP(input_dim * (blocks + 1), hidden_dim, output_dim, dropout=dropout)




    def forward(self, x): # (B, T, N, D)
        res = x[0]  # 64,12,130,2 =(B, T, N, D)
        week = x[1]  # 64,12,7
        day = x[2]  # 64,12,48
        # day = day.long()
        # week = week.long()
        node_idx = torch.arange(0, self.nodes).to(res.device)  # N   130
        feat = res.permute(0, 3, 2, 1)


        input_emb = self.input_embedding(feat)  #
        input_emb = input_emb.permute(0, 3, 2, 1)
        input_emb = self.input_embed_conv(input_emb)
        # input_emb = input_emb.permute(0, 3, 2, 1)

        day = day.unsqueeze(dim=-1)
        day = day.permute(0, 1, 3, 2)
        day = day.repeat(1, 1, res.shape[2], 1)
        time_emb = self.time_embedding(day) # 64 12 130 128
        time_emb = time_emb.permute(0, 3, 2, 1)
        time_emb = self.day_time_embed_conv(time_emb)
        time_emb = time_emb.permute(0, 3, 2, 1)

        week = week.unsqueeze(dim=-1)
        week = week.permute(0, 1, 3, 2)
        week = week.repeat(1, 1, res.shape[2], 1)
        date_emb = self.date_embedding(week)  # B x T x N x hD
        date_emb = date_emb.permute(0, 3, 2, 1)
        date_emb = self.week_time_embed_conv(date_emb)
        date_emb = date_emb.permute(0, 3, 2, 1)

        node_emb = self.node_embedding(node_idx).unsqueeze(0).unsqueeze(0)  # 1 1 170 64
        node_emb = node_emb.repeat(res.shape[0], res.shape[1], 1, 1)
        node_emb = node_emb.permute(0, 3, 2, 1)
        node_emb = self.node_embed_conv(node_emb)
        node_emb = node_emb.permute(0, 3, 2, 1)


        feature = input_emb + time_emb + date_emb + node_emb  # B x T x N x D
        # feature = input_emb + time_emb + node_emb  # B x T x N x D
        # feature = input_emb + node_emb  # B x T x N x D

        ST_out = []
        st_out = feature
        for st in self.st:
            st_out = st(st_out)
            ST_out.append(st_out)
            st_out = st_out + res

        ST_out.append(res)

        out = torch.cat(ST_out,dim=1) # (B, T', N, D)
        x = self.mlp(out)
        return x


