import numpy as np
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
        # self.linear = nn.Linear(hidden_dim , nodes)
        self.dropout1 = nn.Dropout(dropout)



    def forward(self, x): # (B, T, N, D)
        B, T, N, D = x.shape
        x = x.reshape(B, T, N*D)
        res = x

        # bitcn
        x = x.transpose(1,2) # (B, N*D, T)
        x_p = self.network_p(x)
        x_re = torch.flip(x,dims=[2]) #  (B, N*D, T)
        x_b = self.network_b(x_re)
        x_b = torch.flip(x_b,dims=[2])
        x = x_p + x_b
        # x =  x_p
        x = x.transpose(1,2) # (B, T, N*D)
        x = x + res

        # bigru
        x_p, hidden = self.GRU_p(x)

        x_p = self.dropout1(x_p)
        x_re = torch.flip(x, dims=[1])
        x_b, hidden = self.GRU_b(x_re)
        x_b = torch.flip(x_b, dims=[1])
        x_b = self.dropout1(x_b)
        # x = x_p + x_b
        x = torch.cat([x_p, x_b], dim=2)
        # x = x_p
        x = self.linear(x)
        x = x + res
        x = x.reshape(B, T, N, D)
        return x



class ST_Block(nn.Module):
    def __init__(self, args, adj_data, f_nodes, s_nodes, feature, input_dim, hidden_dim, output_dim, dropout, Gs):
        super(ST_Block, self).__init__()

        self.graph = adj_data
        self.f_nodes = f_nodes
        self.s_nodes = s_nodes
        self.nodes = f_nodes + s_nodes
        self.k_neighs = args.knn_neighbor
        self.Gs = Gs
        # self.Gtaxi = Gtaxi
        # self.Gbike = Gbike
        self.winter = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(4)])
        self.wintra = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(2)])
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(6)])
        self.btcn = BTCN(self.nodes * feature, hidden_dim, groups=self.nodes * feature, dropout=dropout)
        self.sbtcn = BTCN(self.nodes, hidden_dim, groups=self.nodes, dropout=dropout)
        self.linear = nn.Linear(feature*2, feature)

        self.HGNN1 = HGNN_conv1(args)
        self.HGNN2 = HGNN_conv1(args)
        self.mlp = MLP(args.input_dim, args.hidden_dim, args.output_dim, args.dropout)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 定义两个可学习的权重参数
        self.weight_f = nn.Parameter(torch.tensor(0.5))
        self.weight_s = nn.Parameter(torch.tensor(0.5))

    def update_hypergraph(self, x):
        # try:
            edge_dict = hgu.construct_edge_list_from_knn(x, k_neighs=[self.k_neighs, self.k_neighs,self.k_neighs, self.k_neighs])
            H, W = hgu._edge_dict_to_H_with_weights(edge_dict)
            Gt = hgu._generate_G_from_H(H)
        # except Exception as e:
        #     print(f"Error during hypergraph update: {e}")
        #     return None

            Gt = torch.Tensor(Gt).to(self.device)
            H = torch.Tensor(H).to(self.device)
            edge_dict = torch.Tensor(edge_dict).to(self.device)
            return Gt, H, edge_dict, W

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
        taxi = x[:, :, :self.f_nodes, :]    #64,12,65,2
        bike = x[:, :, self.f_nodes:, :]    #64,12,65,2
        # 将上下车数据分开
        taxi_pick = taxi[:, :, :, 0]  #64,12,65
        taxi_drop = taxi[:, :, :, 1] #64,12,65

        bike_pick = bike[:, :, :, 0]  #64,12,65
        bike_drop = bike[:, :, :, 1]  #64,12,65

        #inter
        X1 = [taxi_pick, bike_pick]
        X2 = [taxi_drop, bike_drop]
        X3 = [taxi_pick, bike_drop]
        X4 = [taxi_drop, bike_pick]

        #intra
        X5 = [taxi_pick, taxi_drop]
        X6 = [bike_pick, bike_drop]
        X7 = [taxi_pick, taxi_drop, bike_pick, bike_drop]

        Xs = [X1, X2, X3, X4, X5, X6, X7]

        G = []
        E = []
        H = []

        for xs in Xs:
            Gt, Ht, Edict, w = self.update_hypergraph(xs)
            G.append(Gt)
            H.append(Ht)
            E.append(Edict)
            # 超图卷积并加权融合
        # h = Ht
        # np.set_printoptions(threshold=np.inf)
        # e = Edict
        # # np.save('hyperedge.npy', e)
        # g = Gt

        # np.set_printoptions(threshold=np.inf)
        # print(Edict)
        # inter
        X1 = torch.stack([taxi_pick, bike_pick], dim=-1)  # 堆叠在最后一个维度
        X2 = torch.stack([taxi_drop, bike_drop], dim=-1)
        X3 = torch.stack([taxi_pick, bike_drop], dim=-1)
        X4 = torch.stack([taxi_drop, bike_pick], dim=-1)

        # intra
        X5 = torch.stack([taxi_pick, taxi_drop], dim=-1)
        X6 = torch.stack([bike_pick, bike_drop], dim=-1)
        X7 = torch.cat((taxi, bike), dim=-1)
        Xs = [X1, X2, X3, X4, X5, X6]

        gnn_out = []
        for i in range(6):

            gnn_out.append(self.weights[i] * self.HGNN2(G[i], Xs[i]))

        x = sum(gnn_out)

        # F1 = x
        Xall = torch.relu(self.HGNN1(G[6], X7))
        # F2 = Xall
        x1, x2 = torch.split(Xall, 2, dim=-1)
        # np.save('chatujuanjiall.npy', Xall.cpu().detach().numpy())
        # gnn_out = []
        # for i in range(6):
        #
        #     gnn_out.append( self.HGNN2(G[i], Xs[i]))
        #
        # x = torch.cat(gnn_out)
        # leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # X_leaky = leaky_relu(x)
        # # 通过 Softmax 得到注意力系数
        # x = F.softmax(X_leaky, dim=2)
        # attention_weights_list = torch.chunk(x, 6, dim=0)  # (batch_size, feature_dim) for each
        # # 将每个特征矩阵与其对应的注意力系数逐元素相乘，然后相加
        # x = sum([X * w for X, w in zip(Xs, attention_weights_list)])
        # F1 = x
        # np.save('chatujuanji.npy', F1.cpu().detach().numpy())

        s_cr_f = torch.relu(self.HGNN2(self.Gs, taxi))
        s_cr_s = torch.relu(self.HGNN2(self.Gs, bike))
        # tf = self.sbtcn(taxi)
        # ts = self.sbtcn(bike)
        xs_f = x + s_cr_f + res[:, :, :self.f_nodes, :] + x1
        xs_s = x + s_cr_s + res[:, :, self.f_nodes:, :] + x2
        # xs_f = x + s_cr_f + res[:, :, :self.f_nodes, :] + share
        # xs_s = x + s_cr_s + res[:, :, self.f_nodes:, :] + share



        # combined_features = F.softmax(combined_features, dim=2)
        # xs_f = xs_f*combined_features
        # xs_s = xs_s*combined_features

        # gate = torch.relu(self.weight_f * xs_f + self.weight_s * xs_s)
        # xs_f = gate*xs_f + (1-gate)*xs_s
        # xs_s = gate*xs_s + (1-gate)*xs_f

        # xs_f = xs_f + combined_features + tf
        # xs_s = torch.relu(xs_s + combined_features) + ts
        # xs_f = xs_f + combined_features
        # xs_s = torch.relu(xs_s + combined_features)
        tf = self.sbtcn(xs_f)
        ts = self.sbtcn(xs_s)
        t2 = torch.cat([xs_f, xs_s], dim=2)
        t2 = self.btcn(t2)
        # t2 = self.btcn(res)
        F3 = t2
        tf = tf + t2[:, :, :self.f_nodes, :] + res[:, :, :self.f_nodes, :]
        ts = ts + t2[:, :, self.f_nodes:, :] + res[:, :, self.f_nodes:, :]

        t = torch.cat([tf, ts], dim=2)
        # xt = self.mlp(t2)
        xt = self.mlp(t)



        # return xt, F1, F2, F3
        # return xt, h, e, g, w
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

class HGNN_conv1(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, args ):
        super(HGNN_conv1, self).__init__()

        self.dim_in = args.input_dim
        self.dim_out = args.output_dim
        self.fc = MLP(args.input_dim, args.hidden_dim, args.output_dim, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        # self.G = G


    def forward(self, G, x):
        if x.dim() == 4:
            x = (self.fc(x))
            x = nn.ReLU()(G.matmul(x))
        elif x.dim() == 3:
            B, T, N = x.shape
            # 增加一个特征维度 D=1
            x = x.unsqueeze(-1)  # (B, T, N) -> (B, T, N, 1)
            x = (self.fc(x))
            x = nn.ReLU()(G.matmul(x))
            _, _, _, D = x.shape
            x = x.view(B, T, N * D)
        x = self.dropout(x)
        return x


class HGNN_conv2(nn.Module):
    """
    A HGNN layer
    """

    def __init__(self, args):
        super(HGNN_conv2, self).__init__()

        self.dim_in = args.input_dim
        self.dim_out = args.output_dim
        self.fc = MLP(args.input_dim, args.hidden_dim, args.output_dim, args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.batch_normalization1 = nn.BatchNorm2d(args.feature, momentum=0.1)
        self.batch_normalization2 = nn.BatchNorm2d(1, momentum=0.1)
        # 添加一个线性层来设置可学习的权重和偏置
        self.weight1 = nn.Parameter(torch.Tensor(args.feature, args.feature))
        self.bias1 = nn.Parameter(torch.Tensor(args.feature))
        self.weight2 = nn.Parameter(torch.Tensor(1, 1))
        self.bias2 = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))

        self.weight1.data.uniform_(-stdv1, stdv1)
        self.bias1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        self.bias2.data.uniform_(-stdv2, stdv2)

    def forward(self, G, x):
        if x.dim() == 4:
            x = (self.fc(x))
            x = x.permute(0, 3, 1, 2)
            # x = self.batch_normalization1(x)
            x = x.permute(0, 2, 3, 1)  # 恢复形状: (B, T, N, D)
            x = x.matmul(self.weight1)  # 应用权重
            x = G.matmul(x)
            x = x + self.bias1  # 应用偏置
        elif x.dim() == 3:
            B, T, N = x.shape
            # 增加一个特征维度 D=1
            x = x.unsqueeze(-1)  # (B, T, N) -> (B, T, N, 1)
            x = (self.fc(x))
            x = x.permute(0, 3, 1, 2)
            # x = self.batch_normalization2(x)
            x = x.permute(0, 2, 3, 1)  # 恢复形状: (B, T, N, D)
            x = x.matmul(self.weight2)  # 应用权重
            x = G.matmul(x)
            _, _, _, D = x.shape
            x = x.view(B, T, N * D)
            x = x + self.bias2 # 应用偏置
        x = self.dropout(x)
        return x


# class HGNN(nn.Module):
#     def __init__(self, args, momentum=0.1):
#         super(HGNN, self).__init__()
#         self.fc = MLP(args.input_dim, args.hidden_dim, args.output_dim, args.dropout)
#         self.hgc1 = HGNN_conv(args.input_dim, args.hidden_dim)
#         self.hgc2 = HGNN_conv(args.hidden_dim, args.output_dim)
#         self.batch_normalization1 = nn.BatchNorm2d(args.feature, momentum=momentum)
#         self.batch_normalization2 = nn.BatchNorm2d(1, momentum=momentum)
#         self.dropout = nn.Dropout(args.dropout)
#     def forward(self, G, x):
#         if x.dim() == 4:
#             B, T, N, D = x.shape
#             x = nn.ReLU()(self.fc(x))
#             x = x.permute(0, 3, 1, 2)
#             x = self.batch_normalization1(x)
#             x = x.permute(0, 2, 3, 1)  # 恢复形状: (B, T, N, D)
#             x = self.hgc1(x, G)
#             x = x.permute(0, 3, 2, 1)
#             x = nn.ReLU()(x)
#             x = self.dropout(x)
#             x = self.hgc2(x, G)
#             x = x.permute(0, 3, 2, 1)
#
#         elif x.dim() == 3:
#             B, T, N = x.shape
#             # 增加一个特征维度 D=1
#             x = x.unsqueeze(-1)  # (B, T, N) -> (B, T, N, 1)
#             x = nn.ReLU()(self.fc(x))
#             x = x.permute(0, 3, 1, 2)
#             x = self.batch_normalization2(x)
#             x = x.permute(0, 2, 3, 1)  # 恢复形状: (B, T, N, D)
#             x = self.hgc1(x, G)
#             x = x.permute(0, 3, 2, 1)
#             x = nn.ReLU()(x)
#             x = self.dropout(x)
#             x = self.hgc2(x, G)
#             _, D, _, _ = x.shape
#             x = x.view(B, T, N * D)
#         return x

class Network(nn.Module):
    def __init__(self, args, adj_data, f_nodes, s_nodes, blocks, feature, input_dim, hidden_dim, output_dim, dropout, G):
        super(Network, self).__init__()
        self.f_nodes = f_nodes
        self.s_nodes = s_nodes
        self.nodes = f_nodes + s_nodes
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
            st_blocks.append(ST_Block(args, adj_data, f_nodes, s_nodes, feature, input_dim, hidden_dim, output_dim, dropout, G))
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


        # ST_out = []
        # st_out = feature
        # for st in self.st:
        #     st_out = st(st_out)
        #     ST_out.append(st_out)
        #     st_out = st_out + res
        ST_out = []
        # F1_features = []  # 用于存储所有ST模块的F1特征
        # F2_features = []  # 用于存储所有ST模块的F2特征

        st_out = feature
        for st in self.st:
            # st_out, F1, F2, F3 = st(st_out)  # 调用ST模块，得到xt, F1, F2
            # st_out, h, e, g, w = st(st_out)
            st_out = st(st_out)
            ST_out.append(st_out)
            # F1_features.append(F1)  # 存储每个ST模块的F1特征
            # F2_features.append(F2)  # 存储每个ST模块的F2特征
            st_out = st_out + res
        # F1_features = F1
        # F2_features = F2
        # F3_features = F3

        ST_out.append(res)

        out = torch.cat(ST_out,dim=1) # (B, T', N, D)
        x = self.mlp(out)
        # return x, F1_features, F2_features, F3_features
        # return x, h, e, g, w
        # return x, e, w
        return x

