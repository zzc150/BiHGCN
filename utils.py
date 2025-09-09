import math
import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from model import hypergraph_utils as hgu
from sklearn.metrics.pairwise import cosine_distances as cos_dis, euclidean_distances
from sklearn.cluster import KMeans

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# plot train_val_loss
def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# 用于计算平均值
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def look_back(data, input_dim, output_dim):
    train = []
    test = []
    L = len(data)
    for i in range(L - input_dim - output_dim + 1):
        train_seq = data[i:i + input_dim, :, :]
        train_label = data[i + input_dim:i + input_dim + output_dim, :, :]
        train.append(train_seq)
        test.append(train_label)
    train = np.array(train)
    test = np.array(test)
    train = torch.FloatTensor(train)
    test = torch.FloatTensor(test)

    return train, test


def to_slice_with_time_embed(data, len, weekLen, dayLen):
    # 先处理形状，然后切片
    shape = data.shape

    # reshapedData = data.reshape(shape[0], shape[1] * shape[2], shape[3])
    sumLen = shape[0]

    dayEmbed = np.zeros((sumLen, dayLen))
    index = 0
    for i in range(sumLen):
        dayEmbed[i][index] = 1
        index = index + 1
        if index >= dayLen:
            index = 0

    weekEmbed = np.zeros((sumLen, weekLen))
    index = 0
    for i in range(sumLen):
        weekEmbed[i][index] = 1
        index = index + 1
        if index >= weekLen:
            index = 0


    xData = []
    yData = []
    dayEmbedData = []
    weekEmbedData = []

    for i in range(sumLen - len - len ):
        xData.append(data[i:i + len, :, :])
        dayEmbedData.append(dayEmbed[i:i + len, :])
        weekEmbedData.append(weekEmbed[i:i + len, :])
        yData.append(data[i + len:i + len + len, :, :])
    # start = 0
    # while start + len < shape[0]:
    #     xData.append(reshapedData[start:start + len, :, :])
    #     dayEmbedData.append(dayEmbed[start:start + len, :])
    #     weekEmbedData.append(weekEmbed[start:start + len, :])
    #     # yData.append(reshapedData[start + len,None,:,:])
    #     yData.append(reshapedData[start + 1: start + len + 1, :, :])
    #
    #     start = start + 1
    xData = np.array(xData)
    dayEmbedData = np.array(dayEmbedData)
    weekEmbedData = np.array(weekEmbedData)
    yData = np.array(yData)

    xData = [xData, weekEmbedData, dayEmbedData]

    return (xData, yData)

def split_by_percent(data, train_weight, val_weight, test_weight):
    data_len = data.shape[0]
    train_len = (int)(data_len * (train_weight / ((train_weight + val_weight + test_weight) * 1.0)))
    val_len = (int)(data_len * (val_weight / ((train_weight + val_weight + test_weight) * 1.0)))

    train_data = data[:train_len,:]
    val_data = data[train_len:train_len+val_len,:]
    test_data = data[train_len+val_len:,:]
    return train_data, val_data, test_data

def split_list_by_percent(data, train_weight, val_weight, test_weight):
    train_data = []
    val_data = []
    test_data = []
    for d in data:
        tmp_train_data, tmp_val_data, tmp_test_data = split_by_percent(d, train_weight, val_weight, test_weight)
        train_data.append(tmp_train_data)
        val_data.append(tmp_val_data)
        test_data.append(tmp_test_data)
    return train_data, val_data, test_data

class DataLoaderForMergeList(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
        self.size = len(xs[0])
        self.num_batch = math.ceil(self.size / self.batch_size)

        self.xs = xs
        self.ys = ys

    def shuffle(self):

        permutation = np.random.permutation(self.size)

        xs = []
        for x in self.xs:
            xs.append(x[permutation])
        ys = self.ys[permutation]

        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))

                x_i = []
                for x in self.xs:
                    x_i.append(x[start_ind: end_ind, ...])
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
def d_nyc(args, log):
    taxi_pick = pd.read_csv(r'F:/zzc/模型源代码/GSABT-master/GSABT-master/data/mydata/nyctaxi/taxi_pick.csv', index_col=None, header=None).values  # (T,N)=(4368,203)
    taxi_pick = taxi_pick.T
    taxi_drop = pd.read_csv(r'F:/zzc/模型源代码/GSABT-master/GSABT-master/data/mydata/nyctaxi/taxi_drop.csv', index_col=None, header=None).values  # (T,N)=(4368,203)
    taxi_drop = taxi_drop.T
    bike_pick = pd.read_csv(r'F:/zzc/模型源代码/GSABT-master/GSABT-master/data/mydata/nycbike/bike_pick.csv', index_col=None, header=None).values  # (T,N)=(4368,203)
    bike_pick =  bike_pick.T
    bike_drop = pd.read_csv(r'F:/zzc/模型源代码/GSABT-master/GSABT-master/data/mydata/nycbike/bike_drop.csv', index_col=None, header=None).values  # (T,N)=(4368,203)
    bike_drop = bike_drop.T
    matrix = pd.read_csv(r'data\nyc_matrix.csv', index_col=None, header=None).values
    H = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\taxi_zones_incidence_matrix1.npy')  # (203,32)
    Hbike = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\nycbike\bike_zones_incidence_matrix.npy')
    Htaxi = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\nyctaxi\taxi_zones_incidence_matrix.npy')

    # G = generate_G_from_H(H, variable_weight=False)    # (203,203)
    Gbike = generate_G_from_H(Hbike, variable_weight=False) # (151,151)
    Gtaxi = generate_G_from_H(Htaxi, variable_weight=False)  # (261,261)
    G = np.block([[Gtaxi, np.zeros((261, 151))],
                  [np.zeros((151, 261)), Gbike]])

    f_nodes, s_nodes = len(taxi_pick[0]), len(bike_pick[0])
    nodes = f_nodes + s_nodes   #406

    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 203, 2
    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 203, 2


    f_mean, f_std = taxi.mean(), taxi.std()
    taxi = (taxi - f_mean) / f_std

    s_mean, s_std = bike.mean(), bike.std()
    bike = (bike - s_mean) / s_std

    print('taxi.shape:', taxi.shape)
    print('bike.shape:', bike.shape)
    flow = np.concatenate((taxi, bike), axis=1)  # 4368, 406, 2

    print('flow.shape:', flow.shape)

    train_rate, val_rate, test_rate = int(args.train_rate), int(args.val_rate), int(args.val_rate)

    volume_dataX, volume_dataY = to_slice_with_time_embed(flow, args.input_dim, args.week_len, args.day_len)
    trainX, validX, testX = split_list_by_percent(volume_dataX, train_rate, val_rate, test_rate)

    trainY, validY, testY = split_by_percent(volume_dataY, train_rate, val_rate, test_rate)

    data = {}

    data['x_train'] = trainX
    data['y_train'] = trainY

    data['x_val'] = validX
    data['y_val'] = validY

    data['x_test'] = testX
    data['y_test'] = testY

    train_loader = DataLoaderForMergeList(data['x_train'], data['y_train'], args.batch_size)
    valid_loader = DataLoaderForMergeList(data['x_val'], data['y_val'], args.batch_size)
    test_loader = DataLoaderForMergeList(data['x_test'], data['y_test'], args.batch_size)


    trainx, validx, testx = trainX[0], validX[0], testX[0]
    week, validx, testx = trainX[0], validX[0], testX[0]

    log_string(log, f'trainX: {trainx.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validx.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testx.shape}\t\ttestY:   {testY.shape}')


    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, nodes, f_nodes, s_nodes, matrix, log, G, Gtaxi, Gbike




def manhattan(args, log):
    taxi_pick = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\taxi\taxi_pick.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_pick = taxi_pick.T
    taxi_drop = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\taxi\taxi_drop.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_drop = taxi_drop.T
    bike_pick = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\bike\bike_pick.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    bike_pick = bike_pick.T
    bike_drop = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\bike\bike_drop.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    bike_drop = bike_drop.T
    matrix = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\bike\bike_drop.csv', index_col=None, header=None).values
    H = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\incidence_matrix.npy')  # (65,12)

    # G = generate_G_from_H(H, variable_weight=False) # (65,65)
    # with open(r"E:\python\Graph-WaveNet-master\data\NYCTaxi\adj_NYCTaxi.pkl", "rb") as f:
    #     G = pickle.load(f)
    H = np.load(r'F:\zzc\实验\MHGNN\MHGNN\data\mydata\manhadun\incidence_matrix.npy')
    G = hgu._generate_G_from_H(H)
    f_nodes, s_nodes = len(taxi_pick[0]), len(bike_pick[0])
    nodes = f_nodes + s_nodes   #130

    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 203, 2
    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 203, 2


    f_mean, f_std = taxi.mean(), taxi.std()
    taxi = (taxi - f_mean) / f_std

    s_mean, s_std = bike.mean(), bike.std()
    bike = (bike - s_mean) / s_std

    print('taxi.shape:', taxi.shape)
    print('bike.shape:', bike.shape)
    flow = np.concatenate((taxi, bike), axis=1)  # 4368, 406, 2

    print('flow.shape:', flow.shape)

    train_rate, val_rate, test_rate = int(args.train_rate), int(args.val_rate), int(args.val_rate)

    volume_dataX, volume_dataY = to_slice_with_time_embed(flow, args.input_dim, args.week_len, args.day_len)
    trainX, validX, testX = split_list_by_percent(volume_dataX, train_rate, val_rate, test_rate)

    trainY, validY, testY = split_by_percent(volume_dataY, train_rate, val_rate, test_rate)

    data = {}

    data['x_train'] = trainX
    data['y_train'] = trainY

    data['x_val'] = validX
    data['y_val'] = validY

    data['x_test'] = testX
    data['y_test'] = testY
    # BothFlowScaler = StandardScaler(mean=data['x_train'][0].mean(), std=data['x_train'][0].std(),
    #                                 fill_zeroes=fill_zeroes)
    #
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][0] = BothFlowScaler.transform(data['x_' + category][0])

    train_loader = DataLoaderForMergeList(data['x_train'], data['y_train'], args.batch_size)
    valid_loader = DataLoaderForMergeList(data['x_val'], data['y_val'], args.batch_size)
    test_loader = DataLoaderForMergeList(data['x_test'], data['y_test'], args.batch_size)


    trainx, validx, testx = trainX[0], validX[0], testX[0]
    week, validx, testx = trainX[0], validX[0], testX[0]

    log_string(log, f'trainX: {trainx.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validx.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testx.shape}\t\ttestY:   {testY.shape}')



    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, nodes, f_nodes, s_nodes, matrix, log, G

def chicago(args, log):
    taxi_pick = pd.read_csv(r'F:\zzc\实验\MHGNN\MHGNN\data\mydata\chicago\taxi\Chitaxipick.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_pick = taxi_pick.T
    taxi_drop = pd.read_csv(r'F:\zzc\实验\MHGNN\MHGNN\data\mydata\chicago\taxi\Chitaxidrop.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_drop = taxi_drop.T
    bike_pick = pd.read_csv(r'F:\zzc\实验\MHGNN\MHGNN\data\mydata\chicago\bike\Chibikepick.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    bike_pick = bike_pick.T
    bike_drop = pd.read_csv(r'F:\zzc\实验\MHGNN\MHGNN\data\mydata\chicago\bike\Chibikedrop.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    bike_drop = bike_drop.T
    # matrix = pd.read_csv(r'data\nyc_matrix.csv', index_col=None, header=None).values

    # G = generate_G_from_H(H, variable_weight=False) # (65,65)
    with open(r"F:\zzc\实验\MHGNN\MHGNN\data\mydata\chicago\adj_Chicago.pkl", "rb") as f:
        G = pickle.load(f)
    matrix = G
    f_nodes, s_nodes = len(taxi_pick[0]), len(bike_pick[0])
    nodes = f_nodes + s_nodes   #130

    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 203, 2
    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 203, 2


    f_mean, f_std = taxi.mean(), taxi.std()
    taxi = (taxi - f_mean) / f_std

    s_mean, s_std = bike.mean(), bike.std()
    bike = (bike - s_mean) / s_std

    print('taxi.shape:', taxi.shape)
    print('bike.shape:', bike.shape)
    flow = np.concatenate((taxi, bike), axis=1)  # 4368, 406, 2

    print('flow.shape:', flow.shape)

    train_rate, val_rate, test_rate = int(args.train_rate), int(args.val_rate), int(args.val_rate)

    volume_dataX, volume_dataY = to_slice_with_time_embed(flow, args.input_dim, args.week_len, args.day_len)
    trainX, validX, testX = split_list_by_percent(volume_dataX, train_rate, val_rate, test_rate)

    trainY, validY, testY = split_by_percent(volume_dataY, train_rate, val_rate, test_rate)

    data = {}

    data['x_train'] = trainX
    data['y_train'] = trainY

    data['x_val'] = validX
    data['y_val'] = validY

    data['x_test'] = testX
    data['y_test'] = testY
    # BothFlowScaler = StandardScaler(mean=data['x_train'][0].mean(), std=data['x_train'][0].std(),
    #                                 fill_zeroes=fill_zeroes)
    #
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][0] = BothFlowScaler.transform(data['x_' + category][0])

    train_loader = DataLoaderForMergeList(data['x_train'], data['y_train'], args.batch_size)
    valid_loader = DataLoaderForMergeList(data['x_val'], data['y_val'], args.batch_size)
    test_loader = DataLoaderForMergeList(data['x_test'], data['y_test'], args.batch_size)


    trainx, validx, testx = trainX[0], validX[0], testX[0]
    week, validx, testx = trainX[0], validX[0], testX[0]

    log_string(log, f'trainX: {trainx.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validx.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testx.shape}\t\ttestY:   {testY.shape}')



    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, nodes, f_nodes, s_nodes, matrix, log, G

def bierende(args, log):
    # taxi_pick = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\taxi\taxi_pick.csv', index_col=None, header=None).values  # (T,N)=(4368,65)
    # taxi_pick = taxi_pick.T
    # taxi_drop = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\taxi\taxi_drop.csv', index_col=None, header=None).values  # (T,N)=(4368,65)
    # taxi_drop = taxi_drop.T
    # bike_pick = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\bike\bike_pick.csv', index_col=None, header=None).values  # (T,N)=(4368,65)
    # bike_pick =  bike_pick.T
    # bike_drop = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\bike\bike_drop.csv', index_col=None, header=None).values  # (T,N)=(4368,65)
    # bike_drop = bike_drop.T
    taxi_pick = pd.read_csv(r'data\nyctaxi\taxi_pick.csv', index_col=None, header=None).values  # L, N
    taxi_drop = pd.read_csv(r'data\nyctaxi\taxi_drop.csv', index_col=None, header=None).values
    bike_pick = pd.read_csv(r'data\nycbike\bike_pick.csv', index_col=None, header=None).values
    bike_drop = pd.read_csv(r'data\nycbike\bike_drop.csv', index_col=None, header=None).values

    matrix = pd.read_csv(r'data\nyc_matrix.csv', index_col=None, header=None).values


    G = matrix    # (65,65)

    # G = np.block([[G, np.zeros((65, 65))],
    #               [np.zeros((65, 65))]])

    f_nodes, s_nodes = len(taxi_pick[0]), len(bike_pick[0])
    nodes = f_nodes + s_nodes   #130

    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 203, 2
    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 203, 2


    f_mean, f_std = taxi.mean(), taxi.std()
    taxi = (taxi - f_mean) / f_std

    s_mean, s_std = bike.mean(), bike.std()
    bike = (bike - s_mean) / s_std

    print('taxi.shape:', taxi.shape)
    print('bike.shape:', bike.shape)
    flow = np.concatenate((taxi, bike), axis=1)  # 4368, 406, 2

    print('flow.shape:', flow.shape)

    train_rate, val_rate, test_rate = int(args.train_rate), int(args.val_rate), int(args.val_rate)

    volume_dataX, volume_dataY = to_slice_with_time_embed(flow, args.input_dim, args.week_len, args.day_len)
    trainX, validX, testX = split_list_by_percent(volume_dataX, train_rate, val_rate, test_rate)

    trainY, validY, testY = split_by_percent(volume_dataY, train_rate, val_rate, test_rate)

    data = {}

    data['x_train'] = trainX
    data['y_train'] = trainY

    data['x_val'] = validX
    data['y_val'] = validY

    data['x_test'] = testX
    data['y_test'] = testY
    # BothFlowScaler = StandardScaler(mean=data['x_train'][0].mean(), std=data['x_train'][0].std(),
    #                                 fill_zeroes=fill_zeroes)
    #
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][0] = BothFlowScaler.transform(data['x_' + category][0])

    train_loader = DataLoaderForMergeList(data['x_train'], data['y_train'], args.batch_size)
    valid_loader = DataLoaderForMergeList(data['x_val'], data['y_val'], args.batch_size)
    test_loader = DataLoaderForMergeList(data['x_test'], data['y_test'], args.batch_size)


    trainx, validx, testx = trainX[0], validX[0], testX[0]
    week, validx, testx = trainX[0], validX[0], testX[0]

    log_string(log, f'trainX: {trainx.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validx.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testx.shape}\t\ttestY:   {testY.shape}')

    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, nodes, f_nodes, s_nodes, matrix, log, G
def d_nyc151(args, log):
    taxi_pick = pd.read_csv(r'F:/zzc/模型源代码/GSABT-master/GSABT-master/data/mydata/nyctaxi/taxi_pick.csv', index_col=None, header=None).values  # (T,N)=(4368,203)
    taxi_pick = taxi_pick.T
    taxi_drop = pd.read_csv(r'F:/zzc/模型源代码/GSABT-master/GSABT-master/data/mydata/nyctaxi/taxi_drop.csv', index_col=None, header=None).values  # (T,N)=(4368,203)
    taxi_drop = taxi_drop.T
    bike_pick = pd.read_csv(r'F:/zzc/模型源代码/GSABT-master/GSABT-master/data/mydata/nycbike/bike_pick.csv', index_col=None, header=None).values  # (T,N)=(4368,203)
    bike_pick =  bike_pick.T
    bike_drop = pd.read_csv(r'F:/zzc/模型源代码/GSABT-master/GSABT-master/data/mydata/nycbike/bike_drop.csv', index_col=None, header=None).values  # (T,N)=(4368,203)
    bike_drop = bike_drop.T
    matrix = pd.read_csv(r'data\nyc_matrix.csv', index_col=None, header=None).values
    H = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\taxi_zones_incidence_matrix1.npy')  # (203,32)
    Hbike = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\nycbike\bike_zones_incidence_matrix.npy')
    Htaxi = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\nyctaxi\taxi_zones_incidence_matrix.npy')

    # G = generate_G_from_H(H, variable_weight=False)    # (203,203)
    Gbike = generate_G_from_H(Hbike, variable_weight=False) # (151,151)
    Gtaxi = generate_G_from_H(Htaxi, variable_weight=False)  # (261,261)
    G = np.block([[Gbike, np.zeros((151, 261))],
                  [np.zeros((261, 151)), Gtaxi]])

    f_nodes, s_nodes = len(taxi_pick[0]), len(bike_pick[0])
    nodes = f_nodes + s_nodes   #406

    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 203, 2
    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 203, 2


    f_mean, f_std = taxi.mean(), taxi.std()
    taxi = (taxi - f_mean) / f_std

    s_mean, s_std = bike.mean(), bike.std()
    bike = (bike - s_mean) / s_std

    print('taxi.shape:', taxi.shape)
    print('bike.shape:', bike.shape)
    flow = np.concatenate((taxi, bike), axis=1)  # 4368, 406, 2

    print('flow.shape:', flow.shape)

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    idx_train = list(range(len(train)))
    idx_val = list(range(len(train), len(train) + 649))
    idx_test = list(range(len(train) + 649, len(train) + 649 + 649))
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_en = 0

    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, nodes, f_nodes, s_nodes, matrix, log, G, Gtaxi, Gbike


def manhattan_bike(args, log):
    bike_pick = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\bike\bike_pick.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    bike_pick = bike_pick.T
    bike_drop = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\bike\bike_drop.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    bike_drop = bike_drop.T

    matrix = 0
    H = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\incidence_matrix.npy')  # (65,12)

    G = generate_G_from_H(H, variable_weight=False) # (65,65)
    # with open(r"E:\python\Graph-WaveNet-master\data\NYCTaxi\adj_NYCTaxi.pkl", "rb") as f:
    #     G = pickle.load(f)

    f_nodes = len(bike_pick[0])
    nodes = f_nodes  # 130

    length = len(bike_pick)

    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 203, 2

    f_mean, f_std = bike.mean(), bike.std()
    bike = (bike - f_mean) / f_std

    print('taxi.shape:', bike.shape)

    flow = bike

    print('flow.shape:', flow.shape)

    train_rate, val_rate, test_rate = int(args.train_rate), int(args.val_rate), int(args.val_rate)

    volume_dataX, volume_dataY = to_slice_with_time_embed(flow, args.input_dim, args.week_len, args.day_len)
    trainX, validX, testX = split_list_by_percent(volume_dataX, train_rate, val_rate, test_rate)

    trainY, validY, testY = split_by_percent(volume_dataY, train_rate, val_rate, test_rate)

    data = {}

    data['x_train'] = trainX
    data['y_train'] = trainY

    data['x_val'] = validX
    data['y_val'] = validY

    data['x_test'] = testX
    data['y_test'] = testY

    train_loader = DataLoaderForMergeList(data['x_train'], data['y_train'], args.batch_size)
    valid_loader = DataLoaderForMergeList(data['x_val'], data['y_val'], args.batch_size)
    test_loader = DataLoaderForMergeList(data['x_test'], data['y_test'], args.batch_size)

    trainx, validx, testx = trainX[0], validX[0], testX[0]
    week, validx, testx = trainX[0], validX[0], testX[0]

    log_string(log, f'trainX: {trainx.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validx.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testx.shape}\t\ttestY:   {testY.shape}')

    return train_loader, valid_loader, test_loader, f_mean, f_std, nodes, G, log

def manhattan_taxi(args, log):
    taxi_pick = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\taxi\taxi_pick.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_pick = taxi_pick.T
    taxi_drop = pd.read_csv(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\taxi\taxi_drop.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_drop = taxi_drop.T

    matrix = 0
    H = np.load(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\mydata\manhadun\incidence_matrix.npy')  # (65,12)

    # G = generate_G_from_H(H, variable_weight=False) # (65,65)
    with open(r"E:\python\Graph-WaveNet-master\data\NYCTaxi\adj_NYCTaxi.pkl", "rb") as f:
        G = pickle.load(f)

    f_nodes = len(taxi_pick[0])
    nodes = f_nodes  # 130

    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 203, 2

    f_mean, f_std = taxi.mean(), taxi.std()
    taxi = (taxi - f_mean) / f_std

    print('taxi.shape:', taxi.shape)

    flow = taxi

    print('flow.shape:', flow.shape)

    train_rate, val_rate, test_rate = int(args.train_rate), int(args.val_rate), int(args.val_rate)

    volume_dataX, volume_dataY = to_slice_with_time_embed(flow, args.input_dim, args.week_len, args.day_len)
    trainX, validX, testX = split_list_by_percent(volume_dataX, train_rate, val_rate, test_rate)

    trainY, validY, testY = split_by_percent(volume_dataY, train_rate, val_rate, test_rate)

    data = {}

    data['x_train'] = trainX
    data['y_train'] = trainY

    data['x_val'] = validX
    data['y_val'] = validY

    data['x_test'] = testX
    data['y_test'] = testY

    train_loader = DataLoaderForMergeList(data['x_train'], data['y_train'], args.batch_size)
    valid_loader = DataLoaderForMergeList(data['x_val'], data['y_val'], args.batch_size)
    test_loader = DataLoaderForMergeList(data['x_test'], data['y_test'], args.batch_size)

    trainx, validx, testx = trainX[0], validX[0], testX[0]
    week, validx, testx = trainX[0], validX[0], testX[0]

    log_string(log, f'trainX: {trainx.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validx.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testx.shape}\t\ttestY:   {testY.shape}')

    return train_loader, valid_loader, test_loader, f_mean, f_std, nodes, G, log

def chicago_taxi(args, log):
    taxi_pick = pd.read_csv(r'F:\zzc\实验\GSABT-master\data\mydata\chicago\taxi\Chitaxipick.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_pick = taxi_pick.T
    taxi_drop = pd.read_csv(r'F:\zzc\实验\GSABT-master\data\mydata\chicago\taxi\Chitaxidrop.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_drop = taxi_drop.T

    matrix = 0

    with open(r"F:\zzc\实验\GSABT-master\data\mydata\chicago\adj_Chicago.pkl", "rb") as f:
        G = pickle.load(f)

    f_nodes = len(taxi_pick[0])
    nodes = f_nodes  # 130

    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 203, 2

    f_mean, f_std = taxi.mean(), taxi.std()
    taxi = (taxi - f_mean) / f_std

    print('taxi.shape:', taxi.shape)

    flow = taxi

    print('flow.shape:', flow.shape)

    train_rate, val_rate, test_rate = int(args.train_rate), int(args.val_rate), int(args.val_rate)

    volume_dataX, volume_dataY = to_slice_with_time_embed(flow, args.input_dim, args.week_len, args.day_len)
    trainX, validX, testX = split_list_by_percent(volume_dataX, train_rate, val_rate, test_rate)

    trainY, validY, testY = split_by_percent(volume_dataY, train_rate, val_rate, test_rate)

    data = {}

    data['x_train'] = trainX
    data['y_train'] = trainY

    data['x_val'] = validX
    data['y_val'] = validY

    data['x_test'] = testX
    data['y_test'] = testY

    train_loader = DataLoaderForMergeList(data['x_train'], data['y_train'], args.batch_size)
    valid_loader = DataLoaderForMergeList(data['x_val'], data['y_val'], args.batch_size)
    test_loader = DataLoaderForMergeList(data['x_test'], data['y_test'], args.batch_size)

    trainx, validx, testx = trainX[0], validX[0], testX[0]
    week, validx, testx = trainX[0], validX[0], testX[0]

    log_string(log, f'trainX: {trainx.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validx.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testx.shape}\t\ttestY:   {testY.shape}')

    return train_loader, valid_loader, test_loader, f_mean, f_std, nodes, G, log


def chicago_bike(args, log):
    taxi_pick = pd.read_csv(r'F:\zzc\实验\GSABT-master\data\mydata\chicago\bike\Chibikepick.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_pick = taxi_pick.T
    taxi_drop = pd.read_csv(r'F:\zzc\实验\GSABT-master\data\mydata\chicago\bike\Chibikedrop.csv',
                            index_col=None, header=None).values  # (T,N)=(4368,65)
    taxi_drop = taxi_drop.T

    matrix = 0

    with open(r"F:\zzc\实验\GSABT-master\data\mydata\chicago\adj_Chicago.pkl", "rb") as f:
        G = pickle.load(f)

    f_nodes = len(taxi_pick[0])
    nodes = f_nodes  # 130

    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 203, 2

    f_mean, f_std = taxi.mean(), taxi.std()
    taxi = (taxi - f_mean) / f_std

    print('taxi.shape:', taxi.shape)

    flow = taxi

    print('flow.shape:', flow.shape)

    train_rate, val_rate, test_rate = int(args.train_rate), int(args.val_rate), int(args.val_rate)

    volume_dataX, volume_dataY = to_slice_with_time_embed(flow, args.input_dim, args.week_len, args.day_len)
    trainX, validX, testX = split_list_by_percent(volume_dataX, train_rate, val_rate, test_rate)

    trainY, validY, testY = split_by_percent(volume_dataY, train_rate, val_rate, test_rate)

    data = {}

    data['x_train'] = trainX
    data['y_train'] = trainY

    data['x_val'] = validX
    data['y_val'] = validY

    data['x_test'] = testX
    data['y_test'] = testY

    train_loader = DataLoaderForMergeList(data['x_train'], data['y_train'], args.batch_size)
    valid_loader = DataLoaderForMergeList(data['x_val'], data['y_val'], args.batch_size)
    test_loader = DataLoaderForMergeList(data['x_test'], data['y_test'], args.batch_size)

    trainx, validx, testx = trainX[0], validX[0], testX[0]
    week, validx, testx = trainX[0], validX[0], testX[0]

    log_string(log, f'trainX: {trainx.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validx.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testx.shape}\t\ttestY:   {testY.shape}')

    return train_loader, valid_loader, test_loader, f_mean, f_std, nodes, G, log
def s_bike(args, log):

    bike_pick = pd.read_csv(r'data\nycbike\bike_pick.csv', index_col=None, header=None).values
    bike_drop = pd.read_csv(r'data\nycbike\bike_drop.csv', index_col=None, header=None).values  # L, N
    matrix = pd.read_csv(r'data\nycbike\dis_bb.csv', index_col=None, header=None).values

    nodes = len(bike_pick[0])
    flow = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 256, 2

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]


    mean, std = train.mean(), train.std()
    train, valid, test = (train -mean)/std, (valid -mean)/std, (test -mean)/std

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, mean, std, nodes, matrix, log

def evalution(out, tgt):

    out = out.reshape(-1)
    tgt = tgt.reshape(-1)
    mae = mean_absolute_error(tgt,out)
    mse = mean_squared_error(tgt,out)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(tgt,out)[0][1]
    mape = masked_mape(out, tgt, 0)

    return mae, rmse, pcc, mape



def masked_mape(preds, labels, null_val=np.nan):
    # 创建掩码，忽略 null_val 或 NaN 的位置
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)

    # 计算 MAPE，避免除以零的问题
    with np.errstate(divide='ignore', invalid='ignore'):
        loss = np.abs(preds - labels) / np.abs(labels)
        loss = np.where(np.isfinite(loss), loss, 0)  # 将非有限值替换为0
        loss = loss * mask

    return np.mean(loss)


# metric
def metrics(pred, label):
    mae = mae_(pred, label)
    mse = mse_(pred, label)
    rmse = rmse_(pred, label)
    pcc = pcc_(pred, label)
    return mae, rmse, pcc

def mae_(pred, label):
    loss = torch.abs(pred - label).type(torch.float32)
    return loss.mean()

def mse_(pred, label):
    loss = (pred-label).type(torch.float32)**2
    return loss.mean()

def rmse_(pred, label):
    loss = torch.sqrt(mse_(pred, label))
    return loss

def pcc_(pred, label):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    pcc = torch.corrcoef(label, pred)
    return pcc


#-----------------------------hypergraph utils
def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

# # come from dualhgcn
# def generate_G_from_H(args, H):
# 	H = np.array(H)
# 	n_edge = H.shape[1]
# 	W = np.ones(n_edge)
# 	DV = np.sum(H * W, axis=1)
# 	DE = np.sum(H, axis=0)
# 	DV += 1e-12
# 	DE += 1e-12
# 	invDE = np.mat(np.diag(np.power(DE, -1)))
# 	W = np.mat(np.diag(W))
# 	H = np.mat(H)
# 	HT = H.T
# 	if args.conv == "sym":
# 		DV2 = np.mat(np.diag(np.power(DV, -0.5)))
# 		G = DV2 * H * W * invDE * HT * DV2   #sym
# 	elif args.conv == "asym":
# 		DV1 = np.mat(np.diag(np.power(DV, -1)))
# 		G = DV1 * H * W * invDE * HT   #asym
# 	return G
#
# def generate_Gs_from_Hs(args, Hs):
# 	Gs = dict()
# 	for key,val in Hs.items():
# 		Gs[key] = generate_G_from_H(args, val)
# 	return Gs
#
# # come from DHGNN
# def sample_ids(ids, k):
#     """
#     sample `k` indexes from ids, must sample the centroid node itself
#     :param ids: indexes sampled from
#     :param k: number of samples
#     :return: sampled indexes
#     """
#     df = pd.DataFrame(ids)
#     sampled_ids = df.sample(k - 1, replace=True).values
#     sampled_ids = sampled_ids.flatten().tolist()
#     sampled_ids.append(ids[-1])  # must sample the centroid node itself
#     return sampled_ids
# def _construct_edge_list_from_cluster(X, clusters, adjacent_clusters, k_neighbors) -> np.array:
#     """
#     construct edge list (numpy array) from cluster for single modality
#     :param X: feature
#     :param clusters: number of clusters for k-means
#     :param adjacent_clusters: a node's adjacent clusters
#     :param k_neighbors: number of a node's neighbors
#     :return:
#     """
#     N = X.shape[0]
#     kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
#     centers = kmeans.cluster_centers_
#     dis = euclidean_distances(X, centers)
#     _, cluster_center_dict = torch.topk(torch.Tensor(dis), adjacent_clusters, largest=False)
#     cluster_center_dict = cluster_center_dict.numpy()
#     point_labels = kmeans.labels_
#     point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(clusters)]
#
#     def _list_cat(list_of_array):
#         """
#         example: [[0,1],[3,5,6],[-1]] -> [0,1,3,5,6,-1]
#         :param list_of_array: list of np.array
#         :return: list of numbers
#         """
#         ret = list()
#         for array in list_of_array:
#             ret += array.tolist()
#         return ret
#
#     cluster_neighbor_dict = [_list_cat([point_in_which_cluster[cluster_center_dict[point][i]]
#                                         for i in range(adjacent_clusters)]) for point in range(N)]
#     for point, entry in enumerate(cluster_neighbor_dict):
#         entry.append(point)
#     sampled_ids = [sample_ids(cluster_neighbor_dict[point], k_neighbors) for point in range(N)]
#     return np.array(sampled_ids)
#
#
# def construct_edge_list_from_cluster(Xs, clusters, adjacent_clusters, k_neighbors) -> np.array:
#     """
#     construct concatenated edge list from list of features with cluster from multi-modal
#     :param Xs: list of features of each modality
#     :param clusters: list of number of clusters for k-means of each modality
#     :param adjacent_clusters: list of number of a node's adjacent clusters of each modality
#     :param k_neighbors: list of number of a node's neighbors
#     :return: concatenated edge list (numpy array)
#     """
#     return np.concatenate([_construct_edge_list_from_cluster(Xs[i], clusters[i], adjacent_clusters[i], k_neighbors[i])
#                            for i in range(len(Xs))], axis=1)