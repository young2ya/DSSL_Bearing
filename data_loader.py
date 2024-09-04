import os
import random
import numpy as np

import torch
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

def NPZ_Loader(folder_path, each_num):
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".npz")]
    file_names = random.sample(file_names, each_num)

    npy_list = []
    label_list = []
    for file_name in file_names:
        npz = np.load(file_name)
        X = npz['x']
        y = npz['y']

        npy_list.append(X)
        label_list.append(y)
        npy_arr = np.vstack(npy_list)
        label_arr = np.hstack(label_list)

    return npy_arr, label_arr

def Load(folder_path_list, each_data_num):
    arr_dict = dict()
    label_dict = dict()
    for folder_path in folder_path_list:
        bearing_type = folder_path.split("\\")[-1]
        print(f"loading {bearing_type}..")
        npy_arr, label_arr = NPZ_Loader(folder_path=folder_path, each_num=each_data_num)

        arr_dict[bearing_type] = npy_arr
        label_dict[bearing_type] = label_arr

    return arr_dict, label_dict

def Slice(args, data_dict, label_dict):
    train_data, test_data, val_data, unlabel_data, train_label, test_label, val_label, unlabel_label = [{} for i in range(8)]

    for key, value in zip(data_dict.keys(), data_dict.values()):
        train_data[key] = value[:args.num_labeled]
        test_data[key] = value[args.num_labeled:args.num_labeled+500]
        val_data[key] = value[args.num_labeled+500:args.num_labeled+1000]
        unlabel_data[key] = value[args.num_labeled+1000:args.num_labeled+2000]

    for key, value in zip(label_dict.keys(), label_dict.values()):
        train_label[key] = value[:args.num_labeled]
        test_label[key] = value[args.num_labeled:args.num_labeled+500]
        val_label[key] = value[args.num_labeled+500:args.num_labeled+1000]
        unlabel_label[key] = value[args.num_labeled+1000:args.num_labeled+2000]

    return train_data, test_data, val_data, unlabel_data, train_label, test_label, val_label, unlabel_label

def Concat_data(X_dict, y_dict):
    X_concat = np.concatenate(list(X_dict.values()), axis=0)
    y_concat = np.concatenate(list(y_dict.values()), axis=0)

    X_concat = X_concat.reshape(X_concat.shape + (1,))

    return X_concat, y_concat

class MyDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = torch.FloatTensor(x_data).permute(0,3,2,1)
        self.y_data = torch.LongTensor(y_data)
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        signal, target = self.x_data[index], self.y_data[index]
        if self.transform is not None:
            signal = self.transform(signal)
        return signal, target

    def __len__(self):
        return self.len

class MyDataset_MixMatch(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = torch.FloatTensor(x_data).permute(0,3,2,1)
        self.y_data = torch.LongTensor(y_data)
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        signal_1, signal_2, target = self.x_data[index], self.x_data[index], self.y_data[index]

        if self.transform is not None:
            segnal_1 = self.transform(signal_1)
            signal_2 = self.transform(signal_2)

        signal = signal_1, signal_2
        return signal, target

    def __len__(self):
        return self.len

class MyDataset_FixMatch(Dataset):
    def __init__(self, x_data, y_data, transform_w=None, transform_s=None):
        self.x_data = torch.FloatTensor(x_data).permute(0,3,2,1)
        self.y_data = torch.LongTensor(y_data)
        self.transform_w = transform_w
        self.transform_s = transform_s
        self.len = len(y_data)

    def __getitem__(self, index):
        signal_1, signal_2, target = self.x_data[index], self.x_data[index], self.y_data[index]

        signal_1 = self.transform_w(signal_1)
        signal_2 = self.transform_s(signal_2)

        signal = signal_1, signal_2
        return signal, target

    def __len__(self):
        return self.len

def Data_Loader(args, path_list):
    total_data, total_label = Load(path_list, 5)

    shuffle_data = dict()
    for key, value in zip(total_data.keys(), total_data.values()):
        np.random.shuffle(value)
        shuffle_data[key] = value

    train_data, test_data, val_data, unlabel_data, train_label, test_label, val_label, unlabel_label = Slice(args, shuffle_data, total_label)

    X_train, y_train = Concat_data(train_data, train_label)
    X_test, y_test = Concat_data(test_data, test_label)
    X_val, y_val = Concat_data(val_data, val_label)
    X_unlabel, y_unlabel = Concat_data(unlabel_data, unlabel_label)

    trainset = MyDataset(X_train, y_train)
    testset = MyDataset(X_test, y_test)
    valset = MyDataset(X_val, y_val)
    unlabelset = MyDataset(X_unlabel, y_unlabel)

    return trainset, testset, valset, unlabelset

def Data_Loader_MixMatch(args, path_list, train_transform = None):
    total_data, total_label = Load(path_list, 5)

    shuffle_data = dict()
    for key, value in zip(total_data.keys(), total_data.values()):
        np.random.shuffle(value)
        shuffle_data[key] = value

    train_data, test_data, val_data, unlabel_data, train_label, test_label, val_label, unlabel_label = Slice(args, shuffle_data, total_label)

    X_train, y_train = Concat_data(train_data, train_label)
    X_test, y_test = Concat_data(test_data, test_label)
    X_val, y_val = Concat_data(val_data, val_label)
    X_unlabel, y_unlabel = Concat_data(unlabel_data, unlabel_label)

    trainset = MyDataset(X_train, y_train, transform=train_transform)
    testset = MyDataset(X_test, y_test)
    valset = MyDataset(X_val, y_val)
    unlabelset = MyDataset_MixMatch(X_unlabel, y_unlabel, transform=train_transform)

    return trainset, testset, valset, unlabelset

def Data_Loader_FixMatch(args, path_list, transform_w = None, transform_s = None):
    total_data, total_label = Load(path_list, 5)

    shuffle_data = dict()
    for key, value in zip(total_data.keys(), total_data.values()):
        np.random.shuffle(value)
        shuffle_data[key] = value

    train_data, test_data, val_data, unlabel_data, train_label, test_label, val_label, unlabel_label = Slice(args, shuffle_data, total_label)

    X_train, y_train = Concat_data(train_data, train_label)
    X_test, y_test = Concat_data(test_data, test_label)
    X_val, y_val = Concat_data(val_data, val_label)
    X_unlabel, y_unlabel = Concat_data(unlabel_data, unlabel_label)

    trainset = MyDataset(X_train, y_train, transform=transform_w)
    testset = MyDataset(X_test, y_test)
    valset = MyDataset(X_val, y_val)
    unlabelset = MyDataset_FixMatch(X_unlabel, y_unlabel, transform_w=transform_w, transform_s=transform_s)

    return trainset, testset, valset,  unlabelset

####transform
class Noise(object):
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.rand(c, h, w)
        return x

class Noise_w(object):
    def __call__(self, x):
        c, h, w = x.shape
        x +=np.random.uniform(low=0, high=0.5, size=(c,h,w))
        return x

class Noise_s(object):
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.uniform(low=0, high=2, size=(c,h,w))
        return x

class ToTensor(object):
    def __call__(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x


