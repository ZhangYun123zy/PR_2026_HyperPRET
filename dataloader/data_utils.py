import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
from dataloader.utils1 import load_data
from  dataloader.dataset import sample_gt, HSIDataset,sample_mem

def set_up_datasets(args):
    if args.dataset == 'botswana':
        args.base_class = 9
        args.num_classes=14
        args.way = 5
        args.shot = 5
        args.sessions = 2
        args.batch_size_exemplar=args.base_class*5      
    if args.dataset == 'Houston':
        args.base_class = 9
        args.num_classes=15
        args.way = 6
        args.shot = 5
        args.sessions = 2
        args.batch_size_exemplar=args.base_class*5
    if args.dataset == 'xuzhou':
        args.base_class = 6
        args.num_classes=9
        args.way = 3
        args.shot = 5
        args.sessions = 2
        args.batch_size_exemplar=args.base_class*5
    #args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args,seed):

    class_index = np.arange(args.base_class)
    print(class_index)
    if args.dataset == 'Houston':
        test_data = '/mnt/HDD/data/ZY/HSI_DATA/Houston/Houston.mat'#IP
        test_label = '/mnt/HDD/data/ZY/HSI_DATA/Houston/houston2012_gt.mat'#IP
    if args.dataset == 'xuzhou':
        test_data = '/mnt/HDD/data/datasets/HSICls/xuzhou/xuzhou.mat'#IP
        test_label = '/mnt/HDD/data/datasets/HSICls/xuzhou/xuzhou_gt.mat'#IP
    if args.dataset == 'botswana':
        test_data = '/mnt/HDD/data/ZY/HSI_DATA/botswana/Botswana.mat'#IP
        test_label = '/mnt/HDD/data/ZY/HSI_DATA/botswana/Botswana_gt.mat'#IP
    Data_Band_Scaler, GroundTruth = load_data(test_data, test_label)
    Data_Band_Scaler=np.array(Data_Band_Scaler,dtype='float32')
    # 创建布尔索引
    mask = np.isin(np.array(GroundTruth,dtype='float32'),class_index+1)

    # 使用布尔索引生成结果
    GroundTruth_part = np.where(mask, np.array(GroundTruth,dtype='float32'), 0)
    print(np.min(GroundTruth_part),np.max(GroundTruth_part),np.count_nonzero(GroundTruth_part !=0 ))
    GroundTruth=np.array(GroundTruth,dtype='int64')-1
    print(np.min(GroundTruth),np.max(GroundTruth),np.count_nonzero(GroundTruth != -1))
    sub_gt = np.where(np.isin(GroundTruth, class_index), GroundTruth, -1)
    print(np.min(sub_gt),np.max(sub_gt),np.count_nonzero(sub_gt != -1))
    train_gt, test_gt = sample_gt(sub_gt, args.ratio, seed)
    #print(np.argwhere(train_gt != -1))
    print("training and testing data",np.min(train_gt),np.max(train_gt),np.min(test_gt),np.max(test_gt),np.count_nonzero(train_gt != -1),np.count_nonzero(test_gt != -1))
    trainset = HSIDataset(Data_Band_Scaler, train_gt, patch_size=args.patch_size, data_aug=True)#IP，4类别
    testset = HSIDataset(Data_Band_Scaler, test_gt, patch_size=args.patch_size, data_aug=False,datatype="test")#IP，4类别
    all_set = HSIDataset(Data_Band_Scaler, sub_gt, patch_size=args.patch_size, data_aug=False,datatype="test")#IP，4类别
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True)
    
    all_loader = torch.utils.data.DataLoader(
        dataset=all_set, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True)
    return trainset, trainloader, testloader,all_loader,GroundTruth_part,GroundTruth_part

def get_new_dataloader(args,session,seed):
    if args.dataset == 'Houston':
        test_data = '/mnt/HDD/data/ZY/HSI_DATA/Houston/Houston.mat'#IP
        test_label = '/mnt/HDD/data/ZY/HSI_DATA/Houston/houston2012_gt.mat'#IP
    if args.dataset == 'xuzhou':
        test_data = '/mnt/HDD/data/datasets/HSICls/xuzhou/xuzhou.mat'#IP
        test_label = '/mnt/HDD/data/datasets/HSICls/xuzhou/xuzhou_gt.mat'#IP
    if args.dataset == 'botswana':
        test_data = '/mnt/HDD/data/ZY/HSI_DATA/botswana/Botswana.mat'#IP
        test_label = '/mnt/HDD/data/ZY/HSI_DATA/botswana/Botswana_gt.mat'#IP
    class_index=np.arange(args.base_class+(session-1)*args.way, args.base_class+session*args.way)
    print(class_index)
    Data_Band_Scaler, GroundTruth = load_data(test_data, test_label)
    Data_Band_Scaler=np.array(Data_Band_Scaler,dtype='float32')
    # 创建布尔索引
    all_class_index=np.arange(args.base_class+session*args.way)
    mask = np.isin(np.array(GroundTruth,dtype='float32'),all_class_index+1)
    # 使用布尔索引生成结果
    GroundTruth_part = np.where(mask, np.array(GroundTruth,dtype='float32'), 0)
    print(np.min(GroundTruth_part),np.max(GroundTruth_part),np.count_nonzero(GroundTruth_part !=0 ))

    GroundTruth=np.array(GroundTruth,dtype='int64')-1
    print(np.min(GroundTruth),np.max(GroundTruth),np.count_nonzero(GroundTruth != -1))

    sub_gt = np.where(np.isin(GroundTruth, class_index), GroundTruth, -1)
    print(np.min(sub_gt),np.max(sub_gt),np.count_nonzero(sub_gt != -1))
    train_gt, test_gt = sample_mem(sub_gt, args.shot, seed)

    #all_class_index=np.arange(args.base_classnum+session*args.n_way)
    all_seen_gt=np.where(np.isin(GroundTruth, all_class_index), GroundTruth, -1)
    print(np.min(all_seen_gt),np.max(all_seen_gt),np.count_nonzero(all_seen_gt != -1))
    #all_seen_gt=all_seen_gt+test_gt+1

    test_old_class_index=np.arange(args.base_class+(session-1)*args.way)
    test_old_seen_gt=np.where(np.isin(GroundTruth, test_old_class_index), GroundTruth, -1)
    print(np.min(test_old_seen_gt),np.max(test_old_seen_gt),np.count_nonzero(test_old_seen_gt != -1))
    test_gt=test_gt+test_old_seen_gt+1
    print(np.min(test_gt),np.max(test_gt),np.count_nonzero(test_gt != -1))
    print(np.min(all_seen_gt),np.max(all_seen_gt),np.count_nonzero(all_seen_gt != -1))
    print(np.min(train_gt),np.max(train_gt),np.min(test_gt),np.max(test_gt),np.count_nonzero(train_gt != -1),np.count_nonzero(test_gt != -1))
    trainset = HSIDataset(Data_Band_Scaler, train_gt, patch_size=args.patch_size, data_aug=True)#IP，4类别
    testset = HSIDataset(Data_Band_Scaler, test_gt, patch_size=args.patch_size, data_aug=False,datatype="test")#IP，4类别
    all_set = HSIDataset(Data_Band_Scaler, all_seen_gt, patch_size=args.patch_size, data_aug=False,datatype="test")#IP，4类别
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True)
    
    all_loader = torch.utils.data.DataLoader(
        dataset=all_set, batch_size=args.batch_size_test, shuffle=False, num_workers=8, pin_memory=True)
    return trainset, trainloader, testloader,all_loader,np.array(test_gt+1,dtype='int64'),np.array(all_seen_gt+1,dtype='int64')



def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list
