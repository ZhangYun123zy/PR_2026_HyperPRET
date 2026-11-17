from scipy import io
import os
import numpy as np
import sklearn.model_selection
import torch
import torch.utils.data

def sample_gt(gt, percentage, seed):
    """
    :param gt: 2d int array, -1 for undefined or not selected, index starts at 0
    :param percentage: for example, 0.1 for 10%, 0.02 for 2%, 0.5 for 50%
    :param seed: random seed
    :return:
    """
    indices = np.where(gt >= 0)
    X = list(zip(*indices))
    y = gt[indices].ravel()

    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)

    train_indices, test_indices = sklearn.model_selection.train_test_split(
        X,
        train_size=percentage,
        random_state=seed,
        stratify=y
    )

    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]

    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    return train_gt, test_gt

def sample_mem(gt, memorysize, seed):
    """
    :param gt: 2d int array, -1 for undefined or not selected, index starts at 0
    :param percentage: for example, 0.1 for 10%, 0.02 for 2%, 0.5 for 50%
    :param seed: random seed
    :return:
    """
    np.random.seed(seed)

    mem_gt = np.full_like(gt, -1)

    mem_test_gt=gt.copy()
    # 获取除 -1 外的所有元素
    non_negative_indices = np.nonzero(gt != -1)
    non_negative_values = gt[non_negative_indices]

    # 对每个非 -1 的元素随机选择 memorysize个进行保留
    for value in np.unique(non_negative_values):   
        zero_indices = np.argwhere(gt == value)

        # 随机选择五个位置
        random_indices = np.random.choice(zero_indices.shape[0], memorysize, replace=False)

        # 将选定位置设置为 -1
        for idx in random_indices:
            row, col = zero_indices[idx]
            mem_gt[row, col] = value
            mem_test_gt[row,col]=-1
    return mem_gt,mem_test_gt

class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, image, gt, patch_size, data_aug=True,datatype="train"):
        """
        :param image: 3d float np array of HSI, image
        :param gt: train_gt or val_gt or test_gt
        :param patch_size: 7 or 9 or 11 ...
        :param data_aug: whether to use data augment, default is True
        """
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.ps = self.patch_size // 2  # padding size
        self.data = np.pad(image, ((self.ps, self.ps), (self.ps, self.ps), (0, 0)), mode='reflect')
        self.label = np.pad(gt, ((self.ps, self.ps), (self.ps, self.ps)), mode='reflect')

        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)
                                 if self.ps <= x < image.shape[0] + self.ps
                                 and self.ps <= y < image.shape[1] + self.ps])
        self.labels = [self.label[x, y] for x, y in self.indices]
        if datatype=='train':
            np.random.shuffle(self.indices)


    def hsi_augment(self, data):
        # e.g. (7 7 200) data = numpy array float32
        do_augment = np.random.random()
        if do_augment > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2:
                data = np.fliplr(data)
            elif 0.2 < prob <= 0.4:
                data = np.flipud(data)
            elif 0.4 < prob <= 0.6:
                data = np.rot90(data, k=1)
            elif 0.6 < prob <= 0.8:
                data = np.rot90(data, k=2)
            elif 0.8 < prob <= 1.0:
                data = np.rot90(data, k=3)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.data_aug:
            # Perform data augmentation (only on 2D patches)
            data = self.hsi_augment(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Add a fourth dimension for 3D CNN

        data = data.unsqueeze(0)

        return data, label