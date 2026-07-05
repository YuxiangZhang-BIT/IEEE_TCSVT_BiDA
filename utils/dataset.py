import os
import numpy as np
import sklearn.model_selection
import torch
import torch.utils.data
import tifffile as tiff
import random
import scipy.io as io
from sklearn import preprocessing

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return tiff.imread(dataset)
    else:
        raise ValueError("Unknown file format: {}".format(ext))
    
def load_mat_hsi(dataset_name, dataset_dir, norm='normband'):
    """ load HSI.mat dataset """
    # available sets
    available_sets = [
        'sa',
        'pu',
        'whulk',
        'hrl',
        'Loukia',
        'Dioni',
        'Houston18',
        'Houston13',
    ]
    assert dataset_name in available_sets, "dataset should be one of" + ' ' + str(available_sets)

    image = None
    gt = None
    labels = None

    if (dataset_name == 'sa'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_corrected.mat"))
        image = image['salinas_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_gt.mat"))
        gt = gt['salinas_gt']
        labels = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]

    elif (dataset_name == 'pu'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU.mat"))
        image = image['paviaU']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU_gt.mat"))
        gt = gt['paviaU_gt']
        labels = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]

    elif (dataset_name == 'whulk'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou.mat"))
        image = image['WHU_Hi_LongKou']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]

    elif (dataset_name == 'Loukia'):
        img = open_file(dataset_dir + 'Loukia.mat')['ori_data']
        gt = open_file(dataset_dir + 'Loukia_gt_out68.mat')['map']
        labels = [
            'Undefined',
            'Dense Urban Fabric',
            'Mineral Extraction Sites',
            'Non Irrigated Arable Land',
            'Fruit Trees',
            'Olive Groves',
            'Coniferous Forest',
            'Dense Sclerophyllous Vegetation',
            'Sparce Sclerophyllous Vegetation',
            'Sparcely Vegetated Areas',
            'Rocks and Sand',
            'Water',
            'Coastal Water'
        ]

    elif dataset_name == 'Dioni':
        img = open_file(dataset_dir + 'Dioni.mat')['ori_data']
        gt = open_file(dataset_dir + 'Dioni_gt_out68.mat')['map']
        labels = [
            'Undefined',
            'Dense Urban Fabric',
            'Mineral Extraction Sites',
            'Non Irrigated Arable Land',
            'Fruit Trees',
            'Olive Groves',
            'Coniferous Forest',
            'Dense Sclerophyllous Vegetation',
            'Sparce Sclerophyllous Vegetation',
            'Sparcely Vegetated Areas',
            'Rocks and Sand',
            'Water',
            'Coastal Water'
        ]

    elif dataset_name == 'Houston18':
        img = open_file(dataset_dir + 'Houston18.mat')['ori_data']
        gt = open_file(dataset_dir + 'Houston18_7gt.mat')['map']
        labels = ['0', "1", "2", "3",
                        "4", "5",
                        "6", "7"]

    elif dataset_name == 'Houston13':
        img = open_file(dataset_dir + 'Houston13.mat')['ori_data']
        gt = open_file(dataset_dir + 'Houston13_7gt.mat')['map']
        labels = ['0', "1", "2", "3",
                        "4", "5",
                        "6", "7"]

    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("warning: nan values found in dataset {}, using 0 replace them".format(dataset_name))
        img[nan_mask] = 0
        gt[nan_mask] = 0

    if norm == 'normband':
        img = np.asarray(img, dtype='float32')
        m, n, d = img.shape[0], img.shape[1], img.shape[2]
        img_ori= img.reshape((m*n,-1))
        index = np.where(img_ori.sum(axis=-1)!=0)
        img = img_ori[index]
        img = img/img.max()
        img_temp = np.sqrt(np.asarray((img**2).sum(1)))
        img_temp = np.expand_dims(img_temp,axis=1)
        img_temp = img_temp.repeat(d,axis=1)
        img_temp[img_temp==0]=1
        img = img/img_temp
        img_ori[index] = img
        img = np.reshape(img_ori,(m,n,-1))
    elif norm == 'minmax':
        img = np.asarray(img, dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        mean_by_c = np.mean(img, axis=(0, 1))
        for c in range(img.shape[-1]):
            img[:, :, c] = img[:, :, c] - mean_by_c[c]
    elif norm == 'std':
        meanhsi = np.mean(np.reshape(img, -1))
        sigmahsi = np.sqrt(np.var(np.reshape(img, -1)))
        img = (img - meanhsi) / (sigmahsi)
    elif norm == 'ln':
        img =  img/(np.sqrt(np.sum(img**2,axis=2,keepdims = True))+ 1e-9)
    elif norm =='sklearn':
        data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))  # (111104,204)
        data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
        img = data_scaler.reshape(img.shape[0], img.shape[1], img.shape[2])
    elif norm =='ori':
        pass

    gt = gt.astype('int') - 1
    labels = labels[1:]

    return img, gt, labels

def sample_gt(gt, train_size, seed, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    # indices = np.nonzero(gt)
    indices = np.where(gt >= 0)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random':
        if train_size == 1:
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0],test_indices[1],test_label))

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt

class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, image, gt, patch_size, data_aug=True, flip_augmentation=False, radiation_augmentation=False, mixture_augmentation=False):
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
        self.label = gt
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        
        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if x > self.ps and x <
                                self.data.shape[0] - self.ps and y > self.ps and y < self.data.shape[1] - self.ps])
        self.labels = [self.label[x,y] for x,y in self.indices]
        del image, gt

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise
    
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
            data = self.hsi_augment(data)

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            data, _ = self.flip(data, self.label[x1:x2, y1:y2])
        if self.radiation_augmentation and np.random.random() < 0.5:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
            data = self.mixture_noise(data, self.label[x1:x2, y1:y2])
                
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        data = data.unsqueeze(0)

        return data, label

