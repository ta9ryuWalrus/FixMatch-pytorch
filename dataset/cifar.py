import logging
import math

import numpy as np
from PIL import Image
import pandas as pd
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from .randaugment import RandAugmentMC, RandAugmentMC_Cassava

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_cassava(args, root=None):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=400,
                              padding=int(400*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    base_dataset = Cassava(args=args, train=True)

    train_labeled_data, train_labeled_targets, train_unlabeled_data, train_unlabeled_targets = x_u_split_cassava(args, base_dataset)

    train_labeled_dataset = CassavaSSL(args=args, data=train_labeled_data, targets=train_labeled_targets, transform=transform_labeled)
    train_unlabeled_dataset = CassavaSSL(args=args, data=train_unlabeled_data, targets=train_unlabeled_targets, transform=CassavaFixMatch(mean=normal_mean, std=normal_std))
    test_dataset = Cassava(args=args, train=False, transform=transform_val)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

class Cassava(Dataset):
    def __init__(self, args, train=True, transform=None):
        # dfは全部で21397件
        self.data_dir = args.data_dir
        df = pd.read_csv(self.data_dir + '/train.csv')
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        i = 0
        for train_idx, test_idx in kf.split(df['image_id'], df['label']):
          if i == args.fold:
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
          i += 1

        #train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=1)
        if train:
            self.data = np.array(train_df['image_id'])
            self.targets = np.array(train_df['label'])
        else:
            self.data = np.array(test_df['image_id'])
            self.targets = np.array(test_df['label'])
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.data_dir + '/train_images/' + str(self.data[index])
        label = self.targets[index]

        img = Image.open(img_path)
        img = img.resize((400, 400))

        if self.transform:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.targets)



def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

def x_u_split_cassava(args, dataset):
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_data, unlabeled_targets = dataset.data, dataset.targets

    if args.train_stratify:
        labeled_data, _, labeled_targets, _ = train_test_split(dataset.data, dataset.targets, train_size=0.1, random_state=args.seed, stratify=dataset.targets)
        print(np.unique(labeled_targets, return_counts=True))
    else:
        labeled_data, _, labeled_targets, _ = train_test_split(dataset.data, dataset.targets, train_size=0.1, random_state=args.seed)
    return labeled_data, labeled_targets, unlabeled_data, unlabeled_targets


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class CassavaFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=400,
                                  padding=int(400*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=400,
                                  padding=int(400*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC_Cassava(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CassavaSSL(Cassava):
    def __init__(self, args, data, targets, transform=None):
        super(CassavaSSL, self).__init__(args=args, transform=transform, train=True)
        self.data = data
        self.targets = targets


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cassava': get_cassava}