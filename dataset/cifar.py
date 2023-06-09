import logging
import math
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC
from .mydataset import ImageFolder, ImageFolder_fix

logger = logging.getLogger(__name__)

__all__ = ['TransformOpenMatch', 'TransformFixMatch', 'cifar10_mean',
           'cifar10_std', 'cifar100_mean', 'cifar100_std', 'Tiny_mean', 'Tiny_std',
           'TransformFixMatch_Tiny', 'TransformFixMatch_Tiny_Weak']

DATA_PATH = './data'

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
Tiny_mean = (0.4802, 0.4481, 0.3975)
Tiny_std = (0.2770, 0.2691, 0.2821)


def get_cifar(args, norm=True):
    root = args.root
    name = args.dataset
    if name == "cifar10":
        data_folder = datasets.CIFAR10
        data_folder_main = CIFAR10SSL
        mean = cifar10_mean
        std = cifar10_std
        num_class = 10
    elif name == "cifar100":
        data_folder = datasets.CIFAR100
        data_folder_main = CIFAR100SSL
        mean = cifar100_mean
        std = cifar100_std
        num_class = 100

    else:
        raise NotImplementedError()
    assert num_class > args.num_classes

    base_dataset = data_folder(root, train=True, download=True)
    base_dataset.targets = np.array(base_dataset.targets)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = \
        x_u_split(args, base_dataset.targets)

    norm_func = TransformFixMatch(mean=mean, std=std, norm=norm)
    if norm:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
        ])
      
    train_labeled_dataset = data_folder_main(
        root, train_labeled_idxs, train=True,
        transform=norm_func)
    train_unlabeled_dataset = data_folder_main(
        root, train_unlabeled_idxs, train=True,
        transform=norm_func, return_idx=False)
    val_dataset = data_folder_main(
        root, val_idxs, train=True,
        transform=norm_func_test)   
    test_dataset = data_folder(
        root, train=False, transform=norm_func_test, download=False)
    test_dataset.targets = np.array(test_dataset.targets)

    unique_labeled = np.unique(train_labeled_idxs)
    val_labeled = np.unique(val_idxs)
    logger.info("Dataset: %s"%name)
    logger.info(f"Labeled examples: {len(unique_labeled)}"
                f"Unlabeled examples: {len(train_unlabeled_idxs)}"
                f"Valdation samples: {len(val_labeled)}")
    return train_labeled_dataset, train_unlabeled_dataset, \
           test_dataset, val_dataset


def get_tiny_imagenet(args, train_labeled_idxs=None, train_unlabeled_idxs=None, val_idxs = None, norm=True):
    root = args.root
    name = args.dataset
    train_data_folder = os.path.join(root, 'tiny_imagenet/tin_train.txt')
    test_data_folder = os.path.join(root, 'tiny_imagenet/tin_val.txt')
    mean = Tiny_mean
    std = Tiny_std
    num_class = 200
    assert num_class > args.num_classes
    norm_func = TransformFixMatch_Tiny(mean, std, size_image=64)                                   
    base_dataset = ImageFolder_fix(train_data_folder, transform=norm_func)
    print('-'*40 + ' Get idxs ' + '-'*40)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = \
        x_u_split(args, base_dataset.labels)
    
    print('-'*40 + ' Finished ' + '-'*40)
    if norm:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
        ])
       
    train_labeled_dataset = ImageFolder_fix(
        image_list=train_data_folder, indexes=train_labeled_idxs,
        transform=norm_func)
        
    train_unlabeled_dataset = ImageFolder_fix(
        image_list=train_data_folder, indexes=train_unlabeled_idxs,
        transform=norm_func)

    val_dataset = ImageFolder_fix(
        train_data_folder, val_idxs,
        transform=norm_func_test)
    
    test_dataset = ImageFolder_fix(image_list=test_data_folder, transform=norm_func_test)

    unique_labeled = np.unique(train_labeled_idxs)
    val_labeled = np.unique(val_idxs)
    logger.info("Dataset: %s"%name)
    logger.info(f"Labeled examples: {len(unique_labeled)} "
                f"Unlabeled examples: {len(train_unlabeled_idxs)} "
                f"Valdation samples: {len(val_labeled)} "              
                f"Test samples: {len(test_dataset)} "
                )
    
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, val_dataset

def x_u_split(args, labels):
    label_per_class = args.num_labeled 
    val_per_class = args.num_val 
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        unlabeled_idx.extend(idx)   
        idx = np.random.choice(idx, label_per_class+val_per_class, False)
        labeled_idx.extend(idx[:label_per_class])
        val_idx.extend(idx[label_per_class:])

    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == args.num_labeled * args.num_classes
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    unlabeled_idx = np.array(range(len(labels)))
    unlabeled_idx = list(set(unlabeled_idx) - set(labeled_idx))
    unlabeled_idx = list(set(unlabeled_idx) - set(val_idx))
    return labeled_idx, unlabeled_idx, val_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong

class TransformOpenMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)

        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong

class TransformFixMatch_Tiny(object):
    def __init__(self, mean, std, norm=True, size_image=64):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(weak2)
        else:
            return weak, strong



class TransformFixMatch_Tiny_Weak(object):
    def __init__(self, mean, std, norm=True, size_image=64):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak2(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(weak2)
        else:
            return weak, strong


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)

def get_transform(mean, std, image_size=None):
    if image_size: 
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_ood(dataset, id, image_size=None):
    image_size = (32, 32, 3) if image_size is None else image_size
    if id == "cifar10":
        mean = cifar10_mean
        std = cifar10_std
    elif id == "cifar100":
        mean = cifar100_mean
        std = cifar100_std
    elif "tiny"  in id:
        mean = Tiny_mean
        std = Tiny_std

    _, test_transform = get_transform(mean, std, image_size=image_size)

    if dataset == 'cifar10':
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=False,
                                    transform=test_transform)

    elif dataset == 'cifar100':
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=False,
                                     transform=test_transform)

    elif dataset == 'svhn':
        test_set = datasets.SVHN(DATA_PATH, split='test', download=True,
                                 transform=test_transform)

    elif dataset == 'lsun':
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'stanford_dogs':
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'cub':
        test_dir = os.path.join(DATA_PATH, 'cub')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'caltech_256':
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)


    return test_set

DATASET_GETTERS = {'cifar10': get_cifar,
                   'cifar100': get_cifar,
                   'tiny':get_tiny_imagenet,
                   }
