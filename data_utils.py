from torchvision import transforms, datasets
import torch
from utils import get_feature_dir
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

base_path = ''
# the paths for the datasets
datasets_paths = {
    'imagenet': base_path + 'data/imagenet',
    'cub': base_path + 'data/CUB',
    'cifar10': base_path + 'data/cifar10',
    'cifar100': base_path + 'data/cifar100',
    'places365': base_path + 'data/places365'
}

# the class labels for each set
label_paths = {
    'places365': base_path + 'data/categories_places365.txt',
    'imagenet': base_path + 'data/imagenet_classes.txt',
    'cifar10': base_path + 'data/cifar10_classes.txt',
    'cifar100': base_path + 'data/cifar100_classes.txt',
    'cub': base_path + 'data/cub_classes.txt'
}

concept_paths = {
    'places365': base_path + 'data/concept_sets/places365.txt',
    'imagenet': base_path + 'data/concept_sets/imagenet.txt',
    'cifar10': base_path + 'data/concept_sets/cifar10.txt',
    'cifar100': base_path + 'data/concept_sets/cifar100.txt',
    'cub': base_path + 'data/concept_sets/cub.txt',
}


def get_loaders(args, preprocess=None):
    """
    Create the loaders for the dataset. The name and other parameters are in the arg variable.
    :param args:
    :param preprocess: torchvision Transform to use instead of the default. For example one could use the CLIP preprocess.

    :return: the train and validation loader, the classes of the dataset and the concept set.
    """

    if preprocess:
        train_transform = preprocess
        val_transform = preprocess
    # if we load the similarities from file, we cant easily do augmentations
    elif (args.load_similarities or args.compute_similarities) and args.dataset not in ['cub', 'imagenet']:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    elif (args.load_similarities or args.compute_similarities) and args.dataset in ['cub', 'imagenet']:
        train_transform =  transforms.Compose([
            transforms.ToTensor(),transforms.Resize(256), transforms.CenterCrop(224)])
        val_transform = transforms.Compose([
            transforms.ToTensor(),transforms.Resize(256), transforms.CenterCrop(224)])

    # this is for the case that we use the original data and we want to add some augs.
    else:
        raise ValueError('Wrong Preprocess setting..')

    # Load the dataset
    train_data, val_data, classes, concept_set = data_loader(args,
                                                             preprocess_train=train_transform,
                                                             preprocess_val=val_transform,
                                                             load_similarities=args.load_similarities)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=(not args.compute_similarities),
        num_workers=args.num_workers, pin_memory=False, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=False)

    return train_loader, val_loader, classes, concept_set


def data_loader(args, preprocess_train=None, preprocess_val=None, load_similarities=True):
    """
    Load the datasets for the CDM.

    :param name: string. The name of the dataset. Currently, supports cifar10, cifar100, cub, places365, imagenet
    :param preprocess: Torch Transforms. The transform to use for the particular dataset. Depends on data and model.

    :return: data_train, data_val. Torch tensors. The training and validation data for the chosen dataset.
    """

    name = args.dataset
    print(preprocess_train)
    if not load_similarities:
        if name == 'cifar10':
            data_train = datasets.CIFAR10(root=datasets_paths['cifar10'], download=True,
                                          train=True, transform=preprocess_train)
            data_val = datasets.CIFAR10(root=datasets_paths['cifar10'], download=True,
                                        train=False, transform=preprocess_val)

        elif name == 'cifar100':
            data_train = datasets.CIFAR100(root=datasets_paths['cifar100'], download=True,
                                           train=True, transform=preprocess_train)
            data_val = datasets.CIFAR100(root=datasets_paths['cifar100'], download=True,
                                         train=False, transform=preprocess_val)
        elif name == 'places365':
            data_train = datasets.Places365(root=datasets_paths['places365'], download=False,
                                            split='train-standard', small=True,
                                            transform=preprocess_train)
            data_val = datasets.Places365(root=datasets_paths['places365'], download=False,
                                          split='val', small=True, transform=preprocess_val)
            
        # this is for imagenet and cub
        elif name in datasets_paths.keys():
            print(preprocess_train)
            data_train = datasets.ImageFolder(datasets_paths[name] + '/train/', preprocess_train)
            data_val = datasets.ImageFolder(datasets_paths[name] + '/val/', preprocess_val)

        else:
            raise ValueError('Dataset {} not supported (yet?)..'.format(name))

    else:
        # Need to change this if there are multiple pts for the dataset
        save_dir = get_feature_dir(args)
        save_name_features = save_dir + 'image_{}'.format('feats')
        save_name_features += '_{}.pt'.format(0)
        data_tensor, target_tensor = torch.load(save_name_features)
        data_train = torch.utils.data.TensorDataset(data_tensor.cpu().float(), target_tensor.cpu().float())

        # do the same for validation set
        save_dir = get_feature_dir(args, val=True)
        save_name_features = save_dir + 'image_{}'.format('feats',)
        save_name_features += '_{}.pt'.format(0)

        data_tensor, target_tensor = torch.load(save_name_features)
        data_val = torch.utils.data.TensorDataset(data_tensor.cpu().float(), target_tensor.cpu().float())

    # read the classes from the label files
    classes = []
    with open(label_paths[name], 'r') as f:
        for line in f:
            classes.append(line.strip())

    # also read the concepts from the concept sets
    concept_set = []
    with open(concept_paths[args.concept_name], 'r') as f:
        for line in f:
            concept_set.append(line.strip())

    return data_train, data_val, classes, concept_set
