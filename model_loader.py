import os
import cifar10.model_loader


def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)

    if dataset == 'cifar10_vit':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    
    if dataset == 'mrpc':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)

    if dataset == 'cifar100':
        net = cifar10.model_loader.load(model_name=model_name, model_file=model_file, data_parallel=data_parallel)

    return net
    