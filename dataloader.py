import torch
import torchvision
from torchvision import transforms
import os
import numpy as np
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset as load_dataset_hf
from sklearn.model_selection import train_test_split
from functools import partial
from torch_uncertainty.datamodules import CIFAR100DataModule, CIFAR10DataModule

from torchvision.transforms import v2


def get_relative_path(file):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    return os.path.join(script_dir, file)


def load_dataset(dataset='cifar10', datapath='cifar10/data', batch_size=128, \
                 threads=2, raw_data=False, data_split=1, split_idx=0, \
                 trainloader_path="", testloader_path="", NLP_model="distilbert/distilroberta-base"):
    """
    Setup dataloader. The data is not randomly cropped as in training because of
    we want to esimate the loss value with a fixed dataset.

    Args:
        raw_data: raw images, no data preprocessing
        data_split: the number of splits for the training dataloader
        split_idx: the index for the split of the dataloader, starting at 0

    Returns:
        train_loader, test_loader
    """

    # use specific dataloaders
    if trainloader_path and testloader_path:
        assert os.path.exists(trainloader_path), 'trainloader does not exist'
        assert os.path.exists(testloader_path), 'testloader does not exist'
        train_loader = torch.load(trainloader_path)
        test_loader = torch.load(testloader_path)
        return train_loader, test_loader

    assert split_idx < data_split, 'the index of data partition should be smaller than the total number of split'

    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        data_folder = get_relative_path(datapath)
        if raw_data:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                                download=True, transform=transform)
        # If data_split>1, then randomly select a subset of the data. E.g., if datasplit=3, then
        # randomly choose 1/3 of the data.
        if data_split > 1:
            indices = torch.tensor(np.arange(len(trainset)))
            data_num = len(trainset) // data_split # the number of data in a chunk of the split

            # Randomly sample indices. Use seed=0 in the generator to make this reproducible
            state = np.random.get_state()
            np.random.seed(0)
            indices = np.random.choice(indices, data_num, replace=False)
            np.random.set_state(state)

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False, num_workers=threads)
        else:
            kwargs = {'num_workers': 2, 'pin_memory': True}
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=False, **kwargs)
        testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                               download=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=threads)
    elif dataset == 'mrpc':
        tokenizer = AutoTokenizer.from_pretrained(NLP_model)
        eval_ood = False
        eval_shift = False
        num_classes = 2
        raw = load_dataset_hf("glue", "mrpc")
        
        raw_val_set = raw["validation"]
        raw_train_set = raw["train"]
        raw_test_set = raw["test"]
        encode_fn = encode_mrpc
        raw_val_split = raw_val_set
        raw_test_split = raw_test_set

        # Encode all splits
        tokenize = partial(encode_fn, tokenizer=tokenizer)
        train_dataset = raw_train_set.map(tokenize, batched=True)
        val_dataset = raw_val_split.map(tokenize, batched=True)
        test_dataset = raw_test_split.map(tokenize, batched=True)

        # Add 'labels' field
        train_dataset = train_dataset.map(lambda x: {'labels': x['label']}, batched=True)
        val_dataset = val_dataset.map(lambda x: {'labels': x['label']}, batched=True)
        test_dataset = test_dataset.map(lambda x: {'labels': x['label']}, batched=True)

        # Format for PyTorch
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in train_dataset.column_names:
            columns.append('token_type_ids')

        for ds in [train_dataset, val_dataset, test_dataset]:
            ds.set_format(type='torch', columns=columns)

        # Dataloaders
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        def collate_as_tuple(batch):
            batch_dict = collator(batch)  # use your existing collator
            labels = batch_dict.pop("labels")  # remove labels from input dict
            return batch_dict, labels

        kwargs = {'num_workers': 2, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_as_tuple, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_as_tuple)

    elif dataset == 'cifar100':
        num_classes = 100 
        dm = CIFAR100DataModule(root="/local_storage/users/kadec/data/", batch_size=16, num_workers=1, val_split=0.1, eval_ood=False, eval_shift=False, shift_severity=3, basic_augment=True)
        # Note: this assumes a ViT!
        normalize = v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])


        dm.train_transform = v2.Compose([
            v2.Resize(256),
            v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        dm.test_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        dm.prepare_data()
        dm.setup("test")
        test_loader = dm.test_dataloader()[0]

        dm.setup("fit")
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
    elif dataset == 'cifar10_vit':
        num_classes = 10
        dm = CIFAR10DataModule(root="/local_storage/users/kadec/data/", batch_size=16, num_workers=1, val_split=0.1, eval_ood=False, eval_shift=False, shift_severity=3, basic_augment=True)
        # Note: this assumes a ViT!
        normalize = v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])


        dm.train_transform = v2.Compose([
            v2.Resize(256),
            v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        dm.test_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        dm.prepare_data()
        dm.setup("test")
        test_loader = dm.test_dataloader()[0]

        dm.setup("fit")
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

    return train_loader, test_loader



def encode_mrpc(examples, tokenizer):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=256)#padding='max_length')

###############################################################
####                        MAIN
###############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    args = parser.parse_args()

    trainloader, testloader = load_dataset(args.dataset, args.datapath,
                                args.batch_size, args.threads, args.raw_data,
                                args.data_split, args.split_idx,
                                args.trainloader, args.testloader)

    print('num of batches: %d' % len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print('batch_idx: %d   batch_size: %d'%(batch_idx, len(inputs)))
