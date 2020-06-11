import sys
sys.path.append('robustness')

from robustness import model_utils, datasets, defaults, train
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
from robustness.tools import helpers
from utils import transfer_datasets, fine_tunify, constants as cs
from torchvision import models
from cox import utils
import cox.store
import torch as ch
from torch import nn
import argparse
import os 
import numpy as np


parser = argparse.ArgumentParser(description='Transfer learning via pretrained Imagenet models', 
                                conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Custom arguments
parser.add_argument('--dataset', type=str, default='cifar', help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--resume', action='store_true', help='Whether to resume or not (Overrides the one in robustness.defaults)')
parser.add_argument('--pytorch-pretrained', action='store_true', help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--cifar10-cifar10', action='store_true', help='cifar10 to cifar10 transfer')
parser.add_argument('--subset', type=int, default=None, help='number of training data to use from the dataset')
parser.add_argument('--no-tqdm', type=int, default=1, choices=[0, 1], help='Do not use tqdm.')
parser.add_argument('--no-replace-last-layer', action='store_true', 
                help='Whether to avoid replacing the last layer')
parser.add_argument('--freeze-level', type=int, default=-1, 
                help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
parser.add_argument('--additional-hidden', type=int, default=0, 
                help='How many hidden layers to add on top of pretrained network + classification layer')


def get_dataset_and_loaders(args):
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(only_val=args.eval_only, batch_size=args.batch_size, workers=8)
    elif args.cifar10_cifar10:
        ds = datasets.CIFAR('/tmp')
        train_loader, validation_loader = ds.make_loaders(only_val=args.eval_only, batch_size=args.batch_size, workers=8)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(args.dataset, args.batch_size, 8, args.subset)
        if type(ds) == int: 
            new_ds = datasets.CIFAR("/tmp")
            new_ds.num_classes = ds
            new_ds.mean = ch.tensor([0.,0.,0.])
            new_ds.std = ch.tensor([1.,1.,1.])
            ds = new_ds
    return ds, train_loader, validation_loader

def resume_finetuning_from_checkpoint(args, ds, finetuned_model_path):
    print('[Resuming finetuning from a checkpoint...]')
    if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
        model, _ = model_utils.make_and_restore_model(
                    arch=pytorch_models[args.arch](args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch, 
                    dataset=datasets.ImageNet(''), add_custom_forward=args.arch in pytorch_models.keys())
        while hasattr(model, 'model'):
            model = model.model
        model = fine_tunify.ft(args.arch, model, ds.num_classes, args.additional_hidden)
        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=finetuned_model_path,
                                                    add_custom_forward=args.additional_hidden>0 or args.arch in pytorch_models.keys())
    else:
        model, checkpoint = model_utils.make_and_restore_model(arch=args.arch, dataset=ds, resume_path=finetuned_model_path)
    return model, checkpoint

def get_model(args, ds):
    # An option to resume finetuning from a checkpoint. Only for Imagenet-Imagenet transfer
    finetuned_model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    if args.resume and os.path.isfile(finetuned_model_path):
        model, checkpoint = resume_finetuning_from_checkpoint(args, ds, finetuned_model_path)
    else:

        if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
            model, _ = model_utils.make_and_restore_model(
                        arch=pytorch_models[args.arch](args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch, 
                        dataset=datasets.ImageNet(''), resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained, 
                        add_custom_forward=args.arch in pytorch_models.keys())
            checkpoint = None
        else:
            model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=ds, 
                                                resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained)
            checkpoint = None

        # For all other datasets, replace the last layer then finetine, unless otherwise specified using
        # the args.no_replace_last_layer flag
        if not args.no_replace_last_layer and not args.eval_only:
            print(f'[Replacing the last layer with {args.additional_hidden} '
                    f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(args.arch, model, ds.num_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, 
                                                        add_custom_forward=args.additional_hidden>0 or args.arch in pytorch_models.keys())
        else:
            print('[NOT replacing the last layer]')

    return model, checkpoint

def freeze_model(model, freeze_level):
    '''
    Freezes up to args.freeze_level layers of the model (assumes a resnet model)
    '''
    ## Freeze layers according to args.freeze-level
    update_params = None
    if freeze_level != -1:
        # assumes a resnet architecture
        assert len([name for name,_ in list(model.named_parameters()) \
            if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
        update_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())

            if not freeze and f'layer{freeze_level}' not in name:
                print(f"[Appending the params of {name} to the update list]")
                update_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f'layer{freeze_level}' in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return update_params

def get_class_weights(args, loader):
    '''Returns the distribution of classes in a given dataset.
    '''
    if args.dataset in ['pets','flowers']:
        targets = loader.dataset.targets

    elif args.dataset in ['caltech101','caltech256']:
        targets = np.array([loader.dataset.ds.dataset.y[idx] for idx in loader.dataset.ds.indices])

    elif args.dataset == 'aircraft':
        targets = [s[1] for s in loader.dataset.samples]

    counts =  np.unique(targets, return_counts=True)[1]
    class_weights = counts.sum()/(counts*len(counts))
    return ch.Tensor(class_weights)

def main(args, store):
    
    ds, train_loader, validation_loader = get_dataset_and_loaders(args)

    if args.dataset in ['pets','caltech101','caltech256','flowers', 'aircraft']:
        class_weights = get_class_weights(args, validation_loader)
        def custom_acc(logits, labels):
                '''Returns the top1 accuracy, weighted by the class distribution.
                This is important when evaluating an unbalanced dataset. 
                '''
                batch_size = labels.size(0)
                maxk = min(5, logits.shape[-1])
                prec1, _ = helpers.accuracy(logits, labels, topk=(1, maxk), exact=True)

                normal_prec1 = prec1.sum(0, keepdim=True).mul_(100/batch_size)
                weighted_prec1 = prec1 * class_weights[labels.cpu()].cuda()
                weighted_prec1 = weighted_prec1.sum(0, keepdim=True).mul_(100/batch_size)

                return weighted_prec1.item(), normal_prec1.item()
        args.custom_accuracy = custom_acc # meaningful onlt for validation set. Ignore trainig set prec.

    model, checkpoint = get_model(args, ds)

    if args.eval_only:
        return train.eval_model(args, model, validation_loader, store=store)
      
    update_params = freeze_model(model, freeze_level=args.freeze_level)

    # Checking if freeze is working 
    #(uncomment this part, and check if the weights stay the same or change across iterations)
    # def check_freezed_features_hook(model, i, loop_type, inp, target):
    #     if i%100==0:
    #         for name, param in model.named_parameters():
    #             if name == 'module.model.layer4.1.conv2.weight':
    #                 print(name, param)

    # args.iteration_hook = check_freezed_features_hook

    print(f"Dataset: {args.dataset} | Model: {args.arch}")
    train.train_model(args, model, (train_loader, validation_loader), store=store, 
                                checkpoint=checkpoint, update_params=update_params)

def args_preprocess(args):
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0

    if args.pytorch_pretrained:
        assert not args.model_path, 'You can either specify pytorch_pretrained or model_path, not together.'

    ## CIFAR10 to CIFAR10 assertions
    if args.cifar10_cifar10:
        assert args.dataset == 'cifar10'

    if args.data != '':
        cs.PETS_PATH = cs.CARS_PATH = cs.FGVC_PATH = cs.FLOWERS_PATH = cs.DTD_PATH = cs.SUN_PATH = cs.FOOD_PATH = cs.BIRDS_PATH = args.data    

    ALL_DS = list(transfer_datasets.DS_TO_FUNC.keys()) + ['imagenet', 'breeds_living_9', 'stylized_imagenet']
    assert args.dataset in ALL_DS 

    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None

    # Preprocess args
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, None)
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, None)
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, None)

    return args

if __name__ == "__main__":

    args = parser.parse_args()
    args = args_preprocess(args)

    pytorch_models = {
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'squeezenet': models.squeezenet1_0,
        'densenet': models.densenet161,
        'shufflenet': models.shufflenet_v2_x1_0,
        'mobilenet': models.mobilenet_v2,
        'resnext50_32x4d': models.resnext50_32x4d,
        'mnasnet': models.mnasnet1_0,
    }

    # Create store and log the args
    store = cox.store.Store(args.out_dir, args.exp_name)
    if 'metadata' not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata', schema)
        store['metadata'].append_row(args_dict)
    else:
        print('[Found existing metadata in store. Skipping this part.]')
    main(args, store)
