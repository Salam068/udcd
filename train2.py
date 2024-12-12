from pretrainedmodels.models.torchvision_models import model_name

from models.models2 import get_pretrained_model, BaseModelWithFeatures
from utils.utils import set_logger
from utils.trainer import fit
from models.model import DenseNet121
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import random
import logging
import sys
import os
import argparse
import warnings
warnings.simplefilter('ignore')

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to the training data directory')
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='reduced feature dimension')
    parser.add_argument('--drop_rate', type=float, default=0,
                        help='dropout rate')
    parser.add_argument('--epochs', type=int, default=1,
                        help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--n_distill', type=int, default=2,
                        help='start to use the kld loss')
    parser.add_argument('--seed', type=int, default=2024,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use for training')
    parser.add_argument('--mode', default='exact', type=str,
                        choices=['exact', 'relax', 'multi_pos'],
                        help='training mode')
    parser.add_argument('--nce_p', default=1, type=int,
                        help='number of positive samples for NCE')
    parser.add_argument('--nce_k', default=4096, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float,
                        help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='momentum for non-parametric updates')
    parser.add_argument('--CCD_mode', type=str, default="sup",
                        choices=['sup', 'unsup'], help='CCD mode')
    parser.add_argument('--rel_weight', type=float, default=25,
                        help='relation loss weight')
    parser.add_argument('--ccd_weight', type=float, default=0.1,
                        help='CCD loss weight')
    parser.add_argument('--anchor_type', type=str, default="center",
                        choices=['center', 'class'], help='anchor type')
    parser.add_argument('--class_anchor', default=30, type=int,
                        help='number of anchors in each class')
    parser.add_argument('--s_dim', type=int, default=128,
                        help='feature dimension of the student model')
    parser.add_argument('--t_dim', type=int, default=128,
                        help='feature dimension of the EMA teacher')
    parser.add_argument('--n_data', type=int, default=3662,
                        help='total number of training samples.')
    parser.add_argument('--t_decay', type=float, default=0.99,
                        help='EMA decay rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type')
    parser.add_argument('--scheduler', type=str, default='OneCycleLR',
                        help='scheduler type')
    parser.add_argument('--consistency', type=float, default=1,
                        help='consistency weight')
    parser.add_argument('--consistency_rampup', type=float, default=30,
                        help='consistency ramp-up period')

    args = parser.parse_args()
    return args


def set_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # Get arguments
    args = get_args()
    print(args)
    # Set seed
    # set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set Logger
    # if not os.path.exists(args.logdir):
    #     os.makedirs(args.logdir)
    # logger = set_logger(args)
    # logger.info(args)
    print(args.root_dir)

    # Define data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_path = args.root_dir+"/Data/Train/fold_1"
    test_path = args.root_dir + "/Data/Test/fold_1"

    # Load datasets
    train_ds = datasets.ImageFolder(train_path, transform=train_transform)
    test_ds = datasets.ImageFolder(test_path, transform=test_transform)

    n_classes = len(train_ds.classes)
    print(f"Number of classes: {n_classes}")
    print(f"Classes: {train_ds.classes}")

    # Create data loaders
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=12, pin_memory=True,
                          worker_init_fn=worker_init_fn)

    test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=12, pin_memory=True,
                         worker_init_fn=worker_init_fn)

    # Compute class weights
    class_counts = np.array([train_ds.targets.count(i) for i in range(n_classes)])
    weights = class_counts.sum() / class_counts
    print(f"Class weights: {weights}")
    print(f"number of classes: {len(class_counts)}")
    num_classes = len(class_counts)
    # Load models
    # student = DenseNet121(hidden_units=args.feat_dim,
    #                       out_size=n_classes, drop_rate=args.drop_rate)
    # teacher = DenseNet121(hidden_units=args.feat_dim,
    #                       out_size=n_classes, drop_rate=args.drop_rate)

    # Instantiate the models
    # Student: Shallow model (e.g., densenet121)
    # student = get_pretrained_model(model_name="densenet121" , model_mode="import_Torch", isPretrained=True, input_ch=3, class_num=10, final_activation_func='Softmax',
    #                      train_on_gpu=True, multi_gpu=False, q_order=1)
    #
    # # Teacher: Deep model (e.g., densenet201)
    # teacher = get_pretrained_model(model_name="densenet201", model_mode="import_Torch", isPretrained=True, input_ch=3, class_num=10, final_activation_func='Softmax',
    #                      train_on_gpu=True, multi_gpu=False, q_order=1)

    student  = BaseModelWithFeatures(model_name="mobilenet_v3_small",hidden_units = 128, out_size = num_classes, drop_rate = 0.2,pretrained=True)
    teacher = BaseModelWithFeatures(model_name="mobilenet_v3_small", hidden_units=128, out_size=num_classes, drop_rate=0.2,
                                    pretrained=True)

    # # Detach teacher model's parameters
    # for param in teacher.parameters():
    #     param.detach_()

    # Fit the model
    fit(student, teacher, train_dl, test_dl, weights,
        train_ds.class_to_idx, args, device=args.device)


