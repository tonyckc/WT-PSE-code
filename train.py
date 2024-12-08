from datetime import datetime
import os
import os.path as osp

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
import algorithms
import shape_networks
import random
import numpy as np
import Trainer
import hparams_registry
import fundus_dataloader as DL
import custom_transforms as tr
from tqdm import tqdm
from algorithms import WT_PSE



# dir to save model
local_path = '/*/*/*/founds_training'
import shutil
import glob
from utils import save_code,seed_initialization




def main(args,left_argv,trial_name,day):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    result = {'DSC_OC': [], 'HD_OC': [],'ASD_OC': [],'DSC_OD': [], 'HD_OD': [],'ASD_OD': []}

    for _ in range(args.running_times):
        print('-'*10,'{}'.format(args.algorithm),'-'*10)
        # setup hparams
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
        keys = ["config.yaml"] + args.configs
        keys = [open(key, encoding="utf8") for key in keys]
        from sconf import Config
        hparams = Config(*keys, default=hparams)
        hparams.argv_update(left_argv)
        
        # if  the gpu is available  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cuda = torch.cuda.is_available()
        
        
        # save code and write the .ymal file
        save_code(args, hparams, local_path, day, trial_name)
        # seed initialization for good reproduction
        seed_initialization(args)
        
        # set the data augmentation for training (tr) and testing (ts) datalaoder
        # test data is no random scale crop
        composed_transforms_tr,composed_transforms_ts = transforms.Compose([
            tr.Resize(256),
            tr.RandomScaleCrop(256),
            tr.Normalize_tf(),
            tr.ToTensor()
        ]), transforms.Compose([
            tr.Resize(256),
            tr.Normalize_tf(),
            tr.ToTensor()
        ])


        # construct multi-domain training loader
        dataset_list = {}
        for index,i in enumerate(args.datasetTrain):
            domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='train', splitid=[i],
                                                                 transform=composed_transforms_tr)
            dataset_list['site'+str(i)] = domain
        # construct single testing loader, we only use the testing set of each domain for evaluation
        domain_val = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                           transform=composed_transforms_ts,state='prediction',label=args.label)
        val_loader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # construct model
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        if args.label == 'OC' or args.label == 'OD':
            n_classes = 1
            print('-', 'class: 1 - {}'.format(args.label), '-')
        else:
            n_classes = 2
            print('-','class: 2','-')
        per_domain_batch = int(args.batch_size / len(args.datasetTrain))
        # two stage denotes the second phase of a coarse-to-fine segmentation strategy, OD is segmented first, so the two stage is False here
        model_OD = algorithm_class(n_channels=3,
                                 n_classes=n_classes,
                                 hparams=hparams,
                                 device=device,
                                 two_step=False,
                                 per_domain_batch=per_domain_batch, source_domain_num=len(args.datasetTrain)
                                 )
        model_OD.to(device)
        shape_regularization_OD = shape_networks.ShapeVariationalDist_x(hparams, device, n_classes=1,
                                                        number_source_domain=len(args.datasetTrain),batch_size=per_domain_batch)
        shape_regularization_OD.to(device)
        # OC is segmented at the second phase based on OD results, so the two stage is True here
        model_OC = algorithm_class(n_channels=3,
                                   n_classes=n_classes,
                                   hparams=hparams,
                                   device=device,
                                   two_step=True,
                                   per_domain_batch=per_domain_batch, source_domain_num=len(args.datasetTrain)
                                   )

        model_OC.to(device)

        shape_regularization_OC = shape_networks.ShapeVariationalDist_x(hparams, device,  n_classes=1, number_source_domain=len(args.datasetTrain),batch_size=per_domain_batch)
        shape_regularization_OC.to(device)

        start_epoch = 0
        start_iteration = 0

        # optimizer initialization
        optim_od,optim_shape_od = torch.optim.Adam(
            model_OD.parameters(),
            lr=args.lr_od,
            betas=(0.9, 0.99)
        ),torch.optim.Adam(
            shape_regularization_OD.parameters(),
            lr=args.lr_od_shape,
            betas=(0.9, 0.99)
        )

        optim_oc, optim_shape_oc = torch.optim.Adam(
            model_OC.parameters(),
            lr=args.lr_oc,
            betas=(0.9, 0.99)
        ), torch.optim.Adam(
            shape_regularization_OC.parameters(),
            lr=args.lr_oc_shape,
            betas=(0.9, 0.99)
        )

        trainer = Trainer.Trainer(
            algo=args.algorithm,
            cuda=cuda,
            hparams=hparams,
            args=args,
            model=model_OD,
            model_shape=shape_regularization_OD,
            model_oc=model_OC,
            model_shape_oc=shape_regularization_OC,
            lr=args.lr_od,
            lr_shape=args.lr_od_shape,
            lr_oc=args.lr_oc,
            lr_shape_oc=args.lr_oc_shape,
            lr_decrease_rate=args.lr_decrease_rate,
            train_loader=dataset_list,
            val_loader=val_loader,
            optim=optim_od,
            optim_shape=optim_shape_od,
            optim_oc=optim_oc,
            optim_shape_oc=optim_shape_oc,
            out=args.out,
            max_epoch=args.max_epoch,
            stop_epoch=args.stop_epoch,
            interval_validate=args.interval_validate,
            batch_size=args.batch_size,
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        # training loop
        best_value_list = trainer.train()
        index = 0
        for key in result.keys():
            result[key].append(best_value_list[index])
            index += 1
    for key in result.keys():
        print('Domain:{}'.format(args.datasetTest[0]),'{}:{}+_{}'.format(key,np.mean(result[key]),np.std(result[key],ddof=1)))

if __name__ == '__main__':
    # target domain: 1 - 4



    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')  #

    parser.add_argument('--datasetTrain', nargs='+', type=int, default=[1,2,4],
                        help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=[3],
                        help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--batch-size', type=int, default=9, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max-epoch', type=int, default=200, help='max epoch')
    parser.add_argument('--stop-epoch', type=int, default=200, help='stop epoch')
    parser.add_argument('--interval-validate', type=int, default=1, help='interval epoch number to valide the model')
    parser.add_argument('--lr_od', type=float, default=5e-4, help='learning rate for OD', )
    parser.add_argument('--lr_od_shape', type=float, default=5e-4, help='learning rate for OD', )
    parser.add_argument('--lr_oc', type=float, default=5e-4, help='learning rate for OC', )
    parser.add_argument('--lr_oc_shape', type=float, default=5e-4, help='learning rate for OC', )
    parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
    parser.add_argument('--lam', type=float, default=0.9, help='momentum of memory update', )
    parser.add_argument('--data-dir', default='/*/*/dataset/', help='data root path')
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+', )
    parser.add_argument("--dataset", type=str, default="fundus")  #
    parser.add_argument("--algorithm", type=str, default='WT_PSE')  #
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--label", type=str, default='OD',help='determine the objective of validation for best model choice')  #
    parser.add_argument("--seed", type=int, default=1)  #
    parser.add_argument("--running_times", type=int, default=3,help='multiple running for compute the std and mean of final results')  #

    args = parser.parse_args()
    args, left_argv = parser.parse_known_args()
    day = datetime.today().date()
    name = 'Unet2D_{}'.format(args.label)
    day = '{}'.format(day)  #

    main(args,left_argv,name,day)
