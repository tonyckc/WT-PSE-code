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
from train_process import Trainer
import hparams_registry
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.Unet import *
from tqdm import tqdm
from algorithms import Unet_nips2023
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Target = 4
local_path = '/hdd/ckc/founds_training'
import shutil
import glob

def save_file(target_dir):
    for ext in ('py','pyproj','sln'):
        for fn in glob.glob('*.'+ext):
            shutil.copy2(fn,target_dir)
        if os.path.isdir('src'):
            for fn in glob.glob(os.path.join('src','*.'+ext)):
                shutil.copy2(fn,target_dir)


def copy_allfiles(src,dest):
#src:原文件夹；dest:目标文件夹
  src_files = os.listdir(src)
  for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)

def centroids_init(model, data_dir, datasetTrain, composed_transforms):
    centroids = torch.zeros(3, 304, 64, 64).cuda() # 3 means the number of source domains
    model.eval()

    # Calculate initial centroids only on training data.
    with torch.set_grad_enabled(False):
        count = 0
        # tranverse each training source domain
        for index in datasetTrain:
            domain = DL.FundusSegmentation(base_dir=data_dir, phase='train', splitid=[index],
                                           transform=composed_transforms)
            dataloder = DataLoader(domain, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

            for id, sample in tqdm(enumerate(dataloder)):
                sample=sample[0]
                inputs = sample['image'].cuda()
                features = model(inputs, extract_feature=True)

                # Calculate the sum features from the same domain
                centroids[count:count+1] += features

            # Average summed features with class count
            centroids[count] /= torch.tensor(len(dataloder)).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
            count += 1
    # Calculate the mean features for each domain
    ave = torch.mean(torch.mean(centroids, 3, True), 2, True) # size [3, 304]
    return ave.expand_as(centroids).contiguous()  # size [3, 304, 64, 64]

def main(target_domain,trial_name,day):
    training_domains = [1,2,3,4]
    training_domains.remove(target_domain)
    seed = [1,1,1]
    lr_list = [5e-4]
    result = {'DSC_OC': [], 'HD_OC': [],'ASD_OC': [],'DSC_OD': [], 'HD_OD': [],'ASD_OD': []}
    #for seed_value in seed:
    for lr_ in lr_list:
      for seed_value in seed:
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
        parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
        parser.add_argument('--resume', default=None, help='checkpoint path') #

        parser.add_argument('--datasetTrain', nargs='+', type=int, default=training_domains, help='train folder id contain images ROIs to train range from [1,2,3,4]')
        parser.add_argument('--datasetTest', nargs='+', type=int, default=[target_domain], help='test folder id contain images ROIs to test one of [1,2,3,4]')
        parser.add_argument('--batch-size', type=int, default=9, help='batch size for training the model')
        parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
        parser.add_argument('--max-epoch', type=int, default=200, help='max epoch')
        parser.add_argument('--stop-epoch', type=int, default=200, help='stop epoch')
        parser.add_argument('--interval-validate', type=int, default=1, help='interval epoch number to valide the model')
        parser.add_argument('--lr', type=float, default=lr_, help='learning rate',)
        parser.add_argument('--lr_shape', type=float, default=lr_, help='learning rate', )
        parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
        parser.add_argument('--lam', type=float, default=0.9, help='momentum of memory update',)
        parser.add_argument('--data-dir', default='/hdd/ckc/fundus/', help='data root path')
        #parser.add_argument('--pretrained-model', default='./pretrained-weight/test4-epoch40.pth.tar', help='pretrained model of FCN16s',)
        parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+',)
        parser.add_argument("--dataset", type=str, default="fundus")  # PACS
        parser.add_argument("--algorithm", type=str, default='Unet_nips2023')  # ICML22_MMD
        parser.add_argument("configs", nargs="*")
        parser.add_argument("--label", type=str, default='oc and od')  # ICML22_MMD

        parser.add_argument("--label_oc", type=float, default=1)  # ICML22_MMD
        parser.add_argument("--label_od", type=float, default=1)  # ICML22_MMD

        args = parser.parse_args()
        args, left_argv = parser.parse_known_args()

        # setup hparams
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)


        # print(hparams['local_loss'])
        #  exit()
        # hparams['shape_weight'] = shape_embedding_coff
        keys = ["config.yaml"] + args.configs
        keys = [open(key, encoding="utf8") for key in keys]
        from sconf import Config
        hparams = Config(*keys, default=hparams)
        hparams.argv_update(left_argv)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"


        now = datetime.now()
        args.out = osp.join(local_path, day,trial_name, 'test'+str(args.datasetTest[0])+'_seed_'+str(seed_value)+'_lr_'+str(lr_)+'_attention_'+str(hparams['shape_attention_coeffient'])+'_t2_{}_domain_{}_instance_{}_wt_{}_'.format(hparams['domain_wt_gm'],hparams['instance_wt_sc'],float((hparams['whitening'])), now.strftime('%Y%m%d_%H%M%S.%f')))
        os.makedirs(args.out)
        code_dir = os.path.join(args.out, 'code')
        os.makedirs(code_dir)
        save_file(code_dir)
        copy_allfiles('./train_process', code_dir)
        copy_allfiles('./dataloaders', code_dir)
        copy_allfiles('./networks', code_dir)

        with open(osp.join(args.out, 'config.yaml'), 'w') as f:
            yaml.safe_dump(args.__dict__, f, default_flow_style=False)


        cuda = torch.cuda.is_available()
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 1. dataset
        composed_transforms_tr = transforms.Compose([
            tr.Resize(256),
            tr.RandomScaleCrop(256),
            # tr.RandomCrop(512),
            # tr.RandomRotate(),
            # tr.RandomFlip(),
            # tr.elastic_transform(),
            # tr.add_salt_pepper_noise(),
            # tr.adjust_light(),
            # tr.eraser(),
            tr.Normalize_tf(label=args.label),
            tr.ToTensor()
        ])

        composed_transforms_ts = transforms.Compose([
            tr.Resize(256),
            tr.Normalize_tf(label=args.label),
            tr.ToTensor()
        ])


        # construct multi-domain loader
        dataset_list = {}

        for index,i in enumerate(args.datasetTrain):

            domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='train', splitid=[i],
                                                                 transform=composed_transforms_tr,label_oc=args.label_oc,label_od=args.label_od)
            #train_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

            dataset_list['site'+str(i)] = domain


        domain_val = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                           transform=composed_transforms_ts,state='prediction',label=args.label)
        val_loader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # 2. model
        #model = DeepLab(num_classes=2, num_domain=3, backbone='mobilenet', output_stride=args.out_stride, lam=args.lam).cuda()
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        if args.label == 'OC' or args.label == 'OD':
            n_classes = 1
            print('-', 'class: 1 - {}'.format(args.label), '-')
        else:
            n_classes = 2
            print('-','class: 2','-')
        per_domain_batch = 3
        model = algorithm_class(n_channels=3,
                                 n_classes=n_classes,
                                 hparams=hparams,
                                 device=device,
                                 two_step=False,
                                 per_domain_batch=3, source_domain_num=3
                                 )
        model.to(device)
        net_gm_learn_x = shape_networks.ShapeVariationalDist_x(hparams, device, 3, True, 1, wt=hparams['whitening'],
                                                        prior=False,
                                                        number_source_domain=3,batch_size=per_domain_batch)
        net_gm_learn_x.to(device)


        print('parameter numer:', sum([p.numel() for p in model.parameters()]))

        # load weights
        if args.resume:
            print('-'*10,'loading pretrained model','-'*10)
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            #print('Before ', model.centroids.data)
            model.centroids.data = centroids_init(model, args.data_dir, args.datasetTrain, composed_transforms_ts)
            #print('Before ', model.centroids.data)
            # model.freeze_para()

        start_epoch = 0
        start_iteration = 0

        # 3. optimizer
        optim = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99)
        )
        optim_shape = torch.optim.Adam(
            net_gm_learn_x.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99)
        )


        trainer = Trainer.Trainer(
            cuda=cuda,
            hparams=hparams,
            args = args,
            model=model,
            model_shape=net_gm_learn_x,
            lr=args.lr,
            lr_shape=args.lr_shape,
            lr_decrease_rate=args.lr_decrease_rate,
            train_loader=dataset_list,
            val_loader=val_loader,
            optim=optim,
            optim_shape=optim_shape,
            out=args.out,
            max_epoch=args.max_epoch,
            stop_epoch=args.stop_epoch,
            interval_validate=args.interval_validate,
            batch_size=args.batch_size,
        )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        best_value_list = trainer.train()
        index = 0
        for key in result.keys():
            result[key].append(best_value_list[index])
            index += 1
    for key in result.keys():
        print('Domain:{}'.format(target_domain),'{}:{}+_{}'.format(key,np.mean(result[key]),np.std(result[key],ddof=1)))
if __name__ == '__main__':
    target_domain = Target
    day = datetime.today().date()
    name = 'our_Unet2D_OD_and_OC'
    day = 'revision_OC_OD_Joint_{}_P_0.3_wasserstein_ERM'.format(day)#'ablation_0514_P_0.5'#

    main(target_domain,name,day)
