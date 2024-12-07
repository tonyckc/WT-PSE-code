from datetime import datetime
import os
import os.path as osp
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import savgol_filter
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
import algorithms
import shape_networks
import random
import numpy as np
import hparams_registry
import fundus_dataloader as DL
import custom_transforms as tr
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import shutil
import glob
from datetime import datetime
import cv2
from utils import *
from metrics import *
import scipy.io as sci
from algorithms import Unet_nips2023

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
Target = 3
# dataset
local_path = '/*/*/founds'
# model file
model_file = '*/checkpoint_**.pth.tar'





def main(target_domain,trial_name,day):
    training_domains = [1,2,3,4]
    training_domains.remove(target_domain)


    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path') #

    parser.add_argument('--datasetTrain', nargs='+', type=int, default=training_domains, help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=[target_domain], help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--batch-size', type=int, default=9, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max-epoch', type=int, default=201, help='max epoch')
    parser.add_argument('--stop-epoch', type=int, default=201, help='stop epoch')
    parser.add_argument('--interval-validate', type=int, default=1, help='interval epoch number to valide the model')
    parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
    parser.add_argument('--lam', type=float, default=0.9, help='momentum of memory update',)
    parser.add_argument('--data-dir', default='/home/ckc/dataset/', help='data root path')
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+',)
    parser.add_argument("--dataset", type=str, default="fundus")  # PACS
    parser.add_argument("--algorithm", type=str, default='Unet_nips2023')  # ICML22_MMD
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--label", type=str, default='OD')  #
    parser.add_argument("--seed", type=int, default=1)  #


    args = parser.parse_args()
    args, left_argv = parser.parse_known_args()


    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    from sconf import Config
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"



    now = datetime.now()
    args.out = osp.join(local_path, day,trial_name, 'test_domain_'+str(args.datasetTest[0])+ now.strftime('_%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)

    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    seed_initialization(args)


    composed_transforms_ts = transforms.Compose([
        tr.Resize(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])


    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                                   transform=composed_transforms_ts, state='prediction',
                                                   label=args.label)
    val_loader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)


    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    if args.label == 'OC' or args.label == 'OD':
        n_classes = 1
        print('-', 'class: 1 - {}'.format(args.label), '-')
    else:
        n_classes = 2
        print('-','class: 2','-')
    per_domain_batch = 3

    checkpoint = torch.load(model_file)
    model = algorithm_class(n_channels=3,
                             n_classes=n_classes,
                             hparams=hparams,
                             device=device,
                             two_step=False,
                             per_domain_batch=per_domain_batch, source_domain_num=3
                             )
    model.to(device)
    # od segmentation network
    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


    # load od shape
    od_shape = shape_networks.ShapeVariationalDist_x(hparams, device, n_classes=1,
                                                        number_source_domain=len(args.datasetTrain),batch_size=per_domain_batch)
    od_shape.to(device)

    pretrained_dict = checkpoint['model_shape']
    model_dict = od_shape.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    od_shape.load_state_dict(model_dict)





    model_OC = algorithm_class(n_channels=3,
                               n_classes=n_classes,
                               hparams=hparams,
                               device=device,
                               two_step=True,
                               per_domain_batch=per_domain_batch, source_domain_num=3
                               )

    model_OC.to(device)

    pretrained_dict = checkpoint['model_oc']
    model_dict = model_OC.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model_OC.load_state_dict(model_dict)




    oc_shape = shape_networks.ShapeVariationalDist_x(hparams, device,  n_classes=1, number_source_domain=len(args.datasetTrain),batch_size=per_domain_batch)
    oc_shape.to(device)

    pretrained_dict = checkpoint['model_oc_shape']
    model_dict = oc_shape.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    oc_shape.load_state_dict(model_dict)
    model.eval()
    od_shape.eval()
    model_OC.eval()
    oc_shape.eval()

    batch = 2
    num_name_global = 0
    for batch_idx, (sample) in tqdm(enumerate(val_loader), total=len(val_loader), ncols=80, leave=False):
        num_name = 0
        image = sample['image']
        label = sample['original_od']

        label_oc = sample['original_oc']
        data = image.cuda()
        target_map = label.cuda()
        target_map_oc = label_oc.cuda()


        with torch.no_grad():
            predictions, shape_sampling_result_gm = model.predict(od_shape, data)

            od_pred = (torch.sigmoid(predictions) > 0.75).float().detach().float()  # N*C*W*H

            data += 1  # 0-2
            image_roi = data * od_pred
            image_roi -= 1  # -1 - 1

            two_stage_input = image_roi
            x_xs = torch.stack((image_roi, two_stage_input), 0)
            predictions_oc, shape_sampling_result_gm_oc = model_OC.predict(oc_shape, x_xs)
            predictions_oc = predictions_oc * od_pred


        predictions = torch.nn.functional.interpolate(predictions, size=(label.size()[2], label.size()[3]),
                                                      mode="bilinear")
        predictions_oc = torch.nn.functional.interpolate(predictions_oc, size=(label.size()[2], label.size()[3]),
                                                         mode="bilinear")
        data = sample['image']
        data = torch.nn.functional.interpolate(data, size=(label.size()[2], label.size()[3]), mode="bilinear")

        target_numpy = target_map.data.cpu()
        target_numpy_oc = target_map_oc.data.cpu()
        imgs = data.data.cpu()

        for i in range(predictions.shape[0]):

            num_name_global += 1
            prediction_post = postprocessing(predictions[i], dataset='test', label=args.label)
            prediction_post_oc = postprocessing(predictions_oc[i], dataset='test', label=args.label)
            mask = np.zeros((prediction_post.shape[1], prediction_post.shape[2], 2))
            prediction_od = np.squeeze(prediction_post)
            prediction_oc = np.squeeze(prediction_post_oc)
            mask[prediction_od == 1] = [0, 1]
            mask[prediction_oc == 1] = [1, 1]
            mask = mask.transpose(2,0,1)
            prediction_post = mask

            target = np.zeros((prediction_post.shape[1], prediction_post.shape[2], 2))
            mask_od = np.squeeze(target_numpy[i])
            mask_oc = np.squeeze(target_numpy_oc[i])
            target[mask_od == 1] = [0, 1]
            target[mask_oc == 1] = [1, 1]
            target = target.transpose(2, 0, 1)
            target = torch.tensor(target)

            if num_name == batch:
                show = True
            else:
                show = False
            num_name += 1
            for img, lt, lp in zip([imgs[i]], [target], [prediction_post]):
                img, lt = untransform(img, lt)
                save_per_img(img.numpy().transpose(1, 2, 0),
                             args.out,
                             str(num_name_global),
                             lp, lt, mask_path=None, ext="bmp",batch=True)



if __name__ == '__main__':
    target_domain = Target
    name = 'our_Unet2D_OD'
    day = 'visulize'
    main(target_domain,name,day)
