from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.nn.functional as F
import pytz
from tensorboardX import SummaryWriter
import random
import tqdm
import socket
from metrics import *
from utils import *
from utils import joint_val_image, postprocessing, save_per_img
from medpy.metric import binary

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
softmax = torch.nn.Softmax(-1)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_batch(dataset, batch_size):
    x_list = []
    spinal_mask_list = []
    gm_mask_list = []
    for _ in range(batch_size):
        domain = dataset[0]
        x = domain[0]['image']
        label = domain[0]['label_od']
        domain_code = domain[0]['label_oc']
        x_list.append(x)
        spinal_mask_list.append(label)
        gm_mask_list.append(domain_code)
    return x_list, spinal_mask_list, gm_mask_list
    # return torch.stack(x_list, dim=0).cuda(), torch.stack(spinal_mask_list, dim=0).cuda(), torch.stack(gm_mask_list,dim=0).cuda()


def get_multi_batch(dataset_list, batch_size):
    x_list = []
    spinal_mask_list = []
    gm_mask_list = []
    for dataset in dataset_list:
        # print('ckc')
        x, spinal_cord_mask, gm_mask = get_batch(dataset, batch_size)
        x_list.extend(x)
        spinal_mask_list.extend(spinal_cord_mask)
        gm_mask_list.extend(gm_mask)
    return torch.stack(x_list, dim=0).cuda(), torch.stack(spinal_mask_list, dim=0).cuda(), torch.stack(gm_mask_list,dim=0).cuda()


class Trainer(object):

    def __init__(self, algo,cuda, hparams, args, model, model_shape, model_oc,
                 model_shape_oc, lr, lr_shape, lr_oc, lr_shape_oc, val_loader, train_loader, out, max_epoch, optim,
                 optim_shape, optim_oc,
                 optim_shape_oc, stop_epoch=None,
                 lr_decrease_rate=0.1, interval_validate=None, batch_size=8):
        self.algorithm = algo
        self.cuda = cuda
        self.hparams = hparams
        self.model = model
        self.args = args
        self.model_shape = model_shape

        self.model_oc = model_oc
        self.model_shape_oc = model_shape_oc

        self.optim = optim
        self.optim_shape = optim_shape

        self.optim_oc = optim_oc
        self.optim_shape_oc = optim_shape_oc

        self.lr = lr
        self.lr_shape = lr_shape

        self.lr_oc = lr_oc
        self.lr_shape_oc = lr_shape_oc

        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0

        self.best_mean_dice = 0.0
        self.best_epoch = -1
        self.dataset = 'test'
        self.validation_objective = args.label
    def validate(self):
        training = self.model.training
        self.model.eval()
        self.model_shape.eval()
        self.model_oc.eval()
        self.model_shape_oc.eval()
        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        val_disc_hd = 0
        val_cup_hd = 0
        val_disc_asd = 0
        val_cup_asd = 0
        metrics = []
        total_num = 0
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):

                image = sample['image']
                label = sample['original_od']

                label_oc = sample['original_oc']


                data = image.cuda()
                target_map = label.cuda()
                target_map_oc = label_oc.cuda()

                with torch.no_grad():
                    predictions,shape_sampling_result_gm = self.model.predict(self.model_shape,data)



                    od_pred = (torch.sigmoid(predictions) > 0.75).float().detach().float()  # N*C*W*H

                    data += 1  # 0-2
                    image_roi = data * od_pred
                    image_roi -= 1  # -1 - 1

                    two_stage_input = image_roi
                    x_xs = torch.stack((image_roi, two_stage_input), 0)
                    predictions_oc, shape_sampling_result_gm_oc = self.model_oc.predict(self.model_shape_oc, x_xs)

                    predictions_oc = predictions_oc * od_pred

                if self.plot_show == 0:

                    grid_image = make_grid(
                        data[0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('test/input', grid_image, self.iteration)

                    grid_image = make_grid(
                        image_roi[0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('test/input_roi', grid_image, self.iteration)

                    grid_image = make_grid(
                        (torch.sigmoid(predictions_oc[0, 0, ...]) > 0.75).float().clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('test/predict', grid_image, self.iteration)


                loss_data = 0
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                predictions = torch.nn.functional.interpolate(predictions, size=(label.size()[2], label.size()[3]),
                                                             mode="bilinear")
                predictions_oc = torch.nn.functional.interpolate(predictions_oc, size=(label.size()[2], label.size()[3]),
                                                              mode="bilinear")

                
                target_numpy = target_map.data.cpu()
                target_numpy_oc = target_map_oc.data.cpu()
                for i in range(predictions.shape[0]):
                    prediction_post = postprocessing(predictions[i], dataset=self.dataset,label=self.args.label)
                    _, dice_disc = dice_coeff_2label(prediction_post, target_map[i],self.args.label)

                    prediction_post_oc = postprocessing(predictions_oc[i], dataset=self.dataset, label=self.args.label)
                    dice_cup, _ = dice_coeff_2label(prediction_post_oc, target_map_oc[i], self.args.label)

                    if np.sum(prediction_post_oc[0, ...]) < 1e-4:
                        hd_OC = 100
                        asd_OC = 100

                    else:
                        hd_OC = binary.hd95(np.asarray(prediction_post_oc[0, ...], dtype=np.bool_),
                                            np.asarray(target_numpy_oc[i,0, ...], dtype=np.bool_))
                        asd_OC = binary.asd(np.asarray(prediction_post_oc[0, ...], dtype=np.bool_),
                                            np.asarray(target_numpy_oc[i, 0, ...], dtype=np.bool_))

                    if np.sum(prediction_post[0, ...]) < 1e-4:
                        hd_OD = 100
                        asd_OD = 100

                    else:
                        hd_OD = binary.hd95(np.asarray(prediction_post[0, ...], dtype=np.bool_),
                                            np.asarray(target_numpy[i, 0, ...], dtype=np.bool_))
                        asd_OD = binary.asd(np.asarray(prediction_post[0, ...], dtype=np.bool_),
                                            np.asarray(target_numpy[i, 0, ...], dtype=np.bool_))


                    val_cup_dice += dice_cup
                    val_disc_dice += dice_disc
                    val_cup_hd += hd_OC
                    val_disc_hd += hd_OD
                    val_cup_asd += asd_OC
                    val_disc_asd += asd_OD

                    total_num +=1
            val_loss /= len(self.val_loader)
            val_cup_dice /= total_num
            val_disc_dice /= total_num
            val_disc_hd /= total_num
            val_cup_hd /= total_num
            val_disc_asd /= total_num
            val_cup_asd /= total_num

            metrics.append((val_loss, val_cup_dice, val_disc_dice))

            self.writer.add_scalar('val_data/loss', val_loss, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_hd', val_cup_hd, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_hd', val_disc_hd, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_asd', val_cup_asd, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_asd', val_disc_asd, self.epoch * (len(self.train_loader)))

            # validation objective
            if self.validation_objective == 'OD':
                mean_dice = (val_disc_dice)
            elif self.validation_objective  == 'OC':
                mean_dice = (val_cup_dice)
            else:
                mean_dice =  (val_cup_dice + val_disc_dice) / 2
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice
                file = open(os.path.join(self.out,'score.txt'),'a')
                file.write('cd:{} dd:{} c_hd:{} d_hd:{} c_asd:{} d_asd:{}\n'.format(val_cup_dice,val_disc_dice,val_cup_hd,val_disc_hd,val_cup_asd,val_disc_asd))
                file.close()
                torch.save({
                    'model': self.model.state_dict(),
                    'model_shape': self.model_shape.state_dict(),
                    'model_oc': self.model_oc.state_dict(),
                    'model_oc_shape': self.model_shape_oc.state_dict(),

                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
                if training:
                    self.model.train()
                return 1, val_cup_dice,val_cup_hd,val_cup_asd,val_disc_dice,val_disc_hd,val_disc_asd

            else:
                if (self.epoch + 1) % 300 == 0:
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': self.model.state_dict(),
                        'learning_rate_gen': get_lr(self.optim),
                        'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))


            if training:
                self.model.train()
                self.model_shape.train()
                self.model_oc.train()
                self.model_shape_oc.train()
            return 0, 0, 0, 0, 0,0,0


    def validate_joint_shape_reg(self):
        training = self.model.training
        self.model.eval()
        self.model_oc.eval()
        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        val_disc_hd = 0
        val_cup_hd = 0
        val_disc_asd = 0
        val_cup_asd = 0
        metrics = []
        loss_cls = 0
        total_num = 0
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):

                image = sample['image']
                label = sample['original_od']

                label_oc = sample['original_oc']
                #domain_code = sample['dc']

                data = image.cuda()
                target_map = label.cuda()
                target_map_oc = label_oc.cuda()
                #domain_code = domain_code.cuda()

                with torch.no_grad():
                    predictions,shape_sampling_result_gm = self.model.predict(None,data)




                    od_pred = (torch.sigmoid(predictions) > 0.75).float().detach().float()  # N*C*W*H


                    data += 1  # 0-2
                    image_roi = data * od_pred
                    image_roi -= 1  # -1 - 1

                    two_stage_input = image_roi
                    x_xs = torch.stack((image_roi, two_stage_input), 0)
                    predictions_oc, shape_sampling_result_gm_oc = self.model_oc.predict(None, x_xs)

                    predictions_oc = predictions_oc * od_pred

                if self.plot_show == 0:

                    grid_image = make_grid(
                        data[0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('test/input', grid_image, self.iteration)

                    grid_image = make_grid(
                        image_roi[0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('test/input_roi', grid_image, self.iteration)

                    grid_image = make_grid(
                        (torch.sigmoid(predictions_oc[0, 0, ...]) > 0.75).float().clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('test/predict', grid_image, self.iteration)

                loss_seg = 0
                loss_cls = 0
                loss_data = 0
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                predictions = torch.nn.functional.interpolate(predictions, size=(label.size()[2], label.size()[3]),
                                                             mode="bilinear")
                predictions_oc = torch.nn.functional.interpolate(predictions_oc, size=(label.size()[2], label.size()[3]),
                                                              mode="bilinear")


                target_numpy = target_map.data.cpu()
                target_numpy_oc = target_map_oc.data.cpu()
                hd_OC = 100
                asd_OC = 100
                hd_OD = 100
                asd_OD = 100
                for i in range(predictions.shape[0]):
                    prediction_post = postprocessing(predictions[i], dataset=self.dataset,label=self.args.label)
                    _, dice_disc = dice_coeff_2label(prediction_post, target_map[i],self.args.label)

                    prediction_post_oc = postprocessing(predictions_oc[i], dataset=self.dataset, label=self.args.label)
                    dice_cup, _ = dice_coeff_2label(prediction_post_oc, target_map_oc[i], self.args.label)

                    if np.sum(prediction_post_oc[0, ...]) < 1e-4:
                        hd_OC = 100
                        asd_OC = 100

                    else:
                        hd_OC = binary.hd95(np.asarray(prediction_post_oc[0, ...], dtype=np.bool_),
                                            np.asarray(target_numpy_oc[i,0, ...], dtype=np.bool_))
                        asd_OC = binary.asd(np.asarray(prediction_post_oc[0, ...], dtype=np.bool_),
                                            np.asarray(target_numpy_oc[i, 0, ...], dtype=np.bool_))

                    if np.sum(prediction_post[0, ...]) < 1e-4:
                        hd_OD = 100
                        asd_OD = 100

                    else:
                        hd_OD = binary.hd95(np.asarray(prediction_post[0, ...], dtype=np.bool_),
                                            np.asarray(target_numpy[i, 0, ...], dtype=np.bool_))
                        asd_OD = binary.asd(np.asarray(prediction_post[0, ...], dtype=np.bool_),
                                            np.asarray(target_numpy[i, 0, ...], dtype=np.bool_))


                    val_cup_dice += dice_cup
                    val_disc_dice += dice_disc
                    val_cup_hd += hd_OC
                    val_disc_hd += hd_OD
                    val_cup_asd += asd_OC
                    val_disc_asd += asd_OD

                    total_num +=1
            val_loss /= len(self.val_loader)
            val_cup_dice /= total_num
            val_disc_dice /= total_num
            val_disc_hd /= total_num
            val_cup_hd /= total_num
            val_disc_asd /= total_num
            val_cup_asd /= total_num

            metrics.append((val_loss, val_cup_dice, val_disc_dice))

            self.writer.add_scalar('val_data/loss', val_loss, self.epoch * (len(self.train_loader)))
            #self.writer.add_scalar('val_data/loss_cls', loss_cls.data.item(), self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_hd', val_cup_hd, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_hd', val_disc_hd, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_asd', val_cup_asd, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_asd', val_disc_asd, self.epoch * (len(self.train_loader)))


            mean_dice = (val_cup_dice)
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice
                file = open(os.path.join(self.out,'score.txt'),'a')
                file.write('cd:{} dd:{} c_hd:{} d_hd:{} c_asd:{} d_asd:{}\n'.format(val_cup_dice,val_disc_dice,val_cup_hd,val_disc_hd,val_cup_asd,val_disc_asd))
                file.close()
                torch.save({
                    'model': self.model.state_dict(),
                    'model_shape': self.model_shape.state_dict(),
                    'model_oc': self.model_oc.state_dict(),
                    'model_oc_shape': self.model_shape_oc.state_dict(),

                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
                if training:
                    self.model.train()
                return 1, val_cup_dice,val_cup_hd,val_cup_asd,val_disc_dice,val_disc_hd,val_disc_asd

            else:
                if (self.epoch + 1) % 300 == 0:
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': self.model.state_dict(),
                        'learning_rate_gen': get_lr(self.optim),
                        'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))


            if training:
                self.model.train()
                self.model_shape.train()
                self.model_oc.train()
                self.model_shape_oc.train()
            return 0, 0, 0, 0, 0,0,0
    def train_epoch_joint_shape_reg(self):
        self.model.train()
        self.model_oc.train()
        self.running_seg_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        self.running_cls_loss = 0
        self.running_kl = 0.0
        self.running_instance_wt = 0.0
        self.running_domain_wt = 0.0

        self.running_instance_wt_shape = 0.0
        self.running_instance_wt_ii = 0.0
        self.running_instance_wt_ij = 0.0

        self.running_domain_wt_shape = 0.0

        start_time = timeit.default_timer()

        self.running_seg_loss_oc = 0.0
        self.running_total_loss_oc = 0.0
        self.running_cup_dice_tr_oc = 0.0
        self.running_disc_dice_tr_oc = 0.0
        self.running_cls_loss_oc = 0
        self.running_kl_oc = 0.0
        self.running_instance_wt_oc = 0.0

        self.running_domain_wt_oc = 0.0
        self.running_instance_wt_shape_oc = 0.0
        self.running_domain_wt_shape_oc = 0.0

        for batch_idx in range(self.iter_per_epoch):
            iteration = batch_idx + self.epoch * self.iter_per_epoch
            self.iteration = iteration

            assert self.model.training
            self.optim.zero_grad()
            self.model.zero_grad()
            random.shuffle(self.source_domain_datasets)

            # print('ckc')
            image, target_od, target_oc = get_multi_batch(self.source_domain_datasets,
                                                          self.per_domain_batch)

            # if self.args.label == 'OC' or self.args.label == 'OD':
            #    input_mask = target_map
            # else:
            #    input_mask = shape_mask

            output, shape_prior_gm, shape_prior_gm_mask, instance_wt_gm, domain_wt_gm = self.model.update(image,
                                                                                                          target_od,
                                                                                                          step=self.epoch,
                                                                                                          plot_show=self.plot_show,
                                                                                                          two_stage_inputs=image,
                                                                                                          sp_mask=target_od,
                                                                                                          two_step=True)

            # od_weight = torch.tensor(1.) / torch.mean(input_mask.detach()) * self.hparams['p_weight1']
            # if torch.isinf(od_weight) or torch.isnan(od_weight):
            #    od_weight = torch.tensor(1.).cuda()

            loss_seg = bceloss(torch.sigmoid(output),
                               target_od)  # F.binary_cross_entropy_with_logits(output, input_mask,
            # pos_weight=od_weight)
            loss_cls = loss_seg
            loss_ins_wt = instance_wt_gm
            loss_dom_wt = domain_wt_gm

            self.running_seg_loss += loss_seg.item()
            self.running_cls_loss += loss_cls.item()

            if self.hparams['whitening']:
                self.running_instance_wt += loss_ins_wt.item()
                self.running_domain_wt += loss_dom_wt.item()
                loss_data = (loss_seg + loss_ins_wt + loss_dom_wt).data.item()
            else:
                self.running_instance_wt += loss_ins_wt
                self.running_domain_wt += loss_dom_wt
                loss_data = (loss_seg).data.item()

            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss_main = loss_seg + self.hparams['instance_wt_gm'] * loss_ins_wt + self.hparams[
                'domain_wt_gm'] * loss_dom_wt
            loss_main.backward()
            self.optim.step()

            # updata shape network

            od_pred = (torch.sigmoid(output) > 0.75).float().detach().float()  # N*C*W*H

            # oc network
            assert self.model_oc.training

            self.optim_oc.zero_grad()
            self.model_oc.zero_grad()

            image += 1  # 0-2
            image_roi = image * od_pred
            image_roi -= 1  # -1 - 1

            two_stage_input = image_roi
            output_oc, shape_prior_gm_oc, shape_prior_gm_mask_oc, instance_wt_gm_oc, domain_wt_gm_oc = self.model_oc.update(
                image_roi,
                target_oc,
                step=self.epoch,
                plot_show=self.plot_show,
                two_stage_inputs=two_stage_input,
                sp_mask=target_od,
                two_step=True)

            # roi bec loss
            gm_pos_weight = torch.sum(od_pred) / torch.sum(od_pred * target_oc)
            if torch.isinf(gm_pos_weight) or torch.isnan(gm_pos_weight):
                gm_pos_weight = torch.tensor(1.).cuda()

            loss_seg_oc = F.binary_cross_entropy_with_logits(
                output_oc * od_pred,
                target_oc,
                pos_weight=gm_pos_weight)  # bceloss(torch.sigmoid(output_oc * od_pred),
            #        target_oc)

            output_oc = output_oc * od_pred
            loss_ins_wt_oc = instance_wt_gm_oc
            loss_dom_wt_oc = domain_wt_gm_oc

            self.running_seg_loss_oc += loss_seg_oc.item()

            if self.hparams['whitening']:
                self.running_instance_wt_oc += loss_ins_wt_oc.item()
                self.running_domain_wt_oc += loss_dom_wt_oc.item()
                loss_data_oc = (loss_seg_oc + loss_ins_wt_oc + loss_dom_wt_oc).data.item()
            else:
                self.running_instance_wt_oc += loss_ins_wt_oc
                self.running_domain_wt_oc += loss_dom_wt_oc
                loss_data_oc = (loss_seg_oc).data.item()

            if np.isnan(loss_data_oc):
                raise ValueError('loss is nan while training')

            loss_main_oc = loss_seg_oc + self.hparams['instance_wt_gm'] * loss_ins_wt_oc + self.hparams[
                'domain_wt_gm'] * loss_dom_wt_oc
            loss_main_oc.backward()
            self.optim_oc.step()



            # write image log

            if iteration % 30 == 0:
                grid_image = make_grid(
                    image_roi[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/image', grid_image, iteration)

                # why channel = 0 ? OC denotes [1,1], if value =1, the pixel will
                # belongs to the OC
                if self.hparams['whitening']:
                    grid_image = make_grid(
                        shape_prior_gm_oc[0, 0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('train/shape_cup', grid_image, iteration)

                grid_image = make_grid(
                    target_oc[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target_cup', grid_image, iteration)

                grid_image = make_grid(
                    ((torch.sigmoid(output_oc)[0, 0, ...]) > 0.75).float().clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/pre_cup', grid_image, iteration)

                # why channel = 1 ? OD denotes [0,1], if value = 1 the pixel will belongs
                # to the OD
                if self.args.label == None:
                    grid_image = make_grid(
                        target_od[0, 0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('train/target_disc', grid_image, iteration)

                grid_image = make_grid(torch.sigmoid(output)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                if self.args.label == None:
                    self.writer.add_image('train/prediction_cup', grid_image, iteration)
                    grid_image = make_grid(torch.sigmoid(output)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('train/prediction_disc', grid_image, iteration)

            # write loss log
            self.writer.add_scalar('train_gen/loss', loss_data, iteration)

            self.writer.add_scalar('train_gen/loss_seg', loss_seg.data.item(), iteration)
            self.writer.add_scalar('train_gen/loss_seg_oc', loss_seg_oc.data.item(), iteration)
            self.writer.add_scalar('train_gen/loss_cls', loss_cls.data.item(), iteration)

            if self.hparams['whitening']:
                #self.writer.add_scalar('train_shape/loss_kl_oc', loss_kl_oc.data.item(), iteration)

                #self.writer.add_scalar('train_shape/loss_kl', loss_kl.data.item(), iteration)

                self.writer.add_scalar('train_shape/loss_ins_wt', loss_ins_wt.data.item(), iteration)

                self.writer.add_scalar('train_shape/loss_dom_wt', loss_dom_wt.data.item(), iteration)


                self.writer.add_scalar('train_shape/loss_ins_wt_shape_all', loss_ins_wt.data.item(), iteration)
                self.writer.add_scalar('train_shape/loss_dom_wt_shape', loss_dom_wt.data.item(), iteration)

        self.running_seg_loss /= len(self.train_loader)
        self.running_cls_loss /= len(self.train_loader)
        self.running_kl_oc /= len(self.train_loader)
        self.running_domain_wt_oc /= len(self.train_loader)
        self.running_instance_wt_oc /= len(self.train_loader)

        self.running_instance_wt_shape /= len(self.train_loader)
        self.running_domain_wt_shape /= len(self.train_loader)
        self.running_instance_wt_ii /= len(self.train_loader)
        self.running_instance_wt_ij /= len(self.train_loader)
        file = open(os.path.join(self.out, 'ii.txt'), 'a')
        file.write('{}\n'.format(self.running_instance_wt_ii))
        file.close()

        file = open(os.path.join(self.out, 'ij.txt'), 'a')
        file.write('{}\n'.format(self.running_instance_wt_ij))
        file.close()

        file = open(os.path.join(self.out, 'all_ins.txt'), 'a')
        file.write('{}\n'.format(self.running_instance_wt_shape))
        file.close()

        file = open(os.path.join(self.out, 'domain.txt'), 'a')
        file.write('{}\n'.format(self.running_domain_wt_shape))
        file.close()

        stop_time = timeit.default_timer()

        print(
            '\n[Epoch: %d] lr:%f,  Average segLoss: %f, Average kl: %f,Average ins wt: %f,Average dom wt: %f, Execution time: %.5f' %
            (self.epoch, get_lr(self.optim), self.running_seg_loss_oc, self.running_kl_oc, self.running_instance_wt_oc,
             self.running_domain_wt_oc, stop_time - start_time))

    def train_epoch(self):
        self.model.train()
        self.model_shape.train()
        self.model_oc.train()
        self.model_shape_oc.train()
        self.running_seg_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        self.running_cls_loss = 0
        self.running_kd = 0.0
        self.running_instance_wt = 0.0
        self.running_domain_wt = 0.0

        self.running_instance_wt_shape = 0.0
        self.running_instance_wt_ii = 0.0
        self.running_instance_wt_ij = 0.0
        self.running_domain_wt_shape = 0.0

        start_time = timeit.default_timer()

        self.running_seg_loss_oc = 0.0
        self.running_total_loss_oc = 0.0
        self.running_cup_dice_tr_oc = 0.0
        self.running_disc_dice_tr_oc = 0.0
        self.running_cls_loss_oc = 0
        self.running_kd_oc = 0.0
        self.running_instance_wt_oc = 0.0
        self.running_domain_wt_oc = 0.0
        self.running_instance_wt_shape_oc = 0.0
        self.running_domain_wt_shape_oc = 0.0


        for batch_idx in range(self.iter_per_epoch):
            iteration = batch_idx + self.epoch * self.iter_per_epoch
            self.iteration = iteration

            assert self.model.training
            self.optim.zero_grad()
            self.model.zero_grad()
            random.shuffle(self.source_domain_datasets)

            # image data and label
            image, target_od, target_oc = get_multi_batch(self.source_domain_datasets,
                                                                    self.per_domain_batch)

            ###########################################################################################
            ################    step #1 for OD
            ##########################################################################################
            # update segmentation network
            output, shape_prior_od, shape_prior_od_mask, loss_ins_wt, loss_dom_wt = self.model.update(image, target_od,
                                                                                                       step=self.epoch,
                                                                                                       plot_show=self.plot_show,
                                                                                                       two_stage_inputs=image,
                                                                                                       sp_mask=target_od,
                                                                                                       two_step=True)

            # compute the loss
            loss_seg = bceloss(torch.sigmoid(output),target_od)
            self.running_seg_loss += loss_seg.item()

            # wt loss
            if self.hparams['whitening']:
                self.running_instance_wt += loss_ins_wt.item()
                self.running_domain_wt += loss_dom_wt.item()
                loss_data = (loss_seg+loss_ins_wt+loss_dom_wt).data.item()
            else:
                self.running_instance_wt += loss_ins_wt
                self.running_domain_wt += loss_dom_wt
                loss_data = (loss_seg).data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            # total loss = segmentation loss + total wt loss
            loss_main = loss_seg +  self.hparams['instance_wt_gm']*loss_ins_wt + self.hparams['domain_wt_gm']*loss_dom_wt
            # update the segmentation network
            loss_main.backward()
            self.optim.step()
            ###################################################################################################
            ############### step #2 for OD
            ###################################################################################################
            # updata shape network
            if self.hparams['whitening']:
                for _ in range(self.hparams['multi-turn']):
                    assert self.model_shape.training
                    self.optim_shape.zero_grad()
                    self.model_shape.zero_grad()

                    loss_kd, loss_ins_wt_shape, loss_ins_wt_shape_ij,loss_ins_wt_shape_ii,loss_dom_wt_shape = self.model_shape.update(self.model, image, target_od, step=self.epoch,
                                                                                plot_show=self.plot_show,
                                                                                two_stage_inputs=image,
                                                                                two_step=True)

                    # update the shape network
                    loss_shape = loss_kd + self.hparams['instance_wt_gm'] * loss_ins_wt_shape + self.hparams[
                        'domain_wt_gm'] * loss_dom_wt_shape
                    loss_shape.backward()
                    self.optim_shape.step()

                if self.hparams['whitening']:
                    self.running_kd += loss_kd.item()
                    self.running_instance_wt_shape += loss_ins_wt_shape.item()
                    self.running_domain_wt_shape += loss_dom_wt_shape.item()
                    self.running_instance_wt_ii += loss_ins_wt_shape_ii.item()
                    self.running_instance_wt_ij += loss_ins_wt_shape_ij.item()

                else:
                    self.running_kd += loss_kl
                    self.running_instance_wt_shape += loss_ins_wt_shape
                    self.running_domain_wt_shape += loss_dom_wt_shape
            ##############################################################################################################################################


            # coarse-to-fine strategy to segment the OC
            od_pred = (torch.sigmoid(output) > 0.75).float().detach().float()  # N*C*W*H

            # activate oc network
            assert self.model_oc.training
            self.optim_oc.zero_grad()
            self.model_oc.zero_grad()
            # use od prediction to get od ROI image
            # for numberical stablization
            image += 1  # 0-2
            image_roi = image * od_pred
            image_roi -= 1  # -1 - 1
            two_stage_input = image_roi

            # then repeat same steps as conducted in OD, except the input as image roi
            output_oc, shape_prior_oc, shape_prior_mask_oc, loss_ins_wt_oc, loss_dom_wt_oc = self.model_oc.update(
                image_roi,
                target_oc,
                step=self.epoch,
                plot_show=self.plot_show,
                two_stage_inputs=two_stage_input,
                two_step=True)

            # roi bec loss using a reweighted strategy
            oc_pos_weight = torch.sum(od_pred) / torch.sum(od_pred * target_oc)
            if torch.isinf(oc_pos_weight) or torch.isnan(oc_pos_weight):
                oc_pos_weight = torch.tensor(1.).cuda()
            loss_seg_oc = F.binary_cross_entropy_with_logits(
                output_oc * od_pred,
                target_oc,
                pos_weight=oc_pos_weight)

            output_oc = output_oc * od_pred
            self.running_seg_loss_oc += loss_seg_oc.item()
            if self.hparams['whitening']:
                self.running_instance_wt_oc += loss_ins_wt_oc.item()
                self.running_domain_wt_oc += loss_dom_wt_oc.item()
                loss_data_oc = (loss_seg_oc + loss_ins_wt_oc + loss_dom_wt_oc).data.item()
            else:
                self.running_instance_wt_oc += loss_ins_wt_oc
                self.running_domain_wt_oc += loss_dom_wt_oc
                loss_data_oc = (loss_seg_oc).data.item()

            if np.isnan(loss_data_oc):
                raise ValueError('loss is nan while training')


            # update oc segmentation network
            loss_main_oc = loss_seg_oc + self.hparams['instance_wt_gm'] * loss_ins_wt_oc + self.hparams[
                'domain_wt_gm'] * loss_dom_wt_oc
            loss_main_oc.backward()
            self.optim_oc.step()

            if self.hparams['whitening']:
                # updata shape network
                for _ in range(self.hparams['multi-turn']):
                    assert self.model_shape_oc.training
                    self.optim_shape_oc.zero_grad()
                    self.model_shape_oc.zero_grad()

                    loss_kd_oc, loss_ins_wt_shape_oc,instance_ij_shape_oc, instance_ii_shape_oc,loss_dom_wt_shape_oc = self.model_shape_oc.update(self.model_oc,
                                                                                                       image_roi,
                                                                                                       target_oc,
                                                                                                       step=self.epoch,
                                                                                                       plot_show=self.plot_show,
                                                                                                       two_stage_inputs=two_stage_input,
                                                                                                       two_step=True)


                    # compute total loss and update
                    loss_shape_oc = loss_kd_oc + self.hparams['instance_wt_gm'] * loss_ins_wt_shape_oc + self.hparams[
                        'domain_wt_gm'] * loss_dom_wt_shape_oc
                    loss_shape_oc.backward()
                    self.optim_shape_oc.step()

                if self.hparams['whitening']:
                    self.running_kd_oc += loss_kd_oc.item()
                    self.running_instance_wt_shape_oc += loss_ins_wt_shape_oc.item()
                    self.running_domain_wt_shape_oc += loss_dom_wt_shape_oc.item()

                else:
                    self.running_kd_oc += loss_kd_oc
                    self.running_instance_wt_shape_oc += loss_ins_wt_shape_oc
                    self.running_domain_wt_shape_oc += loss_dom_wt_shape_oc


            ##############################################################################################################
            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(
                    image_roi[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/image', grid_image, iteration)

                if self.hparams['whitening']:
                    grid_image = make_grid(
                        shape_prior_oc[0, 0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('train/shape_cup', grid_image, iteration)
                grid_image = make_grid(
                    target_oc[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target_cup', grid_image, iteration)
                grid_image = make_grid(
                    ((torch.sigmoid(output_oc)[0, 0, ...]) > 0.75).float().clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/pre_cup', grid_image, iteration)

                if self.args.label == None:
                    grid_image = make_grid(
                        target_od[0, 0, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('train/target_disc', grid_image, iteration)

                grid_image = make_grid(torch.sigmoid(output)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                if self.args.label == None:
                    self.writer.add_image('train/prediction_cup', grid_image, iteration)
                    grid_image = make_grid(torch.sigmoid(output)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                    self.writer.add_image('train/prediction_disc', grid_image, iteration)

            # write loss log
            self.writer.add_scalar('train_gen/loss', loss_data, iteration)
            self.writer.add_scalar('train_gen/loss_seg', loss_seg.data.item(), iteration)
            self.writer.add_scalar('train_gen/loss_seg_oc', loss_seg_oc.data.item(), iteration)

            if self.hparams['whitening']:
                self.writer.add_scalar('train_shape/loss_kd_oc', loss_kd_oc.data.item(), iteration)
                self.writer.add_scalar('train_shape/loss_kd', loss_kd.data.item(), iteration)
                self.writer.add_scalar('train_shape/loss_ins_wt', loss_ins_wt.data.item(), iteration)
                self.writer.add_scalar('train_shape/loss_dom_wt', loss_dom_wt.data.item(), iteration)
                self.writer.add_scalar('train_shape/loss_ins_wt_shape_ii', loss_ins_wt_shape_ii.data.item(), iteration)
                self.writer.add_scalar('train_shape/loss_ins_wt_shape_ij', loss_ins_wt_shape_ij.data.item(), iteration)
                self.writer.add_scalar('train_shape/loss_ins_wt_shape_all', loss_ins_wt_shape.data.item(), iteration)
                self.writer.add_scalar('train_shape/loss_dom_wt_shape', loss_dom_wt_shape.data.item(), iteration)




        self.running_seg_loss /= len(self.train_loader)
        self.running_cls_loss /= len(self.train_loader)
        self.running_kd_oc /=  len(self.train_loader)
        self.running_domain_wt_oc /= len(self.train_loader)
        self.running_instance_wt_oc /= len(self.train_loader)
        self.running_instance_wt_shape /= len(self.train_loader)
        self.running_domain_wt_shape /= len(self.train_loader)
        self.running_instance_wt_ii /= len(self.train_loader)
        self.running_instance_wt_ij /= len(self.train_loader)


        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, Average kl: %f,Average ins wt: %f,Average dom wt: %f, Execution time: %.5f' %
              (self.epoch, get_lr(self.optim), self.running_seg_loss_oc, self.running_kd_oc, self.running_instance_wt_oc,self.running_domain_wt_oc,stop_time - start_time))
    def lr_update(self,epoch,warmup_steps,warmup_factor,base_lr_od,base_lr_oc,gamma,Steps):
        import bisect
        from bisect import bisect_right
        alpha = epoch / warmup_steps
        warmup_factor = warmup_factor * (1 - alpha) + alpha
        learning_rate_od = base_lr_od * warmup_factor * gamma ** bisect_right(Steps, epoch)
        learning_rate_oc = base_lr_oc * warmup_factor * gamma ** bisect_right(Steps, epoch)
        for param_group in self.optim.param_groups:
            param_group['lr'] = learning_rate_od
        for param_group in self.optim_shape.param_groups:
            param_group['lr'] = learning_rate_od
        for param_group in self.optim_oc.param_groups:
            param_group['lr'] = learning_rate_oc

        for param_group in self.optim_shape_oc.param_groups:
            param_group['lr'] = learning_rate_oc

    def train(self):

        best_cup_dice, best_cup_hd, best_cup_asd,best_disc_dice, best_disc_hd,best_disc_asd = 0, 0, 0, 0,0,0


        self.source_domain_datasets = list(self.train_loader.values())
        self.source_domain_num = len(self.source_domain_datasets)
        self.per_domain_batch = self.batch_size // self.source_domain_num
        total_sample_num = sum([len(source_domain_dataset) for source_domain_dataset in self.source_domain_datasets])
        self.iter_per_epoch = total_sample_num // self.batch_size

        warmup_factor = 0.001
        Steps = (100, 150)
        gamma = 0.5
        warmup_steps  = self.max_epoch * 2
        base_lr_od = self.lr
        base_lr_oc = self.lr_oc


        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.plot_show = 0

            # train model
            if self.algorithm == 'Unet_nips2023_joint_shape_regularization':
                self.train_epoch_joint_shape_reg()
            else:
                self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break
            # lr update; You can choose if the lr is updated with the training. Sometimes, better performance will be achieved using lr update.
            #self.lr_update(epoch,warmup_steps,warmup_factor,base_lr_od,base_lr_oc,gamma,Steps)



            self.writer.add_scalar('lr', get_lr(self.optim), self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('lr_shape', get_lr(self.optim_shape), self.epoch * (len(self.train_loader)))

            # validate
            if (self.epoch + 1) % self.interval_validate == 0 and self.epoch > 2:
                print('-'*10,'start to validate','-'*10)
                if self.algorithm == 'Unet_nips2023_joint_shape_regularization':
                    label, best_cup_dice_, best_cup_hd_, best_cup_asd_, best_disc_dice_, best_disc_hd_, best_disc_asd_ = self.validate_joint_shape_reg()
                else:
                    label, best_cup_dice_, best_cup_hd_, best_cup_asd_, best_disc_dice_, best_disc_hd_,best_disc_asd_ = self.validate()
                # label ==1, i.e., new best results
                if label == 1:
                    best_cup_dice, best_cup_hd, best_cup_asd,best_disc_dice, best_disc_hd,best_disc_asd = best_cup_dice_, best_cup_hd_, best_cup_asd_,best_disc_dice_, best_disc_hd_,best_disc_asd_


        self.writer.close()
        return [best_cup_dice, best_cup_hd, best_cup_asd,best_disc_dice, best_disc_hd,best_disc_asd]



