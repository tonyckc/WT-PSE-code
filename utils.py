from datetime import datetime
import os
import os.path as osp
import shutil
import glob
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import os
import cv2
from skimage import morphology
import scipy
from PIL import Image
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from skimage import measure, draw
import torch
from skimage.morphology import disk, erosion, dilation, opening, closing, white_tophat

import matplotlib.pyplot as plt

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


def save_code(args,hparams,local_path,day,trial_name):
    now = datetime.now()
    args.out = osp.join(local_path, day, trial_name,'test_domain_' + str(args.datasetTest[0]) + '_seed_' + str(args.seed) +now.strftime('_%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)
    code_dir = os.path.join(args.out, 'code')
    os.makedirs(code_dir)
    save_file(code_dir)
    copy_allfiles('./', code_dir)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)
def seed_initialization(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def untransform(img, lt):
    img = (img + 1) * 127.5
    lt = lt * 128
    return img, lt


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou

def get_dice(pred, gt):
    total_dice = 0.0
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
        total_dice += dice

    return total_dice

from skimage import measure
import scipy.ndimage as nd

def post_processing(prediction):
    prediction = nd.binary_fill_holes(prediction)
    label_cc, num_cc = measure.label(prediction,return_num=True)
    total_cc = np.sum(prediction)
    measure.regionprops(label_cc)
    for cc in range(1,num_cc+1):
        single_cc = (label_cc==cc)
        single_vol = np.sum(single_cc)
        if single_vol/total_cc<0.2:
            prediction[single_cc]=0

    return prediction




def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def postprocessing(prediction, threshold=0.75, dataset='G',label=None):
    if dataset[0] == 'D':
        # prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        # for i in range(5):
        #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy
    else:
        prediction = torch.sigmoid(prediction).data.cpu().numpy()

        # disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        # cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.erosion(disc_mask, morphology.diamond(3))  # return 0,1
        # cup_mask = morphology.erosion(cup_mask, morphology.diamond(3))  # return 0,1

        prediction_copy = np.copy(prediction)
        prediction_copy = (prediction_copy > threshold)  # return binary mask
        prediction_copy = prediction_copy.astype(np.uint8)
        if label == None:
            disc_mask = prediction_copy[1]
            cup_mask = prediction_copy[0]
            disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
            cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
            prediction_copy[0] = cup_mask
            prediction_copy[1] = disc_mask
        else:

            cup_mask = prediction_copy[0]

            cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
            prediction_copy[0] = cup_mask

        # selem = disk(6)
        # disc_mask = morphology.closing(disc_mask, selem)
        # cup_mask = morphology.closing(cup_mask, selem)
        # print(sum(disc_mask))


        return prediction_copy


def joint_val_image(image, prediction, mask):
    ratio = 0.5
    _pred_cup = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _pred_disc = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _mask = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    image = np.transpose(image, (1, 2, 0))

    _pred_cup[:, :, 0] = prediction[0]
    _pred_cup[:, :, 1] = prediction[0]
    _pred_cup[:, :, 2] = prediction[0]
    _pred_disc[:, :, 0] = prediction[1]
    _pred_disc[:, :, 1] = prediction[1]
    _pred_disc[:, :, 2] = prediction[1]
    _mask[:,:,0] = mask[0]
    _mask[:,:,1] = mask[1]

    pred_cup = np.add(ratio * image, (1 - ratio) * _pred_cup)
    pred_disc = np.add(ratio * image, (1 - ratio) * _pred_disc)
    mask_img = np.add(ratio * image, (1 - ratio) * _mask)

    joint_img = np.concatenate([image, mask_img, pred_cup, pred_disc], axis=1)
    return joint_img


def save_val_img(path, epoch, img):
    name = osp.join(path, "visualization", "epoch_%d.png" % epoch)
    out = osp.join(path, "visualization")
    if not osp.exists(out):
        os.makedirs(out)
    img_shape = img[0].shape
    stack_image = np.zeros([len(img) * img_shape[0], img_shape[1], img_shape[2]])
    for i in range(len(img)):
        stack_image[i * img_shape[0] : (i + 1) * img_shape[0], :, : ] = img[i]
    imsave(name, stack_image)





def save_per_img(patch_image, data_save_path, img_name, prob_map, gt=None, mask_path=None, ext="bmp",batch=True):
    if batch:
        path1 = os.path.join(data_save_path, 'overlay', img_name+'.png')
        path0 = os.path.join(data_save_path, 'original_image', img_name+'.png')
        if not os.path.exists(os.path.dirname(path0)):
            os.makedirs(os.path.dirname(path0))
        if not os.path.exists(os.path.dirname(path1)):
            os.makedirs(os.path.dirname(path1))
        patch_image_o = patch_image
        patch_image_o = patch_image_o.astype(np.uint8)
        patch_image_o = Image.fromarray(patch_image_o)

        patch_image_o.save(path0)

        disc_map = prob_map[0]
        cup_map = prob_map[1]
        size = disc_map.shape
        disc_map[:, 0] = np.zeros(size[0])
        disc_map[:, size[1] - 1] = np.zeros(size[0])
        disc_map[0, :] = np.zeros(size[1])
        disc_map[size[0] - 1, :] = np.zeros(size[1])
        size = cup_map.shape
        cup_map[:, 0] = np.zeros(size[0])
        cup_map[:, size[1] - 1] = np.zeros(size[0])
        cup_map[0, :] = np.zeros(size[1])
        cup_map[size[0] - 1, :] = np.zeros(size[1])

        #disc_mask = (disc_map > 0.75) # return binary mask
        #cup_mask = (cup_map > 0.75)
        #disc_mask = disc_mask.astype(np.uint8)
        #cup_mask = cup_mask.astype(np.uint8)


        contours_disc = measure.find_contours(disc_map, 0.5)
        contours_cup = measure.find_contours(cup_map, 0.5)


        for n, contour in enumerate(contours_cup):
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
            patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
            patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
            patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
            patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

        for n, contour in enumerate(contours_disc):
            patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
            patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
            patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
            patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
            patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

        disc_mask = get_largest_fillhole(gt[0].numpy()).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(gt[1].numpy()).astype(np.uint8)

        contours_disc = measure.find_contours(disc_mask, 0.5)
        contours_cup = measure.find_contours(cup_mask, 0.5)
        red = [255, 0, 0]
        for n, contour in enumerate(contours_cup):
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = red
            patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
            patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
            patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
            patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red

        for n, contour in enumerate(contours_disc):
            patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = red
            patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
            patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
            patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
            patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
            patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red


        patch_image = patch_image.astype(np.uint8)
        patch_image = Image.fromarray(patch_image)

        patch_image.save(path1)

    else:
        pass


def untransform(img, lt):
    img = (img + 1) * 127.5
    lt = lt * 128
    return img, lt


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou

def get_dice(pred, gt):
    total_dice = 0.0
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
        total_dice += dice

    return total_dice

from skimage import measure
import scipy.ndimage as nd

def post_processing(prediction):
    prediction = nd.binary_fill_holes(prediction)
    label_cc, num_cc = measure.label(prediction,return_num=True)
    total_cc = np.sum(prediction)
    measure.regionprops(label_cc)
    for cc in range(1,num_cc+1):
        single_cc = (label_cc==cc)
        single_vol = np.sum(single_cc)
        if single_vol/total_cc<0.2:
            prediction[single_cc]=0

    return prediction
