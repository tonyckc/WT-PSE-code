from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import copy

def to_multilabel(pre_mask, classes = 2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [0, 1]
    mask[pre_mask == 2] = [1, 1]
    return mask
import torch
class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 4 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='None',
                 phase='train',
                 splitid=[2, 3, 4],
                 transform=None,
                 state='train',
                 label = None
                 ):
        # super().__init__()
        self.label = label
        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.image_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.label_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.img_name_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid

        for id in splitid:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(id), phase, 'ROIs/image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path})

        self.transform = transform
        self._read_img_into_memory()
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        # Display stats
        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label_od': _target, 'label_oc': _target, 'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)

                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label_od': _target, 'label_oc': _target, 'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                if self.label == 'OC' or self.label == 'OD':

                    __mask = np.array(_target).astype(np.uint8)
                    _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
                    _mask[__mask > 200] = 255
                    _mask[(__mask > 50) & (__mask < 201)] = 128
                    __mask[_mask < 255] = 1
                    __mask[_mask == 255] = 0
                    mask = np.expand_dims(__mask, axis=2)
                    mask = mask.transpose(2, 0, 1)
                    mask = torch.from_numpy(np.array(mask)).float()

                    anco_sample['original_od'] = mask

                    __mask = np.array(_target).astype(np.uint8)
                    _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
                    _mask[__mask > 200] = 255
                    _mask[(__mask > 50) & (__mask < 201)] = 128
                    __mask[_mask == 0] = 1
                    __mask[_mask > 0] = 0
                    mask = np.expand_dims(__mask, axis=2)
                    mask = mask.transpose(2, 0, 1)
                    mask = torch.from_numpy(np.array(mask)).float()

                    anco_sample['original_oc'] = mask
                    sample = anco_sample

                else:
                    __mask = np.array(_target).astype(np.uint8)
                    _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
                    _mask[__mask > 200] = 255
                    _mask[(__mask > 50) & (__mask < 201)] = 128
                    __mask[_mask < 255] = 1
                    __mask[_mask == 255] = 0
                    mask = np.expand_dims(__mask, axis=2)
                    mask = mask.transpose(2, 0, 1)
                    mask = torch.from_numpy(np.array(mask)).float()

                    anco_sample['original_od'] = mask

                    __mask = np.array(_target).astype(np.uint8)
                    _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
                    _mask[__mask > 200] = 255
                    _mask[(__mask > 50) & (__mask < 201)] = 128
                    __mask[_mask == 0] = 1
                    __mask[_mask > 0] = 0
                    mask = np.expand_dims(__mask, axis=2)
                    mask = mask.transpose(2, 0, 1)
                    mask = torch.from_numpy(np.array(mask)).float()

                    anco_sample['original_oc'] = mask
                    sample = anco_sample
        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_DGS:
                Flag = 'DGS'
            elif basename[0] in self.flags_REF:
                Flag = 'REF'
            elif basename[0] in self.flags_RIM:
                Flag = 'RIM'
            elif basename[0] in self.flags_REF_val:
                Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            if self.splitid[0] == '4':
                self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').crop((144, 144, 144+512, 144+512)).resize((256, 256), Image.LANCZOS))
                _target = np.asarray(Image.open(self.image_list[index]['label']).convert('L'))
                _target = _target[144:144+512, 144:144+512]
                _target = Image.fromarray(_target)
            else:
                self.image_pool[Flag].append(
                    Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
                # self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB'))
                _target = Image.open(self.image_list[index]['label'])

            if _target.mode is 'RGB':
                _target = _target.convert('L')
            if self.state != 'prediction':
                _target = _target.resize((256, 256))
            self.label_pool[Flag].append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name)



    def __str__(self):
        return 'Fundus(phase=' + self.phase+str(args.datasetTest[0]) + ')'


if __name__ == '__main__':
    import custom_transforms as tr
    from utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    voc_train = FundusSegmentation(split='train1',
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = tmp
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

            break
    plt.show(block=True)
