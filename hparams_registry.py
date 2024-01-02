# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}
    hparams['eval_steps'] = (400, random_state.choice([1, 0.1, 0.01]))
    hparams['training_fraction'] = (0.8, random_state.choice([1, 0.1, 0.01]))
    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet_dropout"] = (0.5, random_state.choice([0.0, 0.1, 0.5]))
    hparams["class_balanced"] = (False, False)
    hparams["optimizer"] = ("adam", "adam")
    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    if algorithm == "DNA":
        hparams["bottleneck_dim"] = (1024, random_state.choice([1024, 2048]))
        hparams["dropout_rate"] = (0.5, random_state.choice([0.5, 0.1]))
        hparams["dropout_type"] = ('Bernoulli', 'Bernoulli')
        hparams["lambda_v"] = (0.1, random_state.choice([0.01, 0.1, 1.0]))

    if dataset not in SMALL_IMAGES:
        hparams["lr_gm"] = (1e-3, 10 ** random_state.uniform(-5, -3.5)) # 5e-5for segmentation 1e-4
        hparams["lr_sc"] = (1e-3, 10 ** random_state.uniform(-5, -3.5))  # 5e-5for segmentation 1e-4
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (9, int(2 ** random_state.uniform(3, 5.5)))
        if algorithm == "ARM":
            hparams["batch_size"] = (8, 8)
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))

    if algorithm in ["DANN", "CDANN"]:
        if dataset not in SMALL_IMAGES:
            hparams["lr_g"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
            hparams["lr_d"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        else:
            hparams["lr_g"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
            hparams["lr_d"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))

        if dataset in SMALL_IMAGES:
            hparams["weight_decay_g"] = (0.0, 0.0)
        else:
            hparams["weight_decay_g"] = (0.0, 10 ** random_state.uniform(-6, -2))

        hparams["lambda"] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams["weight_decay_d"] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams["d_steps_per_g_step"] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams["grad_penalty"] = (0.0, 10 ** random_state.uniform(-2, 1))
        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
        hparams["mlp_width"] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams["mlp_depth"] = (3, int(random_state.choice([3, 4, 5])))
        hparams["mlp_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))

    elif algorithm == "RSC":
        hparams["rsc_f_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
        hparams["rsc_b_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))

    elif algorithm == "SagNet":
        hparams["sag_w_adv"] = (0.1, 10 ** random_state.uniform(-2, 1))

    elif algorithm == 'ICML22_MMD':
        hparams['num_mc'] = (20, int(random_state.choice([1,1,1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20,30,40,50])))
        hparams['moped_delta_factor'] = (0.1,random_state.choice([0.1,0.2,0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1,-2,-3,-4,-5]))
        hparams['kl_weight'] = (1, random_state.choice([1,0.1,0.5,0.25,0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['local_weight'] = (0.1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['global_weight'] = (1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['local_loss'] = (True, True)  # 0.01 for
        hparams['classifier'] = ('SGP', random_state.choice(['SGP','NO']))
        hparams['contrastive_type'] =  ('contrastive_plain_v2', random_state.choice(['contrastive','triplet','contrastive_plain']))
        hparams['margin'] = (1, random_state.choice([1,0.1,0.01])) # 0.01 for
        hparams['training_fraction'] = (0.8, random_state.choice([1, 0.1, 0.01]))
        hparams['pairs_number'] = (100, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['global_loss'] = (True, True)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['level2_gamma'] = ([1], random_state.choice([1,10,0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1,0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1,0.1,10]))
        hparams['eval_steps'] = (1000, random_state.choice([1, 0.1, 0.01]))
    elif algorithm == 'ICML22_MMD_mean':
        hparams['num_mc'] = (20, int(random_state.choice([1,1,1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20,30,40,50])))
        hparams['moped_delta_factor'] = (0.1,random_state.choice([0.1,0.2,0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1,-2,-3,-4,-5]))
        hparams['kl_weight'] = (1, random_state.choice([1,0.1,0.5,0.25,0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['local_weight'] = (0.1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['global_weight'] = (0.7, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['local_loss'] = (True, False)  # 0.01 for
        hparams['classifier'] = ('SGP', random_state.choice(['SGP','NO']))
        hparams['contrastive_type'] =  ('contrastive_plain_v2_mean', random_state.choice(['contrastive','triplet','contrastive_plain']))
        hparams['margin'] = (1, random_state.choice([1,0.1,0.01])) # 0.01 for
        hparams['pairs_number'] = (100, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['global_loss'] = (False, False)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['level2_gamma'] = ([1], random_state.choice([1,10,0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1,0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1,0.1,10]))
        hparams['eval_steps'] = (400, random_state.choice([1, 0.1, 0.01]))
    elif algorithm == 'ICML21_bayesian':
        hparams['num_mc'] = (10, int(random_state.choice([1,1,1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20,30,40,50])))
        hparams['moped_delta_factor'] = (0.1,random_state.choice([0.1,0.2,0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1,-2,-3,-4,-5]))
        hparams['kl_weight'] = (1, random_state.choice([1,0.1,0.5,0.25,0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['local_weight'] = (0.1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['global_weight'] = (0.05, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['local_loss'] = (True, False)  # 0.01 for
        hparams['classifier'] = ('SGP', random_state.choice(['SGP','NO']))
        hparams['contrastive_type'] =  ('kl_distance', random_state.choice(['contrastive','triplet','contrastive_plain']))
        hparams['margin'] = (1, random_state.choice([1,0.1,0.01])) # 0.01 for
        hparams['pairs_number'] = (100, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['training_fraction'] = (0.8, random_state.choice([1,0.1,0.01]))
        hparams['global_loss'] = (True, False)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['level2_gamma'] = ([1], random_state.choice([1,10,0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1,0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1,0.1,10]))
        hparams['eval_steps'] = (400, random_state.choice([1, 0.1, 0.01]))
    elif algorithm == 'ICML22_MMD_Unet':
        hparams['num_mc'] = (10, int(random_state.choice([1,1,1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20,30,40,50])))
        hparams['moped_delta_factor'] = (0.1,random_state.choice([0.1,0.2,0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1,-2,-3,-4,-5]))
        hparams['kl_weight'] = (1, random_state.choice([1,0.1,0.5,0.25,0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['local_weight'] = (0.1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['global_weight'] = (0.7, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['local_loss'] = (True, False)  # 0.01 for
        hparams['p_weight1'] = (2, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['classifier'] = ('SGP', random_state.choice(['SGP','NO']))
        hparams['contrastive_type'] =  ('contrastive_plain_v2_segmentation', random_state.choice(['contrastive','triplet','contrastive_plain']))
        hparams['contrastive_type_global'] = (
        'contrastive_plain_v2_segmentation', random_state.choice(['contrastive', 'triplet', 'contrastive_plain']))
        hparams['margin'] = (1, random_state.choice([1,0.1,0.01])) # 0.01 for
        hparams['pairs_number'] = (100, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['global_loss'] = (True, False)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['level2_gamma'] = ([1], random_state.choice([1,10,0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1,0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1,0.1,10]))
        hparams['eval_steps'] = (100, random_state.choice([1, 0.1, 0.01]))




    elif algorithm == 'ICML22_MMD_Unet_Ablation':
        hparams['num_mc'] = (10, int(random_state.choice([1,1,1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20,30,40,50])))
        hparams['moped_delta_factor'] = (0.1,random_state.choice([0.1,0.2,0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1,-2,-3,-4,-5]))
        hparams['kl_weight'] = (1, random_state.choice([1,0.1,0.5,0.25,0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['local_weight'] = (0.1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['global_weight'] = (1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['local_loss'] = (True, False)  # 0.01 for
        hparams['p_weight1'] = (2, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['classifier'] = ('SGP', random_state.choice(['SGP','NO']))
        hparams['contrastive_type'] =  ('contrastive_plain_v2_segmentation', random_state.choice(['contrastive','triplet','contrastive_plain']))
        hparams['contrastive_type_global'] = (
        'contrastive_plain_v2_segmentation', random_state.choice(['contrastive', 'triplet', 'contrastive_plain']))
        hparams['margin'] = (1, random_state.choice([1,0.1,0.01])) # 0.01 for
        hparams['pairs_number'] = (100, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['global_loss'] = (True, False)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['level2_gamma'] = ([1], random_state.choice([1,10,0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1,0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1,0.1,10]))
        hparams['eval_steps'] = (10, random_state.choice([1, 0.1, 0.01]))

    elif algorithm == 'ICML22_MMD_Unet_DSU':
        hparams['num_mc'] = (10, int(random_state.choice([1, 1, 1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20, 30, 40, 50])))
        hparams['moped_delta_factor'] = (0.1, random_state.choice([0.1, 0.2, 0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1, -2, -3, -4, -5]))
        hparams['kl_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['local_weight'] = (0, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['global_weight'] = (0.001, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['local_loss'] = (True, False)  # 0.01 for
        hparams['p_weight1'] = (2, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['classifier'] = ('SGP', random_state.choice(['SGP', 'NO']))
        hparams['contrastive_type'] = (
        'contrastive_plain_v2_segmentation', random_state.choice(['contrastive', 'triplet', 'contrastive_plain']))
        hparams['contrastive_type_global'] = (
            'contrastive_plain_v2_segmentation', random_state.choice(['contrastive', 'triplet', 'contrastive_plain']))
        hparams['margin'] = (1, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['pairs_number'] = (400, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['global_loss'] = (True, False)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['metric_dimension'] = (8, random_state.choice([1, 10, 0.1]))
        hparams['level2_gamma'] = ([1], random_state.choice([1, 10, 0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1, 0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1, 0.1, 10]))
        hparams['eval_steps'] = (50, random_state.choice([1, 0.1, 0.01]))


    elif algorithm == 'ICML22_MMD_Unet_Alignment':
        '''
        parser.add_argument('--vgg1', default='/root/autodl-tmp/models/vgg_normalised_conv1_1.t7',
                            help='Path to the VGG conv1_1')
        parser.add_argument('--vgg2', default='/root/autodl-tmp/models/vgg_normalised_conv2_1.t7',
                            help='Path to the VGG conv2_1')
        parser.add_argument('--vgg3', default='/root/autodl-tmp/models/vgg_normalised_conv3_1.t7',
                            help='Path to the VGG conv3_1')
        parser.add_argument('--vgg4', default='/root/autodl-tmp/models/vgg_normalised_conv4_1.t7',
                            help='Path to the VGG conv4_1')
        parser.add_argument('--vgg5', default='/root/autodl-tmp/models/vgg_normalised_conv5_1.t7',
                            help='Path to the VGG conv5_1')
        parser.add_argument('--decoder5', default='/root/autodl-tmp/models/feature_invertor_conv5_1.t7',
                            help='Path to the decoder5')
        parser.add_argument('--decoder4', default='/root/autodl-tmp/models/feature_invertor_conv4_1.t7',
                            help='Path to the decoder4')
        parser.add_argument('--decoder3', default='/root/autodl-tmp/models/feature_invertor_conv3_1.t7',
                            help='Path to the decoder3')
        parser.add_argument('--decoder2', default='/root/autodl-tmp/models/feature_invertor_conv2_1.t7',
                            help='Path to the decoder2')
        parser.add_argument('--decoder1', default='/root/autodl-tmp/models/feature_invertor_conv1_1.t7',
                            help='Path to the decoder1')
        '''
        hparams['vgg1'] = ('/hdd/ckc/models/vgg_normalised_conv1_1.t7', random_state.choice(['/root/autodl-tmp/models/vgg_normalised_conv1_1.t7']))
        hparams['decoder1'] = ('/hdd/ckc/models/feature_invertor_conv1_1.t7', random_state.choice(['/root/autodl-tmp/models/feature_invertor_conv1_1.t7']))
        hparams['vgg2'] = ('/hdd/ckc/models/vgg_normalised_conv2_1.t7', random_state.choice(['/root/autodl-tmp/models/vgg_normalised_conv2_1.t7']))
        hparams['decoder2'] = ('/hdd/ckc/models/feature_invertor_conv2_1.t7', random_state.choice(['/root/autodl-tmp/models/feature_invertor_conv2_1.t7']))

        hparams['vgg4'] = ('/root/autodl-tmp/models/vgg_normalised_conv4_1.t7',
                           random_state.choice(['/root/autodl-tmp/models/vgg_normalised_conv4_1.t7']))
        hparams['decoder4'] = ('/root/autodl-tmp/models/feature_invertor_conv4_1.t7',
                               random_state.choice(['/root/autodl-tmp/models/feature_invertor_conv4_1.t7']))



        hparams['num_mc'] = (10, int(random_state.choice([1,1,1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20,30,40,50])))
        hparams['moped_delta_factor'] = (0.1,random_state.choice([0.1,0.2,0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1,-2,-3,-4,-5]))
        hparams['kl_weight'] = (1, random_state.choice([1,0.1,0.5,0.25,0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['shape_weight'] = (0.1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['global_weight'] = (0.1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['shape_prior'] = (False, True)  # 0.01 for
        hparams['p_weight1'] = (2, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['classifier'] = ('SGP', random_state.choice(['SGP','NO']))
        hparams['contrastive_type'] =  ('contrastive_plain_v2_segmentation', random_state.choice(['contrastive','triplet','contrastive_plain']))
        hparams['contrastive_type_global'] = (
        'contrastive_plain_v2_segmentation', random_state.choice(['contrastive', 'triplet', 'contrastive_plain']))
        hparams['margin'] = (1, random_state.choice([1,0.1,0.01])) # 0.01 for
        hparams['pairs_number'] = (200, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['global_loss'] = (False, False)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['metric_dimension'] = (8, random_state.choice([1, 10, 0.1]))
        hparams['level2_gamma'] = ([1], random_state.choice([1,10,0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1,0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1,0.1,10]))
        hparams['eval_steps'] = (90, random_state.choice([1, 0.1, 0.01]))
        hparams['shape_attention'] = (False, True)
        hparams['cat_shape'] = (False, False)
        hparams['shape_attention_coeffient'] = (0.1, random_state.choice([1, 0.1, 0.01])) # how original embeddings reserve
        hparams['shape_start'] = (0, random_state.choice([1, 0.1, 0.01]))
        hparams['whitening'] = (False, False)
        hparams['whitening_type'] = ('instance_wt', random_state.choice(['cca','instance_wt']))
        hparams['wt_type_inference'] = ('instance_wt', random_state.choice(['instance_wt','instance_wt']))
        hparams['CCA_type'] = ('cca_all', random_state.choice(['cca_all','caa_random_2']))
        hparams['CCA_transform_type'] = ('ZCA', random_state.choice(['ZCA', 'CCA']))
        hparams['posterior_transform_follow_prior'] = (False,True)


    elif algorithm == 'Unet_nips2023':


        hparams['num_mc'] = (10, int(random_state.choice([1,1,1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20,30,40,50])))
        hparams['moped_delta_factor'] = (0.1,random_state.choice([0.1,0.2,0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1,-2,-3,-4,-5]))
        hparams['kl_weight'] = (1, random_state.choice([1,0.1,0.5,0.25,0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))

        hparams['global_weight'] = (0.1, random_state.choice([1, 0.1, 0.25, 0.75]))

        hparams['p_weight1'] = (2, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['classifier'] = ('SGP', random_state.choice(['SGP','NO']))
        hparams['contrastive_type'] =  ('contrastive_plain_v2_segmentation', random_state.choice(['contrastive','triplet','contrastive_plain']))
        hparams['contrastive_type_global'] = (
        'contrastive_plain_v2_segmentation', random_state.choice(['contrastive', 'triplet', 'contrastive_plain']))
        hparams['margin'] = (0, random_state.choice([1,0.1,0.01])) # 0.01 for
        hparams['pairs_number'] = (200, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['global_loss'] = (False, False)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['metric_dimension'] = (8, random_state.choice([1, 10, 0.1]))
        hparams['level2_gamma'] = ([1], random_state.choice([1,10,0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1,0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1,0.1,10]))
        hparams['eval_steps'] = (90, random_state.choice([1, 0.1, 0.01]))
        hparams['dropout'] = (False, False)
        hparams['shape_attention'] = (True, True)
        hparams['shape_prior'] = (True, True)  # 0.01 for
        hparams['cat_shape'] = (False, False)
        hparams['shape_attention_coeffient'] = (0.3, random_state.choice([1, 0.1, 0.01])) # how original embeddings reserve

        hparams['shape_start'] = (0.5, random_state.choice([1, 0.1, 0.01]))
        hparams['whitening'] = (True, False)
        hparams['fusing_mode'] = ('P', random_state.choice(['2P']))
        hparams['shape_weight'] = (0, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['instance_wt_gm'] = (1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['domain_wt_gm'] = (1,random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['instance_wt_sc'] = (1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['domain_wt_sc'] = (1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['multi-turn'] = (1, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['layer_wt'] = (4, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['sg_type'] = (None, random_state.choice(['oc','od']))




        hparams['whitening_type'] = ('instance_wt', random_state.choice(['cca','instance_wt']))
        hparams['wt_type_inference'] = ('instance_wt', random_state.choice(['instance_wt','instance_wt']))
        hparams['CCA_type'] = ('cca_all', random_state.choice(['cca_all','caa_random_2']))
        hparams['CCA_transform_type'] = ('ZCA', random_state.choice(['ZCA', 'CCA']))
        hparams['posterior_transform_follow_prior'] = (False,True)






    elif algorithm == 'ICML21_Unet_Alignment':
        hparams['num_mc'] = (10, int(random_state.choice([1,1,1])))
        hparams['num_monte_carlo'] = (40, int(random_state.choice([20,30,40,50])))
        hparams['moped_delta_factor'] = (0.1,random_state.choice([0.1,0.2,0.3]))
        hparams['bnn_rho_init'] = (-3, random_state.choice([-1,-2,-3,-4,-5]))
        hparams['kl_weight'] = (1, random_state.choice([1,0.1,0.5,0.25,0.75]))
        hparams['ce_weight'] = (1, random_state.choice([1, 0.1, 0.5, 0.25, 0.75]))
        hparams['local_weight'] = (0.01, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['global_weight'] = (0.001, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['local_loss'] = (True, False)  # 0.01 for
        hparams['p_weight1'] = (2, random_state.choice([1, 0.1, 0.25, 0.75]))
        hparams['classifier'] = ('SGP', random_state.choice(['SGP','NO']))
        hparams['contrastive_type'] =  ('ICML21_segmentation', random_state.choice(['contrastive','triplet','contrastive_plain']))
        hparams['contrastive_type_global'] = (
        'contrastive_plain_v2_segmentation', random_state.choice(['contrastive', 'triplet', 'contrastive_plain']))
        hparams['margin'] = (1, random_state.choice([1,0.1,0.01])) # 0.01 for
        hparams['pairs_number'] = (400, random_state.choice([1, 0.1, 0.01]))  # 0.01 for
        hparams['global_loss'] = (True, False)  # 0.01 for
        hparams['global_metric'] = ([True], True)  # 0.01 for
        hparams['metric_dimension'] = (8, random_state.choice([1, 10, 0.1]))
        hparams['level2_gamma'] = ([1], random_state.choice([1,10,0.1]))
        hparams['level1_gamma_global'] = ([1], random_state.choice([1,0.01, 0.1]))
        hparams['level1_gamma'] = ([1], random_state.choice([1,0.1,10]))
        hparams['eval_steps'] = (50, random_state.choice([1, 0.1, 0.01]))
    elif algorithm == "IRM":
        hparams["irm_lambda"] = (1e2, 10 ** random_state.uniform(-1, 5))
        hparams["irm_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm in ["Mixup", "OrgMixup"]:
        hparams["mixup_alpha"] = (0.2, 10 ** random_state.uniform(-1, -1))
    elif algorithm == "GroupDRO":
        hparams["groupdro_eta"] = (1e-2, 10 ** random_state.uniform(-3, -1))
    elif algorithm in ("MMD", "CORAL"):
        hparams["mmd_gamma"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm in ("MLDG", "SOMLDG"):
        hparams["mldg_beta"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm == "MTL":
        hparams["mtl_ema"] = (0.99, random_state.choice([0.5, 0.9, 0.99, 1.0]))
    elif algorithm == "VREx":
        hparams["vrex_lambda"] = (1e1, 10 ** random_state.uniform(-1, 5))
        hparams["vrex_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm == "SAM":
        hparams["rho"] = (0.05, random_state.choice([0.01, 0.02, 0.05, 0.1]))
    elif algorithm == "CutMix":
        hparams["beta"] = (1.0, 1.0)
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        hparams["cutmix_prob"] = (1.0, 1.0)

    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
