# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:02:24 2022

@author: loua2
"""

import argparse
import os
from datetime import datetime
from distutils.dir_util import copy_tree
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from data import image_loader
from loss import loss_sup, loss_diff
from metrics import dice_coef
from utils import get_logger, create_dir
from contrastive_loss import ConLoss, contrastive_loss_sup
from model.projector import projectors, classifier
from model.pretrained_unet import preUnet
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=(512,288), help='training dataset size')
parser.add_argument('--dataset', type=str, default='kvasir', help='dataset name')
parser.add_argument('--split', type=float, default=1, help='training data ratio')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--ratio', type=float, default=0.5, help='labeled data ratio')
opt = parser.parse_args()
pixel_wise_contrastive_loss_criter = ConLoss()
contrastive_loss_sup_criter = contrastive_loss_sup()


def adjust_lr(optimizer, init_lr, epoch, max_epoch):
    lr_ = init_lr * (1.0 - epoch / max_epoch) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_


class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.model_1 = preUnet()
        self.model_2 = preUnet()
        self.projector_1 = projectors()
        self.projector_2 = projectors()
        self.classifier_1 = classifier()
        self.classifier_2 = classifier()
        self.best_mIoU, self.best_dice_coeff = 0, 0
        self._init_configure()
        self._init_logger()

    def _init_configure(self):
        with open('configs/config.yml') as fp:
            self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        log_dir = 'logs/' + opt.dataset + '/train/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.image_save_path_1 = log_dir + "/saved_images_1"
        self.image_save_path_2 = log_dir + "/saved_images_2"

        create_dir(self.image_save_path_1)
        create_dir(self.image_save_path_2)

        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        print('Generator Learning Rate: {} Critic Learning Rate'.format(opt.lr))
        self.model_1.cuda()
        self.model_2.cuda()
        
        params = list(self.model_1.parameters()) + list(self.model_2.parameters())

        optimizer = torch.optim.Adam(params,lr=opt.lr)


        image_root = './data/'+ opt.dataset +'/train/image/'
        gt_root = './data/'+ opt.dataset +'/train/mask/'
        val_img_root = './data/'+ opt.dataset +'/test/image/'
        val_gt_root = './data/'+ opt.dataset +'/test/mask/'


        self.logger.info("Split Percentage : {} Labeled Data Ratio : {}".format(opt.split, opt.ratio))
        train_loader_1, train_loader_2, unlabeled_train_loader, val_loader = image_loader(image_root, gt_root,val_img_root,val_gt_root,
                                                                                          opt.batchsize, opt.trainsize,
                                                                                          opt.split, opt.ratio)
        self.logger.info(
            "train_loader_1 {} train_loader_2 {} unlabeled_train_loader {} val_loader {}".format(len(train_loader_1),
                                                                                                 len(train_loader_2),
                                                                                                 len(unlabeled_train_loader),
                                                                                                 len(val_loader)))
        print("Let's go!")
        for epoch in range(1, opt.epoch):

            running_loss = 0.0
            running_dice_val_1 = 0.0
            running_dice_val_2 = 0.0
            

            for i, data in enumerate(zip(train_loader_1, train_loader_2, unlabeled_train_loader)):

                inputs_S1, labels_S1 = data[0][0], data[0][1]
                inputs_S2, labels_S2 = data[1][0], data[1][1]
                inputs_U, labels_U = data[2][0], data[2][1]

                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()
                inputs_S2, labels_S2 = Variable(inputs_S2), Variable(labels_S2)
                inputs_S2, labels_S2 = inputs_S2.cuda(), labels_S2.cuda()
                inputs_U = Variable(inputs_U)
                inputs_U = inputs_U.cuda()

                optimizer.zero_grad()

                # Train Model 1
                prediction_1 = self.model_1(inputs_S1)
                prediction_1_1 = torch.sigmoid(prediction_1)

                feat_1 = self.model_1(inputs_U)
                u_prediction_1 = torch.sigmoid(feat_1)

                # Train Model 2
                prediction_2 = self.model_2(inputs_S2)
                prediction_2_2 = torch.sigmoid(prediction_2)

                feat_2 = self.model_2(inputs_U)
                u_prediction_2 = torch.sigmoid(feat_2)

                self.projector_1.cuda()
                self.projector_2.cuda()
                self.classifier_1.cuda()
                self.classifier_2.cuda()
                feat_q = self.projector_1(feat_1)
                feat_k = self.projector_2(feat_2)
                feat_l_q = self.classifier_1(prediction_1)
                feat_l_k = self.classifier_2(prediction_2)
                Loss_sup = loss_sup(prediction_1_1, prediction_2_2, labels_S1, labels_S2)
                Loss_diff = loss_diff(u_prediction_1, u_prediction_2, opt.batchsize)
                Loss_contrast = pixel_wise_contrastive_loss_criter(feat_q,feat_k)
                Loss_contrast_2 = contrastive_loss_sup_criter(feat_l_q,feat_l_k)
                

                seg_loss = 0.25*Loss_sup +0.25*Loss_diff +0.25*Loss_contrast+0.25*Loss_contrast_2
                
                seg_loss.backward()
                running_loss += seg_loss.item()
                optimizer.step()
                
                adjust_lr(optimizer, opt.lr, epoch, opt.epoch)
                
                    


            epoch_loss = running_loss / (len(train_loader_1) + len(train_loader_2))
            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, opt.epoch, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    images = Variable(images)
                    gts = Variable(gts)
                    images = images.cuda()
                    gts = gts.cuda()

                    prediction_1 = self.model_1(images)
                    prediction_1 = torch.sigmoid(prediction_1)

                    prediction_2 = self.model_2(images)
                    prediction_2 = torch.sigmoid(prediction_2)

                dice_coe_1 = dice_coef(prediction_1, gts)
                running_dice_val_1 += dice_coe_1
                dice_coe_2 = dice_coef(prediction_2, gts)
                running_dice_val_2 += dice_coe_2

            epoch_dice_val_1 = running_dice_val_1 / len(val_loader)

            self.logger.info('Validation dice coeff model 1: {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Validation_1/DSC', epoch_dice_val_1, epoch)

            epoch_dice_val_2 = running_dice_val_2 / len(val_loader)

            self.logger.info('Validation dice coeff model 1: {}'.format(epoch_dice_val_2))
            self.writer.add_scalar('Validation_1/DSC', epoch_dice_val_2, epoch)

            mdice_coeff_1 = epoch_dice_val_1
            mdice_coeff_2 = epoch_dice_val_2

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True

                if not os.path.exists(self.image_save_path_1):
                    os.makedirs(self.image_save_path_1)

                copy_tree(self.image_save_path_1, self.save_path + '/best_model_predictions_1')
                self.patience = 0
            else:
                self.save_best_model_1 = False
                self.patience += 1

            if self.best_dice_coeff_2 < mdice_coeff_2:
                self.best_dice_coeff_2 = mdice_coeff_2
                self.save_best_model_2 = True

                if not os.path.exists(self.image_save_path_2):
                    os.makedirs(self.image_save_path_2)

                copy_tree(self.image_save_path_2, self.save_path + '/best_model_predictions_2')
                self.patience = 0
            else:
                self.save_best_model_2 = False
                self.patience += 1

            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                torch.save(self.model_1.state_dict(), Checkpoints_Path + '/Model_1.pth')
            if self.save_best_model_2:
                torch.save(self.model_2.state_dict(), Checkpoints_Path + '/Model_2.pth')

            self.logger.info(
                'current best dice coef model 1 {}, model 2 {}'.format(self.best_dice_coeff_1, self.best_dice_coeff_2))
            self.logger.info('current patience :{}'.format(self.patience))


if __name__ == '__main__':
    train_network = Network()
    train_network.run()