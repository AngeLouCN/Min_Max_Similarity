import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from metrics import dice_coef

BCE = torch.nn.BCELoss()

def CE(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()

def structure_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()

# def CE(pred,mask):
#     pred = pred.view(-1)
#     mask = mask.view(-1)
#     inter = (pred*mask).sum()
#     union = (pred + mask).sum()
#     iou_loss = 1-(inter+1)/(union-inter+1)

#     return iou_loss
    
# labels for adversarial training
pred_label = 0
gt_label = 1


def make_Dis_label(label, gts):
    D_label = np.ones(gts.shape) * label
    D_label = Variable(torch.FloatTensor(D_label)).cuda()

    return D_label


def calc_loss(pred, target, bce_weight=0.5):
    bce = CE(pred, target)
    # dl = 1 - dice_coef(pred, target)
    # loss = bce * bce_weight + dl * bce_weight

    return bce


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    loss1 = calc_loss(logit_S1, labels_S1)
    loss2 = calc_loss(logit_S2, labels_S2)

    return loss1 + loss2


def loss_sup_1(logit_S1, labels_S1):
    loss1 = calc_loss(logit_S1, labels_S1)
    return loss1

def loss_sup_2(logit_S2, labels_S2):
    loss2 = calc_loss(logit_S2, labels_S2)
    return loss2


def loss_diff(u_prediction_1, u_prediction_2, batch_size):
    a = CE(u_prediction_1, Variable(u_prediction_2, requires_grad=False))
    a = a.item()

    b = CE(u_prediction_2, Variable(u_prediction_1, requires_grad=False))
    b = b.item()

    loss_diff_avg = (a + b)
    return loss_diff_avg / batch_size



