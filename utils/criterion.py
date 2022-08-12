# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):
        # print(score[0].shape)
        # print(target.shape)
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)

        else:
            raise ValueError("lengths of prediction and target are not identical!")


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        # print(pred.shape)
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

        # new ohem
        # crition = OhemCrossEntropy2dTensor(ignore_index=self.ignore_label, thresh=self.thresh, min_kept=self.min_kept,
        #                                    use_weight=True).cuda()
        # return crition(score, target)

    def forward(self, score, target):

        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                        (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])

        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)

        else:
            raise ValueError("lengths of prediction and target are not identical!")


class DiceLoss(nn.Module):
    def __init__(self, num_classes=19, ignore=255):
        super(DiceLoss, self).__init__()
        self.ignore = ignore
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1).exp()
        index = torch.where(target == self.ignore)
        target[index] = 0.

        # print(torch.unique(target))
        # print(torch.unique(target))
        target = F.one_hot(target.to(torch.int64), num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
        gt = target.float()
        pred.float()
        smooth = 0.0001
        iflat = pred.contiguous().view(-1)
        tflat = gt.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class LogCoshDiceLoss(torch.nn.Module):
    def __init__(self, num_classes=19):
        super(LogCoshDiceLoss, self).__init__()
        self.loss = DiceLoss(num_classes=num_classes)

    def forward(self, output, target):
        if not (isinstance(output, list) or isinstance(output, tuple)):
            output = [output]
        # print(len(output))
        balance_weights = [0.4, 1]
        sb_weights = 1
        if len(balance_weights) == len(output):
            loss_1 = balance_weights[0] * self.loss(output[0], target)
            loss_2 = balance_weights[1] * self.loss(output[1], target)
            x = loss_1 + loss_2

            return x
        else:
            return sb_weights * self.loss(output[0], target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class FocalLossCE(nn.Module):
    def __int__(self, gamma=2, alpha=0.5, ignore_index=255):
        pass


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    # print(weight)
    pos_num = pos_index.sum()
    # print(pos_num)
    neg_num = neg_index.sum()
    # print(neg_num)
    sum_num = pos_num + neg_num
    print('pos_index: {}'.format(pos_index))
    print('pos_num: {}'.format(pos_num))
    print('neg_index: {}'.format(neg_index))
    print('neg_sum: {}'.format(neg_num))
    weight[(pos_index)] = (float(neg_num) * 1.0 / float(sum_num))
    weight[(neg_index)] = float(pos_num) * 1.0 / float(sum_num)
    print(weight)
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
    # print('xyz: {}'.format(loss))
    return loss


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            # print('Labels: {}'.format(num_valid))
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss


        return loss


if __name__ == '__main__':
    a = torch.zeros(2, 64, 64)
    a[:, 5, :] = 1
    pre = torch.randn(2, 1, 16, 16)

    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))
