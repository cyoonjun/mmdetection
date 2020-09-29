import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.core import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.utils import get_root_logger



@HEADS.register_module()
class TrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero 
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 with_avg_pool=False,
                 num_fcs=2,
                 in_channels=256,
                 roi_feat_size=7,
                 fc_out_channels=1024,
                 match_coeff=None,
                 bbox_dummy_iou=0,
                 dynamic=True,
                #  loss_track=dict(
                #      type='CrossEntropyLoss',
                #      use_sigmoid=False,
                #      loss_weight=1.0)
                 ):
        super(TrackHead, self).__init__()
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size) 
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (in_channels
                          if i == 0 else fc_out_channels)
            fc = nn.Linear(in_channels, fc_out_channels)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic = dynamic
        # self.fp16_enabled = False
        # self.loss_track = build_loss(loss_track)


    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1, 
                device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta),dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert(len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                + self.match_coeff[2] * label_delta
    
    @auto_fp16()
    def forward(self, x, ref_x, x_n, ref_x_n):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch, 
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        
        # when batch = 1
        # x: [128, 256, 7, 7]
        # ref_x: [1, 256, 7, 7]
        assert len(x_n) == len(ref_x_n)
        device = x.device
        if self.with_avg_pool:
            x = self.avg_pool(x)
            ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)
        ref_x = ref_x.view(ref_x.size(0), -1)
        # x: [128, 12544]
        # ref_x: [1, 12544]
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        # x: [128, 1024]
        # ref_x: [1, 1024]
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        for i in range(n):
            # x_split[i]: [128, 1024]
            # ref_x_split[i]: [1, 1024]
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            # ref_x_split[i]: [128, 1]
            prods.append(prod)
        if self.dynamic:
            match_score = []
            for prod in prods:
                m = prod.size(0)
                dummy = torch.zeros(m, 1, device=device)
                prod_ext = torch.cat([dummy, prod], dim=1)
                match_score.append(prod_ext)
        else:
            dummy = torch.zeros(n, m, device=device)
            prods_all = torch.cat(prods, dim=0)
            match_score = torch.cat([dummy, prods_all], dim=2)
        return match_score

    # @force_fp32(apply_to=('track_pred', ))
    def loss(self,
             match_score,
             ids,
             id_weights,
             reduce=True):
        losses = dict()
        if self.dynamic:
            n = len(match_score) # n is batch size
            x_n = [s.size(0) for s in match_score] # x_n = 128
            ids = torch.split(ids, x_n, dim=0)
            loss_match = 0.
            match_acc = 0.
            n_total = 0
            batch_size = len(ids)
            for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
                valid_idx = torch.nonzero(cur_weights, as_tuple=False).squeeze(dim=1) # TODO:
                if len(valid_idx.size()) == 0: continue
                n_valid = valid_idx.size(0)
                n_total += n_valid
                loss_match += weighted_cross_entropy(
                    score, cur_ids, cur_weights, reduce=reduce)
                match_acc += accuracy(torch.index_select(score, 0, valid_idx), 
                                      torch.index_select(cur_ids, 0, valid_idx)) * n_valid
            losses['loss_match'] = loss_match / n
            if n_total > 0:
                losses['match_acc'] = match_acc / n_total
        else:
            if match_score is not None:
                valid_idx = torch.nonzero(cur_weights, as_tuple=False).squeeze()
                losses['loss_match'] = weighted_cross_entropy(
                    match_score, ids, id_weights, reduce=reduce)
                losses['match_acc'] = accuracy(torch.index_select(match_score, 0, valid_idx), 
                                                torch.index_select(ids, 0, valid_idx))
        return losses


def weighted_cross_entropy(pred, label, weight, avg_factor=None,
                           reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res
