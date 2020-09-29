import torch
import numpy as np

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_overlaps
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin#, TrackTestMixin #TODO:


@HEADS.register_module()
class CustomRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_track_head(self, track_head):
        """Initialize ``track_head``"""
        self.track_head = build_head(track_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
        if self.with_track:
            self.track_head.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore,
                      ref_x,
                      ref_bboxes,
                      gt_pids,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i], gt_pids[i]) # TODO:
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    gt_pids[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        #TODO:
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            ref_rois = bbox2roi(ref_bboxes)
            ref_bbox_img_n = [x.size(0) for x in ref_bboxes]

            # print(rois.shape) # [128, 5]
            # print(bbox_img_n) # [128]
            # print(ref_rois.shape) # [2, 5]
            # print(ref_bbox_img_n) # [2]

            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            ref_bbox_feats = self.bbox_roi_extractor(
                ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)
            
            # print(self.bbox_roi_extractor.num_inputs) # 4
            # print(bbox_feats.shape) # [128, 256, 7, 7]
            # print(ref_bbox_feats.shape) # [2, 256, 7, 7]
            
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            #TODO:
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            # fetch bbox and object_id targets
            # bbox_targets -> labels, label_weights, bbox_targets, bbox_weights
            
            #TODO:
            bbox_targets, (ids, id_weights) = self.bbox_head.get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg)
            
            # print(cls_score.shape) # [128, 41]
            # print(bbox_pred.shape) # [128, 160]
            # print(rois.shape) # [128, 5]
            # print(bbox_targets) 

            #TODO:
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, rois, *bbox_targets)
            losses.update(loss_bbox)

            
            #TODO:
            match_score = self.track_head(bbox_feats, ref_bbox_feats, 
                                          bbox_img_n, ref_bbox_img_n)
            loss_match = self.track_head.loss(match_score, ids, id_weights)
            losses.update(loss_match)


        # mask head forward and loss
        if self.with_mask:
            # loss_bbox: loss_cls, acc loss_bbox
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_feats,
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets, (ids, id_weights) = self.bbox_head.get_targets(sampling_results, 
                                                                     gt_bboxes,
                                                                     gt_labels, 
                                                                     self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results, (ids, id_weights)

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
            # print(mask_results['mask_pred'].shape) # torch.Size([97, 40, 28, 28]) # TODO
            # print(mask_results['mask_feats'].shape) # torch.Size([97, 256, 14, 14])
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels - 1 for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        is_first = (img_metas[0]['frame_id'] == 0) #TODO:
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        #TODO:
        # since det labels are 0(bg), 1~41(class), mask labels are 0~40(class)
        det_labels = det_labels - 1

        if det_bboxes.nelement()==0:
            det_obj_ids=np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes =  None
                self.prev_roi_feats = None
                self.prev_det_labels = None
            return det_bboxes, det_labels, det_obj_ids

        res_det_bboxes = det_bboxes.clone()
        device = res_det_bboxes.device
        if rescale:
            res_det_bboxes[:, :4] *= torch.from_numpy(scale_factor).to(device)

        det_rois = bbox2roi([res_det_bboxes])
        det_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], det_rois)
        # recompute bbox match feature
        
        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels
        else:
            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]
            match_score = self.track_head(det_roi_feats, self.prev_roi_feats,
                                      bbox_img_n, prev_bbox_img_n)[0]
            match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1,1)).float()
            bbox_ious = bbox_overlaps(det_bboxes[:,:4], self.prev_bboxes[:,:4])
            # compute comprehensive score
            comp_scores = self.track_head.compute_comp_scores(match_logprob, 
                det_bboxes[:,4].view(-1, 1),
                bbox_ious,
                label_delta,
                add_bbox_dummy=True)
            match_likelihood, match_ids = torch.max(comp_scores, dim =1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object, 
            # add tracking features/bboxes of new object
            match_ids = match_ids.cpu().numpy().astype(np.int32)
            det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    # add new object
                    det_obj_ids[idx] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score 
                    obj_id = match_id - 1
                    match_score = comp_scores[idx, match_id]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        # udpate feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]

        return det_bboxes, det_labels, det_obj_ids

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         det_obj_ids=None,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)


            mask_rois = bbox2roi([_bboxes])

            mask_results = self._mask_forward(x, mask_rois)
            segm_result = self.mask_head.get_seg_masks(
                mask_results['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale, det_obj_ids=det_obj_ids)
        return segm_result

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels, det_obj_ids = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        
        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, 
                rescale=rescale, det_obj_ids=det_obj_ids)
            
            return bbox_results, segm_results

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

def bbox2result_with_id(bboxes, labels, obj_ids, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return dict()
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        results={}
        for bbox, label, obj_id in zip(bboxes, labels, obj_ids):
          if obj_id >= 0:
            results[obj_id]={'bbox': bbox, 'label':label}
        return results