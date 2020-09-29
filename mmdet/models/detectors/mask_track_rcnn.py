from ..builder import DETECTORS
from .two_stage_for_track import TwoStageDetectorForTrack


@DETECTORS.register_module()
class MaskTrackRCNN(TwoStageDetectorForTrack):
    """Implementation of `MaskTrack R-CNN"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(MaskTrackRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
