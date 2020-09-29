_base_ = [
    '../_base_/models/mask_track_rcnn_r50_fpn_gn.py',
    '../_base_/datasets/ytvos.py',
    '../_base_/schedules/schedule_masktrackrcnn.py', '../_base_/default_runtime.py'
]
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'