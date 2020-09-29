_base_ = [
    '../_base_/models/mask_track_rcnn_x101_32x4d_fpn.py',
    '../_base_/datasets/ytvos.py',
    '../_base_/schedules/schedule_masktrackrcnn.py', '../_base_/default_runtime.py'
]
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth'
data = dict(samples_per_gpu=4)