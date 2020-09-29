_base_ = [
    '../_base_/models/mask_track_rcnn_r50_fpn_v2.py',
    '../_base_/datasets/ytvos.py',
    '../_base_/schedules/schedule_masktrackrcnn.py', '../_base_/default_runtime.py'
]
load_from = "http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)