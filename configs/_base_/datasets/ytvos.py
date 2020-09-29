dataset_type = 'YTVOSDataset'
data_root = 'data/YouTubeVIS/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

#TODO:
train_pipeline = [
    dict(type='LoadVideoFromFiles'), # TODO
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_track=True, poly2mask=False), # TODO with_track
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True), # TODO
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'), # TODO
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_pids',
                               'ref_img', 'ref_bboxes']), # TODO
]

#TODO:
test_pipeline = [
    dict(type='LoadVideoFromFiles'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/JPEGImages',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm']) #TODO
