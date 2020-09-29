import argparse
import torch
import numpy as np

import json
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save_path', 
        type=str,
        help='path to save visual result')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--load_result', 
        action='store_true', 
        help='whether to load existing result')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        help='eval types')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args

def single_test(model, data_loader, show=False, save_path=''):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES,
                                     save_vis = True,
                                     save_path = save_path,
                                     is_video = True)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def results2json_videoseg(dataset, results, out_file):
    json_results = []
    vid_objs = {}
    for idx in range(len(dataset)):
      # assume results is ordered
      vid_id, frame_id = dataset.img_ids[idx]
      if idx == len(dataset) - 1 :
        is_last = True
      else:
        _, frame_id_next = dataset.img_ids[idx+1]
        is_last = frame_id_next == 0
      det, seg = results[idx]
      for obj_id in det:
        bbox = det[obj_id]['bbox']
        segm = seg[obj_id]
        label = det[obj_id]['label']
        if obj_id not in vid_objs:
          vid_objs[obj_id] = {'scores':[], 'cats':[], 'segms':{}}
        vid_objs[obj_id]['scores'].append(bbox[4])
        vid_objs[obj_id]['cats'].append(label)
        segm['counts'] = segm['counts'].decode()
        vid_objs[obj_id]['segms'][frame_id] = segm
      if is_last:
        # store results of  the current video
        for obj_id, obj in vid_objs.items():
          data = dict()
          data['video_id'] = vid_id + 1
          data['score'] = np.array(obj['scores']).mean().item()
          # majority voting for sequence category
          data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item() + 1
          vid_seg = []
          for fid in range(frame_id + 1):
            if fid in obj['segms']:
              vid_seg.append(obj['segms'][fid])
            else:
              vid_seg.append(None)
          data['segmentations'] = vid_seg
          json_results.append(data)
        vid_objs = {}
    mmcv.dump(json_results, out_file)


def ytvos_eval(result_file, result_types, ytvos, max_dets=(100, 300, 1000)):

    if mmcv.is_str(ytvos):
        ytvos = YTVOS(ytvos)
    assert isinstance(ytvos, YTVOS)

    if len(ytvos.anns) == 0:
        print("Annotations does not exist")
        return
    assert result_file.endswith('.json')
    ytvos_dets = ytvos.loadRes(result_file)

    vid_ids = ytvos.getVidIds()
    for res_type in result_types:
        iou_type = res_type
        ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type)
        ytvosEval.params.vidIds = vid_ids
        if res_type == 'proposal':
            ytvosEval.params.useCats = 0
            ytvosEval.params.maxDets = list(max_dets)
        ytvosEval.evaluate()
        ytvosEval.accumulate()
        ytvosEval.summarize()



def main():
    args = parse_args()

    assert args.out or args.eval or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')


    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    # if cfg.model.get('neck'):
    #     if cfg.model.neck.get('rfp_backbone'):
    #         if cfg.model.neck.rfp_backbone.get('pretrained'):
    #             cfg.model.neck.rfp_backbone.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # print("model is")
    # print(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_module(model)
    # if 'CLASSES' in checkpoint['meta']:
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_test(model, data_loader, args.show, save_path=args.save_path)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            if not args.load_result:
                print('writing results to {}'.format(args.out))
                mmcv.dump(outputs, args.out)
        
    
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if not isinstance(outputs[0], dict):
                result_file = args.out + '.json'
                results2json_videoseg(dataset, outputs, result_file)
                ytvos_eval(result_file, eval_types, dataset.ytvos)
            else:
                NotImplemented

if __name__ == '__main__':
    main()
