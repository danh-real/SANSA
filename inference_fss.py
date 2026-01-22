import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join
from collections import defaultdict
from tabulate import tabulate

import opts
from models.sansa.sansa import build_sansa
from datasets import build_dataset
from util.commons import make_deterministic, setup_logging, resume_from_checkpoint
import util.misc as utils
from util.promptable_utils import build_prompt_dict
from util.metrics import AverageMeter, Evaluator

import torch.distributed as dist

def score_cal(seg_map, prd_map):
    '''
    labels B * 1
    seg_map B * H * W
    prd_map B * H * W
    '''
    assert seg_map.ndim == prd_map.ndim
    assert seg_map.ndim >= 2
    if seg_map.ndim == 2:
        seg_map = seg_map.unsqueeze(0)
        prd_map = prd_map.unsqueeze(0)
        
    total_num = seg_map.shape[0]
    
    seg_map = seg_map.reshape(total_num, -1)
    prd_map = prd_map.reshape(total_num, -1)
    dot_product = (seg_map * prd_map)
    b_seg_map = 1 - seg_map
    b_prd_map = 1 - prd_map
    b_dot_product = (b_seg_map * b_prd_map)

    sum_dot = torch.sum(dot_product, dim=-1)
    sum_seg = torch.sum(seg_map, dim=-1)
    sum_prd = torch.sum(prd_map, dim=-1)
    b_sum_dot = torch.sum(b_dot_product, dim=-1)
    b_sum_seg = torch.sum(b_seg_map, dim=-1)
    b_sum_prd = torch.sum(b_prd_map, dim=-1)

    iou_score = sum_dot/((sum_seg + sum_prd)-sum_dot)
    dice_score = 2.*sum_dot / (sum_seg+sum_prd)
    
    b_iou_score = b_sum_dot/((b_sum_seg + b_sum_prd)-b_sum_dot)
    fb_iou_score = (iou_score + b_iou_score) / 2

    return (iou_score, dice_score, fb_iou_score)

def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    print(args)

    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device)
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)

    print(f"number of params: {n_parameters}")
    print('Start inference')

    mIoU = eval_fss(model, args)
    return mIoU


def eval_fss(model: torch.nn.Module, args: argparse.Namespace) -> float:
    """
    Evaluate SANSA on the few-shot segmentation benchmark.
    Computes and prints mIoU across the validation set.
    """
    # load data
    validation_ds = 'coco' if args.dataset_file == 'multi' else args.dataset_file
    print(f'Evaluating {validation_ds} - fold: {args.fold}')
    ds = build_dataset(validation_ds, image_set='val', args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    device = next(model.parameters()).device
    average_meter = AverageMeter(args.dataset_file, ds.class_ids, ds.nclass)
    score_per_class = {}

    pbar = tqdm(dataloader, ncols=80, desc='runn avg.', disable=(utils.get_rank() != 0), file=sys.stderr, dynamic_ncols=True)
    for idx, batch in enumerate(pbar):
        query_img, query_mask = batch['query_img'], batch['query_mask']
        support_imgs, support_masks = batch['support_imgs'], batch['support_masks']
        class_id = batch['class_id'][0]
        
        if isinstance(class_id, float):
            class_id = int(class_id)
        elif isinstance(class_id, torch.Tensor):
            class_id = int(class_id.item())
        
        # NOTE
        imgs = torch.cat([support_imgs[0], query_img]).unsqueeze(0) # b t c h w
        # imgs = torch.cat([support_imgs, query_img], dim=1) # b t c h w
        img_h, img_w = imgs.shape[-2:]

        imgs = imgs.to(args.device)
        prompt_dict = build_prompt_dict(support_masks, args.prompt, n_shots=args.shots, train_mode=False, device=model.device)

        with torch.no_grad():
            outputs = model(imgs, prompt_dict)

        pred_masks = outputs["pred_masks"].unsqueeze(0)  # [1, T, h, w]
        pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode='bilinear', align_corners=False) 
        pred_masks = (pred_masks.sigmoid() > args.threshold).float()[0]
        
        query_mask = query_mask.to(device=pred_masks.device, non_blocking=True)
        # NOTE
        iou_score, dice_score, fb_iou_score = score_cal(query_mask, pred_masks[[-1], :, :])
        # iou_score, dice_score, fb_iou_score = score_cal(query_mask[0], pred_masks[args.shots:, :, :])
        
        if class_id not in score_per_class.keys():
            score_per_class[class_id] = {
                "iou": torch.FloatTensor([]).to(device=args.device),
                "dice": torch.FloatTensor([]).to(device=args.device),
                "fb_iou": torch.FloatTensor([]).to(device=args.device),
            }
        score_dict = score_per_class[class_id]
        score_dict["iou"] = torch.cat([score_dict["iou"], iou_score.detach()])
        score_dict["dice"] = torch.cat([score_dict["dice"], dice_score.detach()])
        score_dict["fb_iou"] = torch.cat([score_dict["fb_iou"], fb_iou_score.detach()])
    
    avg = {
        "iou": torch.FloatTensor([]).to(device=args.device),
        "dice": torch.FloatTensor([]).to(device=args.device),
        "fb_iou": torch.FloatTensor([]).to(device=args.device),
    }
    
    table_data = []
    
    for name, metrics_dict in score_per_class.items():
        miou = metrics_dict["iou"].mean(dim=0, keepdim=True)
        mdice = metrics_dict["dice"].mean(dim=0, keepdim=True)
        mfb_iou = metrics_dict["fb_iou"].mean(dim=0, keepdim=True)
        
        table_data.append((
            name, 
            miou.item(), 
            mdice.item(), 
            mfb_iou.item(),
        ))
        
        avg["iou"] = torch.cat([avg["iou"], miou])
        avg["dice"] = torch.cat([avg["dice"], mdice])
        avg["fb_iou"] = torch.cat([avg["fb_iou"], mfb_iou])
        
    avg["iou"] = avg["iou"].mean()
    avg["dice"] = avg["dice"].mean()
    avg["fb_iou"] = avg["fb_iou"].mean()
            
    table_data.append((
        "Average",
        avg["iou"].item(),
        avg["dice"].item(),
        avg["fb_iou"].item(),
    ))

    print(tabulate(table_data, headers=["name", "iou", "dice", "fb_iou"], floatfmt=".4f", tablefmt="grid"))
    
    # average_meter.write_result(args.dataset_file)
    # miou, fb_iou, _ = average_meter.compute_iou()
    miou = avg["iou"]
    mdice = avg["dice"]
    mfbiou = avg["fb_iou"]
    if args.distributed:
        dist.all_reduce(mdice), dist.all_reduce(miou), dist.all_reduce(mfbiou)
    print('Fold %d mIoU: %5.4f \t mDice: %5.4f  \t mFB-IoU: %5.4f' % (
        args.fold,
        miou,
        mdice,
        mfbiou,
    ))
    print('==================== Finished Testing ====================')

    return mdice


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
