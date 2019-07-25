# -*- coding: utf-8 -*-

import _init_paths

import torch
import cv2
import os
import json
import math
import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import torch.utils.data as data

class AIChallengeKpOri(data.Dataset):
    # 0:  不可见
    # 1： 可见
    # 2： 不在图内或不可推测
    # [365, 345, 2, 279, 454, 2, 374, 489, 2, 172, 437, 2, 324, 616, 2, 178, 596, 2, 267, 762, 2, 203, 753, 2, 270, 687, 1, 160, 646, 1, 355, 783, 2, 146, 741, 2, 445, 935, 2, 105, 885, 1]
    # 1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，13/头顶，14/脖子

    num_classes = 1
    num_joints = 14
    default_resolution = [192, 256] # width height
    mean = np.array([0, 0, 0], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([1, 1, 1], dtype=np.float32).reshape(1, 1, 3)
    flip_idx = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]

    def __init__(self, opt, split, down_ratio=2, mse_loss=False, max_objs=3, debug=False):
        super(AIChallengeKpOri, self).__init__()
        self.down_ratio = down_ratio
        self.mse_loss = mse_loss
        self.max_objs = max_objs
        self.split = split
        self.opt = opt
        self.debug = debug

        self.data_dir = os.path.join(opt.data_dir, 'ai_challenger')
        self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
        self.annot_path = os.path.join(self.data_dir, '{}.json').format(split)

        print('==> initializing ai_challenger {} data.'.format(split))

        with open(self.annot_path) as f:
            gt = json.load(f)

        self.images = []
        for idx in range(len(gt)):
            anno = gt[idx]
            kp_anno = anno['keypoint_annotations']
            bbox_anno = anno['human_annotations']
            anno_keys = anno['human_annotations'].keys()
            file_name = anno['image_id'] + '.jpg'
            img_path = os.path.join(self.img_dir, file_name)

            if len(anno_keys) <= self.max_objs and os.path.exists(img_path):
                self.images.append({
                    "image_id": anno['image_id'],
                    "file_name": file_name,
                    "keypoints": kp_anno,
                    "bbox": bbox_anno,
                    "keys": anno_keys,
                    "category_id": 1,
                })

        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

        if self.debug:
            for idx in range(40, 60):
                self.get_anno(idx)

    def get_anno(self, index):
        anno = self.images[index]
        file_name = anno['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)
        num_objs = min(len(anno['bbox']), self.max_objs)

        if False:
            cv2.imwrite('demo_imgs/ai_challenge_kp_ori_{}.png'.format(index), img)
            kps = anno['keypoints']
            for kp in kps.values():
                for idx in range(len(kp)//3):
                    if kp[idx*3+2] < 2:
                        cv2.circle(img, (int(kp[idx*3+0]), int(kp[idx*3+1])), 3, (255, 0, 255), -1)
                        cv2.putText(img,'%s:%f'%(idx, kp[idx*3+2]), (int(kp[idx*3+0]), int(kp[idx*3+1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.imwrite('demo_imgs/ai_challenge_kp_ori_kps_{}.png'.format(index), img)

            bboxs = anno['bbox']
            for bbox in bboxs.values():
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imwrite('demo_imgs/ai_challenge_kp_ori_bbox_{}.png'.format(index), img)

        height, width = img.shape[0], img.shape[1]
        center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        max_length = int(abs(height - width) * 0.3 + min(height, width))
        rot = 90 if height < width else 0
        if np.random.random() < 0.5:
            rot += int((np.random.random() - 0.5) * 10)

        flipped = False
        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] =  width - center[0] - 1

        trans_input = get_affine_transform(center, max_length, rot, [self.default_resolution[0], self.default_resolution[1]])
        inp_ori = cv2.warpAffine(img, trans_input, (self.default_resolution[0], self.default_resolution[1]), flags=cv2.INTER_LINEAR)
        if self.debug:
            cv2.imwrite('demo_imgs/ai_challenge_kp_input_ori_{}.png'.format(index), inp_ori)
        inp = (inp_ori.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_w, output_h = self.default_resolution[0] // self.down_ratio, self.default_resolution[1] // self.down_ratio
        output_res = max(output_w, output_h)
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(center, max_length, rot, [output_w, output_h])
        trans_output = get_affine_transform(center, max_length, 0, [output_w, output_h])

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_h, output_w), dtype=np.float32)
        dense_kps = np.zeros((num_joints, 2, output_h, output_w), dtype=np.float32)
        dense_kps_mask = np.zeros((num_joints, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.mse_loss else draw_umich_gaussian
        gt_det = []
        for k,val in enumerate(anno['keys']):
            cls_id = int(anno['category_id']) - 1
            bbox = np.array(anno['bbox'][val]).astype(np.float32)
            pts = np.array(anno['keypoints'][val], np.float32).reshape(num_joints, 3)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
            bbox[:2] = affine_transform(bbox[:2], trans_output_rot)
            bbox[2:] = affine_transform(bbox[2:], trans_output_rot)
            bbox[0::2] = np.clip(bbox[0::2], 0, output_w - 1)
            bbox[1::2] = np.clip(bbox[1::2], 0, output_h - 1)
            h, w = abs(bbox[3] - bbox[1]), abs(bbox[2] - bbox[0])
            if self.debug:
                bbox_ = (bbox).astype(np.int32) * self.down_ratio
                cv2.rectangle(inp_ori, (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (255, 0, 0), 2)
                cv2.imwrite('demo_imgs/ai_challenge_kp_bbox_{}.png'.format(index), inp_ori)

            radius = gaussian_radius((math.ceil(h), math.ceil(w)), 0.5)
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            num_kpts = pts[:, 2].sum()
            if num_kpts == 0:
                hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                reg_mask[k] = 0

            hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)), 0.85)
            hp_radius = max(0, int(hp_radius))
            for j in range(num_joints):
                if pts[j, 2] < 2:
                    pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                    if pts[j, 0] >= 0 and pts[j, 0] < output_w and \
                       pts[j, 1] >= 0 and pts[j, 1] < output_h:
                        kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                        kps_mask[k, j * 2: j * 2 + 2] = 1
                        pt_int = pts[j, :2].astype(np.int32)
                        hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                        hp_ind[k * num_joints + j] = pt_int[1] * output_w + pt_int[0]
                        hp_mask[k * num_joints + j] = 1
                        draw_gaussian(hm_hp[j], pt_int, hp_radius)

            draw_gaussian(hm[cls_id], ct_int, radius)
            gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                           ct[0] + w / 2, ct[1] + h / 2, 1] +
                           pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

        if self.debug:
            cv2.imwrite('demo_imgs/ai_challenge_kp_bbox_{}.png'.format(index), inp_ori)
            kp_gausian = cv2.addWeighted(cv2.resize(inp_ori, (output_w, output_h)).astype(np.float32), 0.5, np.repeat(hm_hp.sum(axis=0).reshape(output_h, output_w, 1), 3, axis=2)*255, 0.5, 0)
            cv2.imwrite('demo_imgs/ai_challenge_kp_kp_gaussian_{}.png'.format(index), kp_gausian)

            bbox_gausian = cv2.addWeighted(cv2.resize(inp_ori, (output_w, output_h)).astype(np.float32), 0.5, np.repeat(hm.sum(axis=0).reshape(output_h, output_w, 1), 3, axis=2)*255, 0.5, 0)
            cv2.imwrite('demo_imgs/ai_challenge_kp_bbox_gaussian_{}.png'.format(index), bbox_gausian)

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'hps': kps, 'hps_mask': kps_mask}

        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 40), dtype=np.float32)
            meta = {'c': center, 's': max_length, 'gt_det': gt_det, 'img_id': anno['image_id']}
            ret['meta'] = meta

        return ret

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.get_anno(index)

if __name__ == '__main__':
    # usage: python -m lib.datasets.ai_challenger_kp_ori multi_pose --arch squeeze --aug_rot 0.5 --rotate 90
    from opts import opts
    opt = opts().init()

    AIChallengeKpOri(opt, 'train', debug=True)
