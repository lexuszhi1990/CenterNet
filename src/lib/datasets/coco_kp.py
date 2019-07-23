# -*- coding: utf-8 -*-

import _init_paths

import torch
import json
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

class COCOKP(data.Dataset):
  num_classes = 1
  num_joints = 17
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
              [11, 12], [13, 14], [15, 16]]

  def __init__(self, opt, split):
    super(COCOKP, self).__init__()
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                  [4, 6], [3, 5], [5, 6],
                  [5, 7], [7, 9], [6, 8], [8, 10],
                  [6, 12], [5, 11], [11, 12],
                  [12, 14], [14, 16], [11, 13], [13, 15]]

    self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    self.data_dir = os.path.join(opt.data_dir, 'coco')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'image_info_test-dev2017.json').format(split)
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations',
        'person_keypoints_{}2017.json').format(split)
    self.max_objs = 32
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    image_ids = self.coco.getImgIds()

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          self.images.append(img_id)
    else:
      self.images = image_ids
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

    index = 2;
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)
    cv2.imwrite('coco_kp_ori.png', img)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0

    flipped = False
    # if self.split == 'val':
    #   if not self.opt.not_rand_crop:
    #     s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
    #     w_border = self._get_border(128, img.shape[1])
    #     h_border = self._get_border(128, img.shape[0])
    #     c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
    #     c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
    #   else:
    #     sf = self.opt.scale
    #     cf = self.opt.shift
    #     c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
    #     c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
    #     s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

    # rf = self.opt.rotate
    # rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)
    rot = 90
    print('rorated %f' % rot)


    flipped = True
    img = img[:, ::-1, :]
    c[0] =  width - c[0] - 1
    print('flipped')

    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_res, self.opt.input_res])
    inp_ori = cv2.warpAffine(img, trans_input,
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)
    cv2.imwrite('coco_kp_rand_crop_aug_rot_flip.png', inp_ori)
    inp = (inp_ori.astype(np.float32) / 255.)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_res = self.opt.output_res
    num_joints = self.num_joints
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_res, output_res),
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res),
                               dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(ann['category_id']) - 1
      pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        pts[:, 0] = width - pts[:, 0] - 1
        for e in self.flip_idx:
          pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
      bbox[:2] = affine_transform(bbox[:2], trans_output_rot)
      bbox[2:] = affine_transform(bbox[2:], trans_output_rot)
      bbox = np.clip(bbox, 0, output_res - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

      bbox_ = (bbox * 4).astype(np.int32)
      cv2.rectangle(inp_ori, (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (255, 0, 0), 2)

      if (h > 0 and w > 0) or (rot != 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_res + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        num_kpts = pts[:, 2].sum()
        if num_kpts == 0:
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
          reg_mask[k] = 0

        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
        for j in range(num_joints):
          if pts[j, 2] > 0:
            pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
               pts[j, 1] >= 0 and pts[j, 1] < output_res:
              kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
              kps_mask[k, j * 2: j * 2 + 2] = 1
              pt_int = pts[j, :2].astype(np.int32)
              hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
              hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
              hp_mask[k * num_joints + j] = 1
              if self.opt.dense_hp:
                # must be before draw center hm gaussian
                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                               pts[j, :2] - ct_int, radius, is_offset=True)
                draw_gaussian(dense_kps_mask[j], ct_int, radius)
              draw_gaussian(hm_hp[j], pt_int, hp_radius)

        kp_gausian = cv2.addWeighted(cv2.resize(inp_ori, (128, 128)).astype(np.float32), 0.5, np.repeat(hm_hp.sum(axis=0).reshape(128, 128, 1), 3, axis=2)*255, 0.5, 0)
        cv2.imwrite('coco_kp_kp_gaussian.png', kp_gausian)

        draw_gaussian(hm[cls_id], ct_int, radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1] +
                       pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

    cv2.imwrite('coco_kp_bbox.png', inp_ori)
    bbox_gausian = cv2.addWeighted(cv2.resize(inp_ori, (128, 128)).astype(np.float32), 0.5, np.repeat(hm.sum(axis=0).reshape(128, 128, 1), 3, axis=2)*255, 0.5, 0)
    cv2.imwrite('coco_kp_bbox_gaussian.png', bbox_gausian)
    cv2.imwrite('coco_kp_bbox.png', inp_ori)
    import pdb; pdb.set_trace()

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      if np.random.random() < self.opt.aug_rot:
        rf = self.opt.rotate
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1


    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_res, self.opt.input_res])
    inp = cv2.warpAffine(img, trans_input,
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_res = self.opt.output_res
    num_joints = self.num_joints
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_res, output_res),
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res),
                               dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(ann['category_id']) - 1
      pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        pts[:, 0] = width - pts[:, 0] - 1
        for e in self.flip_idx:
          pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox = np.clip(bbox, 0, output_res - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0) or (rot != 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_res + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        num_kpts = pts[:, 2].sum()
        if num_kpts == 0:
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
          reg_mask[k] = 0

        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
        for j in range(num_joints):
          if pts[j, 2] > 0:
            pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
               pts[j, 1] >= 0 and pts[j, 1] < output_res:
              kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
              kps_mask[k, j * 2: j * 2 + 2] = 1
              pt_int = pts[j, :2].astype(np.int32)
              hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
              hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
              hp_mask[k * num_joints + j] = 1
              if self.opt.dense_hp:
                # must be before draw center hm gaussian
                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                               pts[j, :2] - ct_int, radius, is_offset=True)
                draw_gaussian(dense_kps_mask[j], ct_int, radius)
              draw_gaussian(hm_hp[j], pt_int, hp_radius)
        draw_gaussian(hm[cls_id], ct_int, radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1] +
                       pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])


    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'hps': kps, 'hps_mask': kps_mask}
    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 40), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = dets[4]
          bbox_out  = list(map(self._to_float, bbox))
          keypoints = np.concatenate([
            np.array(dets[5:39], dtype=np.float32).reshape(-1, 2),
            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
          keypoints  = list(map(self._to_float, keypoints))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "keypoints": keypoints
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results),
              open('{}/results.json'.format(save_dir), 'w'))


  def run_eval(self, results, save_dir):
    # result_json = os.path.join(opt.save_dir, "results.json")
    # detections  = convert_eval_format(all_boxes)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



if __name__ == '__main__':


    # usage: python -m lib.datasets.coco_kp multi_pose --arch squeeze --aug_rot 0.5 --rotate 90
    from opts import opts
    opt = opts().init()

    COCOKP(opt, 'val')
