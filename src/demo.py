from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import torch
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def pre_process(im_pth, opt):
  # im_pth = "../images/33823288584_1d21cf0a26_k.jpg"
  desired_size = 512

  im_ori = cv2.imread(im_pth)
  old_size = im_ori.shape[:2] # old_size is in (height, width) format

  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])

  # new_size should be in (width, height) format

  im = cv2.resize(im_ori, (new_size[1], new_size[0]))

  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)

  color = [0, 0, 0]
  inp_image = cv2.copyMakeBorder(im, top-2, bottom+2, left, right, cv2.BORDER_CONSTANT, value=color)

  mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
  std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
  inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

  inp_height, inp_width = opt.input_h, opt.input_w
  images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
  if opt.flip_test:
    images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)

  new_height, new_width = old_size
  c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
  s = max(old_size) * 1.0

  images_, meta_ = {}, {}
  images_[1.0] = torch.from_numpy(images)
  meta_[1.0] = {'c': c, 's': s,
          'out_height': inp_height // opt.down_ratio,
          'out_width': inp_width // opt.down_ratio}
  ret = {'images': images_, 'image': im_ori, 'meta': meta_}
  import pdb; pdb.set_trace()
  return ret

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

    for (image_name) in image_names:

      # inp_img = pre_process(image_name, opt)
      # ret = detector.run(inp_img)
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
