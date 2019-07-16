# load the trained model and convert it to onnx, then test it.


import _init_paths

import cv2
import numpy as np
import torch
import torch.nn as nn

from opts import opts
from models.model import create_model, load_model
from utils.image import get_affine_transform
from models.decode import ctdet_decode, multi_pose_decode, _topk, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from utils.post_process import multi_pose_post_process

def resize_img(image, desired_size=512):

    old_size = image.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im

def pre_process(im_path, mean, std, desired_size=512, down_ratio=4):
    # im_path = "../images/33823288584_1d21cf0a26_k.jpg"
    image = cv2.imread(im_path)
    height, width = image.shape[0:2]
    inp_height, inp_width = desired_size, desired_size
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    resized_img = resize_img(image, desired_size)

    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    inp_image = ((resized_img / 255. - mean) / std).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    # images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}

    return images, meta

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = torch.le(hmax, heat)
    return heat * keep.float()

class PoseFullNet(nn.Module):
    def __init__(self, arch, heads, head_conv):
        super(PoseFullNet, self).__init__()

        self.heads = heads
        self.backbone = create_model(arch, heads, head_conv)

    def forward(self, x):
        output = self.backbone(x)
        hm, wh, kps, reg, hm_hp, hp_offset = output[0]['hm'], output[0]['wh'], output[0]['hps'], output[0]['reg'], output[0]['hm_hp'], output[0]['hp_offset']

        _hm = _nms(hm.sigmoid_())
        _hm_hp = _nms(hm_hp.sigmoid_())

        return torch.cat([_hm, wh, kps, reg, _hm_hp, hp_offset], 1)

        # batch, K = 1, 1
        # heat = hm.sigmoid_()

        # _score = torch.max(heat)
        # _inds = torch.argmax(heat)
        # _xs = (_inds % 128).float()
        # _ys = (_inds / 128).float()
        # _wh = wh[0, :, _ys.int(), _xs.int()]
        # _reg = reg[0, :, _ys.int(), _xs.int()]
        # _xs = _xs + _reg[0]
        # _ys = _ys + _reg[1]

        # _score, _inds = heat.view(batch, -1).max(dim=1)
        # _xs = (_inds % 128).float()[0]
        # _ys = (_inds / 128).float()[0]
        # _wh = wh[0, :, _ys.int(), _xs.int()]
        # _reg = reg[0, :, _ys.int(), _xs.int()]
        # _xs = _xs + _reg[0]
        # _ys = _ys + _reg[1]

        # return torch.stack([_xs - _wh[0] / 2, _ys - _wh[1] / 2, _xs + _wh[0] / 2, _ys + _wh[1] / 2, _score[0]])

        # import pdb; pdb.set_trace()

        # # heat = _nms(hm.sigmoid_())

        # _score, _ind = heat.view(batch, -1).max(dim=1)
        # # _ind = heat.argmax(keepdim=True)
        # # _score = heat.max()
        # _inds = torch.cat([_ind.unsqueeze(dim=0)])
        # _scores = torch.cat([_score.unsqueeze(dim=0)])
        # _xs = (_inds % 128).int().float()
        # _ys = (_inds / 128).int().float()

        # _reg = _tranpose_and_gather_feat(reg, _inds)
        # _reg = _reg.view(batch, K, 2)
        # _xs = _xs.view(batch, K, 1) + _reg[:, :, 0:1]
        # _ys = _ys.view(batch, K, 1) + _reg[:, :, 1:2]
        # _wh = _tranpose_and_gather_feat(wh, _inds)
        # _wh = _wh.view(batch, K, 2)
        # _scores = _scores.view(batch, K, 1)
        # _bboxes = torch.cat([_xs - _wh[..., 0:1] / 2, _ys - _wh[..., 1:2] / 2, _xs + _wh[..., 0:1] / 2, _ys + _wh[..., 1:2] / 2], dim=2)
        # detections = torch.cat([_bboxes, _scores], dim=2)

        # return detections

        # # ret = []
        # # for head in self.heads:
        # #     res = output[0][head]
        # #     ret.append(res)
        # # return torch.cat(ret, 1)

    def init(self, path):
        self.backbone = load_model(self.backbone, opt.load_model)


def test_onnx(opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')

    print('Creating model...')
    model = PoseFullNet(opt.arch, opt.heads, opt.head_conv)
    model.init(opt.load_model)
    model = model.to(opt.device)
    model.eval()

    inputs, meta = pre_process(opt.demo, opt.mean, opt.std)
    output = model(torch.from_numpy(inputs))

    torch.onnx.export(model, torch.from_numpy(inputs), "squeezenet.onnx", verbose=False, input_names=["data"], output_names=["output"])

    import os; os.system('python -m onnxsim squeezenet.onnx squeezenet-sim.onnx')

    import onnx
    onnx_model = onnx.load("squeezenet-sim.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    import onnxruntime
    session = onnxruntime.InferenceSession("squeezenet-sim.onnx")
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    result = session.run(None, {input_name: inputs})

    # (Pdb) output[0, 0, 0:4, 0]
    # tensor([0.0208, 0.0000, 0.0000, 0.0000], grad_fn=<SelectBackward>)
    # (Pdb) result[0][0, 0, 0:4, 0]
    # array([0.02079931, 0.        , 0.        , 0.        ], dtype=float32)


def main(opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')

    image = cv2.imread(opt.demo)

    print('Creating model...')
    model = PoseFullNet(opt.arch, opt.heads, opt.head_conv)
    model.init(opt.load_model)
    model = model.to(opt.device)
    model.eval()

    inputs, meta = pre_process(opt.demo, opt.mean, opt.std)
    output = model(torch.from_numpy(inputs))

    hm = output[:, 0:1, :, :]
    wh = output[:, 1:3, :, :]
    kps = output[:, 3:37, :, :]
    reg = output[:, 37:39, :, :]
    hm_hp = output[:, 39:56, :, :]
    hp_offset = output[:, 56:58, :, :]

    K = 10
    heat = hm
    batch, cat, height, width = heat.size()

    kp_dets = multi_pose_decode(heat, wh, kps, reg, hm_hp, hp_offset, K)
    # kp_dets = multi_pose_decode(heat, wh, kps, reg, None, hp_offset, K)
    kp_dets = kp_dets.detach().cpu().numpy().reshape(1, -1, kp_dets.shape[2])
    kp_dets = multi_pose_post_process(kp_dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'])
    for j in range(1, 1 + 1):
        kp_dets[0][j] = np.array(kp_dets[0][j], dtype=np.float32).reshape(-1, 39)
        # kp_dets[0][j][:, :4] /= scale
        # kp_dets[0][j][:, 5:] /= scale

    npimg = image.copy()
    for p in range(K):
        if kp_dets[0][1][p][4] < 0.2:
            continue
        person = kp_dets[0][1][p].astype(np.int)
        cv2.rectangle(npimg, (person[0], person[1]), (person[2], person[3]), (255, 0, 0), 2)
        for idx in range(17):
            cv2.circle(npimg, (person[idx*2 + 5], person[idx*2 + 6]), 2, (0, 0, 255), 2)
    cv2.imwrite('result-kp.png', npimg)


    # official bbox implement
    dets = ctdet_decode(hm, wh, reg=reg, K=K)
    dets_ = dets.detach().cpu().numpy()
    dets_ = dets_.reshape(1, -1, dets_.shape[2])
    dets_ = ctdet_post_process(dets_.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], 80)

    npimg = image.copy()
    cv2.rectangle(npimg, (731, 50), (950, 497), (255, 0, 0), 2)
    cv2.imwrite('result-bbox.png', npimg)




    # batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    # heat = _nms(heat)
    num_joints = kps.shape[1] // 2
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _tranpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

    reg = _tranpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2,
                      ys + wh[..., 1:2] / 2], dim=2)

    thresh = 0.1
    kps = kps.view(batch, K, num_joints, 2).permute(
      0, 2, 1, 3).contiguous() # b x J x K x 2
    reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
    hp_offset = _tranpose_and_gather_feat(
      hp_offset, hm_inds.view(batch, -1))
    hp_offset = hp_offset.view(batch, num_joints, K, 2)
    hm_xs = hm_xs + hp_offset[:, :, :, 0]
    hm_ys = hm_ys + hp_offset[:, :, :, 1]

    mask = (hm_score > thresh).float()
    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
      2).expand(batch, num_joints, K, K, 2)
    dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
    min_dist, min_ind = dist.min(dim=3) # b x J x K
    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
      batch, num_joints, K, 1, 2)
    hm_kps = hm_kps.gather(3, min_ind)
    hm_kps = hm_kps.view(batch, num_joints, K, 2)
    l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
         (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
         (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
    mask = (mask > 0).float().expand(batch, num_joints, K, 2)
    kps = (1 - mask) * hm_kps + mask * kps
    kps = kps.permute(0, 2, 1, 3).contiguous().view(
      batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    return detections
    # implement post post process via numpy
    # def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):

    # _ind = heat.argmax(keepdim=True)
    # _score = heat.max()
    # _inds = torch.stack([_ind.unsqueeze(dim=0)])
    # _scores = torch.stack([_score.unsqueeze(dim=0)])
    # _xs = (_inds % 128).int().float()
    # _ys = (_inds / 128).int().float()

    # _reg = _tranpose_and_gather_feat(reg, _inds)
    # _reg = _reg.view(batch, K, 2)
    # _xs = _xs.view(batch, K, 1) + _reg[:, :, 0:1]
    # _ys = _ys.view(batch, K, 1) + _reg[:, :, 1:2]
    # _wh = _tranpose_and_gather_feat(wh, _inds)
    # _wh = _wh.view(batch, K, 2)
    # _scores = _scores.view(batch, K, 1)
    # _bboxes = torch.cat([_xs - _wh[..., 0:1] / 2, _ys - _wh[..., 1:2] / 2, _xs + _wh[..., 0:1] / 2, _ys + _wh[..., 1:2] / 2], dim=2)
    # detections = torch.cat([_bboxes, _scores], dim=2)
    # _score = torch.max(heat)
    # _inds = torch.argmax(heat)
    # _xs = (_inds % 128).float()
    # _ys = (_inds / 128).float()
    # _wh = wh[0, :, _ys.int(), _xs.int()]
    # _reg = reg[0, :, _ys.int(), _xs.int()]
    # _xs = _xs + _reg[0]
    # _ys = _ys + _reg[1]


    # tensor([[[ 97.9625,  30.6586, 127.3161,  90.4007,   0.9394,   0.0000]]],
    #        grad_fn=<CatBackward>)

    # scores, inds, clses, ys, xs = _topk(heat, K=1)
    # reg = _tranpose_and_gather_feat(reg, inds)
    # reg = reg.view(batch, K, 2)
    # xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    # ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    # wh = _tranpose_and_gather_feat(wh, inds)
    # wh = wh.view(batch, K, 2)
    # clses  = clses.view(batch, K, 1).float()
    # scores = scores.view(batch, K, 1)
    # bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)
    # detections = torch.cat([bboxes, scores, clses], dim=2)





if __name__ == '__main__':
    opt = opts().init()
    main(opt)
    # test_onnx(opt)
