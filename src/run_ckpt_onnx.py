# load the trained model and convert it to onnx, then test it.


import _init_paths

import cv2
import numpy as np
import scipy.stats as st

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

    resized_img = cv2.resize(image, (desired_size, desired_size))
    # resized_img = resize_img(image, desired_size)

    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    inp_image = ((resized_img / 255. - mean) / std).astype(np.float32)
    inp_image = cv2.resize(image, (512, 512)).astype(np.float32)
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

def get_gauss_kernel(kernlen=5, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

class PoseFullNet(nn.Module):
    def __init__(self, arch, heads, head_conv):
        super(PoseFullNet, self).__init__()

        self.heads = heads
        self.backbone = create_model(arch, heads, head_conv)

        self.gassuian_filter = nn.Conv2d(1, 1, (11, 11), padding=(5, 5), bias=False)
        self.gassuian_filter.weight.data = torch.from_numpy(get_gauss_kernel(11, 5, 1).transpose(2, 3, 0, 1))

        self.max_pool = nn.MaxPool2d((3, 3), stride=1, padding=1)

    def forward(self, x):
        output = self.backbone(x)
        hm, wh, kps, reg, hm_hp, hp_offset = output[0]['hm'], output[0]['wh'], output[0]['hps'], output[0]['reg'], output[0]['hm_hp'], output[0]['hp_offset']

        return hm.sigmoid_()

        # version 0.2:
        batch = 1
        hm = hm.sigmoid_()
        _hm = self.gassuian_filter(hm)
        _score, _inds = _hm.view(batch, _hm.size(1), -1).max(dim=2)
        _wh = wh.view(batch, wh.size(1), -1)[:, :, _inds[0, 0]]
        _reg = reg.view(batch, reg.size(1), -1)[:, :, _inds[0, 0]]
        _kps = kps.view(batch, kps.size(1), -1)[:, :, _inds[0, 0]]
        return torch.cat([_score, _inds.float(), _wh, _reg, _kps], dim=1)

        # version 0.1
        _hm = hm.sigmoid_()
        _hm = self.gassuian_filter(_hm)
        # _hm_max = self.max_pool(_hm)
        # keep = torch.(_hm_max, _hm)
        # hm_final = hm * keep.float()
        _hm_hp = hm_hp.sigmoid_()
        # _hm_hp = _nms(hm_hp.sigmoid_())

        # return _hm
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

    # 69.7109375
    # (Pdb) result[0][0][0].argmax() /%128
    # *** SyntaxError: invalid syntax
    # (Pdb) result[0][0][0].argmax() %128
    # 91
    # (Pdb) result[0][0][0].max()
    # 0.44764897

    print([result[0][0][0].argmax() / 128, result[0][0][0].argmax() % 128, result[0][0][0].max()])

    print("mean correlation error is : %f" % np.mean(output.detach().cpu().numpy() - result[0]))
    print("max correlation error is : %f" % np.max(output.detach().cpu().numpy() - result[0]))

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

    # version 0.2
    person = output.detach().numpy()[0]
    score = person[0]
    center_x = person[1] % 128
    center_y = person[1] / 128
    width = person[2]
    height = person[3]
    offset_x = person[4]
    offset_y = person[5]
    kp_x = person[6::2] + center_x
    kp_y = person[7::2] + center_y

    def center_2_bbox(center_x, center_y, width, height, offset_x, offset_y, scale=4):
        top_x = int(center_x + offset_x - width / 2) * scale
        top_y = int(center_y + offset_y - height / 2) * scale
        buttom_x = int(center_x + offset_x + width / 2) * scale
        buttom_y = int(center_y + offset_y + height / 2) * scale

        return top_x, top_y, buttom_x, buttom_y

    npimg = cv2.resize(image, (512, 512))
    # npimg = resize_img(image, 512)
    if score > 0.3:
        top_x, top_y, buttom_x, buttom_y = center_2_bbox(center_x, center_y, width, height, offset_x, offset_y)
        cv2.rectangle(npimg, (top_x, top_y), (buttom_x, buttom_y), (255, 0, 0), 2)
        for idx in range(len(kp_x)):
            cv2.circle(npimg, (int(kp_x[idx] * 4), int(kp_y[idx] * 4)), 2, (0, 0, 255), 2)
    cv2.imwrite('result-kp-no-hm_hp-v2.png', npimg)
    # (Pdb) center_x, center_y
    # (71.0, 58.5546875)
    # (Pdb) score
    # 0.3278077

    import pdb; pdb.set_trace()

    # version 0.1
    # output is the feature of last layer
    hm = output[:, 0:1, :, :]
    wh = output[:, 1:3, :, :]
    kps = output[:, 3:37, :, :]
    reg = output[:, 37:39, :, :]
    hm_hp = output[:, 39:56, :, :]
    hp_offset = output[:, 56:58, :, :]

    K = 10
    heat = hm
    batch, cat, height, width = heat.size()

    # kp_dets = multi_pose_decode(heat, wh, kps, reg, hm_hp, hp_offset, K)
    kp_dets = multi_pose_decode(heat, wh, kps, reg, None, hp_offset, K)
    kp_dets = kp_dets.detach().cpu().numpy().reshape(1, -1, kp_dets.shape[2])
    npimg = cv2.resize(image, (512, 512))
    for p in range(K):
        if kp_dets[0][p][4] < 0.3:
            continue
        person = kp_dets[0][p].astype(np.int) * 4
        cv2.rectangle(npimg, (person[0], person[1]), (person[2], person[3]), (255, 0, 0), 2)
        for idx in range(17):
            cv2.circle(npimg, (person[idx*2 + 5], person[idx*2 + 6]), 2, (0, 0, 255), 2)
    # cv2.imwrite('result-kp.png', npimg)
    cv2.imwrite('result-kp-no-hm_hp-without-post-process.png', npimg)

    kp_dets = multi_pose_post_process(kp_dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'])
    for j in range(1, 1 + 1):
        kp_dets[0][j] = np.array(kp_dets[0][j], dtype=np.float32).reshape(-1, 39)
        # kp_dets[0][j][:, :4] /= scale
        # kp_dets[0][j][:, 5:] /= scale
    npimg = image.copy()
    for p in range(K):
        if kp_dets[0][1][p][4] < 0.3:
            continue
        person = kp_dets[0][1][p].astype(np.int)
        cv2.rectangle(npimg, (person[0], person[1]), (person[2], person[3]), (255, 0, 0), 2)
        for idx in range(17):
            cv2.circle(npimg, (person[idx*2 + 5], person[idx*2 + 6]), 2, (0, 0, 255), 2)
    # cv2.imwrite('result-kp.png', npimg)
    cv2.imwrite('result-kp-no-hm_hp.png', npimg)

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
    # num_joints = kps.shape[1] // 2
    # scores, inds, clses, ys, xs = _topk(heat, K=K)

    # kps = _tranpose_and_gather_feat(kps, inds)
    # kps = kps.view(batch, K, num_joints * 2)
    # kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    # kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

    # reg = _tranpose_and_gather_feat(reg, inds)
    # reg = reg.view(batch, K, 2)
    # xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    # ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    # wh = _tranpose_and_gather_feat(wh, inds)
    # wh = wh.view(batch, K, 2)
    # clses  = clses.view(batch, K, 1).float()
    # scores = scores.view(batch, K, 1)
    # bboxes = torch.cat([xs - wh[..., 0:1] / 2,
    #                   ys - wh[..., 1:2] / 2,
    #                   xs + wh[..., 0:1] / 2,
    #                   ys + wh[..., 1:2] / 2], dim=2)

    # thresh = 0.1
    # kps = kps.view(batch, K, num_joints, 2).permute(
    #   0, 2, 1, 3).contiguous() # b x J x K x 2
    # reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
    # hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
    # hp_offset = _tranpose_and_gather_feat(
    #   hp_offset, hm_inds.view(batch, -1))
    # hp_offset = hp_offset.view(batch, num_joints, K, 2)
    # hm_xs = hm_xs + hp_offset[:, :, :, 0]
    # hm_ys = hm_ys + hp_offset[:, :, :, 1]

    # mask = (hm_score > thresh).float()
    # hm_score = (1 - mask) * -1 + mask * hm_score
    # hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    # hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    # hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
    #   2).expand(batch, num_joints, K, K, 2)
    # dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
    # min_dist, min_ind = dist.min(dim=3) # b x J x K
    # hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
    # min_dist = min_dist.unsqueeze(-1)
    # min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
    #   batch, num_joints, K, 1, 2)
    # hm_kps = hm_kps.gather(3, min_ind)
    # hm_kps = hm_kps.view(batch, num_joints, K, 2)
    # l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    # t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    # r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    # b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    # mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
    #      (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
    #      (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
    # mask = (mask > 0).float().expand(batch, num_joints, K, 2)
    # kps = (1 - mask) * hm_kps + mask * kps
    # kps = kps.permute(0, 2, 1, 3).contiguous().view(
    #   batch, K, num_joints * 2)
    # detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    # return detections
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
    # main(opt)
    test_onnx(opt)
