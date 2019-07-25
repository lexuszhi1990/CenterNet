# load the trained model and convert it to onnx, then test it.


import _init_paths
import os
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
    # inp_image = ((resized_img / 255.)).astype(np.float32)
    # inp_image = ((resized_img / 255. - mean)).astype(np.float32)
    inp_image = ((resized_img / 255. - mean) / std).astype(np.float32)
    inp_image = inp_image.astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}

    return images, meta

def build(opt):
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model.deploy = True
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

    image = cv2.imread(opt.demo)
    height, width = image.shape[0:2]
    inp_height, inp_width = opt.input_h, opt.input_w
    resized_img = cv2.resize(image, (inp_width, inp_height))
    inp_image = (resized_img / 255.).astype(np.float32)
    inputs = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

    output = model(torch.from_numpy(inputs).to(opt.device))
    torch.onnx.export(model, torch.from_numpy(inputs).to(opt.device), "example.onnx", verbose=False, input_names=["data"], output_names=["output"])

    import os; os.system('python3 -m onnxsim example.onnx example-sim.onnx')

    import onnx
    onnx_model = onnx.load("example.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    import onnxruntime
    session = onnxruntime.InferenceSession("example.onnx")
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    result = session.run(None, {input_name: inputs})

    output = output.detach().cpu().numpy()
    if opt.task == 'ai_challenge':
        hm = output[:, 0:1, :, :]
        wh = output[:, 1:3, :, :]
        kps = output[:, 3:31, :, :]
        reg = output[:, 31:33, :, :]
        hm_hp = output[:, 33:47, :, :]
        hm_offset = output[:, 47:49, :, :]
        num_joints = 14
    else:
        hm = output[:, 0:1, :, :]
        wh = output[:, 1:3, :, :]
        kps = output[:, 3:37, :, :]
        reg = output[:, 37:39, :, :]
        hm_hp = output[:, 39:56, :, :]
        hp_offset = output[:, 56:58, :, :]
        num_joints = 17

    output_h, output_w = hm.shape[2], hm.shape[3]
    center_x, center_y, score = hm.argmax() % output_w, int(hm.argmax() / output_w), hm.max()
    _wh = wh[0, :, center_y, center_x]
    _reg = reg[0, :, center_y, center_x]
    _kps = kps[0, :, center_y, center_x]

    print([center_x, center_y, score])
    print(_wh)
    print(_reg)
    print(_kps)
    print("mean correlation error is : %f" % np.mean(output - result[0]))
    print("max correlation error is : %f" % np.max(output - result[0]))

    import pdb; pdb.set_trace()


def eval(opt):
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model.deploy = True
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

    image = cv2.imread(opt.demo)
    height, width = image.shape[0:2]
    inp_height, inp_width = opt.input_h, opt.input_w
    resized_img = cv2.resize(image, (inp_width, inp_height))
    inp_image = (resized_img / 255.).astype(np.float32)
    inputs = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

    output = model(torch.from_numpy(inputs).to(opt.device))
    if opt.task == 'ai_challenge':
        num_joints = 14
        output = output.detach().cpu().numpy()
        hm = output[:, 0:1, :, :]
        wh = output[:, 1:3, :, :]
        kps = output[:, 3:31, :, :]
        reg = output[:, 31:33, :, :]
        hm_hp = output[:, 33:47, :, :]
        hm_offset = output[:, 47:49, :, :]
    else:
        num_joints = 17
        _output = output[-1]
        hm = _output['hm'].detach().cpu().numpy()
        wh = _output['wh'].detach().cpu().numpy()
        kps = _output['hps'].detach().cpu().numpy()
        reg = _output['reg'].detach().cpu().numpy()
        hm_hp = _output['hm_hp'].detach().cpu().numpy()
        hm_offset = _output['hp_offset'].detach().cpu().numpy()

        # hm = output[:, 0:1, :, :]
        # wh = output[:, 1:3, :, :]
        # kps = output[:, 3:37, :, :]
        # reg = output[:, 37:39, :, :]
        # hm_hp = output[:, 39:56, :, :]
        # hp_offset = output[:, 56:58, :, :]

    output_h, output_w = hm.shape[2], hm.shape[3]
    center_x, center_y, score = hm.argmax() % output_w, int(hm.argmax() / output_w), hm.max()
    _wh = wh[0, :, center_y, center_x]
    _reg = reg[0, :, center_y, center_x]
    _kps = kps[0, :, center_y, center_x]
    _hm_hp = hm_hp[0, :, center_y, center_x]
    _hm_offset = hm_offset[0, :, center_y, center_x]

    print([center_x, center_y, score])
    print(_wh)
    print(_reg)
    print(_kps)

    npimg = image.copy()
    top_x = int((center_x + _reg[0] - _wh[0] / 2) / output_w * width)
    top_y = int((center_y + _reg[1] - _wh[1] / 2) / output_h * height)
    buttom_x = int((center_x + _reg[0] + _wh[0] / 2) / output_w * width)
    buttom_y = int((center_y + _reg[1] + _wh[1] / 2) / output_h * height)
    cv2.rectangle(npimg, (top_x, top_y), (buttom_x, buttom_y), (255, 0, 0), 2)
    for idx in range(num_joints):
        kp_x = int((_kps[idx*2] + center_x + _hm_offset[0] ) / output_w * width)
        kp_y = int((_kps[idx*2 + 1] + center_y + _hm_offset[1] ) / output_h * height)
        cv2.circle(npimg, (kp_x, kp_y), 2, (0, 0, 255), 2)
    cv2.imwrite('result-by-kp.png', npimg)

    npimg = image.copy()
    cv2.rectangle(npimg, (top_x, top_y), (buttom_x, buttom_y), (255, 0, 0), 2)
    for idx in range(num_joints):
        hp_tmp = hm_hp[0][idx]
        center_x, center_y, score = hp_tmp.argmax() % output_w, int(hp_tmp.argmax() / output_w), hp_tmp.max()
        kp_x = int(( center_x ) / output_w * width)
        kp_y = int(( center_y ) / output_h * height)
        cv2.circle(npimg, (kp_x, kp_y), 2, (0, 0, 255), 2)
    cv2.imwrite('result-by-hm_hp.png', npimg)

    import pdb; pdb.set_trace()

    # hm_hp_outp = np.clip(hm_hp[0], 0, 1)
    # kp_gausian = cv2.addWeighted(cv2.resize(resized_img, (output_w, output_h)).astype(np.float32), 0.2, np.repeat(hm_hp_outp.sum(axis=0).reshape(output_h, output_w, 1), 3, axis=2)*255, 0.7, 0)
    # cv2.imwrite('result-kp-with-gaussion.png', kp_gausian)

def eval_dir(opt):
    num_joints = opt.num_joints
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model.deploy = False
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

    prefix = '../images/videos/results_sqeezenetv2'
    assert os.path.isdir(opt.demo)
    imgs = os.listdir(opt.demo)
    for img in imgs:

        image = cv2.imread(os.path.join(opt.demo, img))
        height, width = image.shape[0:2]
        inp_height, inp_width = opt.input_h, opt.input_w
        resized_img = cv2.resize(image, (inp_width, inp_height))
        inp_image = (resized_img / 255.).astype(np.float32)
        inputs = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        output = model(torch.from_numpy(inputs).to(opt.device))
        _output = output[-1]
        hm = _output['hm'].detach().cpu().numpy()
        wh = _output['wh'].detach().cpu().numpy()
        hps = _output['hps'].detach().cpu().numpy()
        reg = _output['reg'].detach().cpu().numpy()
        hm_hp = _output['hm_hp'].detach().cpu().numpy()
        hp_offset = _output['hp_offset'].detach().cpu().numpy()

        output_h, output_w = hm.shape[2], hm.shape[3]
        center_x, center_y, score = hm.argmax() % output_w, int(hm.argmax() / output_w), hm.max()
        _wh = wh[0, :, center_y, center_x]
        _reg = reg[0, :, center_y, center_x]
        _hps = hps[0, :, center_y, center_x]
        _hm_hp = hm_hp[0, :, center_y, center_x]
        _hp_offset = hp_offset[0, :, center_y, center_x]

        print([center_x, center_y, score])
        print(_wh)
        print(_reg)
        print(_hps)

        npimg = image.copy()
        top_x = int((center_x + _reg[0] - _wh[0] / 2) / output_w * width)
        top_y = int((center_y + _reg[1] - _wh[1] / 2) / output_h * height)
        buttom_x = int((center_x + _reg[0] + _wh[0] / 2) / output_w * width)
        buttom_y = int((center_y + _reg[1] + _wh[1] / 2) / output_h * height)
        cv2.rectangle(npimg, (top_x, top_y), (buttom_x, buttom_y), (255, 0, 0), 2)
        for idx in range(num_joints):
            kp_x = int((_hps[idx*2] + center_x + _hp_offset[0] ) / output_w * width)
            kp_y = int((_hps[idx*2 + 1] + center_y + _hp_offset[1] ) / output_h * height)
            cv2.circle(npimg, (kp_x, kp_y), 2, (0, 0, 255), 2)
        cv2.imwrite('result-by-kp.png', npimg)
        cv2.imwrite(os.path.join(prefix, "kp"+img), npimg)

        npimg = image.copy()
        cv2.rectangle(npimg, (top_x, top_y), (buttom_x, buttom_y), (255, 0, 0), 2)
        for idx in range(num_joints):
            hp_tmp = hm_hp[0][idx]
            center_x, center_y, score = hp_tmp.argmax() % output_w, int(hp_tmp.argmax() / output_w), hp_tmp.max()
            kp_x = int(( center_x ) / output_w * width)
            kp_y = int(( center_y ) / output_h * height)
            cv2.circle(npimg, (kp_x, kp_y), 2, (0, 0, 255), 2)
        cv2.imwrite('result-by-hm_hp.png', npimg)
        cv2.imwrite(os.path.join(prefix, "hm_hp"+img), npimg)

    import pdb; pdb.set_trace()

    # hm_hp_outp = np.clip(hm_hp[0], 0, 1)
    # kp_gausian = cv2.addWeighted(cv2.resize(resized_img, (output_w, output_h)).astype(np.float32), 0.2, np.repeat(hm_hp_outp.sum(axis=0).reshape(output_h, output_w, 1), 3, axis=2)*255, 0.7, 0)
    # cv2.imwrite('result-kp-with-gaussion.png', kp_gausian)



if __name__ == '__main__':
    # usage:
    #   python run_ckpt_onnx.py ai_challenge --gpus -1 --input_res -1 --input_h 256 --input_w 192 --arch squeeze --load_model ../exp/ai_challenge/squeeze_0.5_ai_challenge/model_best.pth --demo ../images/example-test.png
    #   python run_ckpt_onnx.py ai_challenge --gpus -1 --input_res -1 --input_h 256 --input_w 192 --arch squeezev1 --load_model ../exp/ai_challenge/squeeze_0.5_ai_challenge_v1/model_best.pth --demo ../images/example-test.png

    opt = opts().init()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    # build(opt)
    # eval(opt)
    eval_dir(opt)



'''


resized_img = cv2.resize(image, (desired_size, desired_size))
inp_image = ((resized_img / 255.)).astype(np.float32)
[49, 112, -0.69862026]
[[75.290535 21.887276]]
[[0.49415135 0.4865199 ]]
[[  6.0268717  -19.860367     7.8258076  -20.605413     5.136759
  -20.906473     9.790175   -19.021986     3.2751024  -19.789131
   13.475291   -11.363523    -4.8127236  -12.576008    16.538713
   -2.4693727   -9.958654    -4.123879    12.080624     0.96762574
   -3.7842124   -0.8098801    5.481064     5.6121926   -7.40891
    5.415961     7.4374175    5.953848    -2.3878777    5.8303423
    3.0252428   14.505438    -1.1344731   13.881374  ]]


resized_img = cv2.resize(image, (desired_size, desired_size))
inp_image = ((resized_img / 255. - mean)).astype(np.float32)
[69, 90, -0.3088241]
[[12.631435 41.64747 ]]
[[0.42041472 0.5093934 ]]
[[  2.0088658  -17.464067     1.9658762  -18.013653     1.3086078
  -17.94765      1.657421   -17.679731    -0.22378223 -17.765205
    2.4471977  -13.97991     -2.1252894  -13.69841      3.6251547
   -8.139481    -3.1706479   -7.441903     3.308625    -7.0497375
   -0.32461512  -6.061578     1.8116724   -1.2859949   -1.2825235
   -1.1196513    1.2725873    8.753617    -1.4597659    9.086942
    0.37079138  17.948229    -1.9442999   18.090464  ]]


resized_img = cv2.resize(image, (desired_size, desired_size))
inp_image = ((resized_img / 255. - mean) / std).astype(np.float32)
[69, 91, 0.05507727]
[[10.628226 42.352142]]
[[0.3485155 0.5409925]]
[[  1.5335134  -16.085285     1.253427   -16.607595     1.0198576
  -16.531975     0.5904298  -16.036745    -0.36234343 -16.103878
    0.59619015 -12.098537    -1.45649    -11.745516     1.1952838
   -7.242378    -1.72565     -6.4785438    1.7387005   -7.168525
    0.29833162  -6.077854     0.56354606   0.2693813   -0.83708215
    0.48358628   0.7702264    9.033973    -0.59760666   9.405047
    0.5232635   17.522482    -0.88558376  17.744093  ]]




resized_img = cv2.resize(image, (desired_size, desired_size))
inp_image = ((resized_img - mean) / std).astype(np.float32)
[54, 113, 6.034814]
[[351.7701  142.73833]]
[[0.54294604 2.643224  ]]
[[-128.23434     32.92517   -170.49823     59.2795     -52.649227
    50.455227  -214.74223     78.84239     72.2035      69.43472
  -201.51599     90.762314    97.89717     65.04135   -223.95894
   -26.68169    156.7228     -45.250546  -177.49687      7.2376857
    47.63789     11.000908   -46.890316   141.45421    134.3464
   130.56107    -70.982254    46.01534    136.60799     35.86316
   -20.987051    16.488102   198.52126      7.223912 ]]


resized_img = cv2.resize(image, (desired_size, desired_size))
inp_image = resized_img.astype(np.float32)
[54, 113, 0.97088]
[[123.11749   52.728474]]
[[0.5076777 1.0843776]]
[[-38.22071    10.868279  -50.731365   18.7904    -15.0190735  15.837307
  -63.8397     24.283243   23.243366   20.782272  -59.651695   27.148743
   29.30888    18.347809  -65.80079    -8.284856   46.913174  -15.075959
  -51.530582    3.9494114  16.668255    4.012199  -12.930778   43.511757
   40.385895   39.929764  -20.627375   13.934929   42.089027   10.805276
   -6.8187943   4.537342   60.253464    1.6526768]]


resized_img = cv2.resize(image, (desired_size, desired_size))
inp_image = ((resized_img - mean)).astype(np.float32)
[54, 113, 0.9696449]
[[122.790474  52.66477]]
[[0.507654  1.0833325]]
[[-38.13101    10.866044  -50.616978   18.771837  -14.965746   15.816618
  -63.70252    24.248873   23.23728    20.734907  -59.548878   27.097239
   29.278452   18.278912  -65.713      -8.249184   46.823505  -15.083103
  -51.43048     3.9672678  16.66291     3.984189  -12.884738   43.429398
   40.351215   39.84513   -20.59064    13.915224   42.0109     10.785912
   -6.819239    4.5100536  60.113907    1.6321871]]



resized_img = cv2.resize(image, (desired_size, desired_size))
[69, 91, 0.05507727]
[[10.628226 42.352142]]
[[0.3485155 0.5409925]]
[[  1.5335134  -16.085285     1.253427   -16.607595     1.0198576
  -16.531975     0.5904298  -16.036745    -0.36234343 -16.103878
    0.59619015 -12.098537    -1.45649    -11.745516     1.1952838
   -7.242378    -1.72565     -6.4785438    1.7387005   -7.168525
    0.29833162  -6.077854     0.56354606   0.2693813   -0.83708215
    0.48358628   0.7702264    9.033973    -0.59760666   9.405047
    0.5232635   17.522482    -0.88558376  17.744093  ]]


resized_img = resize_img(image, desired_size)
[78, 34, 0.7812877]
[[34.142513 26.14727 ]]
[[0.48540735 0.557583  ]]
[[ -2.4960883   -2.6380398   -1.9495628   -2.2942007   -2.47164
   -2.2427602   -1.0498747   -1.9597057   -1.4346856   -1.8491902
    5.4530783   -1.2711922   -5.664255    -1.3531971   11.80295
   -2.5480492  -11.792163    -2.7423568    9.912038    -2.521405
  -13.721993    -2.142022     5.2383757    0.48980692  -2.4984467
    0.48133948   3.1570194   -1.6078571   -2.2164729   -1.8452721
    3.0189598    3.232667     1.8282282    3.008189  ]]


inp_image = ((resized_img - mean) / std).astype(np.float32)
[54, 113, 6.034814]
[[351.7701  142.73833]]
[[0.54294604 2.643224  ]]
[[-128.23434     32.92517   -170.49823     59.2795     -52.649227
    50.455227  -214.74223     78.84239     72.2035      69.43472
  -201.51599     90.762314    97.89717     65.04135   -223.95894
   -26.68169    156.7228     -45.250546  -177.49687      7.2376857
    47.63789     11.000908   -46.890316   141.45421    134.3464
   130.56107    -70.982254    46.01534    136.60799     35.86316
   -20.987051    16.488102   198.52126      7.223912 ]]

















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

    torch.onnx.export(model, torch.from_numpy(inputs), "example.onnx", verbose=False, input_names=["data"], output_names=["output"])

    import os; os.system('python3 -m onnxsim example.onnx example-sim.onnx')

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
    import pdb; pdb.set_trace()

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

'''
