

import sys
CENTERNET_PATH = '/app/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.networks.msra_resnet import get_pose_net
from models.decode import ctdet_decode
from utils.post_process import ctdet_post_process


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
    # (Pdb) aaa=cv2.imread('results/input_v1.png')
    # (Pdb) aaa.shape
    # (512, 512, 3)
    # (Pdb) resized_img.shape
    # (512, 512, 3)
    # (Pdb) np.mean(aaa - resized_img)

    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    inp_image = ((resized_img / 255. - mean) / std).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    # images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}

    return images, meta

def load_model(model, model_path):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
            model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
              print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}.'.format(
                  k, model_state_dict[k].shape, state_dict[k].shape))
              state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))

    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model

def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale

    return dets[0]

def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
        results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
        kth = len(scores) - self.max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, self.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


img_path = '../images/33823288584_1d21cf0a26_k.jpg'
model_path = '../models/resnet18_ctdet.pth'
mean = [0.40789655, 0.44719303, 0.47026116]
std = [0.2886383, 0.27408165, 0.27809834]
num_layers = 18
heads = {'hm': 80, 'wh': 2, 'reg': 2}
head_conv = 64
top_k = 10
desired_size = 512

inputs, meta = pre_process(img_path, mean, std)

model = get_pose_net(num_layers=num_layers, heads=heads, head_conv=head_conv, pretrained=False)
load_model(model, model_path)
output = model(torch.from_numpy(inputs))[-1]
hm = output['hm'].sigmoid_()
wh = output['wh']
reg = output['reg']
dets = ctdet_decode(hm, wh, reg=reg, K=top_k)

dets_ = dets.detach().cpu().numpy()
dets_ = dets_.reshape(1, -1, dets_.shape[2])
dets_ = ctdet_post_process(dets_.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], 80)

# tensor([[[ 94.6174,  27.5132, 128.4623,  95.4632,   0.8150,   0.0000],
#          [ 56.3054,  35.1515,  86.8433,  96.0929,   0.6894,   0.0000],
#          [ 10.3042,  46.8160,  26.5921,  90.3222,   0.6167,   0.0000],
#          [ 11.8997,  53.8905,  57.1729, 103.1914,   0.5577,   0.0000],
#          [ 88.2191,  55.7840,  94.9064,  77.2077,   0.5311,   0.0000],
#          [ 50.8027,  57.7168,  56.1965,  63.7278,   0.4955,   0.0000],
#          [ 94.2124,  53.4928, 100.6876,  77.7233,   0.4255,   0.0000],
#          [ 43.3257,  57.0269,  51.4475,  66.0542,   0.4248,   0.0000],
#          [ -0.0558,  33.1531,  13.1464, 103.8295,   0.3927,   0.0000],
#          [ -0.1001,  39.1863,  12.9736, 105.7441,   0.3549,   0.0000]]],

# tensor([[ 96.6122,  30.9451, 126.5108,  89.9106,   0.8098,   0.0000],
#         [ 10.6189,  47.5344,  26.2005,  89.5588,   0.6523,   0.0000],
#         [ 58.8839,  38.9071,  86.2242,  94.1137,   0.6098,   0.0000],
#         [ 12.9598,  55.1022,  52.0453, 101.8157,   0.5949,   0.0000],
#         [ 95.7140,  56.7733, 127.1870,  94.1714,   0.5103,   1.0000],
#         [ 94.6336,  62.3710, 126.6172,  94.5687,   0.5008,   1.0000],
#         [ -0.2098,  37.4169,  13.3508, 101.5324,   0.4702,   0.0000],
#         [ 88.0050,  57.0956,  95.0799,  77.6867,   0.4417,   0.0000],
#         [ 50.7184,  68.2659,  58.3241,  80.6639,   0.4132,   1.0000],
#         [ 50.8602,  57.8103,  56.2672,  67.0339,   0.4070,   0.0000]],

# array([[ 7.2157233e+02,  5.3121250e+01,  9.4487720e+02,  4.9351978e+02,
#          8.0983472e-01],
#        [ 7.9309967e+01,  1.7702228e+02,  1.9568517e+02,  4.9089212e+02,
#          6.5234667e-01],
#        [ 4.3978928e+02,  1.1258707e+02,  6.4398724e+02,  5.2491174e+02,
#          6.0979694e-01],
#        [ 9.6793526e+01,  2.3354439e+02,  3.8871307e+02,  5.8243579e+02,
#          5.9487402e-01],
#        [-1.5668103e+00,  1.0145759e+02,  9.9714111e+01,  5.8032031e+02,
#          4.7019553e-01],
#        [ 6.5728766e+02,  2.4843257e+02,  7.1012836e+02,  4.0222220e+02,
#          4.4173640e-01],
#        [ 3.7986179e+02,  2.5377065e+02,  4.2024600e+02,  3.2265955e+02,
#          4.0703860e-01],
#        [ 7.0135413e+02,  2.2599469e+02,  7.5569556e+02,  3.9781561e+02,
#          3.8996202e-01],
#        [ 3.2771524e+02,  2.4698991e+02,  3.8034671e+02,  3.3056613e+02,
#          3.5393095e-01],
#        [ 3.1582327e+02,  2.4722124e+02,  3.9422156e+02,  4.3507974e+02,
#          3.3541429e-01]], dtype=float32)


image = cv2.imread(img_path)
resized_img = resize_img(image, desired_size)

import pdb; pdb.set_trace()
det = dets.detach().cpu().numpy()[0][0]

cv2.circle(resized_img, (int(det[1]), int(det[0])), 3, (255, 0, 0))

cv2.rectangle(resized_img, (96, 30), (126, 89), (255, 0, 0), 2)
cv2.imwrite('result.png', resized_img)

print('done...')






def pytorch_2_tf():

    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # torch.onnx.export(model, torch.from_numpy(inputs), 'model_simple.onnx', input_names=['input'], output_names=['output'])

    class detModel(nn.Module):
        def __init__(self, num_layers, heads, head_conv, top_k=100, pretrained=False):
            super(detModel, self).__init__()
            self.model = get_pose_net(num_layers=num_layers, heads=heads, head_conv=head_conv, pretrained=False)
            self.top_k = top_k

        def forward(self, x):
            output = self.model(x)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            # dets = ctdet_decode(hm, wh, reg=reg, K=top_k)

            return hm, wh, reg

    model = detModel(num_layers=num_layers, heads=heads, head_conv=head_conv, pretrained=False, top_k=10)
    load_model(model.model, model_path)
    dets = model(torch.from_numpy(inputs))
    print(dets)

    torch.onnx.export(model, torch.from_numpy(inputs), 'model_simple.onnx', input_names=['input'], output_names=['output'])
    # # torch.onnx.export(model_pytorch, dummy_input, './models/model_simple.onnx', input_names=['input'], output_names=['output'])

    def test_tf_pb():

        def load_pb(path_to_pb):
            with tf.gfile.GFile(path_to_pb, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')
                return graph

        tf_graph = load_pb('model_simple.pb')
        sess = tf.Session(graph=tf_graph)

        output_tensor = tf_graph.get_tensor_by_name('Sigmoid:0')
        input_tensor = tf_graph.get_tensor_by_name('input:0')

        output = sess.run(output_tensor, feed_dict={input_tensor: inputs})
        print(output)

    test_tf_pb()


    model_onnx = onnx.load('model_simple.onnx')
    tf_rep = prepare(model_onnx)
    print(tf_rep.tensor_dict)
    tf_rep.export_graph('model_simple.pb')


