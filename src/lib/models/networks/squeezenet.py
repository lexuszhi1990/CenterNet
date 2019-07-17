import io
import numpy as np
import torch.onnx

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo


# __all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

BN_MOMENTUM = 0.1
model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )


        # # Final convolution is initialized differently form the rest
        # final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     final_conv,
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(13)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.features(x)

        # x = self.features(x)
        # x = self.classifier(x)
        # return x.view(x.size(0), self.num_classes)


class PoseSqueezeNet(nn.Module):
    def __init__(self, heads, head_conv, **kwargs):
        self.inplanes = 512
        self.deconv_with_bias = False
        self.heads = heads
        super(PoseSqueezeNet, self).__init__()

        self.base_model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        # self.deconv_layers = self._make_deconv_layer(
        #     3,
        #     [512, 256, 256],
        #     [4, 4, 4],
        # )

        self.deconv_layers = self._make_deconv_layer(
            2,
            [512, 256],
            [4, 4],
        )

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv,
                      kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output,
                      kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(
                  in_channels=256,
                  out_channels=num_output,
                  kernel_size=1,
                  stride=1,
                  padding=0
                )
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base_model(x)
        x = self.deconv_layers(x)
        # ret = {}
        # for head in self.heads:
        #     ret[head] = self.__getattr__(head)(x)
        # return [ret]
        ret = []
        for head in self.heads:
            ret.append(self.__getattr__(head)(x))
        return torch.cat(ret, 1)

    def init_weights(self, pretrained=True):
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                print('=> init {}.weight as normal(0, 0.001)'.format(m))
                print('=> init {}.bias as 0'.format(m))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                print('=> init {}.weight as 1'.format(m))
                print('=> init {}.bias as 0'.format(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for head in self.heads:
          final_layer = self.__getattr__(head)
          for i, m in enumerate(final_layer.modules()):
              if isinstance(m, nn.Conv2d):
                  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                  # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                  # print('=> init {}.bias as 0'.format(name))
                  if m.weight.shape[0] == self.heads[head]:
                      if 'hm' in head:
                          nn.init.constant_(m.bias, -2.19)
                      else:
                          nn.init.normal_(m.weight, std=0.001)
                          nn.init.constant_(m.bias, 0)

        if pretrained:
            url = model_urls['squeezenet1_1']
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(model_zoo.load_url(url), strict=False)
        else:
            print('=> init models with uniform')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    print('=> init {}.weight as kaiming_uniform_'.format(m))
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


def get_squeeze_pose_net(heads, head_conv, pretrained=False, num_layers=-1):
    model = PoseSqueezeNet(heads, head_conv=head_conv)
    model.init_weights(pretrained=pretrained)
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('squeezenet1_1-f364aa15.pth', map_location=lambda storage, loc: storage))
        # model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    # (Pdb) topk_score
    # tensor([[4.1940e-05]], grad_fn=<TopkBackward>)
    # (Pdb) topk_inds
    # tensor([[457]])
    # (Pdb) topk_clses
    # tensor([[0]], dtype=torch.int32)
    # (Pdb) topk_ys
    # tensor([[3.]])
    # (Pdb) topk_xs
    # tensor([[73.]])

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs



class PoseSqueezeFullNet(nn.Module):
    def __init__(self, heads, head_conv, **kwargs):
        super(PoseSqueezeFullNet, self).__init__()

        self.model = get_squeeze_pose_net(heads, head_conv, pretrained=False)


    def forward(self, x):
        output = self.model(x)
        heat, wh, kps, reg, hm_hp, hp_offset = output[0]['hm'], output[0]['wh'], output[0]['hps'], output[0]['reg'], output[0]['hm_hp'], output[0]['hp_offset']

        _topk(heat, 1)
        aa = torch.argmax(heat)
        height = aa / 128
        width = aa % 128

        return heat[0, 0, height, width]

        # import pdb; pdb.set_trace()
        # from lib.models.decode import multi_pose_decode_dev
        # dets = multi_pose_decode_dev(heat, wh, kps, reg, hm_hp, hp_offset, K=10)

        # return dets

        # K = 10
        # batch, cat, height, width = 1, 1, 128, 128


        # num_joints = kps.shape[1] // 2
        # # heat = torch.sigmoid(heat)
        # # perform nms on heatmaps
        # heat = _nms(heat)
        # scores, inds, clses, ys, xs = _topk(heat, batch, cat, height, width, K=K)

        # import pdb; pdb.set_trace()

        # x = self.deconv_layers(x)
        # ret = {}
        # for head in self.heads:
        #     ret[head] = self.__getattr__(head)(x)
        # return [ret]
        # ret = []
        # for head in self.heads:
        #     ret.append(self.__getattr__(head)(x))


if __name__ == '__main__':
    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
    head_conv = 64
    batch_size = 1
    # dummy_input = torch.randn(batch_size, 3, 256, 256)

    import cv2
    image = cv2.imread('example.jpg')
    inp_image = cv2.resize(image, (512, 512)).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, 512, 512)

    # full_model = PoseSqueezeFullNet(heads, head_conv=head_conv)
    # full_model.eval()
    # torch_outputs = full_model(dummy_input)
    # import pdb; pdb.set_trace()

    # input_names = ["data"]
    # output_names = [ "outputs"]
    # torch.onnx.export(full_model, dummy_input, "squeezenet.onnx", verbose=True, input_names=input_names, output_names=output_names)

    torch_model = get_squeeze_pose_net(heads, head_conv, pretrained=False)
    torch_model.eval()
    torch_outputs = torch_model(torch.from_numpy(images))

    # from torch.autograd import Variable
    # dummy_input_var = Variable(dummy_input, requires_grad=True)
    input_names = ["data"]
    output_names = [ "outputs"]
    torch.onnx.export(torch_model, torch.from_numpy(images), "squeezenet.onnx", verbose=True, input_names=input_names, output_names=output_names)

    import onnx
    # Load the ONNX model
    onnx_model = onnx.load("squeezenet.onnx")
    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(onnx_model.graph))

    import onnxruntime
    session = onnxruntime.InferenceSession("squeezenet.onnx")
    ximg = images
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    print("The shape of the Image is: ", ximg.shape)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    result = session.run(None, {input_name: ximg})
    print(torch_outputs[0, 0, 0, :10].detach().numpy())
    print(result[0][0, 0, 0, :10])

    import pdb; pdb.set_trace()
