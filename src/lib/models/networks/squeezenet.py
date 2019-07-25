
import io
import numpy as np
import torch.onnx

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}

BN_MOMENTUM = 0.1


def get_gauss_kernel(kernlen=5, nsig=3, channels=1):
    import numpy as np
    import scipy.stats as st

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter.astype(np.float32)


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

class PoseSqueezeNet(nn.Module):
    def __init__(self, heads, head_conv=64, deploy=False, multi_exp=1.0, **kwargs):
        self.head_conv = head_conv
        self.deconv_with_bias = False
        self.heads = heads
        self.deploy = deploy
        self.gaussian_filter_size = 5
        self.gaussian_filter_padding = int((self.gaussian_filter_size - 1) / 2)
        super(PoseSqueezeNet, self).__init__()

        scale = lambda x: int(x * multi_exp)

        self.base_model = nn.Sequential(
            nn.Conv2d(3, scale(64), kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(64), scale(16), scale(64), scale(64)),
            Fire(scale(128), scale(16), scale(64), scale(64)),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(128), scale(32), scale(128), scale(128)),
            Fire(scale(256), scale(32), scale(128), scale(128)),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(256), scale(48), scale(192), scale(192)),
            Fire(scale(384), scale(48), scale(192), scale(192)),
            Fire(scale(384), scale(64), scale(256), scale(256)),
            Fire(scale(512), scale(64), scale(256), scale(256)),
        )
        self.inplanes = scale(512)

        # self.deconv_layers = nn.PixelShuffle(4)
        # self.deconv_layers = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Upsample(scale_factor=2, mode='nearest')
        # )

        self.deconv_layers = self._make_deconv_layer(
            2,
            [head_conv*2, head_conv*2],
            [4, 4],
        )

        self.gassuian_filter = nn.Conv2d(1, 1, (self.gaussian_filter_size, self.gaussian_filter_size), padding=(self.gaussian_filter_padding, self.gaussian_filter_padding), bias=False)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(head_conv*2, head_conv*2, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv*2, num_output,
                  kernel_size=1, stride=1, padding=0))
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base_model(x)
        x = self.deconv_layers(x)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        if self.deploy:
            hm = ret['hm'].sigmoid_()
            # _hm = self.gassuian_filter(hm)
            # hmax = nn.functional.max_pool2d(_hm, (5, 5), stride=1, padding=2)
            # keep = torch.le(hmax, _hm)
            # hm = ret['hm'] * keep.float()

            hm_hp = ret['hm_hp'].sigmoid_()
            # _hm_hp = self.gassuian_filter(hm_hp)
            # _hm_hp = hm_hp
            # hm_hp_max = nn.functional.max_pool2d(_hm_hp, (3, 3), stride=1, padding=1)
            # keep = torch.le(hm_hp_max, _hm_hp)
            # hm_hp = hm_hp * keep.float()

            return torch.cat([hm, ret['wh'], ret['hps'], ret['reg'], hm_hp, ret['hp_offset']], dim=1)
        else:
            return [ret]

        # else:
        #     # ret = {}
        #     # ret['hm'] = output[:, 0:1, :, :]
        #     # ret['wh'] = output[:, 1:3, :, :]
        #     # ret['kps'] = output[:, 3:37, :, :]
        #     # ret['reg'] = output[:, 37:39, :, :]
        #     # ret['hm_hp'] = output[:, 39:56, :, :]
        #     # ret['hp_offset'] = output[:, 56:58, :, :]

        #     result = {}
        #     start = 0
        #     for key,value in self.heads.items():
        #         end = start + value
        #         indices = torch.tensor(range(start, end))
        #         torch.index_select(output, dim=1, indices)
        #         result[key] = output[:, start:end, :, :]
        #         print("{} {}:{}".format(key, start, end))
        #         start = end
        #     return [result]

    def init_weights(self, pretrained=True):
        for _, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                print('=> init {}.weight as normal(0, 0.001)'.format(m))
                print('=> init {}.bias as 0'.format(m))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                print('=> init {}.weight as kaiming_uniform_'.format(m))
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                print('=> init {}.weight as 1'.format(m))
                print('=> init {}.bias as 0'.format(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            # model.load_state_dict(torch.load('squeezenet1_1-f364aa15.pth', map_location=lambda storage, loc: storage))
            self.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']), strict=False)

        self.gassuian_filter.weight.data.copy_(torch.from_numpy(get_gauss_kernel(self.gaussian_filter_size, 3, 1).transpose(2, 3, 0, 1)))

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
                    bias=self.deconv_with_bias,
                    groups=planes//4))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            self.inplanes = planes

        return nn.Sequential(*layers)

class PoseSqueezeNetV1(nn.Module):
    def __init__(self, heads, head_conv=64, deploy=False, multi_exp=1.0, **kwargs):
        self.head_conv = head_conv
        self.deconv_with_bias = False
        self.heads = heads
        self.deploy = deploy
        self.gaussian_filter_size = 5
        self.gaussian_filter_padding = int((self.gaussian_filter_size - 1) / 2)
        super(PoseSqueezeNetV1, self).__init__()

        scale = lambda x: int(x * multi_exp)

        self.base_model = nn.Sequential(
            nn.Conv2d(3, scale(64), kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(64), scale(16), scale(64), scale(64)),
            Fire(scale(128), scale(16), scale(64), scale(64)),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(128), scale(32), scale(128), scale(128)),
            Fire(scale(256), scale(32), scale(128), scale(128)),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(256), scale(48), scale(192), scale(192)),
            Fire(scale(384), scale(48), scale(192), scale(192)),
            Fire(scale(384), scale(64), scale(256), scale(256)),
            Fire(scale(512), scale(64), scale(256), scale(256)),
        )
        self.inplanes = scale(512)
        # self.deconv_layers = nn.PixelShuffle(4)
        # self.deconv_layers = nn.Sequential(
        #     nn.PixelShuffle(2),
        #     nn.PixelShuffle(2)
        # )
        # self.deconv_layers = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Upsample(scale_factor=2, mode='nearest')
        # )
        self.deconv_layers = self._make_deconv_layer(
            2,
            [head_conv*2, head_conv*2],
            [4, 4],
        )

        self.output_dims = sum(self.heads.values())
        self.fc = nn.Sequential(
           nn.Conv2d(head_conv*2, head_conv*2, kernel_size=3, padding=1, bias=True),
           nn.ReLU(inplace=True),
           nn.Conv2d(head_conv*2, self.output_dims,
             kernel_size=1, stride=1, padding=0))

        # for head in sorted(self.heads):
        #     num_output = self.heads[head]
        #     fc = nn.Sequential(
        #         nn.Conv2d(head_conv*2, head_conv*2, kernel_size=3, padding=1, bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(head_conv*2, num_output,
        #           kernel_size=1, stride=1, padding=0))
        #     self.__setattr__(head, fc)

        self.gassuian_filter = nn.Conv2d(1, 1, (self.gaussian_filter_size, self.gaussian_filter_size), padding=(self.gaussian_filter_padding, self.gaussian_filter_padding), bias=False)

    def forward(self, x):
        x = self.base_model(x)
        x = self.deconv_layers(x)
        x = self.fc(x)

        if self.deploy:
            return x
            # hm = ret['hm'].sigmoid_()
            # _hm = self.gassuian_filter(hm)
            # hmax = nn.functional.max_pool2d(_hm, (5, 5), stride=1, padding=2)
            # keep = torch.le(hmax, _hm)
            # hm = ret['hm'] * keep.float()
            # hm_hp = ret['hm_hp'].sigmoid_()
            # _hm_hp = self.gassuian_filter(hm_hp)
            # _hm_hp = hm_hp
            # hm_hp_max = nn.functional.max_pool2d(_hm_hp, (3, 3), stride=1, padding=1)
            # keep = torch.le(hm_hp_max, _hm_hp)
            # hm_hp = hm_hp * keep.float()
            # return torch.cat([hm, ret['wh'], ret['hps'], ret['reg'], hm_hp, ret['hp_offset']], dim=1)
        else:
            result = {}
            start = 0
            for key,value in self.heads.items():
                end = start + value
                x_splited = x.split(1, dim=1)
                result[key] = torch.cat(x_splited[start:end], dim=1)
                # print("{} {}:{}".format(key, start, end))
                start = end
            return [result]

    def init_weights(self, pretrained=True):
        for _, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                print('=> init {}.weight as normal(0, 0.001)'.format(m))
                print('=> init {}.bias as 0'.format(m))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                print('=> init {}.weight as kaiming_uniform_'.format(m))
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                print('=> init {}.weight as 1'.format(m))
                print('=> init {}.bias as 0'.format(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            # model.load_state_dict(torch.load('squeezenet1_1-f364aa15.pth', map_location=lambda storage, loc: storage))
            self.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']), strict=False)

        self.gassuian_filter.weight.data.copy_(torch.from_numpy(get_gauss_kernel(self.gaussian_filter_size, 3, 1).transpose(2, 3, 0, 1)))

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
                    bias=self.deconv_with_bias,
                    groups=planes))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            self.inplanes = planes

        return nn.Sequential(*layers)


class PoseSqueezeNetV2(nn.Module):
    def __init__(self, heads, head_conv=64, deploy=False, multi_exp=1.0, **kwargs):
        self.head_conv = head_conv
        self.deconv_with_bias = False
        self.heads = heads
        self.deploy = deploy
        self.gaussian_filter_size = 5
        self.gaussian_filter_padding = int((self.gaussian_filter_size - 1) / 2)
        super(PoseSqueezeNetV2, self).__init__()

        scale = lambda x: int(x * multi_exp)

        self.base_model = nn.Sequential(
            nn.Conv2d(3, scale(64), kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(64), scale(16), scale(64), scale(64)),
            Fire(scale(128), scale(16), scale(64), scale(64)),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(128), scale(32), scale(128), scale(128)),
            Fire(scale(256), scale(32), scale(128), scale(128)),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False, padding=1),
            Fire(scale(256), scale(48), scale(192), scale(192)),
            Fire(scale(384), scale(48), scale(192), scale(192)),
            Fire(scale(384), scale(64), scale(256), scale(256)),
            Fire(scale(512), scale(64), scale(256), scale(256)),
        )
        self.inplanes = scale(512)
        # self.deconv_layers = nn.PixelShuffle(4)
        # self.deconv_layers = nn.Sequential(
        #     nn.PixelShuffle(2),
        #     nn.PixelShuffle(2)
        # )
        # self.deconv_layers = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Upsample(scale_factor=2, mode='nearest')
        # )
        self.deconv_layers = self._make_deconv_layer(
            2,
            [scale(512)//4, scale(512)//16],
            [4, 4],
        )

        self.output_dims = sum(self.heads.values())
        self.fc_1 = nn.Sequential(
           nn.Conv2d(scale(512)//16, head_conv, kernel_size=3, padding=1, bias=True),
           nn.ReLU(inplace=True),
           nn.Conv2d(head_conv, self.output_dims,
             kernel_size=1, stride=1, padding=0))

        fc2_input = scale(512)//16 + self.output_dims
        self.fc_2 = nn.Sequential(
           nn.Conv2d(fc2_input, head_conv, kernel_size=3, padding=1, bias=True),
           nn.ReLU(inplace=True),
           nn.Conv2d(head_conv, self.output_dims,
             kernel_size=1, stride=1, padding=0))

        self.gassuian_filter = nn.Conv2d(1, 1, (self.gaussian_filter_size, self.gaussian_filter_size), padding=(self.gaussian_filter_padding, self.gaussian_filter_padding), bias=False)

    def forward(self, x):
        x = self.base_model(x)
        x = self.deconv_layers(x)
        output_1 = self.fc_1(x)
        output_2 = self.fc_2(torch.cat([output_1, x], dim=1))

        if self.deploy:
            return output_2
        else:
            ret = []
            for res in [output_1, output_2]:
                result = {}
                start = 0
                for key,value in self.heads.items():
                    end = start + value
                    x_splited = res.split(1, dim=1)
                    result[key] = torch.cat(x_splited[start:end], dim=1)
                    # print("{} {}:{}".format(key, start, end))
                    start = end
                ret.append(result)
            return ret

    def init_weights(self, pretrained=True):
        for _, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                print('=> init {}.weight as normal(0, 0.001)'.format(m))
                print('=> init {}.bias as 0'.format(m))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                print('=> init {}.weight as kaiming_uniform_'.format(m))
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                print('=> init {}.weight as 1'.format(m))
                print('=> init {}.bias as 0'.format(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained:
            # model.load_state_dict(torch.load('squeezenet1_1-f364aa15.pth', map_location=lambda storage, loc: storage))
            self.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']), strict=False)
            print('load pretrained model from %s' % model_urls['squeezenet1_1'])

        self.gassuian_filter.weight.data.copy_(torch.from_numpy(get_gauss_kernel(self.gaussian_filter_size, 3, 1).transpose(2, 3, 0, 1)))

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
                    bias=self.deconv_with_bias,
                    groups=self.inplanes//4))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            self.inplanes = planes

        return nn.Sequential(*layers)


def get_squeeze_pose_net(heads, head_conv, multi_exp=1.0, num_layers=0, deploy=False, pretrained=True):
    model = PoseSqueezeNet(heads, head_conv, multi_exp=multi_exp, deploy=deploy)
    model.init_weights(pretrained)
    return model


def get_squeeze_pose_net_v1(heads, head_conv, multi_exp=1.0, num_layers=0, deploy=False, pretrained=True):
    model = PoseSqueezeNetV1(heads, head_conv, multi_exp=multi_exp, deploy=deploy)
    model.init_weights(pretrained)
    return model


def get_squeeze_pose_net_v2(heads, head_conv, multi_exp=1.0, num_layers=0, deploy=False, pretrained=True):
    model = PoseSqueezeNetV2(heads, head_conv, multi_exp=multi_exp, deploy=deploy)
    model.init_weights(pretrained)
    return model


if __name__ == '__main__':
    # usage: python -m lib.models.networks.squeezenet

    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
    # heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2}
    head_conv = 64
    batch_size = 1
    input_w, input_h = 192, 256

    model = PoseSqueezeNetV2(heads, head_conv, multi_exp=1.0, deploy=True)
    model.init_weights(True)
    model.eval()

    import cv2
    image = cv2.imread('example.jpg')
    inp_image = cv2.resize(image, (input_w, input_h)).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)

    torch_outputs = model(torch.from_numpy(images))
    torch.onnx.export(model, torch.from_numpy(images), "example.onnx", verbose=True, input_names=["data"], output_names=[ "output"])

    import onnx
    onnx_model = onnx.load("example.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    import onnxruntime
    session = onnxruntime.InferenceSession("example.onnx")
    ximg = images
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    print("The shape of the Image is: ", ximg.shape)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    result = session.run(None, {input_name: ximg})
    print(torch_outputs[0, 0, 0, :10].detach().numpy())
    print(result[0][0, 0, 0, :10])

    import os; os.system('python3 -m onnxsim example.onnx example-sim.onnx')

    # model size 3.5M
    import pdb; pdb.set_trace()


