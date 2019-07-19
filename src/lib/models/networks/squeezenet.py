
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
    return out_filter



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
    def __init__(self, heads, head_conv=64, deploy=False, **kwargs):
        self.inplanes = 512
        self.deconv_with_bias = False
        self.heads = heads
        self.deploy = deploy
        self.gaussian_filter_size = 5
        self.gaussian_filter_padding = int((self.gaussian_filter_size - 1) / 2)
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

        # self.deconv_layers = nn.PixelShuffle(2)
        # self.deconv_layers = nn.Upsample(scale_factor=2)

        self.deconv_layers = self._make_deconv_layer(
            2,
            [512, 256],
            [4, 4],
        )

        self.gassuian_filter = nn.Conv2d(1, 1, (self.gaussian_filter_size, self.gaussian_filter_size), padding=(self.gaussian_filter_padding, self.gaussian_filter_padding), bias=False)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(256, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
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
            hm = self.gassuian_filter(hm)
            # hmax = nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1)
            # keep = torch.le(hmax, hm)
            # hm = hm * keep.float()
            return torch.cat([hm, ret['wh'], ret['hps'], ret['reg']], dim=1)
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
        #         import pdb; pdb.set_trace()
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
                    groups=planes))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


def get_squeeze_pose_net(heads, head_conv, num_layers=0, deploy=False, pretrained=True):
    model = PoseSqueezeNet(heads, head_conv, deploy=deploy)
    model.init_weights(pretrained)

    return model

if __name__ == '__main__':
    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
    head_conv = 64
    batch_size = 1

    model = PoseSqueezeNet(heads, head_conv, deploy=True)
    model.init_weights(True)
    model.eval()

    import cv2
    image = cv2.imread('example.jpg')
    inp_image = cv2.resize(image, (512, 512)).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, 512, 512)

    torch.onnx.export(model, torch.from_numpy(images), "example.onnx", verbose=True, input_names=["data"], output_names=[ "outputs"])
    torch_outputs = model(torch.from_numpy(images))

    import onnx
    # Load the ONNX model
    onnx_model = onnx.load("example.onnx")
    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
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

    # model size 3.5M
    import pdb; pdb.set_trace()


