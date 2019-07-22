import torch
import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, heads, n_class=1000, input_size=224, width_mult=1., deploy=False):
        super(MobileNetV2, self).__init__()
        self.heads = heads
        self.deploy = deploy
        block = InvertedResidual
        input_channel = 32
        last_channel = 128
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, n_class),
        # )

        self.inplanes = self.last_channel
        self.deconv_layers = self._make_deconv_layer(
            3,
            [self.last_channel, self.last_channel//2, self.last_channel//4],
            [4, 4, 4],
        )

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                # nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                # nn.ReLU(inplace=True),
                nn.Conv2d(self.last_channel//4, num_output,
                  kernel_size=1, stride=1, padding=0))
            self.__setattr__(head, fc)

        self._initialize_weights()

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
                    bias=False,
                    groups=planes))
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.deconv_layers(x)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        if self.deploy:
            hm = ret['hm'].sigmoid_()
            # hm = self.gassuian_filter(hm)
            # hmax = nn.functional.max_pool2d(hm, (3, 3), stride=1, padding=1)
            # keep = torch.le(hmax, hm)
            # hm = hm * keep.float()
            return torch.cat([hm, ret['wh'], ret['hps'], ret['reg']], dim=1)
        else:
            return [ret]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':

    heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2}
    batch_size = 1
    input_w, input_h = 224, 224

    model = MobileNetV2(heads, n_class=1000, input_size=256, width_mult=0.35, deploy=True)
    model.eval()

    import cv2
    import numpy as np
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

    torch_op = torch_outputs.detach().numpy()
    onnx_op = result[0]
    print(torch_op[0, :10])
    print(onnx_op[0, :10])

    import pdb; pdb.set_trace()

    import os; os.system('python3 -m onnxsim example.onnx example-sim.onnx')
