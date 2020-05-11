import torch.nn as nn


def Make_layer(block, num_filters, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block(num_filters))
    return nn.Sequential(*layers)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def resnet28(feature_dim, use_pool, use_dropout):
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super(ResBlock, self).__init__()
            self.resblock = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.PReLU(channels)
            )
        def forward(self, x):
            return x + self.resblock(x)

    filters = [64, 128, 256, 512]
    units = [1, 2, 5, 3]
    net_list = []
    for i, (num_units, num_filters) in enumerate(zip(units, filters)):
        if i == 0:
            net_list += [nn.Conv2d(3, 64, 3),
                         nn.PReLU(64),
                         nn.Conv2d(64, 64, 3),
                         nn.PReLU(64),
                         nn.MaxPool2d(2)]
        elif i == 1:
            net_list += [nn.Conv2d(64, 128, 3),
                         nn.PReLU(128),
                         nn.MaxPool2d(2)]
        elif i == 2:
            net_list += [nn.Conv2d(128, 256, 3),
                         nn.PReLU(256),
                         nn.MaxPool2d(2)]
        elif i == 3:
            net_list += [nn.Conv2d(256, 512, 3),
                         nn.PReLU(512),
                         nn.MaxPool2d(2)]
        if num_units > 0:
            net_list += [Make_layer(ResBlock, num_filters=num_filters, num_of_layer=num_units)]
    if use_pool:
        net_list += [nn.AdaptiveAvgPool2d((1, 1))]
    net_list += [Flatten()]
    if use_dropout:
        net_list += [nn.Dropout()]
    if use_pool:
        net_list += [nn.Linear(512, feature_dim)]
    else:
        net_list += [nn.Linear(512*2*2, 2048)]
        net_list += [nn.BatchNorm1d(2048)]
        net_list += [nn.PReLU(2048)]
        net_list += [nn.Linear(2048, 2048)]
        net_list += [nn.BatchNorm1d(2048)]
        net_list += [nn.PReLU(2048)]
        net_list += [nn.Linear(2048, feature_dim)]
    return nn.Sequential(*net_list)


def spherenet(num_layers, feature_dim, image_size, double_depth, use_batchnorm, use_pool, use_dropout):
    """SphereNets.
    We follow the paper, and the official caffe code:
        SphereFace: Deep Hypersphere Embedding for Face Recognition, CVPR, 2017.
        https://github.com/wy1iu/sphereface
    """
    class SphereResBlock(nn.Module):
        def __init__(self, channels):
            super(SphereResBlock, self).__init__()
            resblock_net_list = []
            for _ in range(2):
                resblock_net_list += [nn.Conv2d(channels, channels, kernel_size=3, padding=1)]
                if use_batchnorm:
                    resblock_net_list += [nn.BatchNorm2d(channels)]
                resblock_net_list += [nn.PReLU(channels)]

            self.resblock = nn.Sequential(*resblock_net_list)

        def forward(self, x):
            return x + self.resblock(x)

    filters = [64, 128, 256, 512]
    if num_layers == 4:
        units = [0, 0, 0, 0]
    elif num_layers == 10:
        units = [0, 1, 2, 0]
    elif num_layers == 16:
        units = [1, 2, 2, 1]
    elif num_layers == 20:
        units = [1, 2, 4, 1]
    elif num_layers == 36:
        units = [2, 4, 8, 2]
    elif num_layers == 64:
        units = [3, 8, 16, 3]
    net_list = []
    for i, (num_units, num_filters) in enumerate(zip(units, filters)):
        if i == 0:
            net_list += [nn.Conv2d(3, 64, 3, 2, 1)]
            if use_batchnorm:
                net_list += [nn.BatchNorm2d(64)]
            net_list += [nn.PReLU(64)]
            if double_depth:
                net_list += [nn.Conv2d(64, 64, kernel_size=3, padding=1)]
                if use_batchnorm:
                    net_list += [nn.BatchNorm2d(64)]
                net_list += [nn.PReLU(64)]
        elif i == 1:
            net_list += [nn.Conv2d(64, 128, 3, 2, 1)]
            if use_batchnorm:
                net_list += [nn.BatchNorm2d(128)]
            net_list += [nn.PReLU(128)]
            if double_depth:
                net_list += [nn.Conv2d(128, 128, kernel_size=3, padding=1)]
                if use_batchnorm:
                    net_list += [nn.BatchNorm2d(128)]
                net_list += [nn.PReLU(128)]
        elif i == 2:
            net_list += [nn.Conv2d(128, 256, 3, 2, 1)]
            if use_batchnorm:
                net_list += [nn.BatchNorm2d(256)]
            net_list += [nn.PReLU(256)]
            if double_depth:
                net_list += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
                if use_batchnorm:
                    net_list += [nn.BatchNorm2d(256)]
                net_list += [nn.PReLU(256)]
        elif i == 3:
            net_list += [nn.Conv2d(256, 512, 3, 2, 1)]
            if use_batchnorm:
                net_list += [nn.BatchNorm2d(512)]
            net_list += [nn.PReLU(512)]
            if double_depth:
                net_list += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
                if use_batchnorm:
                    net_list += [nn.BatchNorm2d(512)]
                net_list += [nn.PReLU(512)]
        if num_units > 0:
            net_list += [Make_layer(SphereResBlock, num_filters=num_filters, num_of_layer=num_units)]
    if use_pool:
        net_list += [nn.AdaptiveAvgPool2d((1, 1))]
    net_list += [Flatten()]
    if use_dropout:
        net_list += [nn.Dropout()]
    if use_pool:
        net_list += [nn.Linear(512, feature_dim)]
    else:
        # output dimension = input dimension / 16
        output_size = int(image_size / 16)
        net_list += [nn.Linear(512*output_size*output_size, 2048)]
        net_list += [nn.BatchNorm1d(2048)]
        net_list += [nn.PReLU(2048)]
        net_list += [nn.Linear(2048, 2048)]
        net_list += [nn.BatchNorm1d(2048)]
        net_list += [nn.PReLU(2048)]
        net_list += [nn.Linear(2048, feature_dim)]

    return nn.Sequential(*net_list)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 feature_dim=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(self.last_channel, feature_dim),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)