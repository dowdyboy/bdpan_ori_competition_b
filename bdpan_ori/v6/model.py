import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import ConvNormActivation
from .shufflenetv2 import ShuffleNetV2
from functools import partial


class SqueezeExpandLayer(nn.Layer):

    def __init__(self, in_channel, out_channel, split_count):
        super(SqueezeExpandLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.split_count = split_count
        assert out_channel % split_count == 0
        self.squeeze_dp_conv = nn.Conv2D(in_channel, in_channel, 3, 1, 1, groups=in_channel, )
        self.expand_convs = nn.LayerList([
            ConvNormActivation(
                in_channel, out_channel // split_count,
                kernel_size=1, stride=1, padding=0,
            )
            for _ in range(split_count)
        ])

    def forward(self, x):
        outs = []
        x = self.squeeze_dp_conv(x)
        for expand_conv in self.expand_convs:
            outs.append(
                expand_conv(x)
            )
        out = paddle.concat(outs, axis=1)
        return out


class TransConvLayer(nn.Layer):

    def __init__(self, in_channel, out_channel, ):
        super(TransConvLayer, self).__init__()
        self.conv_a = SqueezeExpandLayer(in_channel, in_channel, 4)
        self.final_conv = nn.Conv2D(in_channel, out_channel, 1, 1, 0, )

    def forward(self, x):
        x = x + self.conv_a(x)
        x = self.final_conv(x)
        return x


class DownSampleConvLayer(nn.Layer):

    def __init__(self, in_channel, out_channel):
        super(DownSampleConvLayer, self).__init__()
        self.conv_a = SqueezeExpandLayer(in_channel, 2 * in_channel, 4)
        self.conv_b = SqueezeExpandLayer(2 * in_channel, 4 * in_channel, 2)
        self.conv_c = SqueezeExpandLayer(4 * in_channel, 2 * in_channel, 4)
        self.conv_d = SqueezeExpandLayer(2 * in_channel, in_channel, 2)
        self.mp = nn.MaxPool2D(3, 2, 1)
        self.last_conv = nn.Conv2D(in_channel, out_channel, 1, 1, 0, )

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(a)
        c = a + self.conv_c(b)
        d = self.conv_d(c)
        x = x + d
        x = self.mp(x)
        x = self.last_conv(x)
        return x


class OriModelV6(nn.Layer):

    def __init__(self, num_classes=4, is_pretrain=True):
        super(OriModelV6, self).__init__()
        self.shuffle33_net = ShuffleNetV2(scale=0.33, act='swish', num_classes=4, with_pool=True)
        self.shuffle5_net = ShuffleNetV2(scale=0.5, act='relu', num_classes=4, with_pool=True)
        self.shuffle25_net = ShuffleNetV2(scale=0.25, act='swish', num_classes=4, with_pool=True)
        if is_pretrain:
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_5.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.shuffle5_net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_33.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.shuffle33_net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_25.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.shuffle25_net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')

        self.shuffle33_select_feats = [1, 5, 13, 17]
        self.shuffle5_select_feats = [1, 5, 13, 17]
        self.shuffle25_select_feats = [1, 5, 13, 17]

        shuffle33_feats_in_channel = [24, 32, 64, 128]
        shuffle5_feats_in_channel = [24, 48, 96, 192]
        shuffle25_feats_in_channel = [24, 24, 48, 96]
        all_feats_out_channel = [4, 8, 16, 24]

        self.shuffle33_trans_convs = nn.LayerList([
            TransConvLayer(in_channel, all_feats_out_channel[i])
            for i, in_channel in enumerate(shuffle33_feats_in_channel)
        ])
        self.shuffle5_trans_convs = nn.LayerList([
            TransConvLayer(in_channel, all_feats_out_channel[i])
            for i, in_channel in enumerate(shuffle5_feats_in_channel)
        ])
        self.shuffle25_trans_convs = nn.LayerList([
            TransConvLayer(in_channel, all_feats_out_channel[i])
            for i, in_channel in enumerate(shuffle25_feats_in_channel)
        ])
        self.downsample_convs = nn.LayerList()
        for i in range(len(all_feats_out_channel) - 1):
            self.downsample_convs.append(
                DownSampleConvLayer(all_feats_out_channel[i], all_feats_out_channel[i + 1])
            )
        self.last_conv = ConvNormActivation(
            in_channels=all_feats_out_channel[-1],
            out_channels=all_feats_out_channel[-1] * 8,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_layer=partial(nn.BatchNorm2D, epsilon=0.001, momentum=0.99),
            activation_layer=nn.Hardswish)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.classifier = nn.Sequential(
            nn.Linear(all_feats_out_channel[-1] * 8, all_feats_out_channel[-1] * 2),
            nn.Hardswish(),
            nn.Dropout(p=0.15),
            nn.Linear(all_feats_out_channel[-1] * 2, num_classes))

    def forward(self, x):
        out1, feats1 = self.shuffle33_net.extract_feats(x)
        out2, feats2 = self.shuffle5_net.extract_feats(x)
        out3, feats3 = self.shuffle25_net.extract_feats(x)
        shuffle33_feats = []
        shuffle5_feats = []
        shuffle25_feats = []
        for i, idx in enumerate(self.shuffle33_select_feats):
            shuffle33_feats.append(self.shuffle33_trans_convs[i](feats1[idx]))
        for i, idx in enumerate(self.shuffle5_select_feats):
            shuffle5_feats.append(self.shuffle5_trans_convs[i](feats2[idx]))
        for i, idx in enumerate(self.shuffle25_select_feats):
            shuffle25_feats.append(self.shuffle25_trans_convs[i](feats3[idx]))
        fuse_feats = []
        for i in range(len(shuffle33_feats)):
            fuse_feats.append(
                shuffle33_feats[i] + shuffle5_feats[i] + shuffle25_feats[i]
            )
        fuse_x = fuse_feats[0]
        for i in range(1, len(fuse_feats)):
            fuse_x = self.downsample_convs[i - 1](fuse_x) + fuse_feats[i]
        out_fuse = self.last_conv(fuse_x)
        out_fuse = self.avg_pool(out_fuse)
        out_fuse = paddle.flatten(out_fuse, 1)
        out_fuse = self.classifier(out_fuse)
        return out_fuse, out1, out2, out3


