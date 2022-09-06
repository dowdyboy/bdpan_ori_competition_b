import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import ConvNormActivation
from .shufflenetv2 import ShuffleNetV2
from .mobilenetv3 import MobileNetV3Small
from functools import partial


class OriModelV3(nn.Layer):

    def __init__(self, num_classes=4, is_pretrain=True):
        super(OriModelV3, self).__init__()
        self.shuffle33_net = ShuffleNetV2(scale=0.33, act='swish', num_classes=4, with_pool=True)
        self.shuffle5_net = ShuffleNetV2(scale=0.5, act='relu', num_classes=4, with_pool=True)
        self.mobile33_net = MobileNetV3Small(scale=0.33, num_classes=4, )
        if is_pretrain:
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_5.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.shuffle5_net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_33.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.shuffle33_net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')

        self.shuffle33_select_feats = [1, 5, 13, 17]
        self.shuffle5_select_feats = [1, 5, 13, 17]
        self.mobile33_select_feats = [1, 3, 8, 11]

        shuffle33_feats_in_channel = [24, 32, 64, 128]
        shuffle5_feats_in_channel = [24, 48, 96, 192]
        mobile33_feats_in_channel = [8, 8, 16, 32]
        all_feats_out_channel = [8, 8, 16, 32]

        self.shuffle33_trans_convs = nn.LayerList([
            ConvNormActivation(
                in_channels=in_channel,
                out_channels=all_feats_out_channel[i],
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for i, in_channel in enumerate(shuffle33_feats_in_channel)
        ])
        self.shuffle5_trans_convs = nn.LayerList([
            ConvNormActivation(
                in_channels=in_channel,
                out_channels=all_feats_out_channel[i],
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for i, in_channel in enumerate(shuffle5_feats_in_channel)
        ])
        self.mobile33_trans_convs = nn.LayerList([
            nn.Identity()
            for i, in_channel in enumerate(mobile33_feats_in_channel)
        ])
        self.downsample_convs = nn.LayerList()
        for i in range(len(all_feats_out_channel) - 1):
            self.downsample_convs.append(
                ConvNormActivation(
                    in_channels=all_feats_out_channel[i],
                    out_channels=all_feats_out_channel[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
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
            nn.Linear(all_feats_out_channel[-1] * 8, all_feats_out_channel[-1]),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(all_feats_out_channel[-1], num_classes))

    def forward(self, x):
        out1, feats1 = self.shuffle33_net.extract_feats(x)
        out2, feats2 = self.shuffle5_net.extract_feats(x)
        out3, feats3 = self.mobile33_net.extract_feats(x)
        shuffle33_feats = []
        shuffle5_feats = []
        mobile33_feats = []
        for i, idx in enumerate(self.shuffle33_select_feats):
            shuffle33_feats.append(self.shuffle33_trans_convs[i](feats1[idx]))
        for i, idx in enumerate(self.shuffle5_select_feats):
            shuffle5_feats.append(self.shuffle5_trans_convs[i](feats2[idx]))
        for i, idx in enumerate(self.mobile33_select_feats):
            mobile33_feats.append(self.mobile33_trans_convs[i](feats3[idx]))
        fuse_feats = []
        for i in range(len(shuffle33_feats)):
            fuse_feats.append(
                shuffle33_feats[i] + shuffle5_feats[i] + mobile33_feats[i]
            )
        fuse_x = fuse_feats[0]
        for i in range(1, len(fuse_feats)):
            fuse_x = self.downsample_convs[i - 1](fuse_x) + fuse_feats[i]
        out_fuse = self.last_conv(fuse_x)
        out_fuse = self.avg_pool(out_fuse)
        out_fuse = paddle.flatten(out_fuse, 1)
        out_fuse = self.classifier(out_fuse)
        return out_fuse, out1, out2, out3


