import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models.shufflenetv2 import ShuffleNetV2
from paddle.vision.models.mobilenetv3 import MobileNetV3Small


# class OriModel(nn.Layer):
#
#     def __init__(self, ):
#         super(OriModel, self).__init__()
#         # self.net = ShuffleNetV2(scale=0.33, act='relu', num_classes=4, with_pool=True)
#         # net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_33.pdparams'
#         # net_weight = paddle.load(net_weight_path)
#         # self.net.set_state_dict(net_weight)
#         # print(f'successful load {net_weight_path}')
#         self.conv = nn.Conv2D(3, 4, 3, 1, 1)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = F.adaptive_avg_pool2d(x, 1, )
#         x = paddle.squeeze(x)
#         return x


# windows下跑不起来。晕了。
class OriModel(nn.Layer):

    def __init__(self, is_train=True):
        super(OriModel, self).__init__()
        self.net = ShuffleNetV2(scale=0.33, act='relu', num_classes=4, with_pool=True)
        if is_train:
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_33.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')

    def forward(self, x):
        return self.net(x)


class OriModelV2(nn.Layer):

    def __init__(self, is_train=True):
        super(OriModelV2, self).__init__()
        self.shuffle33_net = ShuffleNetV2(scale=0.33, act='swish', num_classes=4, with_pool=True)
        self.shuffle5_net = ShuffleNetV2(scale=0.5, act='relu', num_classes=4, with_pool=True)
        self.mobile33_net = MobileNetV3Small(scale=0.33, num_classes=4, )
        if is_train:
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_5.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.shuffle5_net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_33.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.shuffle33_net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')

    def forward(self, x):
        out1 = self.shuffle33_net(x)
        out2 = self.shuffle5_net(x)
        out3 = self.mobile33_net(x)
        out = out1 + out2 + out3
        return out


