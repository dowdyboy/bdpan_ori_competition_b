import os
import sys

from paddle.io import DataLoader
import paddle

from bdpan_ori.v6.model import OriModelV6
from bdpan_ori.v6.dataset import OriTestDataset

assert len(sys.argv) == 3

src_image_dir = sys.argv[1]
save_dir = sys.argv[2]


def build_data():
    # batch size 可能影响速度？
    test_dataset = OriTestDataset(src_image_dir, scale_size=230, img_size=224)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=0, drop_last=False)
    return test_loader, test_dataset


def build_model():
    model = OriModelV6(is_pretrain=False)
    model = [model]
    model = paddle.amp.decorate(model, level='O2')
    model = model[0]
    return model


def test_step(model, bat):
    ret = []
    bat_x, bat_path = bat
    pred_y, aux_y1, aux_y2, aux_y3 = model(bat_x)
    # pred_y = pred_y + aux_y1
    # pred_y = pred_y + aux_y2
    # pred_y = pred_y + aux_y3
    pred_y = 2.5 * pred_y + aux_y1 + aux_y2 + aux_y3
    pred_label = paddle.argmax(pred_y, axis=1)
    for i in range(bat_x.shape[0]):
        ret.append(f'{os.path.basename(bat_path[i])} {pred_label[i].item()}')
    # return f'{os.path.basename(bat_path[0])} {pred_label[0].item()}'
    return ret


def process():
    test_loader, test_dataset = build_data()

    # chk_dir = 'checkpoint/ori/0827/v6_epoch_1023/model_0.pd'
    # chk_dir = 'checkpoint/ori/0829/v6ms_epoch_972/model_0.pd'
    # chk_dir = 'checkpoint/ori/0830/v6_256_epoch_840/model_0.pd'
    # chk_dir = 'checkpoint/ori/0831/v6_fp_epoch_1318/model_0.pd'  # best
    chk_dir = 'checkpoint/ori/0901/v6_224_epoch_1117/model_0.pd'
    model = build_model()
    weight = paddle.load(chk_dir)
    model.load_dict(weight)
    model.eval()

    res = []
    for bat in test_loader:
        with paddle.no_grad():
            with paddle.amp.auto_cast(level='O2'):
                res.extend(
                    test_step(model, bat)
                )
    with open(save_dir, 'w+') as f:
        f.write('\n'.join(res))


if __name__ == '__main__':
    process()

