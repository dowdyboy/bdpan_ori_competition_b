import paddle
from paddle.io import DataLoader

import os
import argparse
from PIL import Image
import numpy as np

from dowdyboy_lib.paddle.trainer import Trainer, TrainerConfig

from bdpan_ori.baseline.model import OriModel
from bdpan_ori.baseline.dataset import OriDataset

parser = argparse.ArgumentParser(description='ori net train baseline')
# data config
parser.add_argument('--train-data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--val-data-dir', type=str, required=True, help='val data dir')
parser.add_argument('--scale-size', type=int, default=520, help='input img size')
parser.add_argument('--img-size', type=int, default=512, help='input img size')
parser.add_argument('--num-workers', type=int, default=4, help='num workers')
# optimizer config
parser.add_argument('--lr', type=float, default=1e-3, help='lr')
parser.add_argument('--use-scheduler', default=False, action='store_true', help='use schedule')
parser.add_argument('--use-warmup', default=False, action='store_true', help='use warmup')
parser.add_argument('--weight-decay', type=float, default=5e-5, help='model weight decay')
# train config
parser.add_argument('--epoch', type=int, default=10, help='epoch num')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--out-dir', type=str, default='./output', help='out dir')
parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
parser.add_argument('--last-epoch', type=int, default=-1, help='last epoch')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--log-interval', type=int, default=None, help='log process')
args = parser.parse_args()


def build_data():
    train_dataset = OriDataset(
        source_list=[
            os.path.join(args.train_data_dir, 'doc_deblur'),
            os.path.join(args.train_data_dir, 'doc_dehw'),
            os.path.join(args.train_data_dir, 'icdar'),
            os.path.join(args.train_data_dir, 'imagenet'),
            os.path.join(args.train_data_dir, 'ocr'),
        ],
        source_weight_list=[0.25, 0.2, 0.15, 0.2, 0.2],
        data_size=500,
        scale_size=args.scale_size,
        img_size=args.img_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_dataset = OriDataset(
        source_list=[
            os.path.join(args.val_data_dir, 'doc_deblur'),
            os.path.join(args.val_data_dir, 'doc_dehw'),
            os.path.join(args.val_data_dir, 'icdar'),
            os.path.join(args.val_data_dir, 'imagenet'),
            os.path.join(args.val_data_dir, 'ocr'),
        ],
        source_weight_list=[0.25, 0.2, 0.15, 0.2, 0.2],
        data_size=100,
        scale_size=args.scale_size,
        img_size=args.img_size,
        hflip_p=None,
        color_jitter=False,
        is_val=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_loader, train_dataset, val_loader, val_dataset


def build_model():
    model = OriModel()
    return model


def build_optimizer(model: paddle.nn.Layer):
    lr = args.lr
    lr_scheduler = None
    if args.use_scheduler:
        lr = paddle.optimizer.lr.CosineAnnealingDecay(lr, args.epoch, last_epoch=args.last_epoch, verbose=True)
        lr_scheduler = lr
    if args.use_warmup:
        lr = paddle.optimizer.lr.LinearWarmup(lr, 5, args.lr * 0.1, args.lr, last_epoch=args.last_epoch, verbose=True)
        lr_scheduler = lr
    optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters(), weight_decay=args.weight_decay)
    return optimizer, lr_scheduler


def train_step(trainer: Trainer, bat, bat_idx, global_step):
    [model] = trainer.get_models()
    [loss_func] = trainer.get_components()
    bat_x, bat_y = bat
    pred_y = model(bat_x)
    pred_label = paddle.argmax(pred_y, axis=1)
    acc_count = paddle.sum(pred_label == bat_y)
    bat_count = bat_x.shape[0]
    loss = loss_func(pred_y, bat_y)
    trainer.log({
        'train_loss': loss.item()
    }, global_step)
    trainer.set_records({
        'train_loss': loss.item(),
        'train_acc_count': acc_count,
        'train_bat_count': bat_count,
    })
    if args.log_interval is not None and global_step % args.log_interval == 0:
        trainer.print(f'train step: {global_step}, loss: {loss.item()}')
    return loss


def val_step(trainer: Trainer, bat, bat_idx, global_step):
    [model] = trainer.get_models()
    [loss_func] = trainer.get_components()
    bat_x, bat_y = bat
    pred_y = model(bat_x)
    pred_label = paddle.argmax(pred_y, axis=1)
    acc_count = paddle.sum(pred_label == bat_y)
    bat_count = bat_x.shape[0]
    loss = loss_func(pred_y, bat_y)
    trainer.log({
        'val_loss': loss.item()
    }, global_step)
    trainer.set_records({
        'val_loss': loss.item(),
        'val_acc_count': acc_count,
        'val_bat_count': bat_count,
    })
    return loss


def on_epoch_end(trainer: Trainer, ep):
    [optimizer], _ = trainer.get_optimizers()
    rec = trainer.get_records()
    val_acc_count = paddle.sum(rec['val_acc_count']).item()
    val_bat_count = paddle.sum(rec['val_bat_count']).item()
    val_acc = float(val_acc_count) / val_bat_count
    train_acc_count = paddle.sum(rec['train_acc_count']).item()
    train_bat_count = paddle.sum(rec['train_bat_count']).item()
    train_acc = float(train_acc_count) / train_bat_count
    ep_val_loss = paddle.mean(rec['val_loss']).item()
    ep_train_loss = paddle.mean(rec['train_loss']).item()
    lr = optimizer.get_lr()
    trainer.log({
        'ep_val_loss': ep_val_loss,
        'ep_train_loss': ep_train_loss,
        'val_acc': val_acc,
        'train_acc': train_acc,
        'lr': lr,
    }, ep)
    trainer.print(f'ep_train_loss: {ep_train_loss} , ep_val_loss: {ep_val_loss}')
    trainer.print(f'train_acc: {train_acc} , val_acc: {val_acc}')
    trainer.print(f'lr: {lr}')


def save_best_calc_func(trainer: Trainer):
    rec = trainer.get_records()
    val_acc_count = paddle.sum(rec['val_acc_count']).item()
    val_bat_count = paddle.sum(rec['val_bat_count']).item()
    return float(val_acc_count) / val_bat_count


def main():
    cfg = TrainerConfig(
        epoch=args.epoch,
        out_dir=args.out_dir,
        mixed_precision='no',
        multi_gpu=False,
        save_interval=5,
        save_best=True,
        save_best_type='max',
        seed=args.seed,
        auto_optimize=True,
        auto_schedule=True,
        auto_free=True,
    )
    trainer = Trainer(cfg)
    trainer.print(args)

    train_loader, train_dataset, val_loader, val_dataset = build_data()
    trainer.print(f'train size: {len(train_dataset)}, val size: {len(val_dataset)}')

    model = build_model()
    loss_func = paddle.nn.CrossEntropyLoss()
    trainer.print(model)
    trainer.print(loss_func)

    optimizer, lr_scheduler = build_optimizer(model)
    trainer.print(optimizer)
    trainer.print(lr_scheduler)

    trainer.set_train_dataloader(train_loader)
    trainer.set_val_dataloader(val_loader)
    trainer.set_model(model)
    trainer.set_component(loss_func)
    trainer.set_optimizer(optimizer, lr_scheduler)

    trainer.set_save_best_calc_func(save_best_calc_func)

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        trainer.print(f'load checkpoint from {args.resume}')

    trainer.fit(
        train_step=train_step,
        val_step=val_step,
        on_epoch_end=on_epoch_end,
    )

    return


if __name__ == '__main__':
    main()
