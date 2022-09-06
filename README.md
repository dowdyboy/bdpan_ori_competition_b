# 百度网盘AI大赛 -文档图像方向识别赛第3名方案
> 这是一个基于PaddlePaddle的文档图像方向识别的解决方案，本方案在B榜上取得了第3名的成绩，本文将介绍本方案的一些细节，以及如何使用本方案进行预测。

## 项目描述
图像方向识别是一个较为简单的任务，即将被旋转的图像恢复到原始方向。
本方案使用了基于CNN网络，通过对输入图像进行不同层的特征提取，然后，
通过特征来进行分类器的训练，最终实现图像方向识别的任务。


## 项目结构
```
-|bdpan_ori
-|checkpoint
-|dataset
-|dowdyboy_lib
-|ori_scripts
-train.py
-predict.py
-run_train.sh
```
- bdpan_ori: 本项目的模型源代码
- checkpoint: 本项目的模型参数
- dataset: 本项目的数据集
- dowdyboy_lib: 自行编写的基于飞桨的深度学习训练器，详见[这里](https://github.com/dowdyboy/dowdyboy_lib)
- ori_scripts: 本项目的历史版本
- train.py: 训练脚本
- predict.py: 预测脚本
- run_train.sh: 启动训练的脚本

## 数据

本项目训练数据和验证数据由以下组成：

- doc_deblur：文档图像，来源于百度网盘AI大赛图像去模糊赛道
- doc_dehw：文档图像，来源于百度网盘AI大赛图像文字擦除赛道
- icdar：场景图像，来源于ICDAR2015竞赛
- imagenet：自然图像，来源于ImageNet1K数据集和ImageNet100数据集
- ocr：文字标识图像，来源于OCR任务相关图像

本项目测试数据由百度网盘AI大赛提供，详见[官网](https://aistudio.baidu.com/aistudio/competition/detail/327/0/datasets) 。

训练验证数据 下载：

链接: https://pan.baidu.com/s/1oHq2_EVELeUxXewdgrUa2g?pwd=f5e8 提取码: f5e8 

## 训练
> 将数据集放在dataset文件夹下；
```
|dataset
-|ori_train
-|ori_val
```
> 运行run_train.sh脚本
```
run_train.sh --train-data-dir 
            ./dataset/ori_train/ 
            --val-data-dir
            ./dataset/ori_val/ 
            --use-scheduler 
            --use-warmup 
            --epoch 1500 
            --batch-size 64 
            --out-dir output_v6_final 
            --lr 4e-3
```
> 竞赛时，我们使用了4卡RTX3080Ti进行了训练；训练时间较长，我们在docs文件夹下存放了我们的训练记录；

## 预测
> 运行predict.py脚本
```
python predict.py 
     <要预测的图像文件夹路径> predict.txt
```

## 项目详情

### 数据处理

在输入时，我们先将图像缩放到520大小，然后在进行一次大小为512*512随机裁切。在数据增强方面，只是简单采用了一定概率的水平翻转和光度变换。 完成这些预处理后，将数据转换为0-1的Tensor，输入网络。

由“数据准备”的内容可知，数据源比较复杂，所以我们设计了代码实现依概率随机采样，使得输入数据的来源尽量均衡。另外，采用了通过配置来确定数据量大小的方式，防止单次epoch训练时间过长。

### 网络设计

我们的方案选择了shufflenetv2作为基础，其论文为：https://arxiv.org/abs/1807.11164  。 

一方面，作为一种轻量化网络，它很适合当前赛题；另一方面，paddle框架对它有多种尺寸的预训练模型。 我们都知道，不同尺寸的模型能够提取不同的特征，这些特征有可能是互补关系，可以相互辅助实现更好的性能。 

基于这样的思考，我们尝试放弃采用大个儿的单个模型，转而采用多个较小模型的合作。在最终的方案中，我们分别使用了一个shufflenet_v2_x0_5、shufflenet_v2_x0_33、shufflenet_v2_x0_25，组成了我们最终的网络。其中shufflenet_v2_x0_5使用relu作为激活函数，其余使用swish作为激活函数。

相对于采用单个模型，每一个分支模型都可以从不同的角度进行特征提取，可以一定程度上促进性能的提升。 然而，由于不同尺寸模型的中间特征图通道数不一，需要有一种方法进行统一转化。对此，我们设计了SqueezeExpandLayer，采用可分离卷积+分组输出的设计，在保证了性能的同时，尽可能的保留原始特征图中的多样化特征。

SqueezeExpandLayer，首先采用一个深度卷积在空间上进行特征转化，而后采用多组点卷积从不同测度提取通道方面的特征，最终的输出为所有点卷积输出的拼接。

通过SqueezeExpandLayer，就可以将不同层的通道数进行统一，这样，中间特征图的通道数和尺寸就都一样了，我们就在此基础上使用“加和”作为中间特征图的融合方式。 方向识别赛题是一个分类题目，因此一般需要深层次的特征作为分类的依据。目前，已经得到了融合的中间特征图，只需要设计一个下采样模块对不同层次的融合特征图进行再次融合，这样就能在深层特征图的基础上兼顾浅层特征图。

为此，我们设计了DownSampleConvLayer，它使用MaxPool2D作为下采样方式，同时，在下采样之前，使用了残差卷积做了中间的特征提取。 对整个网络分类头的设计上，我们参考了mobile_net，并引入了dropout模块。

### 训练方案

由于我们的方案中，每个分支子网络都是一个完整的分类网络，所以在最终的训练中，我们采用了融合训练的方式，即4个分类头同时训练，并且融合分类头占据更高的权重。 

我们配置训练数据个数为10000，验证数据个数为3000，在4卡3080Ti上进行1500 epoch个训练，取在验证集上acc最高的模型为最优模型。

