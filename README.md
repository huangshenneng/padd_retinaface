# padd_retinaface


Paddle复现 RetinaFace: Single-stage Dense Face Localisation in the Wild  https://arxiv.org/pdf/1905.00641v2.pdf )  论文

# 准备工作
```
1. 下载训练集和测试集，放到  ' data/widerface'文件夹下
2. 到  'widerface_evaluate' 文件中，运行命令行 '  python setup.py build_ext --inplace'
3. 下载验证集的结果，解压到 ' data/widerface'文件夹下，用于测试mAP
4.下载模型
```
### 模型文件
[baidu cloud](https://pan.baidu.com/s/1R2_8zGfjQ63e0BmC_aWwhw ) Password: 0ewb

### 训练log
[baidu cloud](https://pan.baidu.com/s/1cNa2d9HNvOXdhlbYPZZtnw  ) Password: ypry

# 模型的训练

直接运行train.py 文件

# 模型的测试
```
1.运行 test_widerface.py  
2. cd ./widerface_evaluate
    python evaluation.py
```

## 模型训练结果 
```
Easy   Val AP: 0.9402364106525578
Medium Val AP: 0.9208495951285756
Hard   Val AP: 0.8005163465688859
```

参考
https://github.com/biubug6/Pytorch_Retinaface
