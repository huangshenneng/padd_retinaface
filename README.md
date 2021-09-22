# padd_retinaface


Paddle复现 RetinaFace: Single-stage Dense Face Localisation in the Wild  https://arxiv.org/pdf/1905.00641v2.pdf )  论文

# 准备工作
```
1. 下载训练集和测试集，放到  ' data/widerface'文件夹下
2. 到  'widerface_evaluate' 文件中，运行命令行 '  python setup.py build_ext --inplace'
3. 下载验证集的结果，解压到 ' data/widerface'文件夹下，用于测试mAP
4.下载模型
```

模型链接（训练13epoch）
[baidu cloud](https://pan.baidu.com/s/1s6po5aQzQMRLzpmBlq-rfA) Password: wihg

验证集预测结果
[baidu cloud](https://pan.baidu.com/s/1p5ffgic8-L_BXd8LawAYSQ)  Password: rm8s  

# 模型的训练

直接运行train.py 文件

# 模型的测试
```
1.运行 test_widerface.py 
2. cd ./widerface_evaluate
    python evaluation.py
```

## 模型训练结果 （因为显卡资源紧张，目前只训练了13个epoch，效果不是很理想，后期训练好了再继续提交）
```
Easy   Val AP: 0.48140963223995936 
Medium Val AP: 0.5141806493221557  
Hard   Val AP: 0.3101786747705212
```

参考
https://github.com/biubug6/Pytorch_Retinaface
