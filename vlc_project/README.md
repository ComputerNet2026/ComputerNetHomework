# 可见光通信（VLC）项目
## 项目说明
基于Python+OpenCV实现二进制文件→视频（发送端）、视频→二进制文件（接收端）的可见光通信，模块化设计，支持拓展升级。

## 快速开始
### 1. 环境准备
首先使用python3.12.0版本，因为numpy新版本的下架了支持不了
然后创建虚拟环境，必备！！
python -m venv venv
当然如果有同学有多个python，则输入py -3.12 -m venv venv
然后打开终端输入下面一行命令
```bash
pip install opencv-python==4.8.0.76 numpy==1.26.0 tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
安装FFmpeg（视频格式转换）


