# 可见光通信（VLC）项目（新手练习）
## 项目说明
基于Python+OpenCV实现二进制文件→视频（发送端）、视频→二进制文件（接收端）的可见光通信，模块化设计，支持拓展升级。

## 快速开始

先谈谈我个人对项目的理解，其实就是遵循同一套逻辑的加密解密，比如把一段信息变成类似摩斯密码，再把摩斯密码变回信息。那么在这个项目中，是通过输入一段文字，把它转化为交替闪烁的黑白灯光，我们再通过拍摄这段黑白灯光让他解析，看是否会返回和我们输入的信息一致，后续再基于此，通过改变颜色等等来增加信息量，因为我也是第一次接触这类代码，能力有限，有错误理解希望同学指正。

### 1. 环境准备

首先使用python3.12.0版本，因为numpy新版本的下架了支持不了


下载ffmeg（把mp4转换为avi）

https://www.gyan.dev/ffmpeg/builds/


选择full那个
<img width="1461" height="1137" alt="image" src="https://github.com/user-attachments/assets/68474358-2a34-46d4-bd87-b6de138496c4" />

解压缩后加到环境变量

### 2. 操作

正式打开vlc_project这个文件夹

打开终端

然后创建虚拟环境，必备！！


python -m venv venv


当然如果有同学有多个python，则输入


py -3.12 -m venv venv


然后打开终端输入下面一行命令


pip install opencv-python==4.8.0.76 numpy==1.26.0 tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple


之后运行run_sender.py会自动生成data子文件夹，包含input_bin/、encoded_video/等目录；


打开data/input_bin/，新建一个文本文档，右键重命名为small_test.bin（注意：要把后缀.txt 改成.bin，需先在电脑显示文件后缀：文件夹→查看→勾选「文件扩展名」）；


双击打开small_test.bin，写入任意简单内容（如test123），保存关闭。


再次run_sender.py


会出现类似的输出


<img width="974" height="270" alt="image" src="https://github.com/user-attachments/assets/25f5cc98-1148-429f-93fb-581fee223241" />


打开data/encoded_video/，能看到生成的small_test_encoded.avi视频


直接运行run_receiver

成功标志
<img width="1088" height="448" alt="image" src="https://github.com/user-attachments/assets/ec55a1d6-1464-48b1-82b7-dc0d225d487d" />


##手机测试
电脑端


打开data/encoded_video/small_test_encoded.avi，用播放器全屏播放（推荐 PotPlayer / 暴风影音，按 F11 全屏，关闭所有边框 / 缩放 / 弹幕）；


手机端：


固定手机（用支架 / 书夹），距离电脑屏幕30-50cm，镜头正对屏幕中央，确保屏幕占满手机拍摄画面（无桌面 / 墙壁露出）；
手机拍摄设置：帧率 30fps、分辨率 1080P，关闭防抖 / 美颜 / 夜景模式（避免视频压缩 / 画面模糊）；

手机开始拍摄 → 电脑重新播放视频（确保拍到完整的 3 秒白→所有黑白闪烁→3 秒黑）→ 视频播放结束后，手机停止拍摄；
拍摄的视频格式一般为MP4（手机默认格式），记住视频名（如VID_20260307_123456.mp4）

打开run_receiver.py

修改对应参数


<img width="976" height="223" alt="image" src="https://github.com/user-attachments/assets/62e33ea9-7172-4254-9f9a-1553dfcba3f2" />




运行run_receiver.py


最后查看结果是否一致即可







