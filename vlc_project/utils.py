# 通用工具类：封装复用功能，拓展时新增函数即可
import logging
import os
import subprocess
from pathlib import Path
import config

# 初始化日志
def init_logger():
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(config.DATA_DIR / "vlc_project.log", encoding="utf-8")  # 文件输出
        ]
    )
    return logging.getLogger(__name__)

logger = init_logger()

# 校验文件是否存在
def check_file_exists(file_path: Path) -> bool:
    if not file_path.exists():
        logger.error(f"文件不存在：{file_path}")
        return False
    return True

# 转换视频格式（拓展：兼容手机拍摄的MP4）
# 转换视频格式（拓展：兼容手机拍摄的MP4）
def convert_video_format(input_path: Path, output_path: Path, fps: int = config.VIDEO_FPS):
    """使用FFmpeg转换视频格式（需提前配置FFmpeg环境变量）"""
    if not check_file_exists(input_path):
        return False
    # 修正核心：将 -fps 改为 -r（FFmpeg 标准帧率参数）
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-c:v", "libxvid",  # 保持编码器不变（AVI兼容）
        "-r", str(fps),     # ✅ 替换原 -fps，这是FFmpeg识别的帧率参数
        "-q:v", "1",        # 新增：最高画质，避免转换后视频模糊
        "-y",               # 覆盖已有文件
        str(output_path)
    ]
    try:
        # 执行命令并捕获输出（保留原有逻辑，仅改参数）
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding="utf-8"  # 新增：指定编码，避免中文路径乱码
        )
        logger.info(f"视频格式转换完成：{input_path} → {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"视频转换失败：{e.stderr}")
        return False
    except Exception as e:
        # 新增：捕获其他异常（如FFmpeg未配置、路径不存在）
        logger.error(f"视频转换异常：{str(e)}")
        return False

# 对比两个二进制文件内容（测试用）
def compare_bin_files(file1: Path, file2: Path) -> bool:
    if not (check_file_exists(file1) and check_file_exists(file2)):
        return False
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        data1 = f1.read()
        data2 = f2.read()
    if data1 == data2:
        logger.info(f"文件内容一致：{file1} vs {file2}")
        return True
    else:
        logger.warning(f"文件内容不一致：{file1} vs {file2}")
        return False
