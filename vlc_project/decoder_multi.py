#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多帧解码器 - 支持帧序号验证和智能拼接
"""

import os
import sys
import re
from decoder_pillow import OptTransDecoderPillow, detect_file_extension

def get_frame_index(filename: str) -> int:
    """从文件名提取帧序号"""
    match = re.search(r'[._-]?frame(\d+)\.png$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def decode_multi_frames(frame_folder: str, output_file: str) -> bool:
    """解码多帧并拼接（利用控制区中的帧序号验证顺序）"""
    decoder = OptTransDecoderPillow()

    if not os.path.isdir(frame_folder):
        print(f"❌ 目录不存在：{frame_folder}")
        return False

    frame_files = [f for f in os.listdir(frame_folder) if f.lower().endswith('.png')]
    frame_files = [f for f in frame_files if re.search(r'frame\d+\.png$', f, re.IGNORECASE)]

    if not frame_files:
        print(f"❌ 未找到匹配的帧文件：{frame_folder}")
        return False

    # 按文件名中的数字排序（作为初步顺序）
    frame_files.sort(key=get_frame_index)

    print(f"📦 找到 {len(frame_files)} 帧")
    print(f"📁 目录：{frame_folder}")
    print(f"📄 输出：{output_file}")
    print("-" * 60)

    # 存储每帧解码结果，以帧序号为键
    frames_data = {}
    expected_total_frames = None
    max_frame_num = 0
    failed_frames = []

    for i, filename in enumerate(frame_files):
        filepath = os.path.join(frame_folder, filename)
        file_idx = get_frame_index(filename)

        print(f"[{i+1}/{len(frame_files)}] 解码：{filename} (文件序号：{file_idx}) ...", end=" ")

        try:
            result = decoder.decode_image_with_info(filepath)
            if result:
                data, frame_num, total_frames = result
                # 验证总帧数一致性
                if expected_total_frames is None:
                    expected_total_frames = total_frames
                elif total_frames != expected_total_frames:
                    print(f"⚠️ 总帧数不一致（期望{expected_total_frames}，实际{total_frames}），可能多源混合")

                if frame_num in frames_data:
                    print(f"⚠️ 帧序号 {frame_num} 重复，保留首次解码")
                else:
                    frames_data[frame_num] = data
                    max_frame_num = max(max_frame_num, frame_num)
                    print(f"✅ (帧{frame_num}/{total_frames}, {len(data)} B)")
            else:
                print(f"❌ 解码失败")
                failed_frames.append(filename)
        except Exception as e:
            print(f"❌ 异常：{e}")
            failed_frames.append(filename)
            import traceback
            traceback.print_exc()

    print("-" * 60)

    if not frames_data:
        print("❌ 没有成功解码任何数据")
        return False

    # 按帧序号顺序拼接
    full_data = bytearray()
    missing_frames = []
    if expected_total_frames is None:
        # 如果没有从控制区获取总帧数，则按检测到的最大帧序号拼接
        expected_total_frames = max_frame_num + 1

    for fn in range(expected_total_frames):
        if fn in frames_data:
            full_data.extend(frames_data[fn])
        else:
            missing_frames.append(fn)
            # 缺失帧补空？这里选择跳过，但提示用户
            print(f"⚠️ 缺失帧 {fn}，跳过")

    try:
        with open(output_file, 'wb') as f:
            f.write(full_data)

        print(f"\n✅ 解码完成！")
        print(f"📊 成功帧数：{len(frames_data)}/{len(frame_files)}")
        print(f"📊 失败帧数：{len(failed_frames)}")
        if missing_frames:
            print(f"📋 缺失帧序号：{missing_frames}")
        if failed_frames:
            print(f"📋 解码失败列表：{failed_frames}")
        print(f"📊 总数据量：{len(full_data)} 字节")
        print(f"📄 输出文件：{output_file}")

        return True
    except Exception as e:
        print(f"❌ 写入输出文件失败：{e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("用法：python decode_multi.py <frame_folder> <output_file>")
        print("示例：python decode_multi.py test_mterm restored.bin")
        sys.exit(1)

    frame_folder = sys.argv[1]
    output_file = sys.argv[2]

    success = decode_multi_frames(frame_folder, output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
