#!/usr/bin/env python3
"""
OptTrans 编码解码工具

用法：
  py opttrans.py encode <input_file> <output_image>
  py opttrans.py decode <input_image> <output_file>
"""

import sys
import os
from src.encoder import OptTransEncoder
from src.decoder import OptTransDecoder

def main():
    if len(sys.argv) != 4:
        print("用法错误：")
        print("  编码：py opttrans.py encode <input_file> <output_image>")
        print("  解码：py opttrans.py decode <input_image> <output_file>")
        return
    
    command = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    if command == "encode":
        # 编码模式
        if not os.path.exists(input_path):
            print(f"错误：输入文件 '{input_path}' 不存在")
            return
        
        encoder = OptTransEncoder()
        try:
            result = encoder.encode_file(input_path, output_path)
            if isinstance(result, int):
                print(f"编码成功！生成了 {result} 帧图像")
            else:
                print(f"编码成功！生成图像：{output_path}")
        except Exception as e:
            print(f"编码失败：{str(e)}")
            return
    
    elif command == "decode":
        # 解码模式
        if not os.path.exists(input_path):
            print(f"错误：输入图像 '{input_path}' 不存在")
            return
        
        decoder = OptTransDecoder()
        try:
            bytes_decoded = decoder.decode_file(input_path, output_path)
            print(f"解码成功！写入了 {bytes_decoded} 字节到 {output_path}")
        except Exception as e:
            print(f"解码失败：{str(e)}")
            return
    
    else:
        print(f"未知命令：{command}")
        print("支持的命令：encode, decode")
        return

if __name__ == "__main__":
    main()
