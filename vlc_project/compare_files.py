#!/usr/bin/env python3
"""对比两个文件是否相同"""

def compare_files(file1, file2):
    with open(file1, 'rb') as f1:
        data1 = f1.read()
    
    with open(file2, 'rb') as f2:
        data2 = f2.read()
    
    if data1 == data2:
        print("✓ 文件完全相同！")
        print(f"文件大小: {len(data1)} 字节")
        return True
    else:
        print("✗ 文件不相同！")
        print(f"文件1大小: {len(data1)} 字节")
        print(f"文件2大小: {len(data2)} 字节")
        
        # 找到第一个不同的字节
        min_len = min(len(data1), len(data2))
        for i in range(min_len):
            if data1[i] != data2[i]:
                print(f"第一个不同的字节在位置 {i}: 0x{data1[i]:02x} vs 0x{data2[i]:02x}")
                break
        
        # 显示前20个字节的对比
        print("\n前20个字节对比:")
        print("原始: ", end="")
        for i in range(min(20, len(data1))):
            print(f"{data1[i]:02x} ", end="")
        print()
        print("解码: ", end="")
        for i in range(min(20, len(data2))):
            print(f"{data2[i]:02x} ", end="")
        print()
        
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法：py compare_files.py <file1> <file2>")
        sys.exit(1)
    compare_files(sys.argv[1], sys.argv[2])
