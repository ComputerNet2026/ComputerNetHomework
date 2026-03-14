#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from encoder_pillow import OptTransEncoderPillow
from decoder_pillow import OptTransDecoderPillow

def main():
    parser = argparse.ArgumentParser(description='OptTrans: Binary file to image encoder/decoder')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    encode_parser = subparsers.add_parser('encode', help='Encode a binary file to an image')
    encode_parser.add_argument('input_file', help='Input binary file')
    encode_parser.add_argument('output_image', help='Output image file')
    encode_parser.add_argument('--module-size', type=int, default=8, help='Size of each module in pixels (default: 8)')
    encode_parser.add_argument('--ecc', type=int, default=10, help='Number of RS error correction symbols (default: 10)')
    encode_parser.add_argument('--matrix-size', type=int, default=145, help='Matrix size (default: 145, larger = more data)')
    
    decode_parser = subparsers.add_parser('decode', help='Decode an image to a binary file')
    decode_parser.add_argument('input_image', help='Input image file')
    decode_parser.add_argument('output_file', help='Output binary file')
    decode_parser.add_argument('--module-size', type=int, default=8, help='Size of each module in pixels (default: 8)')
    decode_parser.add_argument('--ecc', type=int, default=10, help='Number of RS error correction symbols (default: 10)')
    decode_parser.add_argument('--matrix-size', type=int, default=145, help='Matrix size (must match encode)')
    
    args = parser.parse_args()
    
    if args.command == 'encode':
        encoder = OptTransEncoderPillow()
        encoder.encode_file(args.input_file, args.output_image)
        print(f"Encoded {args.input_file} to {args.output_image}")
        print(f"  Matrix size: 166x166")
        print(f"  Module size: 6px")
        print(f"  Output size: 996x996px")
    elif args.command == 'decode':
        decoder = OptTransDecoderPillow(
            module_size=args.module_size, 
            rs_ecc_symbols=args.ecc,
            matrix_size=args.matrix_size
        )
        success = decoder.decode_to_file(args.input_image, args.output_file)
        if success:
            print(f"Decoded {args.input_image} to {args.output_file}")
        else:
            print(f"Failed to decode {args.input_image}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
