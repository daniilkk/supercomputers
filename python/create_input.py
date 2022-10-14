from argparse import Namespace, ArgumentParser
import os

import numpy as np


def save_to_file(matrix: np.ndarray, kernel: np.ndarray, kernel_type: str):
    n = matrix.shape[0]
    k = kernel.shape[0]

    if not os.path.isdir('input'):
        os.makedirs('input', exist_ok=True) 

    with open(f'input/{n}_{k}_{kernel_type}.txt', 'w') as f:
        f.write(f'{n}\n')

        for idx in range(n):
            f.write(f'{" ".join(map(str, matrix[idx]))}\n')

        f.write(f'{k}\n')

        for idx in range(k):
            f.write(f'{" ".join(map(str, kernel[idx]))}\n')


def main(args: Namespace):
    matrix = (np.random.rand(args.matrix_size, args.matrix_size) - 0.5) * 10

    if args.kernel_type == 'id':
        kernel = np.zeros((args.kernel_size, args.kernel_size), dtype=np.int32)
        kernel_central_idx = (args.kernel_size - 1) // 2
        kernel[kernel_central_idx, kernel_central_idx] = 1

    elif args.kernel_type == 'random':
        kernel = (np.random.rand(args.kernel_size, args.kernel_size) - 0.5) * 10

    else:
        raise RuntimeError(f'Wrong kernel type: {args.kernel_type}')

    save_to_file(matrix, kernel, args.kernel_type)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--matrix_size', type=int, default=10)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--kernel_type', type=str, default='id')
    args = parser.parse_args()

    main(args)
