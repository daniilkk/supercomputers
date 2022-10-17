import fileinput
import time
from typing import Callable, Tuple

import numpy as np


def read_input() -> Tuple[np.ndarray, np.ndarray]:
    input_content = fileinput.input()

    n = int(input_content.readline().rstrip())

    A_rows = []
    for _ in range(n):
        line = input_content.readline().rstrip()
        row = np.array(list(map(np.float64, line.split(' '))))
        A_rows.append(row)

    A = np.stack(A_rows)

    k = int(input_content.readline().rstrip())

    kernel_rows = []
    for _ in range(k):
        line = input_content.readline().rstrip()
        row = np.array(list(map(np.float64, line.split(' '))))
        kernel_rows.append(row)

    kernel = np.stack(kernel_rows)

    return A, kernel


def convolve(A: np.ndarray, kernel: np.ndarray) -> np.float64:
    # Also kernel center idx
    kernel_radius = (kernel.shape[0] - 1) // 2
    A_size = A.shape[0]

    result = np.empty((A_size, A_size), dtype=np.float64)

    for row_idx in range(A_size):
        for col_idx in range(A_size):
            accumulator = 0

            for relative_row_idx in range(-kernel_radius, kernel_radius + 1):
                for relative_col_idx in range(-kernel_radius, kernel_radius + 1):
                    A_row_idx_real = max(
                        min(
                            row_idx + relative_row_idx,
                            A_size - 1
                        ),
                        0
                    )

                    A_col_idx_real = max(
                        min(
                            col_idx + relative_col_idx,
                            A_size - 1
                        ),
                        0
                    )

                    accumulator += A[A_row_idx_real, A_col_idx_real] \
                        * kernel[
                            kernel_radius - relative_row_idx,
                            kernel_radius - relative_col_idx
                        ]

            result[row_idx, col_idx] = accumulator

    return result


def benchmark(fn: Callable[[int], np.float64], fn_input: int, n_repeat: int):
    start = time.perf_counter()
    for _ in range(n_repeat):
        result = fn(fn_input)
    end = time.perf_counter()
    ms_per_run = (end - start) * 1000 / n_repeat

    return result, ms_per_run


def main():
    A, kernel = read_input()

    convolution = lambda idx: convolve(A, kernel).flat[idx]
    result, btime = benchmark(convolution, 0, 50)

    print(f'N = {A.shape[0]}')
    print(f'Timing: {btime:.2f} ms')
    print(f'Answer = {result}')


if __name__ == '__main__':
    main()
