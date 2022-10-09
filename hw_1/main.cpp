#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include "../vectors_and_matrices/array_types.hpp"

#include <functional>
#include <chrono>

using ptrdiff_t = std::ptrdiff_t;

template <class T>
struct benchresult {
    T result;
    double btime;
};

template <class T, class input_type>
auto benchmark(std::function<T(input_type)> fn, input_type input, ptrdiff_t nrepeat) {
    T result;
    auto start = std::chrono::steady_clock::now();
    for (ptrdiff_t i = 0; i < nrepeat; i++) {
        result = fn(input);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_s = end - start;
    double ms_per_run = duration_s.count() * 1000 / nrepeat;
    return benchresult<T> {result, ms_per_run};
}

template <class T>
matrix<T> convolve(matrix<T> A, matrix<T> kernel) {
    // Also kernel center idx
    intptr_t kernel_radius = static_cast<int>((kernel.nrows() - 1) / 2);
    intptr_t A_size = A.nrows();

    matrix<T> result(A_size, A_size);

    for (size_t row_idx = 0; row_idx < A_size; ++row_idx) {
        for (size_t col_idx = 0; col_idx < A_size; ++col_idx) {
            T accumulator = 0;

            for (
                long relative_row_idx = -kernel_radius;
                relative_row_idx <= kernel_radius;
                ++relative_row_idx 
            ) {
                for (
                    long relative_col_idx = -kernel_radius;
                    relative_col_idx <= kernel_radius;
                    ++relative_col_idx 
                ) {
                    T A_row_idx_real = std::max<T>(
                        std::min<T>(
                            row_idx + relative_row_idx,
                            A_size - 1
                        ),
                        0
                    );

                    T A_col_idx_real = std::max<T>(
                        std::min<T>(
                            col_idx + relative_col_idx,
                            A_size - 1
                        ),
                        0
                    );

                    accumulator += A(A_row_idx_real, A_col_idx_real) 
                        * kernel(kernel_radius - relative_row_idx, kernel_radius - relative_col_idx);
                }
            }

            result(row_idx, col_idx) = accumulator;
        }
    }

    return result;
}

int main(int argc, char* argv[]) {
    ptrdiff_t n, k;

    std::cin >> n;
    matrix<double> a(n, n);
    for (ptrdiff_t l = 0; l < n * n; l++) {
        std::cin >> a(l);
    }
    
    std::cin >> k;
    matrix<double> kernel(k, k);
    for (ptrdiff_t l = 0; l < k * k; l++) {
        std::cin >> kernel(l);
    }

    auto res = convolve(a, kernel);
    res.dump(std::string("output/out.txt"));

    std::function<double(int)> convolution = [=](int idx) {return convolve(a, kernel)(idx);};

    auto benchresult = benchmark(convolution, 0, 1000);

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << benchresult.btime << " ms\n"
              << "Answer = " << benchresult.result
              << std::endl;
    return 0;
}
