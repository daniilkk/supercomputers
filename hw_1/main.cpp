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
auto benchmark(std::function<T(input_type)> fn, input_type input, ptrdiff_t nrepeat){
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
matrix<T> convolve(matrix<T> A, matrix<T> kernel)
{    
    return A;
}

int main(int argc, char* argv[])
{
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

    std::function<double(int)> convolution = [=](int idx) {return convolve(a, kernel)(idx);};

    auto benchresult = benchmark(convolution, 5, 1000);

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << benchresult.btime << " ms\n"
              << "Answer = " << benchresult.result
              << std::endl;
    return 0;
}
