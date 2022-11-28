#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include "vectors_and_matrices/array_types.hpp"

#include <omp.h>

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

template <class T>
void fill_random(vec<T> x, T xmin, T xmax, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(xmin, xmax);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}


template <class T>
T force(vec<T> m, vec<T> x, vec<T> y, vec<T> z)
{    
    vec<T> F(x.length());
    T Fmax = 0;
    #pragma omp parallel for schedule(static) reduction(max: Fmax)
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        for (ptrdiff_t j = i+1; j < x.length(); j++)
        {
            T dx = x(j) - x(i), dy = y(j) - y(i), dz = z(j) - z(i);
            T force_ij = m(i)*m(j) / (dx*dx + dy*dy + dz*dz);
            if (force_ij > Fmax)
            {
                Fmax = force_ij;
            }
        }
    }
    return Fmax;
}

int main(int argc, char* argv[])
{
    ptrdiff_t n;

    std::cin >> n;
    vec<double> m(n);
    vec<double> x(n);
    vec<double> y(n);
    vec<double> z(n);

    fill_random(m, 1.0, 5.0, 9876);
    fill_random(x, -10.0, 10.0, 9877);
    fill_random(y, -10.0, 10.0, 9878);
    fill_random(z, -10.0, 10.0, 9879);

    double t0 = omp_get_wtime();

    double max_force = force(m, x, y, z);

    double t1 = omp_get_wtime();

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "Answer = " << max_force
              << std::endl;
    return 0;
}
