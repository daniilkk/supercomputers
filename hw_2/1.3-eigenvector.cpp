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
void fill_random(matrix<T> x, T xmin, T xmax, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(xmin, xmax);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

template <class T>
void symmetrize(matrix<T> a)
{
    ptrdiff_t n = a.nrows();
    for (ptrdiff_t i = 0; i < n; i++)
    {
        for (ptrdiff_t j = 0; j < i; j++)
        {
            T sym_elt = (a(i, j) + a(j, i)) / 2;
            a(i, j) = sym_elt;
            a(j, i) = sym_elt;
        }
    }
}

template <class T>
T eigenvalue(matrix<T> A, ptrdiff_t nrepeat)
{
    ptrdiff_t n = A.nrows();
    vec<T> v0(n);
    vec<T> v1(n);

    fill_random(v0, T(-10), T(10), 24680);

    ptrdiff_t iter;
    ptrdiff_t i, j;

    for (iter = 0; iter < n; iter++)
    {
        // normalize v0
        T norm2 = 0;
        for (i=0; i < n; i++)
        {
            norm2 += v0(i) * v0(i);
        }

        for (i=0; i < n; i++)
        {
            v0(i) /= sqrt(norm2);
        }

        // v1 = A * v0
        for (i=0; i<n; i++)
        {
            v1(i) = 0;
            for (j=0; j<n; j++)
            {
                v1(i) += v0(j) * A(i, j);
            }
        }

        // swap v1 and v0
        vec<T> tmp = v0;
        v0 = v1;
        v1 = tmp;
    }

    // compute average of v0[i] / v1[i]
    T eigv = 0;
    for (i=0; i < n; i++)
    {
        eigv += v0(i) / v1(i);
    }
    eigv /= n;
    return eigv;
}

int main(int argc, char* argv[])
{
    ptrdiff_t n;

    std::cin >> n;
    matrix<double> A(n, n);

    fill_random(A, 0.0, 1.0, 9876);
    symmetrize(A);

    double t0 = omp_get_wtime();

    double q = eigenvalue(A, n);

    double t1 = omp_get_wtime();

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "Answer = " << q
              << std::endl;
    return 0;
}
