#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include <cmath>
#include "vectors_and_matrices/array_types.hpp"

#include <omp.h>

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;
const double PI = 3.141592653589793;

template <class T>
void fill_random_sin(vec<T> x, T ampl_max, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(-ampl_max, ampl_max);

    ptrdiff_t n = x.length();
    for (int imode=1; imode<=10; imode++)
    {
        T ampl = dist(rng);
        for (ptrdiff_t i = 0; i < n; i++)
        {
            x(i) += ampl * sin(imode * (PI * i) / n);
        }
    }
}

void fpu_simulation(vec<double> x, vec<double> vx, double alpha, double dt, ptrdiff_t nsteps)
{
    ptrdiff_t istep, i, n = x.length();
    double force;

    for (istep = 0; istep < nsteps; istep++)
    {
        #pragma omp parallel
        {
            #pragma omp for
            for (i=1; i < n-1; i++)
            {
                force = (x(i-1) - 2 * x(i) + x(i+1)) * (1 + alpha * (x(i+1) - x(i-1)));
                vx(i) += 0.5 * dt * force;
            }
            force = (x(n-1) - 2 * x(0) + x(1)) * (1 + alpha * (x(1) - x(n-1)));
            vx(0) += 0.5 * dt * force;
            force = (x(n-2) - 2 * x(n-1) + x(0)) * (1 + alpha * (x(0) - x(n-2)));
            vx(n-1) += 0.5 * dt * force;
            #pragma omp for
            for (i=0; i < n; i++)
            {
                x(i) += dt * vx(i);
            }
            #pragma omp for
            for (i=1; i < n-1; i++)
            {
                force = (x(i-1) - 2 * x(i) + x(i+1)) * (1 + alpha * (x(i+1) - x(i-1)));
                vx(i) += 0.5 * dt * force;
            }
            force = (x(n-1) - 2 * x(0) + x(1)) * (1 + alpha * (x(1) - x(n-1)));
            vx(0) += 0.5 * dt * force;
            force = (x(n-2) - 2 * x(n-1) + x(0)) * (1 + alpha * (x(0) - x(n-2)));
            vx(n-1) += 0.5 * dt * force;
        }
    }
}

double fpu_chain_energy(vec<double> x, vec<double> vx, double alpha)
{
    double ener = 0;
    double dx;
    for (ptrdiff_t i = 0; i < vx.length(); ++i)
    {
        ener += vx(i) * vx(i) / 2;
    }

    for (ptrdiff_t i = 1; i < x.length(); ++i)
    {
        dx = x(i) - x(i-1);
        ener += dx * dx * (0.5 + alpha * dx / 3);
    }

    dx = x(0) - x(x.length() - 1);
    ener += dx * dx * (0.5 + alpha * dx / 3);
    return ener;
}

int main(int argc, char* argv[])
{
    ptrdiff_t n;

    std::cin >> n;
    vec<double> x(n);
    vec<double> vx(n);

    fill_random_sin(x, 5.0, 9876);
    fill_random_sin(vx, 1.0, 9877);

    double alpha = 0.05;
    double dt = 0.1;

    std::cout << "Initial energy: " << fpu_chain_energy(x, vx, alpha) << '\n';

    double t0 = omp_get_wtime();

    fpu_simulation(x, vx, alpha, dt, 100000);

    double t1 = omp_get_wtime();

    std::cout << "Final energy: " << fpu_chain_energy(x, vx, alpha) << '\n';

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "x[n-1] = " << x(n-1) << "; vx[n-1] = " << vx(n-1)
              << std::endl;
    return 0;
}
