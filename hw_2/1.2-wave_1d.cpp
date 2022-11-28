#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include <cmath>
#include "../vectors_and_matrices/array_types.hpp"

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

template <class T>
void fill_random_cos(vec<T> x, T ampl_max, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(-ampl_max, ampl_max);

    ptrdiff_t n = x.length();
    for (int imode=1; imode<=10; imode++)
    {
        T ampl = dist(rng);
        for (ptrdiff_t i = 0; i < n; i++)
        {
            x(i) += ampl * cos(imode * (PI * i) / n);
        }
    }
}

template <class T>
void wave_simulation(vec<T> u, vec<T> dudt, T c_s, T dt, T dx, ptrdiff_t nsteps)
{
    ptrdiff_t istep, i, n = u.length();
    T prefactor = c_s * c_s * dt / (dx * dx);

    for (istep = 0; istep < nsteps; istep++)
    {
        #pragma omp parallel for
        for (i=1; i < n-1; i++)
        {
            dudt(i) += T(0.5) * prefactor * (u(i-1) - 2 * u(i) + u(i+1));
        }
        dudt(0) += T(0.5) * prefactor * (u(n-1) - 2 * u(0) + u(1));
        dudt(n-1) += T(0.5) * prefactor * (u(n-2) - 2 * u(n-1) + u(0));

        #pragma omp parallel for
        for (i=0; i < n; i++)
        {
            u(i) += dt * dudt(i);
        }
        
        #pragma omp parallel for
        for (i=1; i < n-1; i++)
        {
            dudt(i) += T(0.5) * prefactor * (u(i-1) - 2 * u(i) + u(i+1));
        }
        dudt(0) += T(0.5) * prefactor * (u(n-1) - 2 * u(0) + u(1));
        dudt(n-1) += T(0.5) * prefactor * (u(n-2) - 2 * u(n-1) + u(0));
    }
}

template <class T>
T wave_energy(vec<T> u, vec<T> dudt, T c_s, T dx)
{
    T ener = 0;
    T cdu;
    for (ptrdiff_t i = 0; i < dudt.length(); ++i)
    {
        ener += dudt(i) * dudt(i) / 2;
    }

    for (ptrdiff_t i = 1; i < u.length(); ++i)
    {
        cdu = c_s * (u(i) - u(i-1)) / dx;
        ener += cdu * cdu / 2;
    }

    cdu = c_s * (u(0) - u(u.length() - 1)) / dx;
    ener += cdu * cdu / 2;
    return ener;
}

int main(int argc, char* argv[])
{
    ptrdiff_t n;

    std::cin >> n;
    vec<double> u(n);
    vec<double> dudt(n);

    fill_random_sin(u, 5.0, 9876);
    fill_random_cos(dudt, 1.0, 9877);

    double dx = 1.0;
    double c_s = 1.0;
    double dt = 0.5 * dx / c_s;
    ptrdiff_t nsteps = 100000;

    std::cout << "Initial energy: " << wave_energy(u, dudt, c_s, dx) << '\n';

    double t0 = omp_get_wtime();

    wave_simulation(u, dudt, c_s, dt, dx, nsteps);

    double t1 = omp_get_wtime();

    std::cout << "Final energy: " << wave_energy(u, dudt, c_s, dx) << '\n';

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "u[n-1] = " << u(n-1) << "; dudt[n-1] = " << dudt(n-1)
              << std::endl;
    return 0;
}
