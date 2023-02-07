#include <iostream>
#include <chrono>
#include "milestone2.h"
using namespace std;

int main() {
    int N, NT;
    double L, T, u, v;
    N = 400;
    NT = 10000;
    L = 1.0;
    T = 1.0e6;
    u = 5.0e-7;
    v = 2.85e-7;
    cout << "N = " << N << " (Matrix Dimension)" << endl;
    cout << "NT = " << NT << " (Number of timesteps)" << endl;
    cout << "L = " << L << " (Physical Cartesian Domain Length)" << endl;
    cout << "T = " << T << " (Total Physical Timespan)" << endl;
    cout << "u = " << u << " (X velocity Scalar)" << endl;
    cout << "v = " << v << " (Y velocity Scalar)" << endl;
//    cout << T;
    double start = omp_get_wtime(); 
    // std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    mpcs50187::Advection(N, NT, L, T, u, v);
    double end = omp_get_wtime(); 
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    cout<<"Work took "<< end - start <<" seconds\n";
    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;

    return 0;
}
