#include <iostream>
#include "milestone.h"

using namespace std;

int main() {
    int N, NT;
    double L, T, u, v;
    N = 200;
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
    mpcs50187::Advection(N, NT, L, T, u, v);

    return 0;
}
