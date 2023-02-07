#include<iostream>
#include <fstream>
#include<cmath>
#include <random>
#include <algorithm>

using namespace std;

void ray_tracing(int n, int N_rays);
void direction_sampling(double *V);


double dotp( double * a, double * b);
double square(double a);

void sub( double *a, double *b, double *c);
void mult(double *V, double t, double *W);
double mod(double *V);
void show(double *V);
double **alloc_2d_double(int rows, int cols);
void save_to_file(double **C, const string &name, int N);

int main(int argc, char * argv[]) {
    int n, N_rays;
    n  = 1000;
    N_rays = 10000000;
    ray_tracing(n, N_rays);
    cout << "end"<< endl;

}

void ray_tracing(int n, int N_rays) {
    // allocateG[1...n][1...n]
    double ** G = alloc_2d_double(n, n);
    double V[3];
    double L[3] = {4, 4, -1};
    double W[3];
    W[1] = 10;   // Wy = 10
    double Wmax = 10;

    double C[3] = {0, 12, 0};
    double b, R = 6;
    
    double I[3], N[3], S[3], I_C[3], L_I[3];
    //G[i][j] = 0 for all (i, j)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            G[i][j] = 0;
        }
    }

    for(int iter = 1; iter <= N_rays; iter++) {
        cout << iter << endl;
        do {
            // Sample random V from unit sphere
            direction_sampling(V);
            // The intersection of the view ray and the window
            W[0] = W[1] / V[1] * V[0];
            W[2] = W[1] / V[1] * V[2];
            
            double temp = (square(dotp(V, C)) + square(R) - dotp(C, C));
        } while(!( (abs(W[0]) < Wmax) and (abs(W[2]) < Wmax) and ((square(dotp(V, C)) + square(R) - dotp(C, C)) > 0)) );
        double t  = dotp(V, C) - sqrt(square(dotp(V, C)) + square(R) - dotp(C, C));
        mult(V, t, I); //The intersection of the view ray and the sphere
        sub(I, C, I_C);
        mult(I_C, 1 / mod(I_C), N);

        sub(L, I, L_I);
        mult(L_I, 1/mod(L_I), S);
        b = max(0.0, dotp(S, N));
        int i = n - 1 - (W[0] / (2 * Wmax) + 0.5) * n;
        int j = (W[2] / ( 2 * Wmax) + 0.5) * n;
        G[i][j] += b;
    }
    save_to_file(G, "G.csv", n);
    free(G);
}

void direction_sampling(double *V) {
    random_device generator;
    // Sample φ from uniform distribution (0, 2π)
    uniform_real_distribution<double> distribution1(0.0, 2*M_PI);
    double phi = distribution1(generator);
    
    // Sample cos(θ) from uniform distribution (−1, 1)
    uniform_real_distribution<double> distribution2(-1.0, 1.0);
    double cos_theta = distribution2(generator);
    double sin_theta = sqrt(1 - cos_theta * cos_theta);

    V[0] = sin_theta * cos(phi);
    V[1] = sin_theta * sin(phi);
    V[2] = cos_theta;
}

double dotp( double * a, double * b)
{
    double c = 0.0;
    for( long i = 0; i < 3; i++ )
        c += a[i]*b[i];
    return c;
}

double square(double a) {
    return a * a;
}

void mult(double *V, double t, double *W) {
    for(int i = 0; i < 3; i++) {
        W[i] = V[i] * t;
    }
}

void sub( double *a, double *b, double *c) {
    for(int i = 0; i < 3; i++) {
        c[i] = a[i] - b[i];
    }
}
double mod(double *V) {
    double result = 0;
    for(int i = 0; i < 3; i++) {
        result += (V[i] * V[i]);
    }
    return result;
}

void show(double *V) {
    cout << "(" << V[0] << "," << V[1] << "," << V[2] << ")" ;
}

double **alloc_2d_double(int rows, int cols) {
    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

void save_to_file(double **C, const string &name, int N) {
    ofstream outFile;

    outFile.open(
            "/home/wxh/Project3/milestone1/" +
            name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            outFile << C[i][j] << ", ";

        }
        outFile << "\n";
    }

    outFile << "\n";
    outFile.close();
    std::cout << "generate " + name << std::endl;
}
