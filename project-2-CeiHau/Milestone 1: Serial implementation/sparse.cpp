#include<iostream>
#include<cmath>

using namespace std;

double *CG_Algo_Dense(int n, double **A, double *b);

double *CG_Algo_Sparse(int n, double *b);

double *matvec(double **M, double *v, int n);

double *matvec_sparse(double * w, int N);

double dopt(double *b, double *c, int n);

double* axpy(double a, double *w, double b, double *v, int n);

void Generate_A(double ** A, int N);

void fill_b(double * b, int N);

double find_b(int i, int j, int n);

double ** matrix( long N );

void matrix_free( double ** M);

void show_matrix(double ** A, int N);

void show_vector(double * A, int N);

double get_time(void);

void save_vector(double * x, long N, char * fname );

int main() {
    int n = 75;
    int N = n * n;
    double *x;
    double *b = (double *) malloc(N * (sizeof(double)));

    cout << "Dense Memory = " << (5*N)*sizeof(double)/1024/1024.0 << "mb" << endl;


    fill_b(b, N);
    // for(int i = 0; i < N; i++) {
    //     b[i] = 0;
    // }

    double begin = get_time();
    x = CG_Algo_Sparse(n, b);
    double end = get_time();
    cout<<"Dense Runtime = " << end - begin << " seconds" << endl;
    // save_vector(x, N, "sparse.csv");
    free(x);
    free(b);
}



double *CG_Algo_Sparse(int n,  double *b) {
    int N = n * n;
    double *x = (double*) malloc(N * sizeof(double));
    for(int i = 0; i < N; i++) x[i] = 0;    //x = 0

    // Allocate r, r=b−Ax
    double *r = axpy(1.0, b, -1.0, matvec_sparse(x, N), N);
    // show_vector(r, N);
    // Allocate p, p = r
    double *p = (double *) malloc(N * sizeof(double));
    for(int i = 0; i < N; i++) p[i] = r[i];

    // Allocate z
    double *z;

    double rsold = dopt(r, r, N);
    // cout << rsold << endl;
    int i = 0;
    for(; i < N; i++) {
        z = matvec_sparse(p, N);
        // show_vector(z, N);
        double a = rsold / dopt(p, z, N);
        // cout << "alpha "<< a << endl;
        x = axpy(1, x, a, p, N);
        r = axpy(1, r, -a, z, N);
        double rsnew;
        rsnew = dopt(r, r, N);
        // cout << rsnew << endl;
        if (sqrt(rsnew) < 1.0e-10)
            break;
        p = axpy(1.0, r, rsnew/rsold, p, N);
        rsold = rsnew;
    }
    free(r);
    free(p);
    free(z);
    cout << i << " iterations\n";
    return x;
    
}



double* CG_Algo_Dense(int n, double **A, double *b) {
    int N = n * n;
    // Allocagte x
    double *x = (double *) malloc(N * sizeof(double));
    for(int i = 0; i < N; i++) x[i] = 0;    //x = 0

    // Allocate r, r=b−Ax
    // double * temp = matvec(A, x, N);

    double *r = axpy(1.0, b, -1.0, matvec(A, x, N), N);

    // Allocate p, p = r
    double *p = (double *) malloc(N * sizeof(double));
    for(int i = 0; i < N; i++) p[i] = r[i];

    // Allocate z
    double *z;

    double rsold = dopt(r, r, N);
    // cout << rsold << endl;
    int i = 0;
    for(; i < N; i++) {
        z = matvec(A, p, N);
        double a = rsold / dopt(p, z, N);
        // cout << "alpha "<< a << endl;
        x = axpy(1, x, a, p, N);
        r = axpy(1, r, -a, z, N);
        double rsnew;
        rsnew = dopt(r, r, N);
        // cout << rsnew << endl;
        if (sqrt(rsnew) < 1.0e-10)
            break;
        p = axpy(1.0, r, rsnew/rsold, p, N);
        rsold = rsnew;
    }
    free(r);
    free(p);
    free(z);
    cout << i << " iterations\n";
    return x;
}

double *matvec(double **M, double *w, int n) {
    double *v = (double *) malloc(n * sizeof(double));
    for(int i = 0; i < n; i++) {
        double temp = 0;
        for(int j = 0; j < n; j++) {
            temp += (M[i][j] * w[j]);
        }
        v[i] = temp;
    }
    return v;
}

double *matvec_sparse(double * w, int N) {
    double *v = (double *) malloc(N * sizeof(double));
    int n = sqrt(N);
    for(int i = 0; i < N; i++) {
        double temp = 0.0;
        if((i - n) >= 0) {
            temp -= w[i - n];
        }  
        if((i-1) >= 0 and ((i - 1) % n != n - 1)) {
            temp -= w[i - 1];
        }
        temp += 4.0 *w[i];
        if((i + 1) <= (N - 1) and ((i + 1) % n != 0)) {
            temp -= w[i + 1];
        }
        if((i+n) <= N -1){
            temp -= w[i + n];
        }
        v[i] = temp;
    }
    return v;
}

double dopt(double *b, double *c, int n) {
    double a = 0.0;
    for(int i = 0; i < n; i++) {
        a += b[i] * c[i];
    }
    return a;
}

double* axpy(double a, double *w, double b, double *v, int n) {
    double *result = (double *) malloc(n * sizeof(double));  

    for(int i = 0; i < n; i++) {
        result[i] = a * w[i] + b * v[i];
    }
    return result;
}

// reference: code from utils.c in google drive
double find_b(int i, int j, int n)
{
    double delta = 1.0 / (double) n;

    double x = -.5 + delta + delta * j;
    double y = -.5 + delta + delta * i;

    // Check if within a circle
    double radius = 0.1;
    if( x*x + y*y < radius*radius )
        return delta * delta / 1.075271758e-02;
    else
        return 0.0;
}

// reference: code from utils.c in google drive
// Fills a 1-D RHS vector specifying boundary conditions
// by calling the get_b method
// b = 1-D RHS vector
// N = dimension (length of b)
void fill_b(double * b, int N)
{
    int n = sqrt(N);
    for( int i = 0; i < n; i++ )
        for( int j = 0; j < n; j++ )
        {
            int idx = i*n + j;
            b[idx] = find_b(i,j,n);
        }
}




void show_matrix(double ** A, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            cout << A[i][j] <<" ";
        }
        cout << endl;
    }
}

void show_vector(double * A, int N) {
    N = sqrt(N);
    int idx = 0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            cout << A[idx++] <<" ";
        }
        cout << endl;
    }
}


// reference: code from utils.c in google drive
double ** matrix( long N )
{
  double *data = (double *) calloc( N*N, sizeof(double) );
  double **M  = (double **) malloc( N  * sizeof(double*));
  
  for( int i = 0; i < N; i++ )
    M[i] = &data[i*N];
  
  return M;
}

// reference: code from utils.c in google drive
/* Free's 2-D Contiguous Matrix */
void matrix_free( double ** M)
{
  free(M[0]);
  free(M);
}

// reference: code from utils.c in google drive
double get_time(void)
{
#ifdef MPI
  return MPI_Wtime();
#endif
  
#ifdef OPENMP
  return omp_get_wtime();
#endif
  
  time_t time;
  time = clock();
  
  return (double) time / (double) CLOCKS_PER_SEC;
}

void save_vector(double * x, long N, char * fname )
{
  FILE * fp = fopen(fname, "w");
  long n = sqrt(N);
  long idx = 0;
  for( long i = 0; i < n; i++ )
    {
      for( long j = 0; j < n; j++ )
	{
	  fprintf(fp, "%.9le,", x[idx]);
	  idx++;
	  if( j != n-1 )
	    fprintf(fp, " ");
	}
      if( i != n - 1 )
	fprintf(fp, "\n");
    }
  fclose(fp);
}
