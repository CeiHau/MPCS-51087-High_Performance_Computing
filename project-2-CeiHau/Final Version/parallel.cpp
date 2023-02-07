#include<stdlib.h>
#include<iostream>
#include<cmath>
#include<mpi.h>
#include<cstring>
#include<string>

#define DIMENSION 1

using namespace std;


int parallel_cg_sparse_poisson(double * x_l, double * b_l, long N, int mype, int nprocs, MPI_Comm comm1d, int left, int right);

void parallel_matvec_OTF( double * v, double * w, long N_l, int mype, int nprocs, MPI_Comm comm1d, int left, int right);

double parallel_dotp( double * a, double * b, long N_l, int mype, int nprocs);

void parallel_axpy(double alpha, double * w, double beta, double * v, long N, int mype, int nprocs);

void parallel_fill_b(double * b_l, long N_l, int mype, int nprocs);


double find_b(int i, int j, int n);

double ** matrix( long N );

void matrix_free( double ** M);

void show_matrix(double ** A, int N);

void show_vector(double * A, int N);

double get_time(void);

void save_vector(double * x, long N, const char * fname );

// You compile as:
// $> mpic++ -O3 parallel.cpp -o parallel

int main(int argc, char * argv[]) {

    int n = stoi(argv[1]);
    int N = n * n;
    
    // MPI Init
	int mype, nprocs;
	MPI_Init( &argc, &argv );
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
	MPI_Comm_rank( MPI_COMM_WORLD, &mype );
    if(mype == 0 ) {
        cout <<  "n = " << n << ", Nodes = " << nprocs << endl;
    }
    
    // MPI Cartesian Grid Creation
	int dims[DIMENSION], periodic[DIMENSION], coords[DIMENSION];
	int nprocs_per_dim = nprocs; // sqrt or cube root for 2D, 3D
	MPI_Comm comm1d;
	dims[0] = nprocs_per_dim;    // Number of MPI ranks in each dimension
	periodic[0] = 1;             // Turn on/off periodic boundary conditions for each dimension

    // Create Cartesian Communicator
	MPI_Cart_create( MPI_COMM_WORLD, // Starting communicator we are going to draw from
			DIMENSION,      // MPI grid n-dimensionality
			dims,           // Array holding number of MPI ranks in each dimension
			periodic,       // Array indicating if we want to use periodic BC's or not
			1,              // Yes/no to reordering (allows MPI to re-organize for better perf)
			&comm1d );      // Pointer to our new Cartesian Communicator object

    // Extract this MPI rank's N-dimensional coordinates from its place in the MPI Cartesian grid
	MPI_Cart_coords(comm1d,  // Our Cartesian Communicator Object
			mype,            // Our process ID
			DIMENSION,       // The n-dimensionality of our problem
			coords);         // An n-dimensional array that we will write this rank's MPI coordinates to

    // Determine 1D neighbor ranks for this MPI rank
	int left, right;
	MPI_Cart_shift(comm1d,    // Our Cartesian Communicator Object
			0,                // Which dimension we are shifting
			1,                // Direction of the shift
			&left,            // Tells us our Left neighbor
			&right);          // Tells us our Right neighbor

    
    int N_l = N / nprocs_per_dim;

    // Allocate Local Matrices
    double *x_l = (double *) malloc(N_l * (sizeof(double)));
    for(int i = 0; i < N_l; i++) {
        x_l[i] = 0;
    }
    string bname  = "b"+to_string(mype) + ".csv";
    double *b_l = (double *) malloc(N_l * (sizeof(double)));
    
    // Compute elements of boundary condition vector 'b'
    parallel_fill_b(b_l, N_l, mype, nprocs);

    double begin = MPI_Wtime();
    int iter = parallel_cg_sparse_poisson(x_l, b_l, N, mype, nprocs, comm1d, left, right);

    // If rank 0, collect local x from all other processors
    if(mype == 0) {
        double* x = (double *) malloc(N * sizeof(double));
        // add proc 0's own x
    
        for(int i = 0; i < N_l; i++) {
            x[i] = x_l[i];
        }
        // collect local x from others
        for(int r_id = 1; r_id < nprocs; r_id++) {
            MPI_Recv(&x[r_id * N_l],
                N_l,
                MPI_DOUBLE,
                r_id,
                MPI_ANY_TAG,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }
        save_vector(x, N, "parallel_result.csv");
        free(x);
    } else { // if not rank 0, send local x to rank 0
        MPI_Send(&x_l[0],
            N_l,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD
        );
    }
    free(x_l);
    free(b_l);

    double duration, global;

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    
    duration = end - begin;
    
    MPI_Reduce(&duration,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(mype == 0) {
        printf("CG converged in %ld iterations.\n", iter);
        printf("Global runtime is %f seconds\n",global);
    }
    MPI_Finalize();
}

// Parallel Distributed MPI Version of
// Conjugate Gradient Solver Function for Ax = b
// where A is the 2D poisson matrix operator that
// is assembled on the fly (not stored)
// x = 1D solution vector
// b = 1D vector
// N = dimension
int parallel_cg_sparse_poisson(double * x_l, double * b_l, long N, int mype, int nprocs, MPI_Comm comm1d, int left, int right) {
    // r = -A*x + b
    int N_l = N / nprocs;
    double *r = (double *) malloc(N_l * sizeof(double));
    parallel_matvec_OTF(r, x_l, N_l, mype, nprocs, comm1d, left, right); //r = A*x
    parallel_axpy(-1.0, r, 1.0, b_l, N_l, mype, nprocs); // r = -r + b
    // save_vector(r, N_l, "r.csv");
    // p = r
    double *p = (double *) malloc(N_l * sizeof(double));
    memcpy(p, r, N_l * sizeof(double));

    // Allocate z
    double *z = (double *) malloc(N_l * sizeof(double));

    //rsold = r' * r;
    double rsold = parallel_dotp(r, r, N_l, mype, nprocs);

    int i = 0;
    for(; i < N; i++) {
        // z = A * p;
        parallel_matvec_OTF(z, p, N_l, mype, nprocs, comm1d, left, right);
 
        // alpha = rsold / (p' * z);
        double alpha = rsold /parallel_dotp(p, z, N_l, mype, nprocs);
        // x = x + alpha * p;
        parallel_axpy(1.0, x_l, alpha, p, N_l, mype, nprocs);

        // r = r - alpha * z
        parallel_axpy(1.0, r, -alpha, z, N_l, mype, nprocs);

        // rsnew=r' ·r
        double rsnew = parallel_dotp(r, r, N_l, mype, nprocs);

        if (sqrt(rsnew) < 1.0e-10)
            break;

        // p = r + (rsnew/rsold)p
        parallel_axpy(rsnew / rsold, p, 1.0, r, N_l, mype, nprocs);  

        rsold = rsnew;
    }
  

    free(r);
    free(p);
    free(z);
    
    return i;
}


// Parallel Distributed MPI Version of
// Specific "On the fly" Matrix Vector Product for v = A * w
// where 'A' is the 2D Poisson operator matrix
// that is applied without explicit storage
// v, w = 1D vectors
// N = dimension
void parallel_matvec_OTF( double * v, double * w, long N_l, int mype, int nprocs, MPI_Comm comm1d, int left, int right) {
    
    MPI_Status status;
    int N = N_l * nprocs;
    int n = sqrt(N);

    double * far_left_ghost_cells = (double *) malloc(n * sizeof(double));
    double * far_right_ghost_cells = (double *) malloc(n * sizeof(double));

    // send my data to my left neighbor, and I receive from my right neighbor
    MPI_Sendrecv(&w[0],     // Data I am sending
        n,                 // Number of elements to send
        MPI_DOUBLE,        // Type I am sending
        left,              // Who I am sending to
        99,                // Tag (I don't care)
        &far_right_ghost_cells[0], // Data buffer to receive to
        n,                 // How many elements I am receieving
        MPI_DOUBLE,        // Type
        right,             // Who I am receiving from
        MPI_ANY_TAG,       // Tag (I don't care)
        comm1d,            // Our MPI Cartesian Communicator object
        &status);          // Status Variable

    // send my data to my right neighbor, and I receive from my left neighbor
    MPI_Sendrecv(&w[N_l - n],     // Data I am sending
        n,                 // Number of elements to send
        MPI_DOUBLE,        // Type I am sending
        right,              // Who I am sending to
        99,                // Tag (I don't care)
        &far_left_ghost_cells[0], // Data buffer to receive to
        n,                 // How many elements I am receieving
        MPI_DOUBLE,        // Type
        left,             // Who I am receiving from
        MPI_ANY_TAG,       // Tag (I don't care)
        comm1d,            // Our MPI Cartesian Communicator object
        &status);          // Status Variable


    for(int i = 0; i < N_l; i++) {
        double temp = 0.0;
        int index = mype * N_l + i;
        if((index - n) >= 0) {
            // get w[i - n]
            if((i - n) >= 0) {
                temp -= w[i - n];
            } else {
                temp -= far_left_ghost_cells[i];

            }
            
        }
        if((index - 1) >= 0 and ((index - 1) %n != n-1 )) {
            if((i - 1) >= 0) {
                temp -= w[i - 1];
            } else {
                temp -= far_left_ghost_cells[n - 1];
            }
        }
        temp += 4.0 * w[i];
        
        if((index + 1) <= (N - 1) and ((index + 1) % n != 0)) {
            if((i + 1) <= (N_l - 1)) {
                temp -= w[i + 1];
            } else {
                temp -= far_right_ghost_cells[0];
            }
        }

        if((index + n) <= (N - 1)) {
            if((i + n) <= (N_l - 1)) {
                temp -= w[i + n];
            } else {
                temp -= far_right_ghost_cells[i + n - N_l];
            }
        }
        v[i] = temp;
    }

}


// Parallel Distributed MPI Version of
// Dot product of c = a * b
// c = result scalar that's returned
// a, b = 1D Vectors
// N = dimension
double parallel_dotp( double * a, double * b, long N_l, int mype, int nprocs) {
    double c_l = 0.0;
    double c_g = 0.0;

    // compute the dot product of their own local sub-domains
    for( long i = 0; i < N_l; i++ )
        c_l += a[i]*b[i];


    MPI_Allreduce(&c_l, &c_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return c_g;
}


// Parallel Distributed MPI Version of
// Scale and add of two vectors (axpy)
// Solves w = alpha * w + beta * v (overwrites what's in w)
// alpha, beta = scalars
// w, v = 1D vectors
// N = dimension
void parallel_axpy(double alpha, double * w, double beta, double * v, long N, int mype, int nprocs) {
    for( long i = 0; i < N; i++ )
        w[i] = alpha * w[i] + beta * v[i];
}


// reference: code from utils.c in google drive
double find_b(int i, int j, int n) {
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

void parallel_fill_b(double * b_l, long N_l, int mype, int nprocs) {
    int n = sqrt(N_l * nprocs);
    for(int i = 0; i < N_l; i++) {
        int o_i = i + mype * N_l;
        int x = o_i / n;
        int y = o_i - n * x;
        b_l[i] = find_b(x, y, n);
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

void save_vector(double * x, long N, const char * fname )
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
