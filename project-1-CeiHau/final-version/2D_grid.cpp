#include<stdlib.h>
#include<mpi.h>
#include<cmath>
#include <fstream>
#include <string>
#include<iostream>
#include<cassert>
#include<vector>
#include<omp.h>

using namespace std;

#define DIMENSION 2
// #define N 1000
// #define NT 400

// You compile as:
// $> mpic++ -O3 2D_grid.cpp -o 2D_grid

// ref: https://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi
double **alloc_2d_double(int rows, int cols);
void save_to_file(double **C, const string &name, int N);

int main(int argc, char * argv[]){
    

    // MPI Init
	int mype, nprocs;
    
    
	MPI_Init( &argc, &argv );
    double start_omp = omp_get_wtime();
    double start = MPI_Wtime();

	MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
	MPI_Comm_rank( MPI_COMM_WORLD, &mype );

    // MPI Cartesian Grid Creation
	int dims[DIMENSION], periodic[DIMENSION], coords[DIMENSION];
	int nprocs_per_dim = sqrt(nprocs); // sqrt or cube root for 2D, 3D
	MPI_Comm comm1d;
	dims[0] = nprocs_per_dim;    // Number of MPI ranks in each dimension
    dims[1] = nprocs_per_dim;
	periodic[0] = 1;             // Turn on/off periodic boundary conditions for each dimension
    periodic[1] = 1;

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

    // Determine 2D neighbor ranks for this MPI rank
	int left, right, up, down;
	MPI_Cart_shift(comm1d,    // Our Cartesian Communicator Object
			1,                // Which dimension we are shifting
			1,                // Direction of the shift
			&left,            // Tells us our Left neighbor
			&right);          // Tells us our Right neighbor

    MPI_Cart_shift(comm1d,    // Our Cartesian Communicator Object
			0,                // Which dimension we are shifting
			1,                // Direction of the shift
			&up,            // Tells us our Left neighbor
			&down);          // Tells us our Right neighbor
    int N, NT;
    N = 10000;
    NT = 400;
    double L, T, u, v;
    L = 1.0;
    T = 1.0e3;
    u = 5.0e-7;
    v = 2.85e-7;
    // Allocate Local Matrices
    int N_l = N / nprocs_per_dim; 
    double ** C = alloc_2d_double(N_l, N_l);
    double ** C_new = alloc_2d_double(N_l, N_l);

    //Initial conditions
    double delta_x = L / N;
    double delta_t = T / NT;
    double delta = delta_t / (2 * delta_x);
    double x0 = L / 2, y0 = x0;
    double sigma_x = L / 4, sigma_y = sigma_x;
    assert(("parameters meet the Courant stability condition", delta_t <= delta_x / sqrt(2 * (u * u + v * v))));
    //Initialize Cn for all i,j as a Gaussian
    #pragma omp parallel for default(none)  shared(N_l, coords, L, delta_x, C, x0, y0, sigma_x, sigma_y)
    for (int i = 0; i < N_l; i++) {
        // #pragma omp parallel for default(none) shared(N, L, i, delta_x, C_n)
        for (int j = 0; j < N_l; j++) {
            // #pragma omp critical
            {
                double x = delta_x * (i + N_l * coords[0]);
                double y = delta_x * (j + N_l * coords[1]);
                double value = exp(
                        -(pow(x - x0, 2) / (2 * pow(sigma_x, 2)) + (pow(y - y0, 2) / (2 * pow(sigma_y, 2)))));
                C[i][j] = value;
            }
            
        }
    }
    
    // Timestep loop  
    for(int timestep = 0; timestep < NT; timestep++) {
        MPI_Status status;
        double * right_ghost_cells = (double *) malloc(N_l * sizeof(double));
        double * left_ghost_cells = (double *) malloc(N_l * sizeof(double));
        double * up_ghost_cells = (double *) malloc(N_l * sizeof(double));
        double * down_ghost_cells = (double *) malloc(N_l * sizeof(double));

        // Shift all data to left using sendrecv
        double * left_sending_buffer = (double *) malloc(N_l * sizeof(double));
        for(int i = 0; i< N_l; i++) {
            left_sending_buffer[i] = C[i][0];  
          
        }
        // send my data to my left neighbor, and I receive from my right neighbor
        MPI_Sendrecv(&(left_sending_buffer[0]),     // Data I am sending
				N_l,                 // Number of elements to send
				MPI_DOUBLE,        // Type I am sending
				left,              // Who I am sending to
				99,                // Tag (I don't care)
				&(right_ghost_cells[0]), // Data buffer to receive to
				N_l,                 // How many elements I am receieving
				MPI_DOUBLE,        // Type
				right,             // Who I am receiving from
				MPI_ANY_TAG,       // Tag (I don't care)
				comm1d,            // Our MPI Cartesian Communicator object
				&status);          // Status Variable


        // Shift all data to right using sendrecv
		// I.e., I send my data to my right neighbor, and I receive from my left neighbor
        double * right_sending_buffer = (double *) malloc(N_l * sizeof(double));
        for(int i = 0; i< N_l; i++) {
            right_sending_buffer[i] = C[i][N_l - 1];
        }
		MPI_Sendrecv(&(right_sending_buffer[0]), // Data I am sending
				N_l,                 // Number of elements to send
				MPI_DOUBLE,        // Type I am sending
				right,             // Who I am sending to
				99,                // Tag (I don't care)
				&(left_ghost_cells[0]),  // Data buffer to receive to
				N_l,                 // How many elements I am receieving
				MPI_DOUBLE,        // Type
				left,              // Who I am receiving from
				MPI_ANY_TAG,       // Tag (I don't care)
				comm1d,            // Our MPI Cartesian Communicator object
				&status);          // Status Variable

        
		MPI_Sendrecv(C[0], // Data I am sending
				N_l,                 // Number of elements to send
				MPI_DOUBLE,        // Type I am sending
				up,             // Who I am sending to
				99,                // Tag (I don't care)
				down_ghost_cells,  // Data buffer to receive to
				N_l,                 // How many elements I am receieving
				MPI_DOUBLE,        // Type
				down,              // Who I am receiving from
				MPI_ANY_TAG,       // Tag (I don't care)
				comm1d,            // Our MPI Cartesian Communicator object
				&status);          // Status Variable



		MPI_Sendrecv(C[N_l - 1], // Data I am sending
				N_l,                 // Number of elements to send
				MPI_DOUBLE,        // Type I am sending
				down,             // Who I am sending to
				99,                // Tag (I don't care)
				up_ghost_cells,  // Data buffer to receive to
				N_l,                 // How many elements I am receieving
				MPI_DOUBLE,        // Type
				up,              // Who I am receiving from
				MPI_ANY_TAG,       // Tag (I don't care)
				comm1d,            // Our MPI Cartesian Communicator object
				&status);          // Status Variable

   

        #pragma omp parallel for default(none) shared(N_l, C, C_new, delta_t, delta_x, u, v, delta,up_ghost_cells, down_ghost_cells, left_ghost_cells, right_ghost_cells) schedule(static)
        for (int i = 0; i < N_l; i++) {
            // #pragma omp parallel for default(none)  shared(N, C_n, C_n1, delta_t, delta_x, u, v, i) 
            for (int j = 0; j < N_l; j++) {
                
                double left_value, right_value, up_value, down_value;
                
                up_value = (i == 0) ? up_ghost_cells[j] : C[i - 1][j];
                down_value = (i == (N_l - 1)) ? down_ghost_cells[j] : C[i + 1][j];
                left_value = (j == 0) ? left_ghost_cells[i] : C[i][j - 1];
                right_value = (j == (N_l - 1)) ? right_ghost_cells[i] : C[i][j + 1];

                double value = (up_value + down_value + left_value + right_value) * 0.25 - 
                                delta_t / (2 * delta_x) *
                                (u * (down_value - up_value) + v * (right_value - left_value));
                C_new[i][j] = value;
                
            }
        }
        swap(C, C_new);
        
    }

    // Collect global value with MPI
    double ** C_global = (double **) malloc(N * sizeof(*C_global));
    for (int i = 0; i< N; i++) {
        C_global[i] = (double *) malloc(N * sizeof(double));
    }
    
    double ** local_C = alloc_2d_double(N_l, N_l);

    // If rank 0, collect local C from all other processors
    if (mype ==0 ){
        for(int r_id = 1; r_id< nprocs; r_id++) {
            MPI_Recv(&(local_C[0][0]), 
                N_l * N_l, 
                MPI_DOUBLE, 
                r_id,
                MPI_ANY_TAG,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE      
            );
    
      
            for(int i = 0; i < N_l; i++) {
                for(int j = 0; j< N_l; j++) {
                    C_global[r_id/nprocs_per_dim * N_l + i][(r_id%nprocs_per_dim) * N_l + j] = local_C[i][j];

                }
            }

        }
        free(local_C);
        // add proc 0's own C
        for(int i = 0; i < N_l; i++) {
            for(int j = 0; j < N_l; j++){
                C_global[i][j] = C[i][j];
            }
        }
    } else {    // if not rank 0, send local C to rank 0
        MPI_Send(&(C[0][0]),
            N_l * N_l,
            MPI_DOUBLE,
            0,
            0,
            MPI_COMM_WORLD
        );
    }

    // if (mype == 0) {
    //     save_to_file(C_global, "final.csv", N);
    //     cout<<"Finish\n";
    // }
    double duration, global;

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double end_omp = omp_get_wtime(); 
    duration = end -start;
    printf("Runtime at %d is %f ", mype, duration);
    
    cout<<"; omp"<< end_omp - start_omp <<" seconds\n";

    MPI_Reduce(&duration,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(mype == 0) {
        printf("Global runtime is %f\n",global);
    }
    
    MPI_Finalize();

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
                "/home/wxh/MPI/Project1/2D_grid/" +
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