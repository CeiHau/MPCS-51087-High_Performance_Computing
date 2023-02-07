#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAX_LAYERS = 10

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void sigmoid_matrix(double *x, int n, double *r) {
    for (int i = 0; i < n; i++) {
        r[i] = sigmoid(x[i]);
    }
}


double sigmoid_derivative(double x) {
    return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}

void sigmoid_derivative_matrix(double *x, int n, double *r) {
    for (int i = 0; i < n; i++) {
        r[i] = sigmoid_derivative(x[i]);
    }
}

/* dot product of two matrix m and n
 * matrix m with size a * b
 * matrix n with size b * c
 * result saves in *r
 * */
void dot_simple(const double *m, const double *n, int a, int b, int c, double *r) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < c; j++) {
            r[i * c + j] = 0;
        }
    }

    for (int i = 0; i < a; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < b; k++) {
                r[i * c + j] += m[i * b + k] * n[k * c + j];
            }
        }
    }
}

/* multiply two vectors by element wise */
void mult(const double *x, const double *y, int n, double *r) {
    for (int i = 0; i < n; i++)
        r[i] = x[i] * y[i];

}

/*sub two vectors by element wise */
void sub(const double *x, const double *y, int n, double *r) {
    for (int i = 0; i < n; i++)
        r[i] = x[i] - y[i];
}

/*add two vectors by element wise */
void add(const double *x, const double *y, int n, double *r) {
    for (int i = 0; i < n; i++)
        r[i] = x[i] + y[i];
}

/* mult vector by scalar */
void scalar_mult(const double *x, double c, int n, double *r) {
    for (int i = 0; i < n; i++)
        r[i] = x[i] * c;
}

/* one hot */
void one_hot(double x, double *vec) {
    int num = int(x);

    for (int i = 0; i < 10; i++) {
        if (num == i)
            vec[i] = 1;
        else
            vec[i] = 0;
    }
}

double mse(double *y, double *y_hat) {
    double sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
    }
    return sum / 10;
}

/*read file*/
void load_data(double **training_X, double **training_Y, char *filename) {
    FILE *fp = fopen(
            filename,
            "r");
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    if (fp == NULL) {
        printf("\nUnable to load file!\n");
        exit(EXIT_FAILURE);
    }

    int index = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if (index > 0) {
            char *token = strtok(line, ",");
            int id = 0;
            while (token != NULL) {
                double x = atof(token);
                if (id != 0)
                    training_X[index - 1][id] = x / 255;
                else
                    one_hot(x, training_Y[index - 1]);
                token = strtok(NULL, ",");
                id++;
            }
        }
        index++;
    }
}

class NerualNetwork {
public:
    int *layers;
    int L;

    double **biases;    // length =  L-1, size of biases[i] = layers[i+1]
    double **weights;   // length =  L-1, size of weights[i] = layers[i] * layers[i+1]
    double **a;
    double **z;

    /*
     * Constructor function
     * 1. Set the number of layers and the size of each layer
     * 2. Initialize the biases and weights matrix with random number
     * 3. Allocate the space for a and z matrix
     * */
    NerualNetwork(int *layers, int L) {
        // Set the seed
        srand(time(NULL));

        this->L = L;
        this->layers = (int *) malloc(sizeof(int) * L);
        for (int i = 0; i < L; i++)
            this->layers = layers;
        // Allocate  L-1 biases matrix
        this->biases = (double **) malloc(sizeof(double *) * (L - 1));
        for (int i = 0; i < (L - 1); i++)
            // each biases matrix between i and i+1 has size layers[i] * layers[i + 1]
            this->biases[i] = (double *) malloc(sizeof(double) * layers[i + 1]);

        // Allocate L-1 weights matrix
        this->weights = (double **) malloc(sizeof(double *) * (L - 1));
        for (int i = 0; i < (L - 1); i++) {
            // each weight matrix between i and i+1 has size layers[i] * layers[i + 1]
            this->weights[i] = (double *) malloc(sizeof(double) * layers[i] * layers[i + 1]);
        }

        // Allocate L a(activation) matrix, a = sigmoid(z)
        this->a = (double **) malloc(sizeof(double *) * L);
        for (int i = 0; i < L; i++)
            this->a[i] = (double *) malloc(sizeof(double) * layers[i]);

        // Allocate L-1 z matrix
        this->z = (double **) malloc(sizeof(double *) * (L - 1));
        for (int i = 0; i < (L - 1); i++)
            this->z[i] = (double *) malloc(sizeof(double) * layers[i + 1]);


        // Initialize with random number
        for (int i = 0; i < (L - 1); i++)
            for (int j = 0; j < layers[i + 1]; j++)
                biases[i][j] = ((double) rand() / (double) RAND_MAX);

        for (int i = 0; i < (L - 1); i++)
            for (int j = 0; j < layers[i] * layers[i + 1]; j++)
                weights[i][j] = ((double) rand() / (double) RAND_MAX);
    }

    void feedforward(double *x) {
//        memcpy(a[0], x, (sizeof(a[0]) / sizeof(double)));
        for (int i = 0; i < layers[0]; i++) {
            a[0][i] = x[i];
        }
        for (int i = 0; i < (L - 1); i++) {
            dot_simple(a[i], weights[i], 1, layers[i], layers[i + 1], z[i]);
            add(z[i], biases[i], layers[i + 1], z[i]);
            sigmoid_matrix(z[i], layers[i + 1], a[i + 1]);
        }
    }

    void backpropogate(double *y, double **derv_b, double **derv_w) {

        // Output error: δx, L = ∇aC ⊙ σ′(zx, l)
        double *temp = (double *) malloc(sizeof(double) * layers[L - 1]);
        sigmoid_derivative_matrix(z[L - 2], layers[L - 1], temp);

        double *temp2 = (double *) malloc(sizeof(double) * layers[L - 1]);
        sub(a[L - 1], y, layers[L - 1], temp2);

        double *delta = (double *) malloc(sizeof(double) * layers[L - 1]);
        mult(temp, temp2, layers[L - 1], delta);


        for (int i = 0; i < layers[L - 1]; i++)
            derv_b[L - 2][i] = delta[i];
        dot_simple(a[L - 2], delta, layers[L - 2], 1, layers[L - 1], weights[L - 2]);

        for (int l = L - 3; l >= 0; l--) {
            double *temp = (double *) malloc(sizeof(double) * layers[l + 1]);
            sigmoid_derivative_matrix(z[l], layers[l + 1], temp);
            double *temp2 = (double *) malloc(sizeof(double) * layers[l + 1]);
            dot_simple(weights[l + 1], derv_b[l + 1], layers[l + 1], layers[l + 2], 1, temp2);

            mult(temp2, temp, layers[l + 1], derv_b[l]);
            dot_simple(a[l], derv_b[l], layers[l], 1, layers[l + 1], derv_w[l]);
//            free(temp);
            free(temp2);
        }

    }

    void train_small_batch(int batch_len, double **training_X, double **training_Y, double alpha) {
        // Initiate sum_derv_b, sum_derv_w
        double **sum_derv_b = (double **) malloc(sizeof(double *) * (L - 1));
        for (int i = 0; i < (L - 1); i++)
            sum_derv_b[i] = (double *) malloc(sizeof(double) * layers[i + 1]);

        double **sum_derv_w = (double **) malloc(sizeof(double *) * (L - 1));
        for (int i = 0; i < (L - 1); i++)
            sum_derv_w[i] = (double *) malloc(sizeof(double) * layers[i] * layers[i + 1]);

        for (int i = 0; i < (L - 1); i++)
            for (int j = 0; j < layers[i + 1]; j++)
                sum_derv_b[i][j] = 0;
        for (int i = 0; i < (L - 1); i++)
            for (int j = 0; j < layers[i] * layers[i + 1]; j++)
                sum_derv_w[i][j] = 0;

        for (int b = 0; b < batch_len; b++) {
            feedforward(training_X[b]);

//
//            printf("%d: ", b);
//            for (int i = L - 1; i < L; i++) {
//                for (int j = 0; j < layers[i]; j++)
//                    printf("%f ", a[i][j]);
//                printf("\n");
//            }

            // initiate derv_b, derv_w
            double **derv_b = (double **) malloc(sizeof(double *) * (L - 1));
            for (int i = 0; i < (L - 1); i++)
                derv_b[i] = (double *) malloc(sizeof(double) * layers[i + 1]);

            double **derv_w = (double **) malloc(sizeof(double *) * (L - 1));
            for (int i = 0; i < (L - 1); i++)
                derv_w[i] = (double *) malloc(sizeof(double) * layers[i] * layers[i + 1]);

            for (int i = 0; i < (L - 1); i++)
                for (int j = 0; j < layers[i + 1]; j++)
                    derv_b[i][j] = 0;
            for (int i = 0; i < (L - 1); i++)
                for (int j = 0; j < layers[i] * layers[i + 1]; j++)
                    derv_w[i][j] = 0;
            // Backpropogate
            backpropogate(training_Y[b], derv_b, derv_w);

            // Caculate the changing of weights and bias
            for (int i = 0; i < (L - 1); i++) {
                for (int j = 0; j < layers[i + 1]; j++)
                    sum_derv_b[i][j] += derv_b[i][j];
            }
            for (int i = 0; i < (L - 1); i++) {
                for (int j = 0; j < layers[i] * layers[i + 1]; j++)
                    sum_derv_w[i][j] += derv_w[i][j];
            }
        }
        // Gradient Descent
        for (int i = 0; i < (L - 1); i++) {

            double *temp = (double *) malloc(sizeof(double) * layers[i] * layers[i + 1]);
            // temp = (alpha / m) * sum_derw_w[i]
            scalar_mult(sum_derv_w[i], alpha / batch_len, layers[i] * layers[i + 1], temp);
            sub(weights[i], temp, layers[i] * layers[i + 1], weights[i]);
            free(temp);

            double *temp2 = (double *) malloc(sizeof(double) * layers[i + 1]);
            scalar_mult(sum_derv_b[i], alpha / batch_len, layers[i + 1], temp2);
            sub(biases[i], temp2, layers[i + 1], biases[i]);
        }
    }

    void
    train(double **training_X, double **training_Y, int train_size, int batch_len, int epochs, double alpha,
          double **test_X,
          double **test_Y, int test_size) {
        clock_t start, end;
        double cpu_time_used;

        for (int epoch = 0; epoch < epochs; epoch++) {
            start = clock();
            for (int i = 0; i < train_size; i += batch_len) {
                train_small_batch(batch_len, &training_X[i], &training_Y[i], alpha);

            }
            end = clock();
            cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

            printf("Train %f samples per seconds. ", train_size/cpu_time_used);
            evaluate(test_X, test_Y, test_size);

        }

    }

    int predict(const double *x, double *output) const {
        double *next = (double *) malloc(layers[0] * sizeof(double));
        for (int i = 0; i < layers[0]; i++) {
            next[i] = x[i];
        }
        for (int i = 0; i < (L - 1); i++) {
            double *temp = (double *) malloc(layers[i + 1] * sizeof(double));
            dot_simple(next, weights[i], 1, layers[i], layers[i + 1], temp);

//            free(next);
            next = (double *) malloc(layers[i + 1] * sizeof(double));
            add(temp, biases[i], layers[i + 1], next);
        }

        int max_indx = 0;
        for (int i = 0; i < 10; i++) {
            output[i] = next[i];
            if (next[i] > next[max_indx]) {
                max_indx = i;
            }
        }

        return max_indx;
    }

    int evaluate(double **test_X, double **test_Y, int test_size) {
        int correct = 0;
        int y_hat;
        double output[10];
        double sum_mse;
        for (int i = 0; i < test_size; i++) {
            y_hat = predict(test_X[i], output);
//            printf("%d ", y_hat);
            sum_mse += mse(test_Y[i],output);
            correct += int(test_Y[i][y_hat]);
        }
        printf("MSE = %f\n", sum_mse/test_size);
        return correct;
    }

};

void matrix_mult_test() {
    double m[] = {1, 2, 3, 4, 5, 6}; // 2 * 3
    double n[] = {10, 11, 20, 21, 30, 31}; // 3 * 2
    double r[4];
    dot_simple(m, n, 2, 3, 2, r);
    for (int i = 0; i < 4; i++) {
        printf("%f ", r[i]);
    }
}

int main() {


    int nl = 4;
    int nh[] = {78, 30, 20, 10};
    NerualNetwork network(nh, nl);

    
    double **training_X = (double **) malloc(60000 * 784 * sizeof(double));
    double **training_Y = (double **) malloc(60000 * 10 * sizeof(double));
    for (int i = 0; i < 60000; i++) {
        training_X[i] = (double *) malloc(784 * sizeof(double));
        training_Y[i] = (double *) malloc(10 * sizeof(double));
    }
    load_data(training_X, training_Y,
              "/Users/wxh/Documents/UChicago/2022 Spring/MPCS 51087 High Performance Computing/project4_ml/Cplusplus/mnist_data/mnist_train.csv");

    double **test_X = (double **) malloc(10000 * 784 * sizeof(double));
    double **test_Y = (double **) malloc(10000 * 10 * sizeof(double));
    for (int i = 0; i < 10000; i++) {
        test_X[i] = (double *) malloc(784 * sizeof(double));
        test_Y[i] = (double *) malloc(10 * sizeof(double));
    }
    load_data(test_X, test_Y,
              "/Users/wxh/Documents/UChicago/2022 Spring/MPCS 51087 High Performance Computing/project4_ml/Cplusplus/mnist_data/mnist_test.csv");


    network.train(training_X, training_Y, 60000, 10,10, 3.0, test_X, test_Y, 10000);



// show data

    for (int i = 0; i < 100; i += 10) {
        printf("%d ", i);
    }

    // free data
    for (int i = 0; i < 60000; i++) {
        free(training_Y[i]);
        free(training_X[i]);
    }
    free(training_Y);
    free(training_X);

    return 0;
}
