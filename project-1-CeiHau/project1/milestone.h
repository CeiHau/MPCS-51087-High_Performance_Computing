//
// Created by Xihao Wang on 4/1/22.
//

#ifndef PROJECT1_MILESTONE_H
#define PROJECT1_MILESTONE_H

#include <vector>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>

using std::pair;
using std::vector;
using std::swap;
using std::ofstream;
using std::ostream_iterator;
using std::string;
using std::copy;
namespace mpcs50187 {
    void save_to_file(const vector<vector<double>> &C, const string &name);

    void Advection(int N, int NT, double L, double T, double u, double v) {
        vector<vector<double>> C_n(N, vector<double>(N));

        vector<vector<double>> C_n1(N, vector<double>(N));
        double delta_x = L / N;
        double delta_t = T / NT;
        assert(("parameters meet the Courant stability condition", delta_t <= delta_x / sqrt(2 * (u * u + v * v))));
        //Initialize Cn for all i,j as a Gaussian
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double x = delta_x * i;
                double y = delta_x * j;
                double x0 = L / 2, y0 = x0;
                double sigma_x = L / 4, sigma_y = sigma_x;

                C_n[i][j] = exp(-(pow(x - x0, 2) / (2 * pow(sigma_x, 2)) + (pow(y - y0, 2) / (2 * pow(sigma_y, 2)))));
            }
        }

//        save_to_file(C_n, "initialized.csv");

        for (int n = 1; n <= NT; n++) {
//            std::cout << n << std::endl;
            if (n == NT / 2) {
//                save_to_file(C_n, "half.csv");
            }
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {


                    //Update Cn+1 with Apply periodic boundary conditions
                    C_n1[i][j] = (1.0 / 4.0) * (C_n[(((i - 1) % N) + N) % N][j] + C_n[(((i + 1) % N) + N) % N][j] +
                                                C_n[i][(((j - 1) % N) + N) % N] + C_n[i][(((j + 1) % N) + N) % N])
                                 - (delta_t / (2 * delta_x)) *
                                   (u * (C_n[(((i + 1) % N) + N) % N][j] - C_n[(((i - 1) % N) + N) % N][j]) +
                                    v * (C_n[i][(((j + 1) % N) + N) % N] - C_n[i][(((j - 1) % N) + N) % N]));
                }
            }
            //Set Cn = Cn+1 for all i, j
            swap(C_n1, C_n);
        }

//        save_to_file(C_n1, "end.csv");


    }

    void save_to_file(const vector<vector<double>> &C, const string &name) {
        ofstream outFile;

        outFile.open(
                "/Users/wxh/Documents/UChicago/2022 Spring/MPCS 51087 High Performance Computing/project-1-CeiHau/project1/" +
                name);
        for (const auto &i: C) {
            for (auto &j: i) {
                outFile << j << ",";
//                if (j != std::prev(i.end())) {
//                    outFile << ",";
//                }
            }
            outFile << "\n";

        }


        outFile << "\n";
        outFile.close();
        std::cout << "generate " + name << std::endl;
    }

}
#endif //PROJECT1_MILESTONE_H
