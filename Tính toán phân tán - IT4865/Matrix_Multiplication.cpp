#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <time.h>
#define MAX_THREADS 4

using namespace std;

void printMatrix(double** m, int COL, int ROW) {
    for (int i = 0; i < COL; i++) {
        for (int j = 0; j < ROW; j++) {
            cout << m[i][j] << " ";
        }
        cout << endl;
    }
    cout << "----" << endl;
}
void enterMatrix(double** a, int COL, int ROW)
{
    int i, j;
    for (i = 0; i < COL; i++)
        for (j = 0; j < ROW; j++)
        {
            a[i][j] = (double)(i);
        }
}
double** init(int COL, int ROW) {
    double** matrix = new double* [COL];
    for (int i = 0; i < COL; ++i)
        matrix[i] = new double[ROW];
    return matrix;
}
void deleteMatrix(double** a, int COL) {
    for (int i = 0; i < COL; i++)
        delete[] a[i];

    delete[] a;
}
void alg_matmul2D(int m, int n, int p, double** a, double** b, double** c)
{

    int i, j, k;

    for (i = 0; i < m; i = i + 1) {
        for (j = 0; j < n; j = j + 1) {
            a[i][j] = 0.;
            for (k = 0; k < p; k = k + 1) {
                a[i][j] = (a[i][j]) + ((b[i][k]) * (c[k][j]));
            }
        }
    }
}
void matrix_multiplication(int m, int n, int p, double** a, double** b, double** c)
{
    int i, j, k;
    #pragma omp parallel shared(a,b,c) private(i,j,k) 
    {
        #pragma omp for schedule(static)
        for (i = 0; i < m; i = i + 1) {
            for (j = 0; j < n; j = j + 1) {
                a[i][j] = 0.0;
                for (k = 0; k < p; k = k + 1) {
                    a[i][j] = (a[i][j]) + ((b[i][k]) * (c[k][j]));
                }
            }
        }
    }
}
int main(int argc, char* argv[]) {
    int j;
    double start, delta;
    for (j = 1; j <= MAX_THREADS; j++) {

        printf(" running on %d threads: ", j);
        omp_set_num_threads(j);
        

        int COL_A = 1000, ROW_A = 500;
        int COL_B = 500, ROW_B = 1000;
        int COL_C = COL_A, ROW_C = ROW_B;

        double** matrixA = init(COL_A, ROW_A);
        double** matrixB = init(COL_B, ROW_B);
        double** matrixC = init(COL_C, ROW_C);

        enterMatrix(matrixA, COL_A, ROW_A);
        enterMatrix(matrixB, COL_B, ROW_B);
        enterMatrix(matrixC, COL_C, ROW_C);

        
        double start = omp_get_wtime();
        matrix_multiplication(COL_A, ROW_B, ROW_A, matrixC, matrixA, matrixB);        
        delta = omp_get_wtime() - start;
        printf("matrix multiplication computed in %.4g seconds\n", delta);


        deleteMatrix(matrixA, COL_A);
        deleteMatrix(matrixB, COL_B);
        deleteMatrix(matrixC, COL_C);

             
    }
    return 0;
}
