#include <cstdlib>
#include <iostream>
#include <math.h>
#include <sys/time.h>

using namespace std;

double** multiplyMatricesSeq(double** matrixA, double** matrixB, int size);
double** multiplyMatricesPar(double** matrixA, double** matrixB, int size);
double** multiplyMatricesOptimized1(double** matrixA, double** matrixB, int size);
double** multiplyMatricesStrassens(double** matrixA, double** matrixB, int size);
double** multiplyMatricesTiled(double** matrixA, double** matrixB, int size);
double** multiplyMatricesTiledPar(double** matrixA, double** matrixB, int size);

double** createRandomMatrix();
void transpose(double** matrix, int size);
double** concatMatrices(double** A, double** B, double** C, double** D, int size);
double** subMatrix(double** matrixA, int size, int subType);
double** addMatrices(double** matrixA, double** matrixB, int size);
double** subtractMatrices(double** matrixA, double** matrixB, int size);
void printMatrix(double** matrix, int size);
bool isEqual(double** matrixA, double** matrixB, int size);
double** copyMatrix(double** matrix, int size);

static int n;
timeval start, finish;
double duration;

int main()
{
    srand((unsigned)time(0));

    double** matrixA0;
    double** matrixB0;
    double** matrixA1;
    double** matrixB1;
    double** matrixA2;
    double** matrixB2;
    double** matrixA3;
    double** matrixB3;
    double** matrixA4;
    double** matrixB4;

    double** matrixC0;
    double** matrixC1;
    double** matrixC2;
    double** matrixC3;
    double** matrixC4;

    for(int i=0; i<1; i++){
        //n = (i/200 + 1) * 200;
        n = 1024;
        matrixA0 = createRandomMatrix();
        matrixB0 = createRandomMatrix();
        matrixA1 = copyMatrix(matrixA0, n);
        matrixB1 = copyMatrix(matrixB0, n);
        matrixA2 = copyMatrix(matrixA0, n);
        matrixB2 = copyMatrix(matrixB0, n);
        matrixA3 = copyMatrix(matrixA0, n);
        matrixB3 = copyMatrix(matrixB0, n);
        matrixA4 = copyMatrix(matrixA0, n);
        matrixB4 = copyMatrix(matrixB0, n);

//        // Sequential Matrix Multiplication
//        gettimeofday(&start, NULL);
//        matrixC0 = multiplyMatricesSeq(matrixA0, matrixB0, n);
//        gettimeofday(&finish, NULL);
//        duration = ((finish.tv_sec  - start.tv_sec) * 1000000u +
//                           finish.tv_usec - start.tv_usec) / 1.e6;
//        cout << n << "," << duration << "\n";
//
//        // Parallel Matrix Multiplication
//        gettimeofday(&start, NULL);
//        matrixC1 = multiplyMatricesPar(matrixA1, matrixB1, n);
//        gettimeofday(&finish, NULL);
//        duration = ((finish.tv_sec  - start.tv_sec) * 1000000u +
//                           finish.tv_usec - start.tv_usec) / 1.e6;
//        cout << n << "," << duration << "\n";
//
        // Parallel Transposed Matrix Multiplication
        gettimeofday(&start, NULL);
        matrixC2 = multiplyMatricesOptimized1(matrixA2, matrixB2, n);
        gettimeofday(&finish, NULL);
        duration = ((finish.tv_sec  - start.tv_sec) * 1000000u +
                           finish.tv_usec - start.tv_usec) / 1.e6;
        cout << n << "," << duration << "\n";
//
//        // Tiled Transposed Matrix Multiplication
//        gettimeofday(&start, NULL);
//        matrixC3 = multiplyMatricesStrassens(matrixA3, matrixB3, n);
//        gettimeofday(&finish, NULL);
//        duration = ((finish.tv_sec  - start.tv_sec) * 1000000u +
//                    finish.tv_usec - start.tv_usec) / 1.e6;
//        cout << n << "," << duration << "\n";

        // Tiled Parallel Transposed Matrix Multiplication
        gettimeofday(&start, NULL);
        matrixC4 = multiplyMatricesTiledPar(matrixA4, matrixB4, n);
        gettimeofday(&finish, NULL);
        duration = ((finish.tv_sec  - start.tv_sec) * 1000000u +
                    finish.tv_usec - start.tv_usec) / 1.e6;
        cout << n << "," << duration << "\n";
    }
    isEqual(matrixC3,matrixC4,n);
    return 0;
}

double** multiplyMatricesSeq(double** matrixA, double** matrixB, int size) {
    double** matrix = new double*[size];
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new double[size];
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = 0;
            for (int k = 0; k < size; k++)
            {
                matrix[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return matrix;
}

double** multiplyMatricesPar(double** matrixA, double** matrixB, int size) {
    double** matrix = new double*[size];
    int i, j, k, threadCount = 4;
    
    #pragma omp parallel for shared(matrixA,matrixB,matrix) private(i,j,k) schedule(static) num_threads(threadCount)
    for (i = 0; i < size; i++)
    {
        matrix[i] = new double[size];
        for (j = 0; j < size; j++)
        {
            matrix[i][j] = 0;
            for (k = 0; k < size; k++)
            {
                matrix[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return matrix;
}

double** multiplyMatricesOptimized1(double** matrixA, double** matrixB, int size) {
    double** matrix = new double*[size];
    transpose(matrixB, size);
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new double[size];
        for (int j = 0; j < size; j++)
        {
            double temp = 0;
            for (int k = 0; k < size; k++)
            {
               temp += matrixA[i][k] * matrixB[j][k];
            }
            matrix[i][j] = temp;
        }
    }
    return matrix;
}

double** multiplyMatricesStrassens(double** matrixA, double** matrixB, int size) {
    if(size <= 64){
        return multiplyMatricesPar(matrixA, matrixB, size);
    }else{
        double** a = subMatrix(matrixA, size, 1);
        double** b = subMatrix(matrixA, size, 2);
        double** c = subMatrix(matrixA, size, 3);
        double** d = subMatrix(matrixA, size, 4);
        double** e = subMatrix(matrixB, size, 1);
        double** f = subMatrix(matrixB, size, 2);
        double** g = subMatrix(matrixB, size, 3);
        double** h = subMatrix(matrixB, size, 4);

        double** p1 = multiplyMatricesStrassens(a, subtractMatrices(f, h, size/2), size/2);
        double** p2 = multiplyMatricesStrassens(addMatrices(a, b, size/2), h, size/2);
        double** p3 = multiplyMatricesStrassens(addMatrices(c, d, size/2), e, size/2);
        double** p4 = multiplyMatricesStrassens(d, subtractMatrices(g, e, size/2), size/2);
        double** p5 = multiplyMatricesStrassens(addMatrices(a, d, size/2), addMatrices(e, h, size/2), size/2);
        double** p6 = multiplyMatricesStrassens(subtractMatrices(b, d, size/2), addMatrices(g, h, size/2), size/2);
        double** p7 = multiplyMatricesStrassens(subtractMatrices(a, c, size/2), addMatrices(e, f, size/2), size/2);

        double** A = addMatrices(addMatrices(p4, p5, size/2), subtractMatrices(p6, p2, size/2), size/2);
        double** B = addMatrices(p1, p2, size/2);
        double** C = addMatrices(p3, p4, size/2);
        double** D = addMatrices(subtractMatrices(p1, p3, size/2), subtractMatrices(p5, p7, size/2), size/2);

        return concatMatrices(A, B, C, D, size/2);
    }
}

double** multiplyMatricesTiled(double** matrixA, double** matrixB, int size){
    double** matrix = new double*[size];
    double temp;
    int x1, x2, x3, x4, x5, x6, tileSize = 40;

    for(int i=0; i<size; i++){
        matrix[i] = new double[size];
    }
    transpose(matrixB, size);

    for (x1 = 0; x1 < size; x1+=tileSize)
    {
        for (x2 = 0; x2 < size; x2+=tileSize)
        {
            for (x3 = 0; x3 < size; x3+=tileSize)
            {
                // C(x1,x2) = A(x1,x3) X B(x2,x3)
                for (x4 = x1; x4 < x1+tileSize && x4 < size; x4++)
                {
                    for (x5 = x2; x5 < x2+tileSize && x5 < size; x5++)
                    {
                        temp = 0;
                        for (x6 = x3; x6 < x3+tileSize && x6 < size; x6++)
                        {
                            temp += matrixA[x4][x6] * matrixB[x5][x6];
                        }
                        matrix[x4][x5] += temp;
                    }
                }
            }
        }
    }
    return matrix;
}

double** multiplyMatricesTiledPar(double** matrixA, double** matrixB, int size){
    double** matrix = new double*[size];
    double temp;
    int threadCount = 40;
    int x1, x2, x3, x4, x5, x6, tileSize = size/threadCount;

    for(int i=0; i<size; i++){
        matrix[i] = new double[size];
    }
    transpose(matrixB, size);
    #pragma omp parallel for shared(matrixA,matrixB,matrix) private(x4,x5,x6,temp) schedule(dynamic)num_threads
    (threadCount)
    for (x1 = 0; x1 < size; x1+=tileSize)
    {
        for (x2 = 0; x2 < size; x2+=tileSize)
        {
            for (x3 = 0; x3 < size; x3+=tileSize)
            {
                // C(x1,x2) = A(x1,x3) X B(x2,x3)
                for (x4 = x1; x4 < x1+tileSize && x4 < size; x4++)
                {
                    for (x5 = x2; x5 < x2+tileSize && x5 < size; x5++)
                    {
                        temp = 0;
                        for (x6 = x3; x6 < x3+tileSize && x6 < size; x6++)
                        {
                            temp += matrixA[x4][x6] * matrixB[x5][x6];
                        }
                        matrix[x4][x5] += temp;
                    }
                }
            }
        }
    }
    return matrix;
}

void transpose(double** matrix, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            swap(matrix[i][j], matrix[j][i]);
        }
    }
}

double** subMatrix(double** matrixA, int size, int subType) {
    double** matrix = new double *[size/2];
    /*
     * subType defines which part of matrixA should be returned
     *        |
     *    1   |   2
     * _______|_______
     *        |
     *    3   |   4
     *        |
     * */
    if(subType == 1){
        for (int i = 0; i < size/2; i++)
        {
            matrix[i] = new double[size/2];
            for (int j = 0; j < size/2; j++)
            {
                matrix[i][j] = matrixA[i][j];
            }
        }
    }else if(subType == 2){
        for (int i = 0; i < size/2; i++)
        {
            matrix[i] = new double[size/2];
            for (int j = 0; j < size/2; j++)
            {
                matrix[i][j] = matrixA[i][size/2+j];
            }
        }
    }else if(subType == 3){
        for (int i = 0; i < size/2; i++) {
            matrix[i] = new double[size / 2];
            for (int j = 0; j < size / 2; j++) {
                matrix[i][j] = matrixA[i + size / 2][j];
            }
        }
    }else {
        for (int i = 0; i < size/2; i++)
        {
            matrix[i] = new double[size/2];
            for (int j = 0; j < size/2; j++)
            {
                matrix[i][j] = matrixA[i+size/2][j+size/2];
            }
        }
    }
    return matrix;
}

double** concatMatrices(double** A, double** B, double** C, double** D, int size) {
    double** matrix = new double* [size*2];
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new double[size*2];
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = A[i][j];
        }
    }
    for (int i = 0; i < size; i++)
    {
        matrix[size + i] = new double[size*2];
        for (int j = 0; j < size; j++)
        {
            matrix[i][size + j] = B[i][j];
        }
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[size + i][j] = C[i][j];
        }
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[size + i][size + j] = D[i][j];
        }
    }
    return matrix;
}

double** addMatrices(double** matrixA, double** matrixB, int size) {
    double** matrix = new double *[size];
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new double[size];
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }
    return matrix;
}

double** subtractMatrices(double** matrixA, double** matrixB, int size) {
    double** matrix = new double *[size];
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new double[size];
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = matrixA[i][j] - matrixB[i][j];
        }
    }
    return matrix;
}

double** createRandomMatrix() {
    double** randMatrix = new double*[n];

    for (int i = 0; i < n; i++)
    {
        randMatrix[i] = new double[n];
        for (int j = 0; j < n; j++)
        {
            randMatrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return randMatrix;
}

void printMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            cout << matrix[i][j] << ", ";
        }
        cout << "\n";
    }
    cout << "\n";
}

bool isEqual(double** matrixA, double** matrixB, int size) {
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if(fabs(matrixA[i][j] - matrixB[i][j]) > 0.00001){
                cout << "Matrices are not equal!\n";
                return false;
            }
        }
    }
    cout << "Matrices are equal!\n";
    return true;
}

double** copyMatrix(double** matrix, int size){
    double** temp = new double*[n];
    for(int i=0; i<size; i++){
        temp[i] = new double [n];
        for(int j=0; j<size; j++){
            temp[i][j] = matrix[i][j] + 0;
        }
    }
    return temp;
}