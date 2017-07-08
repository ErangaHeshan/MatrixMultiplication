#include <cstdlib>
#include <iostream>
#include <math.h>

using namespace std;

double** multiplyMatricesSeq(double** matrixA, double** matrixB, int size);
double** multiplyMatricesPar(double** matrixA, double** matrixB, int size);
double** multiplyMatricesOptimized1(double** matrixA, double** matrixB, int size);
double** multiplyMatricesStrassens(double** matrixA, double** matrixB, int size);
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

int main()
{
    n = 1024;
    srand((unsigned)time(0));
    double** matrixA0 = createRandomMatrix();
    double** matrixB0 = createRandomMatrix();
    double** matrixA1 = copyMatrix(matrixA0, n);
    double** matrixB1 = copyMatrix(matrixB0, n);
    double** matrixA2 = copyMatrix(matrixA0, n);
    double** matrixB2 = copyMatrix(matrixB0, n);
    double** matrixA3 = copyMatrix(matrixA0, n);
    double** matrixB3 = copyMatrix(matrixB0, n);

    clock_t start = clock();
    double** matrixC0 = multiplyMatricesSeq(matrixA0, matrixB0, n);
    clock_t finish = clock();
    cout << float(finish - start)/CLOCKS_PER_SEC << "s\n";

    start = clock();
    double** matrixC1 = multiplyMatricesPar(matrixA1, matrixB1, n);
    finish = clock();
    cout << float(finish - start)/CLOCKS_PER_SEC << "s\n";

    start = clock();
    double** matrixC2 = multiplyMatricesOptimized1(matrixA2, matrixB2, n);
    finish = clock();
    cout << float(finish - start)/CLOCKS_PER_SEC << "s\n";

    start = clock();
    double** matrixC3 = multiplyMatricesStrassens(matrixA3, matrixB3, n);
    finish = clock();
    cout << float(finish - start)/CLOCKS_PER_SEC << "s\n";

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
    #pragma omp parallel for
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

void transpose(double** matrix, int size) {
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