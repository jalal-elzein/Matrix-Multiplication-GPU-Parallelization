#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

using std::vector, std::cout, std::endl;

constexpr char nl{'\n'};

vector<vector<float>> generateRandomMatrix(int rows, int cols) {
    vector<vector<float>> matrix(rows, vector<float>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    return matrix;
}

vector<vector<float>> matrixMultiplication(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    int N = A.size();
    int M = B[0].size();
    int K = B.size();

    vector<vector<float>> C(N, vector<float>(M, 0.0f));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

void printMatrix(const vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int rows = 1000;
    const int cols = 1000;

    vector<vector<float>> A = generateRandomMatrix(rows, cols);
    vector<vector<float>> B = generateRandomMatrix(rows, cols);

    auto start = std::chrono::high_resolution_clock::now();

    vector<vector<float>> C = matrixMultiplication(A, B);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cout << "Calculation done in " << duration.count() << " milliseconds" << nl;

    cout << "Matrix A:" << endl;
    printMatrix(A);

    cout << "Matrix B:" << endl;
    printMatrix(B);

    cout << "Matrix C = A * B:" << endl;
    printMatrix(C);

    return 0;
}