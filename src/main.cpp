#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cblas.h>

template <typename T>
void trsm(int M, int N, T alpha, const T* A, int lda, T* B, int ldb) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            T sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += A[i + k * lda] * B[k + j * ldb];
            }
            B[i + j * ldb] = (alpha * B[i + j * ldb] - sum) / A[i + i * lda];
        }
    }
}

double geometric(const std::vector<double>& values) {
    double log_sum = 0.0;
    for (double v : values) log_sum += std::log(v);
    return std::exp(log_sum / values.size());
}

int main() {
    const int size = 4200;
    const double alpha = 1.0;
    const int iterations = 10;
    std::vector<int> threads_list = {1, 2, 4, 8, 16};

    std::vector<double> A(size * size, 1.0);
    std::vector<double> B_orig(size * size, 2.0);

    for (int i = 0; i < size; ++i) {
        A[i + i * size] = size * 2.0;
    }

    std::cout << "Подсчет матрицы размером " << size << std::endl;

    std::cout << "\n[1/2] Замер самописной реализации..." << std::endl;
    std::vector<double> my_times;
    for (int i = 0; i < iterations; ++i) {
        std::vector<double> B_my = B_orig;
        auto start = std::chrono::high_resolution_clock::now();
        trsm(size, size, alpha, A.data(), size, B_my.data(), size);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        my_times.push_back(duration);
        std::cout << "  Итерация " << i + 1 << ": " << std::fixed << std::setprecision(4) << duration << " сек" << std::endl;
    }
    double myGeom = geometric(my_times);
    std::cout << ">> Самописное среднее геом: " << myGeom << " сек" << std::endl;

    std::cout << "\n[2/2] Сравнение с OpenBLAS по потокам..." << std::endl;
    
    for (int t : threads_list) {
        openblas_set_num_threads(t);
        std::vector<double> blas_times;
        std::cout << "\n--- Потоков: " << t << " ---" << std::endl;

        for (int i = 0; i < iterations; ++i) {
            std::vector<double> B_blas = B_orig;
            auto start = std::chrono::high_resolution_clock::now();
            
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                        size, size, alpha, A.data(), size, B_blas.data(), size);
            
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(end - start).count();
            
            blas_times.push_back(duration);
            std::cout << "  Итерация " << i + 1 << ": " << std::fixed << std::setprecision(6) << duration << " сек" << std::endl;
        }

        double blasGeom = geometric(blas_times);
        double performance_pct = (blasGeom / myGeom) * 100.0;

        std::cout << "Среднее геом. (BLAS): " << blasGeom << " сек" << std::endl;
        std::cout << "Относительная производительность: " << performance_pct << "%" << std::endl;
    }

    return 0;
}