#include "core/simd.h"
#include <immintrin.h>
#include <vector>
#include <cstddef>
#include <cassert>
#include <iostream>
#include <numeric>
#include <algorithm>


template<typename T>
class Tensor {
public:
    // Constructor for a tensor with an initializer list for shape
    Tensor(std::initializer_list<std::size_t> shape_init)
        : shape(shape_init), data(compute_total_size() / simd<T>::size()) {
        // Ensure each dimension, particularly the innermost one, is a multiple of the SIMD width.
        //assert(shape.back() % simd<T, 8>::size() == 0 && "Innermost dimension must be a multiple of the SIMD width.");
    }

    // Access an element in the tensor using multi-dimensional indices
    T operator()(std::initializer_list<std::size_t> indices) {
        return data[offset(indices) / simd<T>::size()][offset(indices) % simd<T>::size()];
    }

    const T operator()(std::initializer_list<std::size_t> indices) const {
        return data[offset(indices) / simd<T>::size()][offset(indices) % simd<T>::size()];
    }

    // Get the shape of the tensor
    const std::vector<std::size_t>& get_shape() const {
        return shape;
    }

private:
    std::vector<std::size_t> shape;
    std::vector<simd<T>> data;

    // Compute the total size (number of elements) of the tensor
    std::size_t compute_total_size() const {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    }

    // Calculate the offset in the flat data vector for multi-dimensional indices
    size_t offset(std::initializer_list<std::size_t> indices) const {
        assert(indices.size() == shape.size());
        size_t off = 0;
        size_t stride = 1;
        for (auto i = indices.begin(); i != indices.end(); ++i) {
            off += *i * stride;
            stride *= shape[std::distance(indices.begin(), i)];
        }
        return off;
    }
};


// Free function for matrix multiplication
template<typename T>
Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
    assert(a.get_cols() == b.get_rows() && "Matrices dimensions must be compatible for multiplication.");

    Tensor<T> result(a.get_rows(), b.get_cols());
    for (std::size_t i = 0; i < a.get_rows(); ++i) {
        for (std::size_t j = 0; j < b.get_cols() / simd<T>::size(); ++j) {
            simd<T> sum = simd<T>::zero();  // Assuming simd class has a zero() method
            for (std::size_t k = 0; k < a.get_cols() / simd<T>::size(); ++k) {
                sum = sum + a(i, k * simd<T>::size()) * b(k, j * simd<T>::size());
            }
            for (std::size_t l = 0; l < simd<T>::size(); ++l) {
                result(i, j * simd<T>::size() + l) = sum[l];  // Assuming simd class has operator[]
            }
        }
    }
    return result;
}