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

    // Constructor for a tensor with an initializer list for data
    Tensor(std::initializer_list<std::initializer_list<T>> data_init)
        : shape({ data_init.size(), data_init.begin()->size() }), data(compute_total_size() / simd<T>::size()) {
        assert(compute_total_size() == std::accumulate(data_init.begin(), data_init.end(), 0, [](size_t a, const std::initializer_list<T>& b) { return a + b.size(); }) && "Size of initializer list must match total size of tensor.");
        assert(shape[1] % simd<T>::size() == 0 && "Width of tensor must be a multiple of SIMD width.");

        auto it = data.begin();
        for (const auto& row : data_init) {
            assert(row.size() == shape[1] && "Size of each row must match the second dimension of the tensor.");
            for (size_t i = 0; i < row.size(); i += simd<T>::size()) {
                *it++ = simd<T>(&row.begin()[i]);
            }
        }
    }

    // Access an element in the tensor using 2D indices
    simd<T>& operator[](size_t row, size_t column) {
        std::size_t index = (row * shape[1] + column) / simd<T>::size();
        std::size_t element = column % simd<T>::size();

        if (element % simd<T>::size() != 0)
        {
            throw std::invalid_argument("Element must divide cleanly into size: column = (" + std::to_string(column) + "), simd<T>::size() = (" + std::to_string(simd<T>::size()) + ").");
        }

        return data[index];
    }

    // Access an element in the tensor using 2D indices
    simd<T> operator[](size_t row, size_t column) const {
        std::size_t index = (row * shape[1] + column) / simd<T>::size();
        std::size_t element = column % simd<T>::size();

        if (element % simd<T>::size() != 0) 
        {
            throw std::invalid_argument("Element must divide cleanly into size: column = (" + std::to_string(column) + "), simd<T>::size() = (" + std::to_string(simd<T>::size()) + ").");
        }

        assert(element % simd<T>::size() == 0 && "Element index must be aligned with SIMD width.");
        return data[index];
    }

    std::vector<std::size_t> shape;

private:
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



// Multiply function for two 2D Tensors
template<typename T>
Tensor<T> multiply(const Tensor<T>& leftMatrix, const Tensor<T>& rightMatrix) {
	if (leftMatrix.shape[1] != rightMatrix.shape[0]) {
		if (leftMatrix.shape[1] != rightMatrix.shape[0]) {
			throw std::invalid_argument("Matrix dimensions must be compatible for multiplication: left matrix shape = (" + std::to_string(leftMatrix.shape[0]) + ", " + std::to_string(leftMatrix.shape[1]) + "), right matrix shape = (" + std::to_string(rightMatrix.shape[0]) + ", " + std::to_string(rightMatrix.shape[1]) + ").");
		}
	}

	// Creating the result tensor with dimensions derived from the input matrices
    Tensor<T> result({ leftMatrix.shape[0], rightMatrix.shape[1] });

    // Compute multiplication using simd operations for efficiency
    const std::size_t simdWidth = simd<T>::size();
    for (std::size_t i = 0; i < leftMatrix.shape[0]; ++i) 
    {
        for (std::size_t j = 0; j < rightMatrix.shape[1]; j += simdWidth) 
        {
            // Initialize sum as a simd vector of zeros
            simd<T> sum = simd<T>::zero();

            // Iterate over the shared dimension
            for (std::size_t k = 0; k < leftMatrix.shape[1]; ++k) 
            {
                // Load a simd-width vector from the left matrix
                simd<T> matA_simd = leftMatrix[i, k];

                // Accumulate the product into the sum
                for (std::size_t l = 0; l < simdWidth; ++l) 
                {
                    if (k + l < leftMatrix.shape[1] && j + l < rightMatrix.shape[1]) 
                    {
                        float r = reduce(simd<T>{matA_simd[l]} * rightMatrix[k, j + l]);
                        sum[l] = sum[l] + r;
                    }
                }
            }

            // Store the result back into the result tensor
            result[i, j] += sum;
        }
    }
    return result;
}