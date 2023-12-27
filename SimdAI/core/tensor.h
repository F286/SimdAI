#include "core/simd.h"
#include <immintrin.h>
#include <vector>
#include <cstddef>
#include <cassert>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <array>
#include "mdspan.hpp"

namespace
{
    // For usage with std::extents<>
    [[nodiscard]] constexpr size_t total_elements(auto extents)
    {
        size_t total = 1;
        for (size_t i = 0; i < extents.rank(); ++i) {
            total *= extents.extent(i);
        }
        return total;
    }
}

template<typename T>
class Tensor {
public:
    using Extents = std::dextents<int, 2>;

    // Constructor for a tensor with an initializer list for shape
    Tensor(Extents extents)
        : shape(extents), data(total_elements(extents))
    {
        // Ensure each dimension, particularly the innermost one, is a multiple of the SIMD width.
        //assert(shape.back() % simd<T, 8>::size() == 0 && "Innermost dimension must be a multiple of the SIMD width.");
    }

    // Constructor for a tensor with an initializer list for data
    Tensor(std::initializer_list<std::initializer_list<simd<float>>> data_init)
    {
        // Compute and assign shape
        size_t size_x = 1;
        for (const auto& row : data_init)
        {
            size_x = std::max(size_x, row.size());
        }

        const size_t size_y = data_init.size();
		shape = Extents{ size_y, size_x }; // Least significant dimension is stored towards the right

        // Initialize storage
        storage = std::vector<simd<T>>(total_elements(shape));

        // Populate storage
        auto store_to = data();

        for (size_t y = 0; y < data_init.size(); ++y)
        {
            const auto& row = data_init.begin()[y]; // row major

            for (size_t x = 0; x < row.size(); ++x)
            {
                store_to[y, x] = row.begin()[x];
            }
        }
    }

    // Access an element in the tensor using 2D indices. The least significant bit is store to the very right (like arabic numerals). Therefore order is e.g. [batch, y, x]
    simd<T> operator[](size_t row, size_t column) const {

        return data()[row, column];
    }

    // Access an element in the tensor using 2D indices. The least significant bit is store to the very right (like arabic numerals). Therefore order is e.g. [batch, y, x]
    simd<T>& operator[](size_t row, size_t column) {

        return data()[row, column];
    }

    std::mdspan<simd<T>, Extents> data()
    {
        return std::mdspan(storage.data(), shape);
    }

    Extents shape {};
    //std::vector<std::size_t> shape;

private:
    std::vector<simd<T>> storage;
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
                simd<T> matA_simd = leftMatrix.simd_at(i, (k / simdWidth) * simdWidth);

                // Accumulate the product into the sum
                for (std::size_t l = 0; l < simdWidth; ++l) 
                {
                    if (k + l < leftMatrix.shape[1] && j + l < rightMatrix.shape[1]) 
                    {
                        float r = matA_simd[l] * rightMatrix[k, j];
                        sum[l] = sum[l] + r;
                    }
                }
            }

            // Store the result back into the result tensor
            result.simd_at(i, j) += sum;
        }
    }
    return result;
}