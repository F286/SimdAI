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

    /// @brief Rounds a number's size up to given SIMD width
    auto round_up_to_simd_width = [simd_size = simd<float>::size()](size_t size) {
        return ((size + simd_size - 1) / simd_size) * simd_size;
    };

    /// @brief Calculates the minimum size in SIMD width units that contains a given number of elements
    auto elements_to_simd_units = [simd_size = simd<float>::size()](size_t elements) {
        return (elements + simd_size - 1) / simd_size;
    };
}

using Extents = std::dextents<int, 2>;

template<typename T>
class Tensor {
public:

    // Constructor for a tensor with an initializer list for extents (shape)
    Tensor(Extents extents)
        : storage(total_elements(extents)), extents(extents)
    {
        // Ensure each dimension, particularly the innermost one, is a multiple of the SIMD width.
        //assert(shape.back() % simd<T, 8>::size() == 0 && "Innermost dimension must be a multiple of the SIMD width.");
    }

    // Constructor for a tensor with an initializer list for data
    Tensor(std::initializer_list<std::initializer_list<simd<float>>> data_init)
    {
        initialize(std::move(data_init));
    }

    // Constructor taking a nested initializer list of floats
    Tensor(std::initializer_list<std::initializer_list<float>> init_list) 
    {
        constexpr size_t simd_width = simd<float>::size();
        std::vector<std::vector<simd<float>>> simdInitList;

        for (const auto& row : init_list) 
        {
            std::vector<simd<float>> simdRow;

            for (size_t i = 0; i < row.size(); i += simd_width) 
            {
                float values[simd_width] = {}; // Temporary array for SIMD values, initialized to zeros

                // Fill the array with values from the initializer list, pad with zeros if necessary
                for (size_t j = 0; j < simd_width && i + j < row.size(); ++j) {
                    values[j] = *(row.begin() + i + j);
                }

                // Create a simd<float> and add to the row
                simdRow.emplace_back(values[0], values[1], values[2], values[3],
                    values[4], values[5], values[6], values[7]);
            }

            simdInitList.push_back(std::move(simdRow));
        }

        // Call the existing constructor with simd<float> initializer list
        initialize(std::move(simdInitList));
    }

    // Access an element in the tensor using 2D indices. The least significant bit is store to the very right (like arabic numerals). Therefore order is e.g. [batch, y, x]
    simd<T> operator[](size_t row, size_t column) const {

        return data()[row, column];
    }

    // Access an element in the tensor using 2D indices. The least significant bit is store to the very right (like arabic numerals). Therefore order is e.g. [batch, y, x]
    simd<T>& operator[](size_t row, size_t column) {

        return data()[row, column];
    }

    std::mdspan<const simd<T>, Extents> data() const
    {
        return std::mdspan(storage.data(), extents);
    }

    std::mdspan<simd<T>, Extents> data()
    {
        return std::mdspan(storage.data(), extents);
    }

    // Gets the shape of the tensor at a given rank. Specified in order starting from zero and increasing, from least significant (very right) to most significant (very left)
    [[nodiscard]] int shape(int rank) const
    {
        // The mdspan is storing dimensions with the highest rank being the least significant. If negative, we need to access from least significant to match Pytorch syntax.

        // Calculate adjusted rank. If rank is negative, Extents::rank() is added; otherwise, it remains unchanged.
        // This is achieved by using the fact that (rank >= 0) evaluates to 1 for non-negative ranks and 0 for negative ranks.
        rank += (rank < 0) * Extents::rank();

        return extents.extent(rank);
    }

	bool operator==(const Tensor& other) const
	{
		return extents == other.extents && storage == other.storage;
	}
    bool operator!=(const Tensor& other) const
    {
        return !(*this == other);
    }

private:

    // Constructor for a tensor with an initializer list for data
    template<class NestedList = std::initializer_list<std::initializer_list<simd<float>>>>
    void initialize(NestedList data_init)
    {
        // Compute and assign extents
        size_t size_x = 1;
        for (const auto& row : data_init)
        {
            size_x = std::max(size_x, row.size());
        }

        size_t size_y = data_init.size();

        // Shape 'rounds up' to SIMD size
        //size_x = round_up_to_simd_width(size_x);
        size_y = round_up_to_simd_width(size_y);

        extents = Extents{ size_y, size_x }; // Least significant dimension is stored towards the right

        // Initialize storage
        storage = std::vector<simd<T>>(total_elements(extents));

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

	std::vector<simd<T>> storage;
	Extents extents{};
};

// Multiply function for two 2D Tensors

template<typename T>
Tensor<T> matmul_transposed(const Tensor<T>& leftMatrix, const Tensor<T>& rightMatrixTransposed) {
    // Ensure the inner dimensions match for multiplication
    if (leftMatrix.shape(-1) != rightMatrixTransposed.shape(-1)) {
        throw std::invalid_argument("Matrix dimensions must be compatible for multiplication.");
    }

    // Initialize the result tensor
    Tensor<T> result(Extents{ leftMatrix.shape(-2), elements_to_simd_units(rightMatrixTransposed.shape(-2)) });

    constexpr size_t simd_width = simd<float>::size();

    for (size_t i = 0; i < leftMatrix.shape(-2); i++) {
        for (size_t j = 0; j < rightMatrixTransposed.shape(-2); j++) {
            // Initialize sum as a simd vector of zeros
            simd<T> sum = simd<T>::zero();

            // Perform dot product along contiguous memory
            for (size_t k = 0; k < leftMatrix.shape(-1); k++) {
                simd<T> leftValue = leftMatrix[i, k]; // Accessing individual SIMD block
                simd<T> rightValue = rightMatrixTransposed[j, k]; // Accessing individual SIMD block

                // Dot product and accumulate using SIMD
                sum += leftValue * rightValue;
            }

            // Reduce the SIMD vector to a single value and set it in the result tensor
            result[i, j / simd_width][j % simd_width] = reduce(sum);
        }
    }

    return result;
}


template<typename T>
Tensor<T> transpose(const Tensor<T>& tensor)
{
    constexpr size_t simd_width = simd<float>::size();

    const size_t rows = tensor.shape(-2);
    const size_t cols = tensor.shape(-1);

    // Ensure the tensor is compatible with the SIMD width
    if (rows % simd_width != 0) {
        throw std::invalid_argument("Number of columns must be a multiple of SIMD width.");
    }

    Tensor<T> transposedTensor(Extents{ cols * simd_width, rows / simd_width });

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; j += simd_width) {
            // Accessing SIMD blocks and transposing them
            simd<T> block = tensor[i, j / simd_width];

            for (size_t k = 0; k < simd_width; ++k) {
                // Transpose the individual elements in the SIMD block
                // Assuming the Tensor class provides a method to set individual elements
                transposedTensor[j + k, i / simd_width][i % simd_width] = block[k];
            }
        }
    }

    return transposedTensor;
}
