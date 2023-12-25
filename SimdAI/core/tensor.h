#pragma once
#include "core/simd.h"
#include <vector>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <cassert>

template<typename T>
class Tensor {
public:
    // Constructor for a tensor given its shape
    Tensor(std::initializer_list<std::size_t> shape) : shape(shape.begin(), shape.end()) {
        // Compute the total size and allocate memory
        std::size_t total_size = std::accumulate(this->shape.begin(), this->shape.end(),
            1, std::multiplies<std::size_t>());
        data = std::make_unique<T[]>(total_size);
        compute_strides();
    }

    // Access an element in the tensor
    T& operator()(std::initializer_list<std::size_t> indices) {
        assert(indices.size() == shape.size());
        std::size_t offset = 0;
        std::size_t i = 0;
        for (auto index : indices) {
            assert(index < shape[i]); // Bounds check
            offset += index * strides[i++];
        }
        return data[offset];
    }

    // Get the shape of the tensor
    const std::vector<std::size_t>& get_shape() const {
        return shape;
    }

    // ... Additional methods for tensor operations ...

private:
    std::unique_ptr<T[]> data; // Contiguous block of memory for tensor data
    std::vector<std::size_t> shape; // Shape of the tensor
    std::vector<std::size_t> strides; // Strides for indexing

    // Compute strides based on shape
    void compute_strides() {
        strides.resize(shape.size());
        std::size_t stride = 1;
        for (std::size_t i = shape.size(); i > 0; --i) {
            strides[i - 1] = stride;
            stride *= shape[i - 1];
        }
    }
};
