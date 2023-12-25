#include "core/simd.h"
#include "core/tensor.h"
#include <iostream>
#include "catch2/catch.hpp"

// Example usage
int main() {
	{
		simd<float> x{ 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f };

		float sum = reduce(x);
		float min_val = reduce_min(x);

		std::cout << "Sum: " << sum << std::endl;
		std::cout << "Min: " << min_val << std::endl;
	}

	{

		simd<float> x{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
		simd<float> y{ 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };

		// Perform the conditional selection
		x = simd_select(x < y, y, x);

		// Print the result
		for (size_t i = 0; i < x.size(); ++i) {
			std::cout << x[i] << " ";
		}
		std::cout << std::endl;
	}

	{
		simd<float> x{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };

		// Use the subscript operator
		std::cout << "Third element: " << x[2] << std::endl;

		// Modify an element
		x[2] = 10.0f;
		std::cout << "Modified third element: " << x[2] << std::endl;
	}
	
	{
		Tensor<float> tensor({ 2, 3, 4 }); // Create a 3D tensor

		// Access and modify elements
		tensor({ 0, 0, 0 }) = 1.0f;
		tensor({ 1, 2, 3 }) = 2.0f;

		// Print an element
		std::cout << "Tensor element: " << tensor({ 1, 2, 3 }) << std::endl;

	}

	// TODO (fd): Split unit tests out into simd, tensor separate classes with catch 2
	
	// TODO (fd): Train simple LLM network, with 'causal attention' within the embedding.
	// So groups (8, simd size) can only 'see' previous groups within the embedding.
	// This possibly could allow us to train many losses per embedding, forcing the network
	// To pack the 'most important' information into the groups that have the least view of everything else.

	return 0;
}
