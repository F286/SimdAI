#include <catch2/catch.hpp>
#include "core/tensor.h"

TEST_CASE("Tensor operations", "[tensor]") {
	Tensor<float> tensor{ { simd<float>{1}, simd<float>{2}, simd<float>{3} } };

	SECTION("Element Access and Modification") {
		CHECK(tensor[0, 0] == simd<float>{ 1});
		CHECK(tensor[0, 1] == simd<float>{ 2});

		tensor[0, 0] = simd<float>{ 4 };
		tensor[0, 1] = simd<float>{ 5 };

		CHECK(tensor[0, 0] == simd<float>{ 4});
		CHECK(tensor[0, 1] == simd<float>{ 5});
	}
}

TEST_CASE("Tensor multiplication with transposed matrix", "[matmul_transposed]") {
	// Define two tensors with multiple simd<float> elements per row/column
	Tensor<float> a = {
		{ simd<float>{1}, simd<float>{2}, simd<float>{3} },
		{ simd<float>{4}, simd<float>{5}, simd<float>{6} }
	};
	Tensor<float> b = {
		{ simd<float>{2}, simd<float>{2}, simd<float>{2} },
		{ simd<float>{2}, simd<float>{2}, simd<float>{2} },
	};

	// Perform matrix multiplication with the second matrix transposed
	Tensor<float> c = matmul_transposed(a, b);

	// Define the expected result tensor
	Tensor<float> expected = {
		{96, 96},
		{240, 240}
	};

	// Check if the result matches the expected tensor
	CHECK(c == expected);
}


TEST_CASE("Tensor multiplication", "[matmul]") {
	Tensor<float> a = {
		{ 1, 2, 3 },
		{ 4, 5, 6 }
	};
	Tensor<float> b = {
		{ 1, 2 },
		{ 1, 2 },
		{ 1, 2 }
	};

	Tensor<float> c = matmul(a, b);

	Tensor<float> expected = {
		{6, 12},
		{15, 30}
	};

	// Check if the result matches the expected tensor
	CHECK(c == expected);
}

TEST_CASE("Tensor constructed from list of floats", "[Tensor]") {
	// Initialize Tensor with a nested initializer list of floats
	Tensor<float> tensor = {
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9}  // This row is shorter and should be padded with zeros
	};

	// Check the elements of the tensor
	CHECK(tensor[0, 0] == simd<float>(1, 2, 3, 4, 0, 0, 0, 0));
	CHECK(tensor[1, 0] == simd<float>(5, 6, 7, 8, 0, 0, 0, 0));
	CHECK(tensor[2, 0] == simd<float>(9, 0, 0, 0, 0, 0, 0, 0));

	// Add more checks if necessary
}

TEST_CASE("Tensor transpose with simd<float>", "[transpose]") {
	// Initialize a 2x2 Tensor with each simd<float> representing 8 values
	Tensor<float> original = {
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 }
	};

	// Perform the transpose operation
	Tensor<float> transposed = transpose(original);

	// Define the expected result of the transpose
	Tensor<float> expected = {
		{ 1, 5 },
		{ 2, 6 },
		{ 3, 7 },
		{ 4, 8 },
	};

	// Check if the transposed tensor matches the expected tensor
	REQUIRE(transposed == expected);
}