#include <catch2/catch.hpp>
#include "core/tensor.h"

TEST_CASE("Tensor operations", "[tensor]") {
	Tensor<float> tensor{ { simd<float>{1.f}, simd<float>{2.f}, simd<float>{3.f} } };

	SECTION("Element Access and Modification") {
		CHECK(tensor[0, 0] == simd<float>{ 1.0f});
		CHECK(tensor[0, 1] == simd<float>{ 2.0f});

		tensor[0, 0] = simd<float>{ 4.0f };
		tensor[0, 1] = simd<float>{ 5.0f };

		CHECK(tensor[0, 0] == simd<float>{ 4.0f});
		CHECK(tensor[0, 1] == simd<float>{ 5.0f});
	}
}

TEST_CASE("Tensor multiplication with transposed matrix is computed", "[matmul_transposed]") {
    // Define two tensors with multiple simd<float> elements per row/column
    Tensor<float> a = {
        { simd<float>{1.0f}, simd<float>{2.0f}, simd<float>{3.0f} },
        { simd<float>{4.0f}, simd<float>{5.0f}, simd<float>{6.0f} }
    };
    Tensor<float> b = {
        { simd<float>{2.0f}, simd<float>{2.0f}, simd<float>{2.0f} },
        { simd<float>{2.0f}, simd<float>{2.0f}, simd<float>{2.0f} },
        { simd<float>{2.0f}, simd<float>{2.0f}, simd<float>{2.0f} },
        { simd<float>{2.0f}, simd<float>{2.0f}, simd<float>{2.0f} },
        { simd<float>{2.0f}, simd<float>{2.0f}, simd<float>{2.0f} },
        { simd<float>{2.0f}, simd<float>{2.0f}, simd<float>{2.0f} },
        { simd<float>{2.0f}, simd<float>{2.0f}, simd<float>{2.0f} },
        { simd<float>{2.0f}, simd<float>{2.0f}, simd<float>{2.0f} },
    };

    // Perform matrix multiplication with the second matrix transposed
    Tensor<float> c = matmul_transposed(a, b);

    // Define the expected result tensor
    Tensor<float> expected = {
        { simd<float>{96.f} },
        { simd<float>{240.f} }
    };

    // Check if the result matches the expected tensor
    CHECK(c == expected);
}

TEST_CASE("Tensor constructed from list of floats", "[Tensor]") {
    // Initialize Tensor with a nested initializer list of floats
    Tensor<float> tensor = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f}  // This row is shorter and should be padded with zeros
    };

    // Check the elements of the tensor
    CHECK(tensor[0, 0] == simd<float>(1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f));
    CHECK(tensor[1, 0] == simd<float>(5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f));
    CHECK(tensor[2, 0] == simd<float>(9.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));

    // Add more checks if necessary
}

TEST_CASE("Tensor transpose with simd<float>", "[.offbydefault][transpose]") {
    // Initialize a 2x2 Tensor with each simd<float> representing 8 values
    Tensor<float> original = {
        { simd<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
          simd<float>{9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f} },
        { simd<float>{17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
          simd<float>{25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f} }
    };

    // Perform the transpose operation
    Tensor<float> transposed = transpose(original);

    // Define the expected result of the transpose
    Tensor<float> expected = {
        { simd<float>{1.0f, 9.0f, 17.0f, 25.0f, 2.0f, 10.0f, 18.0f, 26.0f},
          simd<float>{3.0f, 11.0f, 19.0f, 27.0f, 4.0f, 12.0f, 20.0f, 28.0f} },
        { simd<float>{5.0f, 13.0f, 21.0f, 29.0f, 6.0f, 14.0f, 22.0f, 30.0f},
          simd<float>{7.0f, 15.0f, 23.0f, 31.0f, 8.0f, 16.0f, 24.0f, 32.0f} }
    };

    // Check if the transposed tensor matches the expected tensor
    REQUIRE(transposed == expected);
}