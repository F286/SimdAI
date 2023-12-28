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
