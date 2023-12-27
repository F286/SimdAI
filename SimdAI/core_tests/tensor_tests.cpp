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

TEST_CASE("Tensor multiplication is computed", "[multiply]") {

	Tensor<float> a{ { simd<float>{1.f} } };
	Tensor<float> b{ { simd<float>{2.f} }, { simd<float>{2.f} } };

	//Tensor<float> a{ {1, 2}, {3, 4} };
	//Tensor<float> b{ {1, 2}, {3, 4} };
	//Tensor<float> b{ {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
					  //{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
					  //{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
					  //{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f} };

	//Tensor<float> a{ {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f} };
	//Tensor<float> b{ {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
	//				  {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
	//				  {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
	//				  {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}};
	//Tensor<float> c = multiply(a, b);

	//Tensor<float> expected_c{ {90.0f, 100.0f, 110.0f, 120.0f, 130.0f, 140.0f, 150.0f, 160.0f},
	//						   {198.0f, 225.0f, 252.0f, 279.0f, 306.0f, 333.0f, 360.0f, 387.0f},
	//						   {288.0f, 325.0f, 362.0f, 399.0f, 436.0f, 473.0f, 510.0f, 547.0f},
	//						   {360.0f, 405.0f, 450.0f, 495.0f, 540.0f, 585.0f, 630.0f, 675.0f} };

	//REQUIRE(c.shape == expected_c.shape);
	//for (size_t i = 0; i < c.shape[0]; ++i) {
	//	for (size_t j = 0; j < c.shape[1]; ++j) {
	//		//CHECK(c[i, j] == expected_c[i, j]);
	//	}
	//}
}