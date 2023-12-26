#include "catch2/catch.hpp"
#include "core/simd.h"

TEST_CASE("SIMD operations", "[simd]") 
{
	SECTION("Sum and Min Reduction") 
	{
		simd<float> x = simd<float>::create_from({ 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f });
		float sum = reduce(x);
		float min_val = reduce_min(x);

		CHECK(sum == Approx(-4.0f));
		CHECK(min_val == Approx(-8.0f));
	}

	SECTION("Conditional Selection") 
	{
		simd<float> x = simd<float>::create_from({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
		simd<float> y = simd<float>::create_from({ 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f });

		simd<float> output = simd_select(x < y, y, x);

		for (size_t i = 0; i < x.size(); ++i) {
			CHECK(output[i] == std::max(x[i], y[i]));
		}
	}

	SECTION("Subscript Operator") 
	{
		simd<float> x = simd<float>::create_from({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
		CHECK(x[2] == Approx(3.0f));
	}
}
