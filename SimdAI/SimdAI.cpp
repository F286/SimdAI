#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "core/simd.h"
#include "core/tensor.h"

TEST_CASE("Tensor operations", "[tensor]") {
    Tensor<float> tensor({ 2, 3, 4 });

    SECTION("Element Access and Modification") {
        //tensor({ 0, 0, 0 }) = 1.0f;
        //tensor({ 1, 2, 3 }) = 2.0f;

        //CHECK(tensor({ 0, 0, 0 }) == Approx(1.0f));
        //CHECK(tensor({ 1, 2, 3 }) == Approx(2.0f));
    }
}

// Additional test cases can be added here, following the same structure.


	// TODO (fd): Split unit tests out into simd, tensor separate classes with catch 2
	
	// TODO (fd): Train simple LLM network, with 'causal attention' within the embedding.
	// So groups (8, simd size) can only 'see' previous groups within the embedding.
	// This possibly could allow us to train many losses per embedding, forcing the network
	// To pack the 'most important' information into the groups that have the least view of everything else.

