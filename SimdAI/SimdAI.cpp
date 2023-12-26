#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "core/simd.h"
#include "core/tensor.h"


// Additional test cases can be added here, following the same structure.


	// TODO (fd): Split unit tests out into simd, tensor separate classes with catch 2
	
	// TODO (fd): Train simple LLM network, with 'causal attention' within the embedding.
	// So groups (8, simd size) can only 'see' previous groups within the embedding.
	// This possibly could allow us to train many losses per embedding, forcing the network
	// To pack the 'most important' information into the groups that have the least view of everything else.

