#pragma once
#include "tensor.h"
#include <vector>

class Operation
{
	virtual void forward(const std::vector<Tensor<float>>& input) = 0;
};

class MatmulOperation : Operation
{
	void forward(const std::vector<Tensor<float>>& input) override
	{

	}
};