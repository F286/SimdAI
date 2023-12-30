#pragma once
#include "tensor.h"
#include <vector>

namespace core
{
	using namespace std;

	/// @brief Tensor value and gradient used for training
	struct TensorWithGradient
	{
		Tensor<float> value;
		Tensor<float> gradient;
	};

	class Operation
	{
		virtual vector<shared_ptr<TensorWithGradient>> forward(const vector<shared_ptr<TensorWithGradient>>& input) = 0;
		virtual vector<shared_ptr<TensorWithGradient>> backward(const vector<TensorWithGradient>& gradient) = 0;
	};

	class MatmulOperation : Operation
	{
		vector<shared_ptr<TensorWithGradient>> forward(const vector<shared_ptr<TensorWithGradient>>& input) override
		{

		}

		vector<shared_ptr<TensorWithGradient>> backward(const vector<shared_ptr<TensorWithGradient>>& gradient) override
		{

		}
	};
}