#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <functional>
#include <initializer_list>

// Forward declare the simd class
template<typename T, std::size_t N>
class simd;

// Define the simd_mask class
template<typename T, std::size_t N>
class simd_mask {
public:
    simd_mask(__m256i mask) : mask(mask) {}

    __m256i get() const {
        return mask;
    }

private:
    __m256i mask;
};

// simd class definition
template<typename T, std::size_t N = 8>
class simd {
public:
    simd() : data(_mm256_setzero_ps()) {}

    simd(std::initializer_list<T> init) {
        assert(init.size() == N);
        alignas(32) T temp[N];
        std::copy(init.begin(), init.end(), temp);
        data = _mm256_loadu_ps(temp);
    }

    explicit simd(__m256 val) : data(val) {}

    static simd load(const T* data) {
        return simd(_mm256_loadu_ps(data));
    }

    void store(T* dest) const {
        _mm256_storeu_ps(dest, data);
    }

    simd operator+(const simd& other) const {
        return simd(_mm256_add_ps(data, other.data));
    }

    simd operator-(const simd& other) const {
        return simd(_mm256_sub_ps(data, other.data));
    }

    simd operator*(const simd& other) const {
        return simd(_mm256_mul_ps(data, other.data));
    }

    simd operator/(const simd& other) const {
        return simd(_mm256_div_ps(data, other.data));
    }

    simd_mask<T, N> operator<(const simd& other) const {
        return simd_mask<T, N>(_mm256_castps_si256(_mm256_cmp_ps(data, other.data, _CMP_LT_OS)));
    }

    __m256 get() const {
        return data;
    }

    constexpr std::size_t size() const {
        return N;
    }

    T operator[](std::size_t index) const {
        assert(index < N);
        alignas(32) T temp[N];
        _mm256_storeu_ps(temp, data);
        return temp[index];
    }

    // Setting an element is more complex and would usually not be recommended for SIMD types.
    // If absolutely needed, it can be implemented using intrinsics, but it's costly in terms of performance.

private:
    __m256 data;
};

// Reduce operations and other functions remain the same


// Reduce operation for sum
template<typename T, std::size_t N>
T reduce(const simd<T, N>& v) {
	__m256 temp = v.get();
	temp = _mm256_hadd_ps(temp, temp);
	temp = _mm256_hadd_ps(temp, temp);
	__m128 high128 = _mm256_extractf128_ps(temp, 1);
	__m128 dotproduct = _mm_add_ps(_mm256_castps256_ps128(temp), high128);
	return _mm_cvtss_f32(dotproduct);
}

// Reduce operation for minimum value
template<typename T, std::size_t N>
T reduce_min(const simd<T, N>& v) {
	__m256 temp = v.get();
	temp = _mm256_min_ps(temp, _mm256_permute2f128_ps(temp, temp, 0x1));
	temp = _mm256_min_ps(temp, _mm256_shuffle_ps(temp, temp, _MM_SHUFFLE(1, 0, 3, 2)));
	temp = _mm256_min_ps(temp, _mm256_shuffle_ps(temp, temp, _MM_SHUFFLE(2, 3, 0, 1)));
	return _mm_cvtss_f32(_mm256_castps256_ps128(temp));
}

// Free function for conditional selection based on simd_mask
template<typename T, std::size_t N>
simd<T, N> simd_select(const simd_mask<T, N>& cond, const simd<T, N>& a, const simd<T, N>& b) {
	// Cast the __m256i mask to __m256 to use with _mm256_blendv_ps
	__m256 mask_ps = _mm256_castsi256_ps(cond.get());
	return simd<T, N>(_mm256_blendv_ps(b.get(), a.get(), mask_ps));
}
