#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iomanip>

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
    explicit simd() : data(_mm256_setzero_ps()) {}

    explicit simd(const T* mem) {
        data = _mm256_loadu_ps(mem);
    }

    explicit simd(T val) : data(_mm256_set1_ps(val)) {}

    simd(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
        data = _mm256_set_ps(f7, f6, f5, f4, f3, f2, f1, f0);
    }

    explicit simd(__m256 val) : data(val) {}

    // Copy constructor
    simd(const simd& other) : data(other.data) {}

    // Copy assignment operator
    simd& operator=(const simd& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }


    // Static zero method
    static simd zero() {
        return simd(_mm256_setzero_ps());
    }

    static simd create_from(std::initializer_list<T> init) {
        if (init.size() != N) {
            throw std::invalid_argument("Invalid size for simd from initializer list: size = (" + std::to_string(init.size()) + "), N = (" + std::to_string(N) + ").");
        }

        return simd(_mm256_loadu_ps(&*init.begin()));
    }

    static simd copy_from(const T* data) {
        return simd(_mm256_loadu_ps(data));
    }

    void copy_to(T* dest) const {
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

    // Compound assignment operators
    simd& operator+=(const simd& other) {
        data = _mm256_add_ps(data, other.data);
        return *this;
    }

    simd& operator-=(const simd& other) {
        data = _mm256_sub_ps(data, other.data);
        return *this;
    }

    simd& operator*=(const simd& other) {
        data = _mm256_mul_ps(data, other.data);
        return *this;
    }

    simd& operator/=(const simd& other) {
        data = _mm256_div_ps(data, other.data);
        return *this;
    }

    friend bool operator==(const simd& lhs, const simd& rhs) {
        // Compare the two __m256 values
        __m256 cmp = _mm256_cmp_ps(lhs.data, rhs.data, _CMP_EQ_OQ);

        // Check if all elements are equal
        return (_mm256_movemask_ps(cmp) == 0xFF);
    }

    friend bool operator!=(const simd& lhs, const simd& rhs) {
        // Use the == operator and invert the result
        return !(lhs == rhs);
    }

    __m256 get() const {
        return data;
    }

    static constexpr std::size_t size() {
        return N;
    }

    T operator[](std::size_t index) const {
        assert(index < N);
        alignas(32) T temp[N];
        _mm256_storeu_ps(temp, data);
        return temp[index];
    }

    // Mutable subscript reference operator
    T& operator[](std::size_t index) {
        assert(index < N);
        float* raw = reinterpret_cast<float*>(&data);
        return *(raw + index);
    }

    // Used for debug output
    friend std::ostream& operator<<(std::ostream& os, const simd& obj) 
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            os << std::right << std::setw(8) << obj[i];
        }
        return os;
    }


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
