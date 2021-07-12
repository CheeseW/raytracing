#pragma once
#include <limits>
#include <curand_kernel.h>

// #include <random>

// #include "Vec3.h"

namespace rayUtilities {
	__device__ inline Vec3 randomVec(curandState& randState) {
		return Vec3(curand_uniform(&randState), curand_uniform(&randState), curand_uniform(&randState));
	}

	__device__ inline Vec3 randomVec(const float min, const float max, curandState& randState) {
		const Vec3 allOne(1, 1, 1);
		return randomVec(randState) * (max - min) + min * allOne;
	}

	__device__ inline Vec3 randomInUnitSphere(curandState& randState) {
		while (true) {
			const auto p = randomVec(-1, 1, randState);
			if (p.dot(p) < 1)
				return p;
		}
	}

	__device__ inline Vec3 randomUnitVector(curandState& randState) {
		return randomInUnitSphere(randState).normalized();
	}

	__device__ inline float randomFloat(curandState& randState) {
		return curand_uniform(&randState);
	}

#if 0

	// constants
	const float PI = 3.1415926535898932385;
	inline double degrees2radians(const double d) { return d / 180 * PI; }

	inline double randomDouble() {
		static std::uniform_real_distribution<double> distribution(0, 1);
		static std::mt19937 generator;
		return distribution(generator);
	}
#endif
}