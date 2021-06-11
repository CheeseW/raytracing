#pragma once
#include <limits>
#include <random>

#include <Eigen/Dense>


namespace rayUtilities {

	using Vec3 = Eigen::Matrix<double, 3, 1>;
	using Point3 = Eigen::Matrix<double, 3, 1>;

	inline Vec3 randomVec() {
		return (Vec3::Random() + Vec3{ 1,1,1 }) / 2.;
	}

	inline Vec3 randomVec(const double min, const double max) {
		const Vec3 allOne{ 1,1,1 };
		return (Vec3::Random() + allOne) / 2. * (max - min) + min * allOne;
	}

	Vec3 radomInUnitSphere() {
		while (true) {
			const auto p = randomVec(-1,1);
			if (p.dot(p) < 1)
				return p;
		}
	}

	// constants
	const double INFINITY_R = std::numeric_limits<double>::infinity();
	const double PI = 3.1415926535898932385;

	inline double degrees2radians(const double d) { return d / 180 * PI; }

	inline double randomDouble() {
		static std::uniform_real_distribution<double> distribution(0, 1);
		static std::mt19937 generator;
		return distribution(generator);
	}
}