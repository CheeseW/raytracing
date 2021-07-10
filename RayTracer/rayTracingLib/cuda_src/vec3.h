#pragma once
#include <iostream>
namespace rayUtilities {

	class Vec3 {
	public:
		__host__ __device__ Vec3() {}
		__host__ __device__ Vec3(float e0, float e1, float e2) : e{ e0,e1,e2 } {}

		__host__ __device__ inline float x() const { return e[0]; }
		__host__ __device__ inline float y() const { return e[1]; }
		__host__ __device__ inline float z() const { return e[2]; }

		__host__ __device__ inline float const& operator[](int i) const { /*assert(i >= 0 && i < 3);*/ return e[i]; }
		__host__ __device__ inline float& operator[](int i) { /*assert(i >= 0 && i < 3);*/ return e[i]; }

		__host__ __device__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
		__host__ __device__ Vec3& operator+=(const Vec3& v) {
			e[0] += v[0];
			e[1] += v[1];
			e[2] += v[2];
			return *this;
		}

		__host__ __device__ Vec3& operator*=(const float v) {
			e[0] *= v;
			e[1] *= v;
			e[2] *= v;
			return *this;
		}

		__host__ __device__ Vec3& operator*=(const Vec3& v) {
			e[0] *= v[0];
			e[1] *= v[1];
			e[2] *= v[2];
			return *this;
		}

		__host__ __device__ Vec3& operator/=(const float t) {
			return *this *= 1 / t;
		}

		__host__ __device__ inline Vec3 operator+(const Vec3& v) const {
			return Vec3(e[0] + v[0], e[1] + v[1], e[2] + v[2]);
		}

		__host__ __device__ inline Vec3 operator-(const Vec3& v) const {
			return operator+(-v);
		}

		__host__ __device__ inline Vec3 operator*(float t) const {
			return Vec3(e[0] * t, e[1] * t, e[2] * t);
		}

		__host__ __device__ inline Vec3 operator*(Vec3& v) const {
			return Vec3(e[0] * v[0], e[1] * v[1], e[2] * v[2]);
		}

		__host__ __device__ inline Vec3 operator/(float t) const {
			return  operator*(1 / t);
		}


		__host__ __device__ inline float dot(const Vec3& v) const {
			return v[0] * e[0] + v[1] * e[1] + v[2] * e[2];
		}

		__host__ __device__ inline Vec3 cross(const Vec3& v) const {
			return Vec3(e[1] * v[2] - e[2] * v[1],
				e[2] * v[0] - e[0] * v[2],
				e[0] * v[1] - e[1] * v[0]
			);
		}

		__host__ __device__ inline Vec3 normalized() {
			return operator/(norm());
		}

		__host__ __device__ float norm() {
			return std::sqrt(dot(*this));
		}

	private:
		float e[3];
	};


	inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
		return out << v[0] << " " << v[1] << " " << v[2];
	}


	__host__ __device__ inline Vec3 operator*(float t, const Vec3& u) {
		return u * t;
	}

	using Point3 = Vec3;
	using Color = Vec3;
}
