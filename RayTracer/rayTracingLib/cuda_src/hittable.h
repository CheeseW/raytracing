#pragma once

#include "ray.h"
namespace rayUtilities {
	struct HitRecord {
		Point3 p;
		Vec3 normal;
		float t;
		bool frontFace;

		__device__ inline void setFaceNormal(const Ray& r, const Vec3& outwardNormal) {
			frontFace = r.direction().dot(outwardNormal) <= 0;
			normal = frontFace ? outwardNormal : -outwardNormal;
		}
	};

	class Hittable {
	public:
		__device__ virtual bool hit(const Ray& r, const float tMin, const float tMax, HitRecord& rec) const = 0;

	};
}