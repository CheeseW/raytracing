#pragma once

#include "ray.h"
namespace rayUtilities {
	struct HitRecord {
		Point3 p;
		Vec3 normal;
		double t;
		bool frontFace;

		inline void setFaceNormal(const Ray& r, const Vec3& outwardNormal) {
			frontFace = r.direction().dot(outwardNormal) <= 0;
			normal = frontFace ? outwardNormal : -outwardNormal;
		}
	};

	class Hittable {
	public:
		virtual bool hit(const Ray& r, const double tMin, const double tMax, HitRecord& rec) const = 0;

	};
}