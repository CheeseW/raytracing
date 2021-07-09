#pragma once

#include "ray.h"
#include <memory>

namespace rayUtilities {
	class Material;

	struct HitRecord {
		Point3 p;
		Vec3 normal;
		double t;
		bool frontFace;

		std::shared_ptr<Material> matPtr;

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