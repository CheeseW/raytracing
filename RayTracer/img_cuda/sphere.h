#pragma once

#include "hittable.h"

namespace rayUtilities {
	class Sphere : public Hittable {
	public: 
		__device__ Sphere() {}
		__device__ Sphere(const Point3& c, const float r) : center(c), radius(r) {}

		__device__ virtual bool hit(const Ray& r, const float t_min, const float t_max, HitRecord& rec) const override;

	private:
		Point3 center;
		float radius;
	};

	__device__ bool Sphere::hit(const Ray& r, const float t_min, const float t_max, HitRecord& rec) const {
		// Compute distance from center to ray
		Vec3 oc = center - r.origin();
		const auto& d = r.direction();

		float a = d.dot(d);
		float b = -d.dot(oc);
		float c = oc.dot(oc) - radius * radius;

		float discriminant = b * b - a * c;
		if (discriminant < 0) return false;

		float rt = sqrt(discriminant);
		auto t = (-b - rt) / a;
		if (t > t_max) return false;

		if (t < t_min) {
			t = (-b + rt) / a;
			if (t > t_max || t < t_min)
				return false;
		}

		rec.t = t;
		rec.p = r.at(t);

		 rec.normal = (rec.p - center)/radius;
		// rec.setFaceNormal(r, normal);
		return true;
	}
}