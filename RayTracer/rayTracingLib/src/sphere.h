#pragma once

#include "hittable.h"
#include <memory>

namespace rayUtilities {
	class Material;

	class Sphere : public Hittable {
	public: 
		Sphere() {}
		Sphere(const Point3& c, const double r, const std::shared_ptr<Material>& m) : 
			center(c), radius(r), matPtr(m){}

		virtual bool hit(const Ray& r, const double t_min, const double t_max, HitRecord& rec) const override;

	private:
		Point3 center;
		double radius;
		std::shared_ptr<Material> matPtr;
	};

	bool Sphere::hit(const Ray& r, const double t_min, const double t_max, HitRecord& rec) const {
		// Compute distance from center to ray
		Vec3 oc = center - r.origin();
		const auto& d = r.direction();

		double a = d.dot(d);
		double b = -2 * d.dot(oc);
		double c = oc.dot(oc) - radius * radius;

		double discriminant = b * b - 4 * a * c;
		double rt = std::sqrt(discriminant);

		if (discriminant < 0) return false;
		auto t = (-b - rt) / 2 / a;
		if (t > t_max) return false;

		if (t < t_min) {
			t = (-b + rt) / 2 / a;
			if (t > t_max || t < t_min)
				return false;
		}

		rec.t = t;
		rec.p = r.at(t);
		Vec3 normal = (rec.p - center)/radius;
		rec.matPtr = matPtr;
		rec.setFaceNormal(r, normal);
		return true;
	}
}