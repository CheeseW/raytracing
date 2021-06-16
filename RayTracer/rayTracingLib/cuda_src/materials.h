#pragma once

#include "utilities.h"

namespace Materials{
	enum materials { skybox, lambertian_s, none};


	__device__ rayUtilities::Color ray_color(const rayUtilities::Ray& ray, const rayUtilities::Hittable* world, const int maxIter, curandState& randState) {
		using namespace rayUtilities;
		const float attenRate = .5;
		const float thresh = 0.001;
		Ray r = ray;
		float att = 1.f;
		HitRecord rec;

		for (int i = 0; i < maxIter; i++) {
			if (world->hit(r, thresh, FLT_MAX, rec)) {
				r = Ray(rec.p, rec.normal + radomInUnitSphere(randState));
				att *= attenRate;
			}
			else {
				Vec3 d = r.direction().normalized();
				auto t = .5f * (d[1] + 1.f);
				const Color top{ .5,.7,1 };
				const Color bottom{ 1,1,1 };
				return (t * top + (1.f - t) * bottom) * att;
			}
		}
		return Vec3(0, 0, 0);
	}	
}