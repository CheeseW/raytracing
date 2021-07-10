#pragma once

#include "utilities.h"

namespace rayUtilities {

	class Material {
	public:
		__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState& randState) const = 0;
	};

	class Lambertian :public Material {
	public:
		__device__ Lambertian(const Color& a): albedo(a) {}
		__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState& randState) const override {
			attenuation = albedo;
			scattered = Ray(rec.p, rec.normal + radomInUnitSphere(randState));
			return true;
		}
	private:
		Color albedo;
	};
	__device__ rayUtilities::Color ray_color(const rayUtilities::Ray& ray, const rayUtilities::Hittable* world, const int maxIter, curandState& randState) {
		using namespace rayUtilities;
		Color attenRate(.8,.8,.8);
		const float thresh = 0.001;
		Ray r = ray;
		Ray scattered;
		Color att(1,1,1);
		HitRecord rec;

		for (int i = 0; i < maxIter; i++) {
			if (world->hit(r, thresh, FLT_MAX, rec)) {
				if (rec.mPtr->scatter(r, rec, attenRate, scattered, randState)) {
					r = scattered;
					att *= attenRate;
				}
				else {
					return Color(0, 0, 0);
				}
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