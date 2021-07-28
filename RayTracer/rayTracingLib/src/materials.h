#pragma once

#include "utilities.h"
#include "ray.h"
#include "hittable.h"

namespace rayUtilities {
	struct HitRecord;
	Vec3 reflect(const Vec3& v, const Vec3& n) {
		return v - 2 * v.dot(n) * n;
	}

	Vec3 refract(const Vec3& v, const Vec3& n, const double etai_over_etao) {
		Vec3 R_perp = etai_over_etao * (v - (v.dot(n) * n));
		const double delta = 1 - R_perp.dot(R_perp);
		assert(delta >= 0);
		return R_perp - std::sqrt(delta) * n;
	}

	class Material {
	public:
		virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const = 0;
	};

	class Dielectric : public Material {
	public:
		Dielectric(const double ri) : refractionIdx(ri) {}

		virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
			const double refract_ratio = rec.frontFace ? 1 / refractionIdx : refractionIdx;
			const Vec3 dirIn = r_in.direction().normalized();
			double cosine = dirIn.dot(-rec.normal);
			bool canRefract = 1 / refract_ratio > (1 - cosine * cosine);
			/*
			if (!canRefract|| reflectance(cosine, refract_ratio) > randomDouble())
				scattered = Ray(rec.p, reflect(dirIn, rec.normal));
			else
			*/
			if (canRefract)
				scattered = Ray(rec.p, refract(dirIn, rec.normal, refract_ratio));
			else
				scattered = Ray(rec.p, reflect(dirIn, rec.normal));


			attenuation = Color{ 1,1,1 };
			return true;
		}
	private:

		static double reflectance(const double cosine, const double rIndex) {
			double r0 = (1 - rIndex) / (1 + rIndex);
			r0 = r0 * r0;
			return r0 + (1 - r0) * std::pow(1 - cosine, 5);
		}
		double refractionIdx;
	};

	class Metal :public Material {
	public:
		Metal(const Color& a, const float f) :albedo(a), fuzz(f<1?f:1) {}

		virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
			rayUtilities::Vec3 reflected = reflect(r_in.direction(), rec.normal.normalized()) + fuzz*randomInUnitSphere();
			double thresh = 1e-10;
			scattered = Ray(rec.p, reflected);
			attenuation = albedo;
			return (reflected.dot(rec.normal)) > 0;
		}
	private:
		Color albedo;
		double fuzz;
	};

	class Lambertian :public Material {
	public:
		Lambertian(const Color& a) :albedo(a) {}

		virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
			rayUtilities::Vec3 scatterDir = rec.normal + randomUnitVector();
			double thresh = 1e-10;
			if (scatterDir.dot(scatterDir) < thresh)
				scatterDir = rec.normal;
			scattered = Ray(rec.p, scatterDir);
			attenuation = albedo;
			return true;
		}
	private:
		Color albedo;
	};
}

namespace Materials{
	enum materials { skybox, lambertian, diffuse, none};

	template<materials MAT=none> 
	rayUtilities::Color ray_color(const rayUtilities::Ray& r, const rayUtilities::Hittable& world, const int depth);

	template<>
	rayUtilities::Color ray_color<skybox>(const rayUtilities::Ray& r, const rayUtilities::Hittable& world, const int depth) {
		using namespace rayUtilities;
		Vec3 direction = r.direction().normalized();
		const auto t = .5 * (direction[1] + 1);
		const Color top{ .5,.7,1 };
		const Color bottom{ 1,1,1 };
		return (1 - t) * bottom + t * top;
	}

	template<>
	rayUtilities::Color ray_color<none>(const rayUtilities::Ray& r, const rayUtilities::Hittable& world, const int depth) {
		using namespace rayUtilities;
		if (depth < 0)
			return Color{ 0,0,0 };
		HitRecord rec;
		double thresh = 0.001;
		if (world.hit(r, thresh, INFINITY_R, rec)) {
			Ray scattered;
			Color attenuation;
			if (rec.matPtr->scatter(r, rec, attenuation, scattered))
				return attenuation.array() * ray_color(scattered, world, depth - 1).array();
			else return Color(0, 0, 0);
		}
		else return ray_color<skybox>(r, world, depth);
	}
}

