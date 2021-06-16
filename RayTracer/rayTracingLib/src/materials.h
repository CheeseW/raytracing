#include "utilities.h"

namespace Materials{
	enum materials {none, skybox, lambertian_s};

	template<materials MAT=none> 
	rayUtilities::Color ray_color(const rayUtilities::Ray& r, const rayUtilities::Hittable& world, const int depth);

	template<>
	rayUtilities::Color ray_color<none>(const rayUtilities::Ray& r, const rayUtilities::Hittable& world, const int depth) {
		using namespace rayUtilities;
		if (depth < 0)
			return Color{ 0,0,0 };
		HitRecord rec;
		double thresh = 0.001;
		if (world.hit(r, thresh, INFINITY_R, rec)) {
			const auto target = rec.p + rec.normal + radomInUnitSphere();
			return 0.5 * ray_color(Ray{ rec.p, target - rec.p }, world, depth - 1);
		}
		else {
			Vec3 direction = r.direction().normalized();
			const auto t = .5 * (direction[1] + 1);
			const Color top{ .5,.7,1 };
			const Color bottom{ 1,1,1 };
			return (1 - t) * bottom + t * top;
		}
	}
}