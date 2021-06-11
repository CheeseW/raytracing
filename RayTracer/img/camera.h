#pragma once

#include "utilities.h"
#include "ray.h"

namespace rayUtilities {
	class Camera {
	public:
		Camera() {
			const auto aspectRatio = 16. / 9.;
			const auto viewportHeight = 2.;
			const auto viewportWidth = aspectRatio * viewportHeight;
			auto focal_length = 1.;

			origin = { 0,0,0 };
			horizontal = { viewportWidth,0,0 };
			vertical = { 0, viewportHeight, 0 };

			lowerLeft = origin - horizontal / 2 - vertical / 2 - rayUtilities::Vec3(0, 0, focal_length);
		}

		Ray getRay(const double u, const double v) const {
			return Ray(origin, lowerLeft + u * horizontal + v * vertical - origin);
		}

	private:
		Point3 origin;
		Point3 lowerLeft;
		Vec3 horizontal;
		Vec3 vertical;
	};
}