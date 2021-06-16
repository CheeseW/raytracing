#pragma once

#include "utilities.h"
#include "ray.h"

namespace rayUtilities {
	class Camera {
	public:
		__device__ Camera(const int width, const int height) {
			// const auto aspectRatio = float(width) / height;
			const auto viewportHeight = 2.;
			const auto viewportWidth = viewportHeight / height * width;
			auto focal_length = 1.;

			origin = { 0,0,0 };
			horizontal = { viewportWidth,0,0 };
			vertical = { 0, viewportHeight, 0 };

			lowerLeft = origin - horizontal / 2 - vertical / 2 - rayUtilities::Vec3(0, 0, focal_length);

			horizontal /= width;
			vertical /= height;
		}

		__device__ inline Ray getRay(const int u, const int v, curandState& randState) const {
			return Ray(origin, lowerLeft + (u + rayUtilities::randomFloat(randState)) * horizontal + (v + rayUtilities::randomFloat(randState)) * vertical - origin);
		}

	private:
		Point3 origin;
		Point3 lowerLeft;
		Vec3 horizontal;
		Vec3 vertical;
	};
}