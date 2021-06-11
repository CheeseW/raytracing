#include <iostream>
#include <fstream>

#include "color.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include "utilities.h"

#if 1
rayUtilities::Color ray_color(const rayUtilities::Ray& r, const rayUtilities::Hittable& world, const int depth) {
	using namespace rayUtilities;
	if (depth < 0)
		return Color{ 0,0,0 };
	HitRecord rec;
	double thresh = 0.001;
	if (world.hit(r, thresh, INFINITY_R, rec)) {
		const auto target = rec.p + rec.normal + radomInUnitSphere();
		return 0.5 * ray_color(Ray{ rec.p, target - rec.p }, world, depth-1);
	} else {
		Vec3 direction = r.direction().normalized();
		const auto t = .5 * (direction[1] + 1);
		const Color top{ .5,.7,1 };
		const Color bottom{ 1,1,1 };
		return (1 - t) * bottom + t * top;
	}
}
#endif

void makeWorld(rayUtilities::HittableList& world) {
	using namespace rayUtilities;
	world.add(std::make_shared<Sphere>(Point3{ 0,0,-1 }, 0.5));
	world.add(std::make_shared<Sphere>(Point3{ 0,-100.5,-1 }, 100));
}

int main(int argc, char* argv[]) {

	// World
	rayUtilities::HittableList world;
	makeWorld(world);

	// Screen
	const auto aspectRatio = 16. / 9.;
	const int img_width = 512;
	const int img_height = static_cast<int>(img_width/aspectRatio);
	std::cout << "Image of size (" << img_width << ", " << img_height << ")" << std::endl;
	const int nSamples = 100;
	const int maxDepth = 50;

	// Camera
	rayUtilities::Camera camera;

	// render to image
	std::ofstream imgWritter("image.ppm");
	imgWritter << "P3" << std::endl 
		<< img_width << " " << img_height << std::endl
		<< "255" << std::endl;
	for (int j = img_height-1; j>=0; j--)
		for (int i = 0; i < img_width; i++) {
		

			rayUtilities::Color pixelColor{ 0,0,0 };
			for (int s = 0; s < nSamples; s++) {
				auto u = (double(i) + rayUtilities::randomDouble()) / (img_width - 1) ;
				auto v = (double(j) + rayUtilities::randomDouble()) / (img_height - 1);

				const auto r = camera.getRay(u, v);
				pixelColor += ray_color(r, world, maxDepth);
			}
			rayUtilities::write_color(imgWritter, pixelColor, nSamples);
		}
	imgWritter.close();
	std::cout << "Done." << std::endl;
}