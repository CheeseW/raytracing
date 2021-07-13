#include <rayTracing.h>

#include "materials.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

rayTracer::rayTracer(int width, int height) :
	image_width(width), image_height(height) {
	using namespace rayUtilities;
	world = new HittableList();
	camera = new Camera(image_width, image_height);
	fb = new T[image_width * image_height * 3];
}


rayTracer::~rayTracer() {
	delete fb;
	delete camera;
	delete world;
}

int rayTracer::create_world()
{
	using namespace rayUtilities;
	auto ground = std::make_shared<Lambertian>(Color(.8, .8, 0));
	auto center = std::make_shared<Lambertian>(Color(.7, .3, .3));
	auto left = std::make_shared<Metal>(Color(.8, .8, .8), .3);
	auto right = std::make_shared<Dielectric>(1.6);

	int count = 0;

	world->add(std::make_shared<Sphere>(Point3{ 0,0,-1 }, 0.5, center)); count++;
	world->add(std::make_shared<Sphere>(Point3{ 0,-100.5,-1 }, 100, ground)); count++;
	world->add(std::make_shared<Sphere>(Point3{ 1, 0,-1 }, .5, left)); count++;
	world->add(std::make_shared<Sphere>(Point3{ -1, 0,-1 }, .5, right)); count++;
	return count;
}

void rayTracer::render(int nSamples, int maxDepth) {
	std::cout << "Rendering image of size (" << image_width << ", " << image_height << ")" << std::endl
		<< "  with nSamples = " << nSamples << ", maxDepth = " << maxDepth << std::endl;
	using namespace rayUtilities;
	for (int j = 0; j < image_height; j++) {
		if (j%10 == 0)
			std::cout << "scanning line " << image_height - j << "..." << std::endl;
		for (int i = 0; i < image_width; i++) {
			Color pixelColor{ 0,0,0 };
			for (int s = 0; s < nSamples; s++) {
				const auto r = camera->getRay(i, image_height - j);
				pixelColor += Materials::ray_color(r, *world, maxDepth);
			}
			pixelColor /= nSamples;

			int idx = j * image_width + i;
			fb[idx * 3] = pixelColor[0];
			fb[idx * 3 + 1] = pixelColor[1];
			fb[idx * 3 + 2] = pixelColor[2];
		}
	}
	std::cout << "Done." << std::endl;
}