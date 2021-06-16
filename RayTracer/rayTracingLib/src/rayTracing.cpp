#include <rayTracing.h>
#include <fstream>
#include <iostream>

#include "hittable_list.h"
#include "sphere.h"
#include "materials.h"

#include "camera.h"

rayTracer::rayTracer(int width, int height):
	image_width(width), image_height(height) {
	using namespace rayUtilities;
	world = new HittableList();
	camera = new Camera(image_width, image_height);
	fb = new T[image_width * image_height*3];
}

int rayTracer::create_world()
{
	using namespace rayUtilities;
	world->add(std::make_shared<Sphere>(Point3{ 0,0,-1 }, 0.5));
	world->add(std::make_shared<Sphere>(Point3{ 0,-100.5,-1 }, 100));
	return 2;
}

void rayTracer::render(int nSamples, int maxDepth) {
	std::cout << "Rendering image of size (" << image_width << ", " << image_height << ")" << std::endl
		<< "  with nSamples = " << nSamples << ", maxDepth = " << maxDepth << std::endl;
	using namespace rayUtilities;
	for (int j = 0; j < image_height; j++)
		for (int i = 0; i < image_width; i++) {
			Color pixelColor{ 0,0,0 };
			for (int s = 0; s < nSamples; s++) {
				const auto r = camera->getRay(i, image_height-j);
				pixelColor += Materials::ray_color(r, *world, maxDepth);
			}
			pixelColor /= nSamples;
			
			int idx = j * image_width + i;
			fb[idx * 3] = pixelColor[0];
			fb[idx * 3 + 1] = pixelColor[1];
			fb[idx * 3 + 2] = pixelColor[2];
		}
	std::cout << "Done." << std::endl;
}

void rayTracer::write_image(std::string filename) const{
	// render to image
	std::cout << "Writting "<<filename<<" of size (" << image_width << ", " << image_height << ")" << std::endl;
	std::ofstream imgWritter(filename);
	imgWritter << "P3" << std::endl
		<< image_width << " " << image_height << std::endl
		<< "255" << std::endl;
	// with gamma correction of gamma = 2
	for (int j = 0; j < image_height * image_width; j++)
		imgWritter << static_cast<int>(sqrt(fb[j * 3])*255.999)
		<< " " << static_cast<int>(sqrt(fb[j * 3 + 1]) * 255.999)
		<< " " << static_cast<int>(sqrt(fb[j * 3 + 2]) * 255.999) << std::endl;
	imgWritter.close();
	std::cout << "Done." << std::endl;
}

const rayTracer::T* rayTracer::get_framebuffer() const {
	return nullptr;
}

rayTracer::~rayTracer() {
	delete fb;
	delete camera;
	delete world;
}