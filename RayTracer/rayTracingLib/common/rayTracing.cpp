#include <rayTracing.h>
#include <fstream>
#include <iostream>

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
