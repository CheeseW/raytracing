#include <rayTracing.h>
#include <fstream>
#include <iostream>

namespace {
   

	void writeBMP(const std::string& filename, const float* fb, const int w, const int h) {
		std::cout << "Writting " << filename << " of size (" << w << ", " << h << ")" << std::endl;
		std::ofstream imgWritter(filename, std::ios::binary);

		const int padding = (4 - (w * 3) % 4) % 4;
		std::cout << padding << std::endl;
		int filesize = 54 + (3 * w + padding) * h;  //w is your image width, h is image height, both int
		std::cout << w * 3 + padding << std::endl;
		std::cout << filesize - 54 << std::endl;

		unsigned char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
		unsigned char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };
		char bmppad[3] = { 0,0,0 };

		bmpfileheader[2] = (unsigned char)(filesize);
		bmpfileheader[3] = (unsigned char)(filesize >> 8);
		bmpfileheader[4] = (unsigned char)(filesize >> 16);
		bmpfileheader[5] = (unsigned char)(filesize >> 24);

		bmpinfoheader[4] = (unsigned char)(w);
		bmpinfoheader[5] = (unsigned char)(w >> 8);
		bmpinfoheader[6] = (unsigned char)(w >> 16);
		bmpinfoheader[7] = (unsigned char)(w >> 24);
		bmpinfoheader[8] = (unsigned char)(h);
		bmpinfoheader[9] = (unsigned char)(h >> 8);
		bmpinfoheader[10] = (unsigned char)(h >> 16);
		bmpinfoheader[11] = (unsigned char)(h >> 24);

		imgWritter.write((char*)bmpfileheader, 14);
		imgWritter.write((char*)bmpinfoheader, 40);
		for (int i = h-1; i >=0; i--)
		{
			for (int j = 0; j < w; j++) {
				unsigned char pixel[3];
				for (int v = 0; v < 3; v++)
					pixel[2 - v] = static_cast<unsigned char>(sqrt(fb[i * w * 3 + j * 3 + v]) * 255.999);
				imgWritter.write((char*)pixel, 3);
			}
			imgWritter.write(bmppad, padding);
		}

		imgWritter.close();
	}
}

void rayTracer::write_image(std::string filename) const{
	// render to image
	auto found = filename.find(".bmp", filename.length() - 4);
	if (found != std::string::npos) {
		// std::cout << "found = " << found << std::endl;
		// write bmp
		writeBMP(filename, fb, image_width, image_height);
	}
	else {
		std::cout << "Writting " << filename << " of size (" << image_width << ", " << image_height << ")" << std::endl;
		std::ofstream imgWritter(filename);
		imgWritter << "P3" << std::endl
			<< image_width << " " << image_height << std::endl
			<< "255" << std::endl;
		// with gamma correction of gamma = 2
		for (int j = 0; j < image_height * image_width; j++)
			imgWritter << static_cast<int>(sqrt(fb[j * 3]) * 255.999)
			<< " " << static_cast<int>(sqrt(fb[j * 3 + 1]) * 255.999)
			<< " " << static_cast<int>(sqrt(fb[j * 3 + 2]) * 255.999) << std::endl;
		imgWritter.close();
	}
	std::cout << "Done." << std::endl;
}

const rayTracer::T* rayTracer::get_framebuffer() const {
	return nullptr;
}
