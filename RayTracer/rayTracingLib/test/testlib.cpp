#include <rayTracing.h>

int main(int argc, char* argv[]) {
	const auto aspectRatio = 16. / 9.;
	const int img_width = 512;
	const int img_height = static_cast<int>(img_width / aspectRatio);

	rayTracer rt(img_width, img_height);
	rt.create_world();
	rt.render(100, 50);
	rt.write_image("image.ppm");
	return 0;
}