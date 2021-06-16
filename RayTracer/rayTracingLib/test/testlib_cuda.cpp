#include <rayTracing.h>
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
	const auto aspectRatio = 16. / 9.;
	const int img_width = 800;
	const int img_height = static_cast<int>(img_width / aspectRatio);

	std::chrono::steady_clock::time_point  t1, t2;
	std::chrono::duration<double> time_span;
	rayTracer rt(img_width, img_height);
	rt.create_world();

	t1 = std::chrono::steady_clock::now();
	rt.render(100, 50);
	t2 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	std::cout << "Rendering took " << time_span.count() << " s" << std::endl;

	t1 = std::chrono::steady_clock::now();
	rt.write_image("image_cuda.ppm");
#if 1

	t2 = std::chrono::steady_clock::now();

	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	std::cout << "File i/o took " << time_span.count() << " s" << std::endl;
#endif
	return 0;
}