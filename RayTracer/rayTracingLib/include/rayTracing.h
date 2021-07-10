#pragma once

#include <string>
namespace rayUtilities {
	class HittableList;
	class Camera;
	class Material;
}

class rayTracer {
	using T = float;
public:
	rayTracer(int width, int height);
	int create_world();
	void render(int nSamples, int maxDepth);
	void write_image(std::string fileName) const;
	const T* get_framebuffer() const;
	~rayTracer();
private:
	int image_width;
	int image_height;

	rayUtilities::HittableList* world;
	rayUtilities::Camera* camera;
	rayUtilities::Material** materials;

	T* fb;
};