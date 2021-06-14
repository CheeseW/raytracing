#include <iostream>
#include <fstream>
#include "vec3.h"
#include "ray.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ": " << line << " '" << func << "'" << std::endl;
		cudaDeviceReset();
		exit(99);
	}
}

__device__ bool hit_sphere(const rayUtilities::Ray& r) {
    using namespace rayUtilities;
    const Point3 center(0, 0, -1);
    const float radius = .5;

    // Compute distance from center to ray
    const Vec3 oc = center - r.origin();
    const auto& d = r.direction();

    double a = d.dot(d);
    double b = -2 * d.dot(oc);
    double c = oc.dot(oc) - radius * radius;
    double discriminant = b * b - 4 * a * c;
    return discriminant >= 0;

}

__device__ rayUtilities::Color ray_color(const rayUtilities::Ray& r) {
    using namespace rayUtilities;
    if (hit_sphere(r))
        return Color(1, 0, 0);
    Vec3 d = r.direction().normalized();
    auto t = .5f * (d[1] + 1.f);
    const Color top{ .5,.7,1 };
    const Color bottom{ 1,1,1 };
    return t* top + (1.f - t) * bottom;
}
#if 1

void write_image(const std::string filename, const rayUtilities::Color* fb, const int width, const int height) {
    using namespace rayUtilities;
    std::cout << "Writing image of size (" << width << ", " << height << ")" << std::endl;
    std::ofstream imgWritter(filename);
    imgWritter << "P3" << std::endl
        << width << " " << height << std::endl
        << "255" << std::endl;
    for (int j = height - 1; j >= 0; --j) 
        for (int i = 0; i < width; ++i) {
            int idx = j * width + i;
            int ir = static_cast<int>(255.999 * fb[idx][0]);
            int ig = static_cast<int>(255.999 * fb[idx][1]);
            int ib = static_cast<int>(255.999 * fb[idx][2]);

            imgWritter << ir << ' ' << ig << ' ' << ib << std::endl;
        }
    imgWritter.close();
    std::cout << "Done writing "<<filename << std::endl;
}

#endif
#if 1
__global__ void render(rayUtilities::Color* fb, int max_x, int max_y, const rayUtilities::Point3 lowerLeft, const rayUtilities::Vec3 horizontal, const rayUtilities::Vec3 vertical, const rayUtilities::Vec3 origin) {
    using namespace rayUtilities;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < max_x && j < max_y) {
        const float u = float(i) / max_x;
        const float v = float(j) / max_y;
        const Ray ray(origin, lowerLeft + u * horizontal + v * vertical - origin);

        int idx = max_x * j + i;

        fb[idx] = ray_color(ray);
       
    }
        
}
#endif

int main(int argc, char* argv[]) {
    using namespace rayUtilities;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Image
    const auto aspectRatio = 16. / 9;
    const int image_width = 800;
    const int image_height = static_cast<int>(image_width/aspectRatio);

    //camera
    const float viewport_height = 2.f;
    const float viewport_width = viewport_height*aspectRatio;
    const float focal_length = 1.f;

    const Point3 origin = Vec3(0, 0, 0);
    const Vec3 horizontal = Vec3(viewport_width, 0, 0);
    const Vec3 vertical = Vec3(0, viewport_height, 0);
    const Point3 lowerLeft = origin - (horizontal + vertical) / 2 - Vec3(0, 0, focal_length);
#if 1
    Color* fb;
   checkCudaErrors(cudaMallocManaged((void**)&fb, image_width*image_height*sizeof(Color)));

   const int tx = 8;
   const int ty = 8;
   dim3 blocks((image_width + tx - 1) / tx, (image_height + ty - 1) / ty);
   dim3 threads(tx, ty);

   cudaEventRecord(start);
   render << <blocks, threads >> > (fb, image_width, image_height, lowerLeft, horizontal, vertical, origin);
   cudaEventRecord(stop);

   checkCudaErrors(cudaGetLastError());
   checkCudaErrors(cudaDeviceSynchronize());
   checkCudaErrors(cudaEventSynchronize(stop));

   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   std::cout << "Time to compute the frame buffer : " << milliseconds << " ms" << std::endl;

   write_image("image.ppm", fb, image_width, image_height);
  
   checkCudaErrors(cudaFree(fb));
#endif
   return 0;
}
