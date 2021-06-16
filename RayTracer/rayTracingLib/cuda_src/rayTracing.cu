#include <rayTracing.h>
#include <curand_kernel.h>

#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include "utilities.h"
#include "materials.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) ::check_cuda( (val), #val, __FILE__, __LINE__ )

namespace {
    const int tx = 8;
    const int ty = 8;

    curandState* d_randState;
    int nStates;
    rayUtilities::Hittable** d_list;

    __global__ void random_init(int n, curandState* rand_state) {
        int index = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
        //Each thread gets same seed, a different sequence number, no offset
        if (index<n)
            curand_init(1984, index, 0, &rand_state[index]);
    }

    __global__ void create_world(rayUtilities::Hittable** d_list, rayUtilities::HittableList* d_world) {
        using namespace rayUtilities;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *(d_list) = new Sphere(Vec3(0, 0, -1), 0.5);
            *(d_list + 1) = new Sphere(Vec3(0, -100.5, -1), 100);
            d_world = new(d_world) HittableList(d_list, 2);
        }
    }

    __global__ void create_camera(rayUtilities::Camera* d_camera, int width, int height) {
        using namespace rayUtilities;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            d_camera = new(d_camera) Camera(width, height);
        }
    }

    __global__ void destroy_camera(rayUtilities::Camera* d_camera) {
        using namespace rayUtilities;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            d_camera -> ~Camera();
        }
    }

    __global__ void destroy_world(rayUtilities::Hittable** d_list, rayUtilities::HittableList* d_world) {
        using namespace rayUtilities;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            delete* (d_list);
            delete* (d_list + 1);
            d_world -> ~HittableList();
        }
    }

    __global__ void render(float* fb, const rayUtilities::Hittable* world, const rayUtilities::Camera* camera, int nSamples, int maxDepth, int max_x, int max_y, curandState* randState) {
        using namespace rayUtilities;

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if (i < max_x && j < max_y) {
            int idx = max_x * j + i;
            Color pixelColor(0, 0, 0);
            curandState localState = randState[idx];
            for (int s = 0; s < nSamples; s++) {
                const float u = (float(i) + curand_uniform(&localState)) / max_x;
                const float v = (float(j) + curand_uniform(&localState)) / max_y;
                // const Ray ray(origin, lowerLeft + u * horizontal + v * vertical - origin);
                Ray r = camera->getRay(i, max_y - j, localState);
                pixelColor += Materials::ray_color(r, world, maxDepth, localState);
            }
            pixelColor /= float(nSamples);

            fb[idx * 3] = pixelColor[0];
            fb[idx * 3 + 1] = pixelColor[1];
            fb[idx * 3 + 2] = pixelColor[2];
        }

    }

    void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                file << ":" << line << " '" << func << "' \n";
            // Make sure we call CUDA Device Reset before exiting
            cudaDeviceReset();
            exit(99);
        }
    }

}

rayTracer::rayTracer(int width, int height) :
    image_width(width), image_height(height) {
    using namespace rayUtilities;

    const dim3 blocks((image_width + ::tx - 1) / ::tx, (image_height + ::ty - 1) / ::ty);
    const dim3 threads(::tx, ::ty);

    checkCudaErrors(cudaMalloc((void**)&::d_randState, image_width * image_height * sizeof(curandState)));

    random_init << <blocks, threads >> > (image_width*image_height, ::d_randState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // world
    checkCudaErrors(cudaMalloc((void**)&::d_list, 2 * sizeof(Hittable*)));
    checkCudaErrors(cudaMalloc((void**)&world, 2 * sizeof(HittableList)));

    ::create_world << <1, 1 >> > (::d_list, world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // camera
    checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera)));
    ::create_camera << <1, 1 >> > (camera, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMallocManaged((void**)&fb, image_width * image_height * 3 * sizeof(T)));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

}

int rayTracer::create_world()
{
    return 2;
}

void rayTracer::render(int nSamples, int maxDepth) {
    std::cout << "Rendering image of size (" << image_width << ", " << image_height << ")" << std::endl
        << "  with nSamples = " << nSamples << ", maxDepth = " << maxDepth << std::endl;
    using namespace rayUtilities;
    const dim3 blocks((image_width + ::tx - 1) / ::tx, (image_height + ::ty - 1) / ::ty);
    const dim3 threads(::tx, ::ty);
    ::render << <blocks, threads >> > (fb, world, camera, nSamples, maxDepth, image_width, image_height, ::d_randState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Done." << std::endl;

}

rayTracer::~rayTracer() {
    checkCudaErrors(cudaFree(fb));
    ::destroy_camera << <1, 1 >> > (camera);
    ::destroy_world << <1, 1 >> > (::d_list, world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(::d_list));
    checkCudaErrors(cudaFree(::d_randState));

}

#undef checkCudaErrors