namespace {
	static curandState* rand_states;
	static int nIdx;

	__global__ void rand_init(const int n, curandState* rand_state) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		if (i < max_x && j < max_y) {
			int idx = j * max_x + i;
			curand_init(1984, idx, 0, &rand_state[idx]);
		}
	}

    
}

namespace cudaRandom {
	void rand_init(const int n) {

	}
}
