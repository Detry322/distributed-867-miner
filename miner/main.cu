#include <iostream>
#include <helper_cuda.h>
#include <helper_string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "./structs.h"
#include "./step_a.h"
#include "./step_b.h"
#include "./sha.h"
#include "./utils.h"

using namespace std;

// static bool initialized = false;
// static bool in_step_A = true;
// static bool switched = false;

void initializeCuda() {
  cout << "=== Initializing CUDA..." << endl;
  int device_count = 0, device = -1;
  checkCudaErrors(cudaGetDeviceCount(&device_count));
  cout << "= Found " << device_count << " device(s)." << endl;
  for (int i = 0 ; i < device_count ; ++i) {
    cudaDeviceProp properties;
    checkCudaErrors(cudaGetDeviceProperties(&properties, i));
    if (properties.major > 2) {
      device = i;
      cout << "= Running on GPU " << i << " (" << properties.name << ")" << endl;
      cout << "= Clock speed: " << properties.clockRate << " KHz, " << properties.multiProcessorCount << " processors." << endl;
      break;
    }
  }
  if (device == -1) {
    std::cerr << "= This miner requires a GPU device with compute SM 3.0 or higher.  Exiting..." << std::endl;
    exit(EXIT_WAIVED);
  }
  cout << "=== CUDA Initialized!" << endl;
  cudaSetDevice(device);
}

void run_step_a(sha_base* host_input) {
  uint64_t base_nonce = GET_RAND();

  sha_base* input;
  cudaMalloc(&input, sizeof(sha_base));
  cudaMemcpy(input, host_input, sizeof(sha_base), cudaMemcpyHostToDevice);

  step_a_kernel<<<16, 512>>>(input, base_nonce);
  cudaDeviceSynchronize();

  cudaFree(input);
}


#define STEP_B_BLOCKS 8
#define STEP_B_THREADS 128
#define MEMORY_SIZE (sizeof(triple)*STEP_B_THREADS*STEP_B_BLOCKS)

void run_step_b(sha_base* host_input, triple* host_triples) {
  sha_base* input;
  cudaMalloc(&input, sizeof(sha_base));
  cudaMemcpy(input, host_input, sizeof(sha_base), cudaMemcpyHostToDevice);

  triple* triples;
  cudaMalloc(&triples, MEMORY_SIZE);
  cudaMemcpy(triples, host_triples, MEMORY_SIZE, cudaMemcpyHostToDevice);

  step_b_kernel<<<STEP_B_BLOCKS, STEP_B_THREADS>>>(input, triples);
  cudaDeviceSynchronize();

  cudaFree(input);
  cudaFree(triples);
  cout << "B_FIN" << endl;
}

sha_base* create_base(uint8_t parentid[32], uint8_t root[32], uint64_t difficulty, uint64_t timestamp, uint8_t version) {
  sha_base* base = (sha_base*) malloc(sizeof(sha_base));
  prepare_base(base, parentid, root, difficulty, timestamp, version);
  return base;
}

int main(int argc, char **argv) {
  srand(time(NULL));
  initializeCuda();
  uint8_t parentid[32] = {47, 254, 149, 3, 14, 129, 220, 200, 18, 210, 158, 85, 188, 8, 177, 74, 156, 33, 244, 140, 177, 197, 230, 199, 62, 65, 169, 208, 25, 70, 106, 26};
  uint8_t root[32] = {21, 98, 32, 101, 67, 218, 118, 65, 35, 194, 27, 213, 36, 103, 79, 10, 138, 175, 73, 200, 168, 151, 68, 201, 115, 82, 254, 103, 127, 126, 64, 6};
  uint64_t difficulty = 42;
  uint64_t timestamp = 1456441489431952896L;
  uint8_t version = 0;
  sha_base* base = create_base(parentid, root, difficulty, timestamp, version);
  run_step_a(base);
  free(base);
};
