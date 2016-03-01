#include <iostream>
#include <helper_cuda.h>
#include <helper_string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "./structs.h"
#include "./step_a.h"
#include "./step_b.h"
#include "./sha.h"

using namespace std;

// static bool initialized = false;
// static bool in_step_A = true;
// static bool switched = false;

void initializeCuda() {
  cout << "=== Initializing CUDA..." << endl;
  int device_count = 0, device = -1;
  checkCudaErrors(cudaGetDeviceCount(&device_count));
  cout << "Found " << device_count << " device(s)." << endl;
  for (int i = 0 ; i < device_count ; ++i) {
    cudaDeviceProp properties;
    checkCudaErrors(cudaGetDeviceProperties(&properties, i));
    if (properties.major > 2) {
      device = i;
      cout << "Running on GPU " << i << " (" << properties.name << ")" << endl;
      cout << "Clock speed: " << properties.clockRate << " KHz, " << properties.multiProcessorCount << " processors." << endl;
      break;
    }
  }
  if (device == -1) {
    std::cerr << "This miner requires a GPU device with compute SM 3.0 or higher.  Exiting..." << std::endl;
    exit(EXIT_WAIVED);
  }
  cout << "=== CUDA Initialized!" << endl;
  cudaSetDevice(device);
}

// void respondToKernelFinish() {
//   if (!initialized)
//     return
//   if (in_step_A) {
//     // Get pair that was found
//     if (!switched)
//       // Send message in queue
//     else
//       switched = false;
//     runStepA();
//   } else {
//     // Check if triplet was found
//     if (!switched)
//       if (triplet_found())
//         // Send message in queue
//     else
//       switched = false;
//     runStepB();
//   }
// }

// void respondToMessage() {
//   if (isHashPerSecondMessage)
//     // send hash per second message
//   else if (isRunStepAMessage) {
//     in_step_A = true;
//     switched = true;
//     // set up step A
//   } else if (isRunStepBMessage) {
//     in_step_A = false;
//     switched = true;
//     // set up step B
//   }
// }

two_way_collision run_step_a(uint8_t parentid[32], uint8_t root[32], uint64_t difficulty, uint64_t timestamp, uint8_t version) {
  sha_base          host_input;
  two_way_collision host_collision = { .found = false, .nonces = {0L, 0L}};
  uint64_t base_nonce = (uint64_t) rand();
  prepare_base(&host_input, parentid, root, difficulty, timestamp, version);

  sha_base* input;
  cudaMalloc(&input, sizeof(sha_base));
  cudaMemcpy(input, &host_input, sizeof(sha_base), cudaMemcpyHostToDevice);

  two_way_collision* collision;
  cudaMalloc(&collision, sizeof(two_way_collision));
  cudaMemcpy(collision, &host_collision, sizeof(two_way_collision), cudaMemcpyHostToDevice);

  dim3 grid(100, 1);
  dim3 block(100, 1);

  step_a_kernel<<<grid, block>>>(input, collision, base_nonce);

  cudaMemcpy(&host_collision, collision, sizeof(two_way_collision), cudaMemcpyDeviceToHost);
  cudaFree(input);
  cudaFree(collision);
  return host_collision;
}

int main(int argc, char **argv) {
  srand(time(NULL));
  initializeCuda();
  uint8_t parentid[32] = {47, 254, 149, 3, 14, 129, 220, 200, 18, 210, 158, 85, 188, 8, 177, 74, 156, 33, 244, 140, 177, 197, 230, 199, 62, 65, 169, 208, 25, 70, 106, 26};
  uint8_t root[32] = {21, 98, 32, 101, 67, 218, 118, 65, 35, 194, 27, 213, 36, 103, 79, 10, 138, 175, 73, 200, 168, 151, 68, 201, 115, 82, 254, 103, 127, 126, 64, 6};
  uint64_t difficulty = 42;
  uint64_t timestamp = 1456441489431952896L;
  uint8_t version = 0;
  run_step_a(parentid, root, difficulty, timestamp, version);
};
