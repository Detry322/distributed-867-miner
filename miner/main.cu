#include <iostream>
#include <helper_cuda.h>
#include <helper_string.h>

#include "./structs.h"
#include "./step_a.h"
#include "./step_b.h"
#include "./sha.h"

using namespace std;

static bool initialized = false;
static bool in_step_A = true;
static bool switched = false;

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

int main(int argc, char **argv) {
  initializeCuda();
  bool kernel_finished = true;
  while (true) {
    // if (messagesAvailable()) {
    //   m = getMessage();
    //   respondToMessage(m);
    // } else if (kernel_finished {
    //   respondToKernelFinish();
    // } else {
    //   yield
    // }
  //}
  }
};
