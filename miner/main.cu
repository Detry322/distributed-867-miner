#include <iostream>
#include <helper_cuda.h>
#include <helper_string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <poll.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <sys/time.h>

#include "./structs.h"
#include "./step_a.h"
#include "./step_b.h"
#include "./sha.h"
#include "./utils.h"

using namespace std;

typedef enum {
  STATE_DOING_NOTHING,
  STATE_WORKING_A,
  STATE_WORKING_B
} work_state;

sha_base* global_base = NULL;
triple* part_b_triples = NULL;
work_state state = STATE_DOING_NOTHING;

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
  if (host_input == NULL) {
    cout << "===== Error: Nothing to copy, step B. Will definitely segfault." << endl;
  }
  if (part_b_triples != NULL) {
    free(part_b_triples);
    part_b_triples = NULL;
  }
  uint64_t base_nonce = GET_RAND();

  sha_base* input;
  cudaMalloc(&input, sizeof(sha_base));
  cudaMemcpy(input, host_input, sizeof(sha_base), cudaMemcpyHostToDevice);

  step_a_kernel<<<16, 256>>>(input, base_nonce);
  cudaDeviceSynchronize();

  cudaFree(input);
}


#define STEP_B_BLOCKS 8
#define STEP_B_THREADS 256
#define MEMORY_SIZE (sizeof(triple)*STEP_B_THREADS*STEP_B_BLOCKS)

void run_step_b(sha_base* host_input, triple* host_triples) {
  if (host_input == NULL || host_triples == NULL) {
    cout << "===== Error: Nothing to copy, step B. Will definitely segfault." << endl;
  }
  cout << "= Running step B..." << endl;
  sha_base* input;
  cudaMalloc(&input, sizeof(sha_base));
  cudaMemcpy(input, host_input, sizeof(sha_base), cudaMemcpyHostToDevice);

  triple* triples;
  cudaMalloc(&triples, MEMORY_SIZE);
  cudaMemcpy(triples, host_triples, MEMORY_SIZE, cudaMemcpyHostToDevice);

  step_b_kernel<<<STEP_B_BLOCKS, STEP_B_THREADS>>>(input, triples);
  cudaDeviceSynchronize();
  cout << "= Step B finished..." << endl;

  cudaFree(input);
  cudaFree(triples);
}

sha_base* create_base(uint8_t parentid[32], uint8_t root[32], uint64_t difficulty, uint64_t timestamp, uint8_t version) {
  sha_base* base = (sha_base*) malloc(sizeof(sha_base));
  prepare_base(base, parentid, root, difficulty, timestamp, version);
  return base;
}

uint64_t parse_uint64(const char* input) {
  char buf[17];
  strncpy(buf, input, 16); // 16 hex bytes
  buf[16] = 0;
  return strtoul(buf, NULL, 16); // Base 16
}

int32_t parse_int32(const char* input) {
  char buf[9];
  strncpy(buf, input, 8);
  buf[8] = 0;
  return (int32_t) strtol(buf, NULL, 16);
}

void parse_byte_array(const char* str, uint8_t result[32]) {
  for (int i = 0; i < 32; i++) {
    sscanf(str, "%02hhx", &result[i]);
    str += 2;
  }
}

void handle_update_base(const char* input) {
  input += 2;
  uint8_t parentid[32];
  uint8_t root[32];
  uint64_t difficulty;
  uint64_t timestamp;
  uint8_t version;

  parse_byte_array(input, parentid);
  input += 65;
  parse_byte_array(input, root);
  input += 65;
  difficulty = parse_uint64(input);
  input += 17;
  timestamp = parse_uint64(input);
  input += 17;
  sscanf(input, "%02hhx", &version);

  if (global_base != NULL) {
    free(global_base);
  }

  global_base = create_base(parentid, root, difficulty, timestamp, version);
  cout << "=== Updating base..." << endl;
  cout << "= Parentid: ";
  for (int i = 0; i < 32; i ++) {
    printf("%02hhx", parentid[i]);
  }
  cout << endl << "= Root: ";
  for (int i = 0; i < 32; i ++) {
    printf("%02hhx", root[i]);
  }
  cout << endl;
  cout << "= Difficulty: " << difficulty << " Timestamp: " << timestamp << " Version: " << (uint32_t) version << endl;
};

void handle_start_a(const char* input) {
  if (global_base == NULL) {
    cout << "= Error: Starting A but haven't received base. Will prolly segfault" << endl;
  }
  state = STATE_WORKING_A;
  cout << "=== Working A..." << endl;
};

void handle_start_b() {
  if (global_base == NULL) {
    cout << "= Error: Starting B but haven't received base. Will prolly segfault" << endl;
  }
  part_b_triples = (triple*) malloc(sizeof(triple)*STEP_B_BLOCKS*STEP_B_THREADS);
  for (int i = 0; i < STEP_B_BLOCKS*STEP_B_THREADS; i++) {
    string s;
    getline(cin, s);
    const char* input = s.c_str();
    for (int j = 0; j < 3; j++) {
      part_b_triples[i].chains[j].start = parse_uint64(input);
      input += 17;
      part_b_triples[i].chains[j].length = parse_int32(input);
      input += 9;
    }
  }
  state = STATE_WORKING_B;
  cout << "=== Working B..." << endl;
};

int main(int argc, char **argv) {
  struct timeval ts;
  gettimeofday(&ts, NULL);
  srand(ts.tv_sec ^ ts.tv_usec);
  cout << ts.tv_sec + ts.tv_usec << endl;
  initializeCuda();
  struct pollfd stdin_poll = { .fd = STDIN_FILENO, .events = POLLIN | POLLRDBAND | POLLRDNORM | POLLPRI };
  while (true) {
    string input;
    if (poll(&stdin_poll, 1, 0) == 1 && !cin.eof()) {
      getline(cin, input);
      if (input.length() == 0)
        continue;
      const char* cstr = input.c_str();
      if (cstr[0] == 'H') {
        handle_update_base(cstr);
      } else if (cstr[0] == 'A') {
        handle_start_a(cstr);
      } else if (cstr[0] == 'B') {
        handle_start_b();
        run_step_b(global_base, part_b_triples);
        state = STATE_WORKING_A;
      } else if (cstr[0] == 'Q') {
        cout << "=== Exiting..." << endl;
        exit(0);
      } else {
        cout << "===== Error: Received a line of garbage input" << endl;
      }
    } else {
      if (cin.eof()) {
        exit(0);
      }
      if (state == STATE_DOING_NOTHING) {
        this_thread::yield();
      } else if (state == STATE_WORKING_A) {
        run_step_a(global_base);
        cout.flush();
      }
    }
  }
};



// uint8_t parentid[32] = {47, 254, 149, 3, 14, 129, 220, 200, 18, 210, 158, 85, 188, 8, 177, 74, 156, 33, 244, 140, 177, 197, 230, 199, 62, 65, 169, 208, 25, 70, 106, 26};
// uint8_t root[32] = {21, 98, 32, 101, 67, 218, 118, 65, 35, 194, 27, 213, 36, 103, 79, 10, 138, 175, 73, 200, 168, 151, 68, 201, 115, 82, 254, 103, 127, 126, 64, 6};
// uint64_t difficulty = 42;
// uint64_t timestamp = 1456441489431952896L;
// uint8_t version = 0;
// sha_base* base = create_base(parentid, root, difficulty, timestamp, version);
// run_step_a(base);
// free(base);
