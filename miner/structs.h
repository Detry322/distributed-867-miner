#ifndef _structs_
#define _structs_

#include <stdint.h>

typedef struct {
  uint32_t h[8]; // Letters -> l[0] = A, l[1] = B, etc.
  uint32_t work[16];
  uint64_t difficulty;
  uint64_t timestamp;
} sha_base;

typedef struct {
  int32_t length;
  uint64_t start;
} hash_chain;

typedef struct {
  hash_chain chains[3];
} triple;

#endif // _structs_
