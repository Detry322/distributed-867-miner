#ifndef _structs_
#define _structs_

#include <stdint.h>

typedef struct {
  uint32_t h[8]; // Letters -> l[0] = A, l[1] = B, etc.
  uint32_t work[16];
  uint64_t difficulty;
} sha_base;

typedef struct {
  bool found;
  uint64_t nonces[2];
} two_way_collision;

#endif // _structs_
