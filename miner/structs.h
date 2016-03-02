#ifndef _structs_
#define _structs_

#include <stdint.h>

typedef struct {
  uint32_t h[8]; // Letters -> l[0] = A, l[1] = B, etc.
  uint32_t work[16];
  uint64_t difficulty;
  uint64_t timestamp;
} sha_base;

#endif // _structs_
