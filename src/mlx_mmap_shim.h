#pragma once
#include <stddef.h>
#include "mlx/c/array.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct msv_mmap_shard msv_mmap_shard;

/* Open a safetensors (or any regular) file as a read-only MAP_PRIVATE mapping
 * and wrap the whole mapping in exactly one MLX/Metal no-copy buffer. */
int msv_mmap_shard_open(msv_mmap_shard **out, const char *path);
void msv_mmap_shard_close(msv_mmap_shard *shard);
size_t msv_mmap_shard_size(const msv_mmap_shard *shard);

/* Create an available MLX leaf array that shares the shard's Metal buffer and
 * starts byte_offset bytes into the file mapping. No tensor bytes are copied. */
int msv_mmap_shard_array(mlx_array *out,
                         msv_mmap_shard *shard,
                         size_t byte_offset,
                         const int *shape,
                         int dim,
                         mlx_dtype dtype);

/* Hint that a byte range is cold. The mapping and MLX arrays remain valid;
 * macOS may discard clean file-backed pages and fault them back on next use. */
int msv_mmap_shard_dontneed(msv_mmap_shard *shard,
                            size_t byte_offset,
                            size_t byte_length);

const char *msv_mmap_last_error(void);

#ifdef __cplusplus
}
#endif
