#include <math.h>
#include "mlx_mmap_shim.h"

#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/dtype.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;
using mlx::core::allocator::Buffer;

static thread_local std::string g_mmap_error;

static int fail(const char *msg) {
    g_mmap_error = msg ? msg : "unknown mmap error";
    return 1;
}

static int fail_errno(const char *what) {
    g_mmap_error = std::string(what) + ": " + std::strerror(errno);
    return 1;
}

static bool to_dtype(mlx_dtype in, Dtype *out) {
    switch (in) {
        case MLX_BOOL: *out = mlx::core::bool_; return true;
        case MLX_UINT8: *out = mlx::core::uint8; return true;
        case MLX_UINT16: *out = mlx::core::uint16; return true;
        case MLX_UINT32: *out = mlx::core::uint32; return true;
        case MLX_UINT64: *out = mlx::core::uint64; return true;
        case MLX_INT8: *out = mlx::core::int8; return true;
        case MLX_INT16: *out = mlx::core::int16; return true;
        case MLX_INT32: *out = mlx::core::int32; return true;
        case MLX_INT64: *out = mlx::core::int64; return true;
        case MLX_FLOAT16: *out = mlx::core::float16; return true;
        case MLX_FLOAT32: *out = mlx::core::float32; return true;
        case MLX_FLOAT64: *out = mlx::core::float64; return true;
        case MLX_BFLOAT16: *out = mlx::core::bfloat16; return true;
        case MLX_COMPLEX64: *out = mlx::core::complex64; return true;
        default: return false;
    }
}

struct MappedFile {
    void *base = nullptr;
    size_t size = 0;
    ~MappedFile() {
        if (base && size) munmap(base, size);
    }
};

struct msv_mmap_shard {
    /* Declaration order is intentional: backing dies before map. Its Data
     * deleter also captures map, so arrays derived from backing keep the mmap
     * alive even after this shard handle is closed. */
    std::shared_ptr<MappedFile> map;
    array backing;

    msv_mmap_shard(std::shared_ptr<MappedFile> m, Buffer buf)
        : map(std::move(m)),
          backing(buf,
                  Shape{1},
                  mlx::core::uint8,
                  [owner = map](Buffer b) mutable {
                      mlx::core::allocator::release(b);
                      owner.reset();
                  }) {}
};

extern "C" const char *msv_mmap_last_error(void) {
    return g_mmap_error.c_str();
}

extern "C" int msv_mmap_shard_open(msv_mmap_shard **out, const char *path) {
    if (!out || !path) return fail("mmap shard: null argument");
    *out = nullptr;
    g_mmap_error.clear();

    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail_errno("mmap shard open");

    struct stat st{};
    if (fstat(fd, &st) != 0) {
        int saved = errno;
        close(fd);
        errno = saved;
        return fail_errno("mmap shard fstat");
    }
    if (st.st_size <= 0) {
        close(fd);
        return fail("mmap shard: empty file");
    }
    const uint64_t usize = static_cast<uint64_t>(st.st_size);
    if (usize > std::numeric_limits<size_t>::max()) {
        close(fd);
        return fail("mmap shard: file too large for size_t");
    }
    const size_t size = static_cast<size_t>(usize);

    void *base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    int saved = errno;
    close(fd);
    if (base == MAP_FAILED) {
        errno = saved;
        return fail_errno("mmap shard mmap");
    }

    auto mapping = std::make_shared<MappedFile>();
    mapping->base = base;
    mapping->size = size;

    Buffer buf = mlx::core::allocator::make_buffer(base, size);
    if (!buf.ptr()) {
        return fail("mmap shard: MLX Metal allocator refused no-copy buffer");
    }

    try {
        *out = new msv_mmap_shard(std::move(mapping), buf);
        return 0;
    } catch (const std::exception& e) {
        mlx::core::allocator::release(buf);
        return fail(e.what());
    } catch (...) {
        mlx::core::allocator::release(buf);
        return fail("mmap shard: unknown constructor failure");
    }
}

extern "C" void msv_mmap_shard_close(msv_mmap_shard *shard) {
    delete shard;
}

extern "C" size_t msv_mmap_shard_size(const msv_mmap_shard *shard) {
    return shard && shard->map ? shard->map->size : 0;
}

extern "C" int msv_mmap_shard_array(mlx_array *out,
                                      msv_mmap_shard *shard,
                                      size_t byte_offset,
                                      const int *shape,
                                      int dim,
                                      mlx_dtype dtype) {
    if (!out || !shard || !shard->map || !shape || dim < 0)
        return fail("mmap array: invalid argument");
    out->ctx = nullptr;
    g_mmap_error.clear();

    Dtype dt = mlx::core::uint8;
    if (!to_dtype(dtype, &dt)) return fail("mmap array: unsupported dtype");

    Shape sh;
    size_t elements = 1;
    try {
        sh.reserve(static_cast<size_t>(dim));
        for (int i = 0; i < dim; ++i) {
            if (shape[i] < 0) return fail("mmap array: negative shape dimension");
            sh.push_back(shape[i]);
            const size_t d = static_cast<size_t>(shape[i]);
            if (d != 0 && elements > std::numeric_limits<size_t>::max() / d)
                return fail("mmap array: shape overflow");
            elements *= d;
        }
        if (dt.size() != 0 && elements > std::numeric_limits<size_t>::max() / dt.size())
            return fail("mmap array: byte size overflow");
        const size_t nbytes = elements * dt.size();
        if (byte_offset > shard->map->size || nbytes > shard->map->size - byte_offset)
            return fail("mmap array: tensor range outside shard");
        if (dt.size() == 0 || (byte_offset % dt.size()) != 0)
            return fail("mmap array: byte offset is not dtype-aligned");

        /* Construct an available leaf descriptor without taking a second
         * ownership reference to the Metal buffer, then replace its Data with
         * the backing array's shared Data and the tensor's byte offset. */
        array a(shard->backing.buffer(), sh, dt, [](Buffer) {});
        const auto strides = a.strides();
        const auto flags = a.flags();
        const size_t data_size = a.data_size();
        a.copy_shared_buffer(shard->backing,
                             strides,
                             flags,
                             data_size,
                             static_cast<int64_t>(byte_offset / dt.size()));
        out->ctx = new array(std::move(a));
        return 0;
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("mmap array: unknown constructor failure");
    }
}

extern "C" int msv_mmap_shard_dontneed(msv_mmap_shard *shard,
                                         size_t byte_offset,
                                         size_t byte_length) {
    if (!shard || !shard->map) return fail("mmap dontneed: invalid shard");
    const size_t size = shard->map->size;
    if (byte_offset > size) return fail("mmap dontneed: offset outside shard");
    if (byte_length == 0) byte_length = size - byte_offset;
    if (byte_length > size - byte_offset) return fail("mmap dontneed: range outside shard");
    if (byte_length == 0) return 0;

    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t start = byte_offset & ~(page - 1);
    size_t end = byte_offset + byte_length;
    if (end > std::numeric_limits<size_t>::max() - (page - 1))
        return fail("mmap dontneed: range overflow");
    end = (end + page - 1) & ~(page - 1);
    if (end > size) end = size;
    if (end <= start) return 0;

    const int rc = posix_madvise(static_cast<char *>(shard->map->base) + start,
                                 end - start,
                                 POSIX_MADV_DONTNEED);
    if (rc != 0) {
        g_mmap_error = std::string("mmap dontneed: ") + std::strerror(rc);
        return 1;
    }
    return 0;
}
