#define FUTHARK_CUDA
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;
typedef uint64_t ulong;
#define __kernel extern "C" __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK)
#define __global
#define __local
#define __private
#define __constant
#define __write_only
#define __read_only
static inline int get_group_id_fn(int block_dim0, int block_dim1,
                                  int block_dim2, int d)
{
    switch (d) {
        
      case 0:
        d = block_dim0;
        break;
        
      case 1:
        d = block_dim1;
        break;
        
      case 2:
        d = block_dim2;
        break;
    }
    switch (d) {
        
      case 0:
        return blockIdx.x;
        
      case 1:
        return blockIdx.y;
        
      case 2:
        return blockIdx.z;
        
      default:
        return 0;
    }
}
#define get_group_id(d) get_group_id_fn(block_dim0, block_dim1, block_dim2, d)
static inline int get_num_groups_fn(int block_dim0, int block_dim1,
                                    int block_dim2, int d)
{
    switch (d) {
        
      case 0:
        d = block_dim0;
        break;
        
      case 1:
        d = block_dim1;
        break;
        
      case 2:
        d = block_dim2;
        break;
    }
    switch (d) {
        
      case 0:
        return gridDim.x;
        
      case 1:
        return gridDim.y;
        
      case 2:
        return gridDim.z;
        
      default:
        return 0;
    }
}
#define get_num_groups(d) get_num_groups_fn(block_dim0, block_dim1, block_dim2, d)
static inline int get_local_id(int d)
{
    switch (d) {
        
      case 0:
        return threadIdx.x;
        
      case 1:
        return threadIdx.y;
        
      case 2:
        return threadIdx.z;
        
      default:
        return 0;
    }
}
static inline int get_local_size(int d)
{
    switch (d) {
        
      case 0:
        return blockDim.x;
        
      case 1:
        return blockDim.y;
        
      case 2:
        return blockDim.z;
        
      default:
        return 0;
    }
}
static inline int get_global_id_fn(int block_dim0, int block_dim1,
                                   int block_dim2, int d)
{
    return get_group_id(d) * get_local_size(d) + get_local_id(d);
}
#define get_global_id(d) get_global_id_fn(block_dim0, block_dim1, block_dim2, d)
static inline int get_global_size(int block_dim0, int block_dim1,
                                  int block_dim2, int d)
{
    return get_num_groups(d) * get_local_size(d);
}
#define CLK_LOCAL_MEM_FENCE 1
#define CLK_GLOBAL_MEM_FENCE 2
static inline void barrier(int x)
{
    __syncthreads();
}
static inline void mem_fence_local()
{
    __threadfence_block();
}
static inline void mem_fence_global()
{
    __threadfence();
}
#define NAN (0.0/0.0)
#define INFINITY (1.0/0.0)
extern volatile __shared__ char shared_mem[];
static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t udiv_up8(uint8_t x, uint8_t y)
{
    return (x + y - 1) / y;
}
static inline uint16_t udiv_up16(uint16_t x, uint16_t y)
{
    return (x + y - 1) / y;
}
static inline uint32_t udiv_up32(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}
static inline uint64_t udiv_up64(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline uint8_t udiv_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint16_t udiv_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint32_t udiv_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint64_t udiv_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint8_t umod_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint16_t umod_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint32_t umod_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint64_t umod_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t sdiv_up8(int8_t x, int8_t y)
{
    return sdiv8(x + y - 1, y);
}
static inline int16_t sdiv_up16(int16_t x, int16_t y)
{
    return sdiv16(x + y - 1, y);
}
static inline int32_t sdiv_up32(int32_t x, int32_t y)
{
    return sdiv32(x + y - 1, y);
}
static inline int64_t sdiv_up64(int64_t x, int64_t y)
{
    return sdiv64(x + y - 1, y);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t sdiv_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : sdiv8(x, y);
}
static inline int16_t sdiv_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : sdiv16(x, y);
}
static inline int32_t sdiv_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : sdiv32(x, y);
}
static inline int64_t sdiv_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : sdiv64(x, y);
}
static inline int8_t sdiv_up_safe8(int8_t x, int8_t y)
{
    return sdiv_safe8(x + y - 1, y);
}
static inline int16_t sdiv_up_safe16(int16_t x, int16_t y)
{
    return sdiv_safe16(x + y - 1, y);
}
static inline int32_t sdiv_up_safe32(int32_t x, int32_t y)
{
    return sdiv_safe32(x + y - 1, y);
}
static inline int64_t sdiv_up_safe64(int64_t x, int64_t y)
{
    return sdiv_safe64(x + y - 1, y);
}
static inline int8_t smod_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : smod8(x, y);
}
static inline int16_t smod_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : smod16(x, y);
}
static inline int32_t smod_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : smod32(x, y);
}
static inline int64_t smod_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : smod64(x, y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t squot_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int16_t squot_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int32_t squot_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int64_t squot_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int8_t srem_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int16_t srem_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int32_t srem_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int64_t srem_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((int8_t) (uint8_t) x)
#define zext_i8_i16(x) ((int16_t) (uint8_t) x)
#define zext_i8_i32(x) ((int32_t) (uint8_t) x)
#define zext_i8_i64(x) ((int64_t) (uint8_t) x)
#define zext_i16_i8(x) ((int8_t) (uint16_t) x)
#define zext_i16_i16(x) ((int16_t) (uint16_t) x)
#define zext_i16_i32(x) ((int32_t) (uint16_t) x)
#define zext_i16_i64(x) ((int64_t) (uint16_t) x)
#define zext_i32_i8(x) ((int8_t) (uint32_t) x)
#define zext_i32_i16(x) ((int16_t) (uint32_t) x)
#define zext_i32_i32(x) ((int32_t) (uint32_t) x)
#define zext_i32_i64(x) ((int64_t) (uint32_t) x)
#define zext_i64_i8(x) ((int8_t) (uint64_t) x)
#define zext_i64_i16(x) ((int16_t) (uint64_t) x)
#define zext_i64_i32(x) ((int32_t) (uint64_t) x)
#define zext_i64_i64(x) ((int64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_ctzz8(int8_t x)
{
    int i = 0;
    
    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int i = 0;
    
    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int i = 0;
    
    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int i = 0;
    
    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_ctzz8(int8_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 8 : y - 1;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 16 : y - 1;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 32 : y - 1;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int y = __ffsll(x);
    
    return y == 0 ? 64 : y - 1;
}
#else
static int32_t futrts_ctzz8(int8_t x)
{
    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz16(int16_t x)
{
    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz32(int32_t x)
{
    return x == 0 ? 32 : __builtin_ctz(x);
}
static int32_t futrts_ctzz64(int64_t x)
{
    return x == 0 ? 64 : __builtin_ctzl(x);
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_cosh64(double x)
{
    return cosh(x);
}
static inline double futrts_sinh64(double x)
{
    return sinh(x);
}
static inline double futrts_tanh64(double x)
{
    return tanh(x);
}
static inline double futrts_acosh64(double x)
{
    return acosh(x);
}
static inline double futrts_asinh64(double x)
{
    return asinh(x);
}
static inline double futrts_atanh64(double x)
{
    return atanh(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_fma64(double a, double b, double c)
{
    return fma(a, b, c);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double fmod64(double x, double y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
static inline double futrts_mad64(double a, double b, double c)
{
    return mad(a, b, c);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
static inline double futrts_mad64(double a, double b, double c)
{
    return a * b + c;
}
#endif
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
}
// Start of atomics.h

inline int32_t atomic_add_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline float atomic_fadd_f32_global(volatile __global float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline float atomic_fadd_f32_local(volatile __local float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __local int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

inline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

// End of atomics.h




__kernel void buildKDtreezireplicate_40593(int32_t nodes_this_lvl_32221,
                                           int32_t pts_per_node_at_lev_32225,
                                           __global unsigned char *mem_39556,
                                           __global unsigned char *mem_39561)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_40593;
    int32_t replicate_ltid_40594;
    int32_t replicate_gid_40595;
    
    replicate_gtid_40593 = get_global_id(0);
    replicate_ltid_40594 = get_local_id(0);
    replicate_gid_40595 = get_group_id(0);
    if (slt64(replicate_gtid_40593, sext_i32_i64(nodes_this_lvl_32221) *
              sext_i32_i64(pts_per_node_at_lev_32225))) {
        ((__global
          int32_t *) mem_39561)[sext_i32_i64(squot32(replicate_gtid_40593,
                                                     pts_per_node_at_lev_32225)) *
                                sext_i32_i64(pts_per_node_at_lev_32225) +
                                sext_i32_i64(replicate_gtid_40593 -
                                squot32(replicate_gtid_40593,
                                        pts_per_node_at_lev_32225) *
                                pts_per_node_at_lev_32225)] = ((__global
                                                                int32_t *) mem_39556)[sext_i32_i64(replicate_gtid_40593 -
                                                                                      squot32(replicate_gtid_40593,
                                                                                              pts_per_node_at_lev_32225) *
                                                                                      pts_per_node_at_lev_32225)];
    }
    
  error_0:
    return;
}
__kernel void buildKDtreeziscan_stage1_37746(__global int *global_failure,
                                             uint scan_arr_mem_40721_backing_offset_0,
                                             uint scan_arr_mem_40719_backing_offset_1,
                                             uint scan_arr_mem_40717_backing_offset_2,
                                             uint scan_arr_mem_40715_backing_offset_3,
                                             int32_t nodes_this_lvl_32221,
                                             int32_t pts_per_node_at_lev_32225,
                                             int32_t lifted_2_radix_sort_step_arg_38052,
                                             int32_t lifted_0_get_bit_arg_38053,
                                             unsigned char res_38054,
                                             unsigned char res_38055, __global
                                             unsigned char *mem_param_39843,
                                             __global unsigned char *mem_39905,
                                             __global unsigned char *mem_39910,
                                             __global unsigned char *mem_39915,
                                             __global unsigned char *mem_39920,
                                             __global unsigned char *mem_39925,
                                             int32_t num_threads_40709)
{
    #define segscan_group_sizze_38155 (buildKDtreezisegscan_group_sizze_37740)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *scan_arr_mem_40721_backing_3 =
                  &shared_mem[scan_arr_mem_40721_backing_offset_0];
    volatile char *scan_arr_mem_40719_backing_2 =
                  &shared_mem[scan_arr_mem_40719_backing_offset_1];
    volatile char *scan_arr_mem_40717_backing_1 =
                  &shared_mem[scan_arr_mem_40717_backing_offset_2];
    volatile char *scan_arr_mem_40715_backing_0 =
                  &shared_mem[scan_arr_mem_40715_backing_offset_3];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40710;
    int32_t local_tid_40711;
    int32_t group_sizze_40714;
    int32_t wave_sizze_40713;
    int32_t group_tid_40712;
    
    global_tid_40710 = get_global_id(0);
    local_tid_40711 = get_local_id(0);
    group_sizze_40714 = get_local_size(0);
    wave_sizze_40713 = LOCKSTEP_WIDTH;
    group_tid_40712 = get_group_id(0);
    
    int32_t phys_tid_37746;
    
    phys_tid_37746 = global_tid_40710;
    
    __local char *scan_arr_mem_40715;
    __local char *scan_arr_mem_40717;
    __local char *scan_arr_mem_40719;
    __local char *scan_arr_mem_40721;
    
    scan_arr_mem_40715 = (__local char *) scan_arr_mem_40715_backing_0;
    scan_arr_mem_40717 = (__local char *) scan_arr_mem_40717_backing_1;
    scan_arr_mem_40719 = (__local char *) scan_arr_mem_40719_backing_2;
    scan_arr_mem_40721 = (__local char *) scan_arr_mem_40721_backing_3;
    
    int32_t x_38163;
    int32_t x_38164;
    int32_t x_38165;
    int32_t x_38166;
    int32_t x_38167;
    int32_t x_38168;
    int32_t x_38169;
    int32_t x_38170;
    
    x_38163 = 0;
    x_38164 = 0;
    x_38165 = 0;
    x_38166 = 0;
    for (int32_t j_40723 = 0; j_40723 < sdiv_up32(nodes_this_lvl_32221 *
                                                  pts_per_node_at_lev_32225,
                                                  num_threads_40709);
         j_40723++) {
        int32_t chunk_offset_40724 = segscan_group_sizze_38155 * j_40723 +
                group_tid_40712 * (segscan_group_sizze_38155 *
                                   sdiv_up32(nodes_this_lvl_32221 *
                                             pts_per_node_at_lev_32225,
                                             num_threads_40709));
        int32_t flat_idx_40725 = chunk_offset_40724 + local_tid_40711;
        int32_t gtid_37735 = squot32(flat_idx_40725, pts_per_node_at_lev_32225);
        int32_t gtid_37745 = flat_idx_40725 - squot32(flat_idx_40725,
                                                      pts_per_node_at_lev_32225) *
                pts_per_node_at_lev_32225;
        
        // threads in bounds read input
        {
            if (slt32(gtid_37735, nodes_this_lvl_32221) && slt32(gtid_37745,
                                                                 pts_per_node_at_lev_32225)) {
                float x_38176 = ((__global
                                  float *) mem_param_39843)[sext_i32_i64(gtid_37735) *
                                                            sext_i32_i64(pts_per_node_at_lev_32225) +
                                                            sext_i32_i64(gtid_37745)];
                int32_t i32_arg_38177;
                
                i32_arg_38177 = futrts_to_bits32(x_38176);
                
                int32_t unsign_arg_38178 = ashr32(i32_arg_38177,
                                                  lifted_0_get_bit_arg_38053);
                int32_t unsign_arg_38179 = 1 & unsign_arg_38178;
                int32_t unsign_arg_38180 = ashr32(i32_arg_38177, 31);
                int32_t unsign_arg_38181 = 1 & unsign_arg_38180;
                bool cond_38182 = unsign_arg_38181 == 1;
                bool x_38183 = !cond_38182;
                bool y_38184 = res_38054 && x_38183;
                bool cond_38185 = cond_38182 || y_38184;
                int32_t res_38186;
                
                if (cond_38185) {
                    int32_t res_38187 = 1 ^ unsign_arg_38179;
                    
                    res_38186 = res_38187;
                } else {
                    res_38186 = unsign_arg_38179;
                }
                
                int32_t x_38188 = mul32(2, res_38186);
                int32_t unsign_arg_38189 = ashr32(i32_arg_38177,
                                                  lifted_2_radix_sort_step_arg_38052);
                int32_t unsign_arg_38190 = 1 & unsign_arg_38189;
                bool y_38191 = res_38055 && x_38183;
                bool cond_38192 = cond_38182 || y_38191;
                int32_t res_38193;
                
                if (cond_38192) {
                    int32_t res_38194 = 1 ^ unsign_arg_38190;
                    
                    res_38193 = res_38194;
                } else {
                    res_38193 = unsign_arg_38190;
                }
                
                int32_t res_38195 = add32(x_38188, res_38193);
                bool cond_38196 = res_38195 == 0;
                int32_t res_38197 = btoi_bool_i32(cond_38196);
                int32_t res_38198;
                int32_t res_38199;
                int32_t res_38200;
                
                if (cond_38196) {
                    res_38198 = 0;
                    res_38199 = 0;
                    res_38200 = 0;
                } else {
                    bool cond_38201 = res_38195 == 1;
                    int32_t res_38202 = btoi_bool_i32(cond_38201);
                    int32_t res_38203;
                    int32_t res_38204;
                    
                    if (cond_38201) {
                        res_38203 = 0;
                        res_38204 = 0;
                    } else {
                        bool cond_38205 = res_38195 == 2;
                        int32_t res_38206 = btoi_bool_i32(cond_38205);
                        bool cond_neg_38207 = !cond_38205;
                        int32_t res_38208 = btoi_bool_i32(cond_neg_38207);
                        
                        res_38203 = res_38206;
                        res_38204 = res_38208;
                    }
                    res_38198 = res_38202;
                    res_38199 = res_38203;
                    res_38200 = res_38204;
                }
                // write to-scan values to parameters
                {
                    x_38167 = res_38197;
                    x_38168 = res_38198;
                    x_38169 = res_38199;
                    x_38170 = res_38200;
                }
                // write mapped values results to global memory
                {
                    ((__global int32_t *) mem_39925)[sext_i32_i64(gtid_37735) *
                                                     sext_i32_i64(pts_per_node_at_lev_32225) +
                                                     sext_i32_i64(gtid_37745)] =
                        res_38195;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_37735, nodes_this_lvl_32221) &&
                      slt32(gtid_37745, pts_per_node_at_lev_32225))) {
                    x_38167 = 0;
                    x_38168 = 0;
                    x_38169 = 0;
                    x_38170 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t res_38171 = add32(x_38163, x_38167);
                int32_t res_38172 = add32(x_38164, x_38168);
                int32_t res_38173 = add32(x_38165, x_38169);
                int32_t res_38174 = add32(x_38166, x_38170);
                
                ((__local
                  int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711)] =
                    res_38171;
                ((__local
                  int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711)] =
                    res_38172;
                ((__local
                  int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711)] =
                    res_38173;
                ((__local
                  int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711)] =
                    res_38174;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_40726;
            int32_t x_40727;
            int32_t x_40728;
            int32_t x_40729;
            int32_t x_40730;
            int32_t x_40731;
            int32_t x_40732;
            int32_t x_40733;
            int32_t x_40738;
            int32_t x_40739;
            int32_t x_40740;
            int32_t x_40741;
            int32_t x_40742;
            int32_t x_40743;
            int32_t x_40744;
            int32_t x_40745;
            int32_t skip_threads_40750;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_40711, segscan_group_sizze_38155)) {
                    x_40730 = ((volatile __local
                                int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711)];
                    x_40731 = ((volatile __local
                                int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711)];
                    x_40732 = ((volatile __local
                                int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711)];
                    x_40733 = ((volatile __local
                                int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711)];
                    if ((local_tid_40711 - squot32(local_tid_40711, 32) * 32) ==
                        0) {
                        x_40726 = x_40730;
                        x_40727 = x_40731;
                        x_40728 = x_40732;
                        x_40729 = x_40733;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_40750 = 1;
                while (slt32(skip_threads_40750, 32)) {
                    if (sle32(skip_threads_40750, local_tid_40711 -
                              squot32(local_tid_40711, 32) * 32) &&
                        slt32(local_tid_40711, segscan_group_sizze_38155)) {
                        // read operands
                        {
                            x_40726 = ((volatile __local
                                        int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711 -
                                                                       skip_threads_40750)];
                            x_40727 = ((volatile __local
                                        int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711 -
                                                                       skip_threads_40750)];
                            x_40728 = ((volatile __local
                                        int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711 -
                                                                       skip_threads_40750)];
                            x_40729 = ((volatile __local
                                        int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711 -
                                                                       skip_threads_40750)];
                        }
                        // perform operation
                        {
                            bool inactive_40751 = slt32(srem32(local_tid_40711 +
                                                               chunk_offset_40724,
                                                               pts_per_node_at_lev_32225),
                                                        local_tid_40711 +
                                                        chunk_offset_40724 -
                                                        (local_tid_40711 -
                                                         skip_threads_40750 +
                                                         chunk_offset_40724));
                            
                            if (inactive_40751) {
                                x_40726 = x_40730;
                                x_40727 = x_40731;
                                x_40728 = x_40732;
                                x_40729 = x_40733;
                            }
                            if (!inactive_40751) {
                                int32_t res_40734 = add32(x_40726, x_40730);
                                int32_t res_40735 = add32(x_40727, x_40731);
                                int32_t res_40736 = add32(x_40728, x_40732);
                                int32_t res_40737 = add32(x_40729, x_40733);
                                
                                x_40726 = res_40734;
                                x_40727 = res_40735;
                                x_40728 = res_40736;
                                x_40729 = res_40737;
                            }
                        }
                    }
                    if (sle32(wave_sizze_40713, skip_threads_40750)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_40750, local_tid_40711 -
                              squot32(local_tid_40711, 32) * 32) &&
                        slt32(local_tid_40711, segscan_group_sizze_38155)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711)] =
                                x_40726;
                            x_40730 = x_40726;
                            ((volatile __local
                              int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711)] =
                                x_40727;
                            x_40731 = x_40727;
                            ((volatile __local
                              int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711)] =
                                x_40728;
                            x_40732 = x_40728;
                            ((volatile __local
                              int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711)] =
                                x_40729;
                            x_40733 = x_40729;
                        }
                    }
                    if (sle32(wave_sizze_40713, skip_threads_40750)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_40750 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_40711 - squot32(local_tid_40711, 32) * 32) ==
                    31 && slt32(local_tid_40711, segscan_group_sizze_38155)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_40715)[sext_i32_i64(squot32(local_tid_40711,
                                                                          32))] =
                        x_40726;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40717)[sext_i32_i64(squot32(local_tid_40711,
                                                                          32))] =
                        x_40727;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40719)[sext_i32_i64(squot32(local_tid_40711,
                                                                          32))] =
                        x_40728;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40721)[sext_i32_i64(squot32(local_tid_40711,
                                                                          32))] =
                        x_40729;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_40752;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_40711, 32) == 0 &&
                        slt32(local_tid_40711, segscan_group_sizze_38155)) {
                        x_40742 = ((volatile __local
                                    int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711)];
                        x_40743 = ((volatile __local
                                    int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711)];
                        x_40744 = ((volatile __local
                                    int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711)];
                        x_40745 = ((volatile __local
                                    int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711)];
                        if ((local_tid_40711 - squot32(local_tid_40711, 32) *
                             32) == 0) {
                            x_40738 = x_40742;
                            x_40739 = x_40743;
                            x_40740 = x_40744;
                            x_40741 = x_40745;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_40752 = 1;
                    while (slt32(skip_threads_40752, 32)) {
                        if (sle32(skip_threads_40752, local_tid_40711 -
                                  squot32(local_tid_40711, 32) * 32) &&
                            (squot32(local_tid_40711, 32) == 0 &&
                             slt32(local_tid_40711,
                                   segscan_group_sizze_38155))) {
                            // read operands
                            {
                                x_40738 = ((volatile __local
                                            int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711 -
                                                                           skip_threads_40752)];
                                x_40739 = ((volatile __local
                                            int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711 -
                                                                           skip_threads_40752)];
                                x_40740 = ((volatile __local
                                            int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711 -
                                                                           skip_threads_40752)];
                                x_40741 = ((volatile __local
                                            int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711 -
                                                                           skip_threads_40752)];
                            }
                            // perform operation
                            {
                                bool inactive_40753 =
                                     slt32(srem32(local_tid_40711 * 32 + 32 -
                                                  1 + chunk_offset_40724,
                                                  pts_per_node_at_lev_32225),
                                           local_tid_40711 * 32 + 32 - 1 +
                                           chunk_offset_40724 -
                                           ((local_tid_40711 -
                                             skip_threads_40752) * 32 + 32 - 1 +
                                            chunk_offset_40724));
                                
                                if (inactive_40753) {
                                    x_40738 = x_40742;
                                    x_40739 = x_40743;
                                    x_40740 = x_40744;
                                    x_40741 = x_40745;
                                }
                                if (!inactive_40753) {
                                    int32_t res_40746 = add32(x_40738, x_40742);
                                    int32_t res_40747 = add32(x_40739, x_40743);
                                    int32_t res_40748 = add32(x_40740, x_40744);
                                    int32_t res_40749 = add32(x_40741, x_40745);
                                    
                                    x_40738 = res_40746;
                                    x_40739 = res_40747;
                                    x_40740 = res_40748;
                                    x_40741 = res_40749;
                                }
                            }
                        }
                        if (sle32(wave_sizze_40713, skip_threads_40752)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_40752, local_tid_40711 -
                                  squot32(local_tid_40711, 32) * 32) &&
                            (squot32(local_tid_40711, 32) == 0 &&
                             slt32(local_tid_40711,
                                   segscan_group_sizze_38155))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711)] =
                                    x_40738;
                                x_40742 = x_40738;
                                ((volatile __local
                                  int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711)] =
                                    x_40739;
                                x_40743 = x_40739;
                                ((volatile __local
                                  int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711)] =
                                    x_40740;
                                x_40744 = x_40740;
                                ((volatile __local
                                  int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711)] =
                                    x_40741;
                                x_40745 = x_40741;
                            }
                        }
                        if (sle32(wave_sizze_40713, skip_threads_40752)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_40752 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_40711, 32) == 0 ||
                      !slt32(local_tid_40711, segscan_group_sizze_38155))) {
                    // read operands
                    {
                        x_40730 = x_40726;
                        x_40731 = x_40727;
                        x_40732 = x_40728;
                        x_40733 = x_40729;
                        x_40726 = ((__local
                                    int32_t *) scan_arr_mem_40715)[sext_i32_i64(squot32(local_tid_40711,
                                                                                        32) -
                                                                   1)];
                        x_40727 = ((__local
                                    int32_t *) scan_arr_mem_40717)[sext_i32_i64(squot32(local_tid_40711,
                                                                                        32) -
                                                                   1)];
                        x_40728 = ((__local
                                    int32_t *) scan_arr_mem_40719)[sext_i32_i64(squot32(local_tid_40711,
                                                                                        32) -
                                                                   1)];
                        x_40729 = ((__local
                                    int32_t *) scan_arr_mem_40721)[sext_i32_i64(squot32(local_tid_40711,
                                                                                        32) -
                                                                   1)];
                    }
                    // perform operation
                    {
                        bool inactive_40754 = slt32(srem32(local_tid_40711 +
                                                           chunk_offset_40724,
                                                           pts_per_node_at_lev_32225),
                                                    local_tid_40711 +
                                                    chunk_offset_40724 -
                                                    (squot32(local_tid_40711,
                                                             32) * 32 - 1 +
                                                     chunk_offset_40724));
                        
                        if (inactive_40754) {
                            x_40726 = x_40730;
                            x_40727 = x_40731;
                            x_40728 = x_40732;
                            x_40729 = x_40733;
                        }
                        if (!inactive_40754) {
                            int32_t res_40734 = add32(x_40726, x_40730);
                            int32_t res_40735 = add32(x_40727, x_40731);
                            int32_t res_40736 = add32(x_40728, x_40732);
                            int32_t res_40737 = add32(x_40729, x_40733);
                            
                            x_40726 = res_40734;
                            x_40727 = res_40735;
                            x_40728 = res_40736;
                            x_40729 = res_40737;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711)] =
                            x_40726;
                        ((__local
                          int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711)] =
                            x_40727;
                        ((__local
                          int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711)] =
                            x_40728;
                        ((__local
                          int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711)] =
                            x_40729;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_40711, 32) == 0) {
                    ((__local
                      int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711)] =
                        x_40730;
                    ((__local
                      int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711)] =
                        x_40731;
                    ((__local
                      int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711)] =
                        x_40732;
                    ((__local
                      int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711)] =
                        x_40733;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_37735, nodes_this_lvl_32221) && slt32(gtid_37745,
                                                                     pts_per_node_at_lev_32225)) {
                    ((__global int32_t *) mem_39905)[sext_i32_i64(gtid_37735) *
                                                     sext_i32_i64(pts_per_node_at_lev_32225) +
                                                     sext_i32_i64(gtid_37745)] =
                        ((__local
                          int32_t *) scan_arr_mem_40715)[sext_i32_i64(local_tid_40711)];
                    ((__global int32_t *) mem_39910)[sext_i32_i64(gtid_37735) *
                                                     sext_i32_i64(pts_per_node_at_lev_32225) +
                                                     sext_i32_i64(gtid_37745)] =
                        ((__local
                          int32_t *) scan_arr_mem_40717)[sext_i32_i64(local_tid_40711)];
                    ((__global int32_t *) mem_39915)[sext_i32_i64(gtid_37735) *
                                                     sext_i32_i64(pts_per_node_at_lev_32225) +
                                                     sext_i32_i64(gtid_37745)] =
                        ((__local
                          int32_t *) scan_arr_mem_40719)[sext_i32_i64(local_tid_40711)];
                    ((__global int32_t *) mem_39920)[sext_i32_i64(gtid_37735) *
                                                     sext_i32_i64(pts_per_node_at_lev_32225) +
                                                     sext_i32_i64(gtid_37745)] =
                        ((__local
                          int32_t *) scan_arr_mem_40721)[sext_i32_i64(local_tid_40711)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_40755 = slt32(srem32(chunk_offset_40724 +
                                                          segscan_group_sizze_38155,
                                                          pts_per_node_at_lev_32225),
                                                   chunk_offset_40724 +
                                                   segscan_group_sizze_38155 -
                                                   (chunk_offset_40724 +
                                                    segscan_group_sizze_38155 -
                                                    1));
                bool should_load_carry_40756 = local_tid_40711 == 0 &&
                     !crosses_segment_40755;
                
                if (should_load_carry_40756) {
                    x_38163 = ((__local
                                int32_t *) scan_arr_mem_40715)[sext_i32_i64(segscan_group_sizze_38155 -
                                                               1)];
                    x_38164 = ((__local
                                int32_t *) scan_arr_mem_40717)[sext_i32_i64(segscan_group_sizze_38155 -
                                                               1)];
                    x_38165 = ((__local
                                int32_t *) scan_arr_mem_40719)[sext_i32_i64(segscan_group_sizze_38155 -
                                                               1)];
                    x_38166 = ((__local
                                int32_t *) scan_arr_mem_40721)[sext_i32_i64(segscan_group_sizze_38155 -
                                                               1)];
                }
                if (!should_load_carry_40756) {
                    x_38163 = 0;
                    x_38164 = 0;
                    x_38165 = 0;
                    x_38166 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_38155
}
__kernel void buildKDtreeziscan_stage2_37746(__global int *global_failure,
                                             uint scan_arr_mem_40768_backing_offset_0,
                                             uint scan_arr_mem_40766_backing_offset_1,
                                             uint scan_arr_mem_40764_backing_offset_2,
                                             uint scan_arr_mem_40762_backing_offset_3,
                                             int32_t nodes_this_lvl_32221,
                                             int32_t pts_per_node_at_lev_32225,
                                             __global unsigned char *mem_39905,
                                             __global unsigned char *mem_39910,
                                             __global unsigned char *mem_39915,
                                             __global unsigned char *mem_39920,
                                             int32_t stage1_num_groups_40708,
                                             int32_t num_threads_40709)
{
    #define segscan_group_sizze_38155 (buildKDtreezisegscan_group_sizze_37740)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *scan_arr_mem_40768_backing_3 =
                  &shared_mem[scan_arr_mem_40768_backing_offset_0];
    volatile char *scan_arr_mem_40766_backing_2 =
                  &shared_mem[scan_arr_mem_40766_backing_offset_1];
    volatile char *scan_arr_mem_40764_backing_1 =
                  &shared_mem[scan_arr_mem_40764_backing_offset_2];
    volatile char *scan_arr_mem_40762_backing_0 =
                  &shared_mem[scan_arr_mem_40762_backing_offset_3];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40757;
    int32_t local_tid_40758;
    int32_t group_sizze_40761;
    int32_t wave_sizze_40760;
    int32_t group_tid_40759;
    
    global_tid_40757 = get_global_id(0);
    local_tid_40758 = get_local_id(0);
    group_sizze_40761 = get_local_size(0);
    wave_sizze_40760 = LOCKSTEP_WIDTH;
    group_tid_40759 = get_group_id(0);
    
    int32_t phys_tid_37746;
    
    phys_tid_37746 = global_tid_40757;
    
    __local char *scan_arr_mem_40762;
    __local char *scan_arr_mem_40764;
    __local char *scan_arr_mem_40766;
    __local char *scan_arr_mem_40768;
    
    scan_arr_mem_40762 = (__local char *) scan_arr_mem_40762_backing_0;
    scan_arr_mem_40764 = (__local char *) scan_arr_mem_40764_backing_1;
    scan_arr_mem_40766 = (__local char *) scan_arr_mem_40766_backing_2;
    scan_arr_mem_40768 = (__local char *) scan_arr_mem_40768_backing_3;
    
    int32_t flat_idx_40770;
    
    flat_idx_40770 = (local_tid_40758 + 1) * (segscan_group_sizze_38155 *
                                              sdiv_up32(nodes_this_lvl_32221 *
                                                        pts_per_node_at_lev_32225,
                                                        num_threads_40709)) - 1;
    
    int32_t gtid_37735;
    
    gtid_37735 = squot32(flat_idx_40770, pts_per_node_at_lev_32225);
    
    int32_t gtid_37745;
    
    gtid_37745 = flat_idx_40770 - squot32(flat_idx_40770,
                                          pts_per_node_at_lev_32225) *
        pts_per_node_at_lev_32225;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_37735, nodes_this_lvl_32221) && slt32(gtid_37745,
                                                             pts_per_node_at_lev_32225)) {
            ((__local
              int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)] =
                ((__global int32_t *) mem_39905)[sext_i32_i64(gtid_37735) *
                                                 sext_i32_i64(pts_per_node_at_lev_32225) +
                                                 sext_i32_i64(gtid_37745)];
            ((__local
              int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)] =
                ((__global int32_t *) mem_39910)[sext_i32_i64(gtid_37735) *
                                                 sext_i32_i64(pts_per_node_at_lev_32225) +
                                                 sext_i32_i64(gtid_37745)];
            ((__local
              int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)] =
                ((__global int32_t *) mem_39915)[sext_i32_i64(gtid_37735) *
                                                 sext_i32_i64(pts_per_node_at_lev_32225) +
                                                 sext_i32_i64(gtid_37745)];
            ((__local
              int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)] =
                ((__global int32_t *) mem_39920)[sext_i32_i64(gtid_37735) *
                                                 sext_i32_i64(pts_per_node_at_lev_32225) +
                                                 sext_i32_i64(gtid_37745)];
        } else {
            ((__local
              int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)] = 0;
            ((__local
              int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)] = 0;
            ((__local
              int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)] = 0;
            ((__local
              int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_38163;
    int32_t x_38164;
    int32_t x_38165;
    int32_t x_38166;
    int32_t x_38167;
    int32_t x_38168;
    int32_t x_38169;
    int32_t x_38170;
    int32_t x_40771;
    int32_t x_40772;
    int32_t x_40773;
    int32_t x_40774;
    int32_t x_40775;
    int32_t x_40776;
    int32_t x_40777;
    int32_t x_40778;
    int32_t skip_threads_40783;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_40758, stage1_num_groups_40708)) {
            x_38167 = ((volatile __local
                        int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)];
            x_38168 = ((volatile __local
                        int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)];
            x_38169 = ((volatile __local
                        int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)];
            x_38170 = ((volatile __local
                        int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)];
            if ((local_tid_40758 - squot32(local_tid_40758, 32) * 32) == 0) {
                x_38163 = x_38167;
                x_38164 = x_38168;
                x_38165 = x_38169;
                x_38166 = x_38170;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_40783 = 1;
        while (slt32(skip_threads_40783, 32)) {
            if (sle32(skip_threads_40783, local_tid_40758 -
                      squot32(local_tid_40758, 32) * 32) &&
                slt32(local_tid_40758, stage1_num_groups_40708)) {
                // read operands
                {
                    x_38163 = ((volatile __local
                                int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758 -
                                                               skip_threads_40783)];
                    x_38164 = ((volatile __local
                                int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758 -
                                                               skip_threads_40783)];
                    x_38165 = ((volatile __local
                                int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758 -
                                                               skip_threads_40783)];
                    x_38166 = ((volatile __local
                                int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758 -
                                                               skip_threads_40783)];
                }
                // perform operation
                {
                    bool inactive_40784 = slt32(srem32((local_tid_40758 + 1) *
                                                       (segscan_group_sizze_38155 *
                                                        sdiv_up32(nodes_this_lvl_32221 *
                                                                  pts_per_node_at_lev_32225,
                                                                  num_threads_40709)) -
                                                       1,
                                                       pts_per_node_at_lev_32225),
                                                (local_tid_40758 + 1) *
                                                (segscan_group_sizze_38155 *
                                                 sdiv_up32(nodes_this_lvl_32221 *
                                                           pts_per_node_at_lev_32225,
                                                           num_threads_40709)) -
                                                1 - ((local_tid_40758 -
                                                      skip_threads_40783 + 1) *
                                                     (segscan_group_sizze_38155 *
                                                      sdiv_up32(nodes_this_lvl_32221 *
                                                                pts_per_node_at_lev_32225,
                                                                num_threads_40709)) -
                                                     1));
                    
                    if (inactive_40784) {
                        x_38163 = x_38167;
                        x_38164 = x_38168;
                        x_38165 = x_38169;
                        x_38166 = x_38170;
                    }
                    if (!inactive_40784) {
                        int32_t res_38171 = add32(x_38163, x_38167);
                        int32_t res_38172 = add32(x_38164, x_38168);
                        int32_t res_38173 = add32(x_38165, x_38169);
                        int32_t res_38174 = add32(x_38166, x_38170);
                        
                        x_38163 = res_38171;
                        x_38164 = res_38172;
                        x_38165 = res_38173;
                        x_38166 = res_38174;
                    }
                }
            }
            if (sle32(wave_sizze_40760, skip_threads_40783)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_40783, local_tid_40758 -
                      squot32(local_tid_40758, 32) * 32) &&
                slt32(local_tid_40758, stage1_num_groups_40708)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)] =
                        x_38163;
                    x_38167 = x_38163;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)] =
                        x_38164;
                    x_38168 = x_38164;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)] =
                        x_38165;
                    x_38169 = x_38165;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)] =
                        x_38166;
                    x_38170 = x_38166;
                }
            }
            if (sle32(wave_sizze_40760, skip_threads_40783)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_40783 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_40758 - squot32(local_tid_40758, 32) * 32) == 31 &&
            slt32(local_tid_40758, stage1_num_groups_40708)) {
            ((volatile __local
              int32_t *) scan_arr_mem_40762)[sext_i32_i64(squot32(local_tid_40758,
                                                                  32))] =
                x_38163;
            ((volatile __local
              int32_t *) scan_arr_mem_40764)[sext_i32_i64(squot32(local_tid_40758,
                                                                  32))] =
                x_38164;
            ((volatile __local
              int32_t *) scan_arr_mem_40766)[sext_i32_i64(squot32(local_tid_40758,
                                                                  32))] =
                x_38165;
            ((volatile __local
              int32_t *) scan_arr_mem_40768)[sext_i32_i64(squot32(local_tid_40758,
                                                                  32))] =
                x_38166;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_40785;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_40758, 32) == 0 && slt32(local_tid_40758,
                                                           stage1_num_groups_40708)) {
                x_40775 = ((volatile __local
                            int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)];
                x_40776 = ((volatile __local
                            int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)];
                x_40777 = ((volatile __local
                            int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)];
                x_40778 = ((volatile __local
                            int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)];
                if ((local_tid_40758 - squot32(local_tid_40758, 32) * 32) ==
                    0) {
                    x_40771 = x_40775;
                    x_40772 = x_40776;
                    x_40773 = x_40777;
                    x_40774 = x_40778;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_40785 = 1;
            while (slt32(skip_threads_40785, 32)) {
                if (sle32(skip_threads_40785, local_tid_40758 -
                          squot32(local_tid_40758, 32) * 32) &&
                    (squot32(local_tid_40758, 32) == 0 && slt32(local_tid_40758,
                                                                stage1_num_groups_40708))) {
                    // read operands
                    {
                        x_40771 = ((volatile __local
                                    int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758 -
                                                                   skip_threads_40785)];
                        x_40772 = ((volatile __local
                                    int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758 -
                                                                   skip_threads_40785)];
                        x_40773 = ((volatile __local
                                    int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758 -
                                                                   skip_threads_40785)];
                        x_40774 = ((volatile __local
                                    int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758 -
                                                                   skip_threads_40785)];
                    }
                    // perform operation
                    {
                        bool inactive_40786 = slt32(srem32((local_tid_40758 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_38155 *
                                                            sdiv_up32(nodes_this_lvl_32221 *
                                                                      pts_per_node_at_lev_32225,
                                                                      num_threads_40709)) -
                                                           1,
                                                           pts_per_node_at_lev_32225),
                                                    (local_tid_40758 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_38155 *
                                                     sdiv_up32(nodes_this_lvl_32221 *
                                                               pts_per_node_at_lev_32225,
                                                               num_threads_40709)) -
                                                    1 - (((local_tid_40758 -
                                                           skip_threads_40785) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_38155 *
                                                          sdiv_up32(nodes_this_lvl_32221 *
                                                                    pts_per_node_at_lev_32225,
                                                                    num_threads_40709)) -
                                                         1));
                        
                        if (inactive_40786) {
                            x_40771 = x_40775;
                            x_40772 = x_40776;
                            x_40773 = x_40777;
                            x_40774 = x_40778;
                        }
                        if (!inactive_40786) {
                            int32_t res_40779 = add32(x_40771, x_40775);
                            int32_t res_40780 = add32(x_40772, x_40776);
                            int32_t res_40781 = add32(x_40773, x_40777);
                            int32_t res_40782 = add32(x_40774, x_40778);
                            
                            x_40771 = res_40779;
                            x_40772 = res_40780;
                            x_40773 = res_40781;
                            x_40774 = res_40782;
                        }
                    }
                }
                if (sle32(wave_sizze_40760, skip_threads_40785)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_40785, local_tid_40758 -
                          squot32(local_tid_40758, 32) * 32) &&
                    (squot32(local_tid_40758, 32) == 0 && slt32(local_tid_40758,
                                                                stage1_num_groups_40708))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)] =
                            x_40771;
                        x_40775 = x_40771;
                        ((volatile __local
                          int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)] =
                            x_40772;
                        x_40776 = x_40772;
                        ((volatile __local
                          int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)] =
                            x_40773;
                        x_40777 = x_40773;
                        ((volatile __local
                          int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)] =
                            x_40774;
                        x_40778 = x_40774;
                    }
                }
                if (sle32(wave_sizze_40760, skip_threads_40785)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_40785 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_40758, 32) == 0 || !slt32(local_tid_40758,
                                                          stage1_num_groups_40708))) {
            // read operands
            {
                x_38167 = x_38163;
                x_38168 = x_38164;
                x_38169 = x_38165;
                x_38170 = x_38166;
                x_38163 = ((__local
                            int32_t *) scan_arr_mem_40762)[sext_i32_i64(squot32(local_tid_40758,
                                                                                32) -
                                                           1)];
                x_38164 = ((__local
                            int32_t *) scan_arr_mem_40764)[sext_i32_i64(squot32(local_tid_40758,
                                                                                32) -
                                                           1)];
                x_38165 = ((__local
                            int32_t *) scan_arr_mem_40766)[sext_i32_i64(squot32(local_tid_40758,
                                                                                32) -
                                                           1)];
                x_38166 = ((__local
                            int32_t *) scan_arr_mem_40768)[sext_i32_i64(squot32(local_tid_40758,
                                                                                32) -
                                                           1)];
            }
            // perform operation
            {
                bool inactive_40787 = slt32(srem32((local_tid_40758 + 1) *
                                                   (segscan_group_sizze_38155 *
                                                    sdiv_up32(nodes_this_lvl_32221 *
                                                              pts_per_node_at_lev_32225,
                                                              num_threads_40709)) -
                                                   1,
                                                   pts_per_node_at_lev_32225),
                                            (local_tid_40758 + 1) *
                                            (segscan_group_sizze_38155 *
                                             sdiv_up32(nodes_this_lvl_32221 *
                                                       pts_per_node_at_lev_32225,
                                                       num_threads_40709)) - 1 -
                                            ((squot32(local_tid_40758, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_38155 *
                                              sdiv_up32(nodes_this_lvl_32221 *
                                                        pts_per_node_at_lev_32225,
                                                        num_threads_40709)) -
                                             1));
                
                if (inactive_40787) {
                    x_38163 = x_38167;
                    x_38164 = x_38168;
                    x_38165 = x_38169;
                    x_38166 = x_38170;
                }
                if (!inactive_40787) {
                    int32_t res_38171 = add32(x_38163, x_38167);
                    int32_t res_38172 = add32(x_38164, x_38168);
                    int32_t res_38173 = add32(x_38165, x_38169);
                    int32_t res_38174 = add32(x_38166, x_38170);
                    
                    x_38163 = res_38171;
                    x_38164 = res_38172;
                    x_38165 = res_38173;
                    x_38166 = res_38174;
                }
            }
            // write final result
            {
                ((__local
                  int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)] =
                    x_38163;
                ((__local
                  int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)] =
                    x_38164;
                ((__local
                  int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)] =
                    x_38165;
                ((__local
                  int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)] =
                    x_38166;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_40758, 32) == 0) {
            ((__local
              int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)] =
                x_38167;
            ((__local
              int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)] =
                x_38168;
            ((__local
              int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)] =
                x_38169;
            ((__local
              int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)] =
                x_38170;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_37735, nodes_this_lvl_32221) && slt32(gtid_37745,
                                                             pts_per_node_at_lev_32225)) {
            ((__global int32_t *) mem_39905)[sext_i32_i64(gtid_37735) *
                                             sext_i32_i64(pts_per_node_at_lev_32225) +
                                             sext_i32_i64(gtid_37745)] =
                ((__local
                  int32_t *) scan_arr_mem_40762)[sext_i32_i64(local_tid_40758)];
            ((__global int32_t *) mem_39910)[sext_i32_i64(gtid_37735) *
                                             sext_i32_i64(pts_per_node_at_lev_32225) +
                                             sext_i32_i64(gtid_37745)] =
                ((__local
                  int32_t *) scan_arr_mem_40764)[sext_i32_i64(local_tid_40758)];
            ((__global int32_t *) mem_39915)[sext_i32_i64(gtid_37735) *
                                             sext_i32_i64(pts_per_node_at_lev_32225) +
                                             sext_i32_i64(gtid_37745)] =
                ((__local
                  int32_t *) scan_arr_mem_40766)[sext_i32_i64(local_tid_40758)];
            ((__global int32_t *) mem_39920)[sext_i32_i64(gtid_37735) *
                                             sext_i32_i64(pts_per_node_at_lev_32225) +
                                             sext_i32_i64(gtid_37745)] =
                ((__local
                  int32_t *) scan_arr_mem_40768)[sext_i32_i64(local_tid_40758)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_38155
}
__kernel void buildKDtreeziscan_stage3_37746(__global int *global_failure,
                                             int32_t nodes_this_lvl_32221,
                                             int32_t pts_per_node_at_lev_32225,
                                             int32_t num_groups_38156, __global
                                             unsigned char *mem_39905, __global
                                             unsigned char *mem_39910, __global
                                             unsigned char *mem_39915, __global
                                             unsigned char *mem_39920,
                                             int32_t num_threads_40709,
                                             int32_t required_groups_40788)
{
    #define segscan_group_sizze_38155 (buildKDtreezisegscan_group_sizze_37740)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40789;
    int32_t local_tid_40790;
    int32_t group_sizze_40793;
    int32_t wave_sizze_40792;
    int32_t group_tid_40791;
    
    global_tid_40789 = get_global_id(0);
    local_tid_40790 = get_local_id(0);
    group_sizze_40793 = get_local_size(0);
    wave_sizze_40792 = LOCKSTEP_WIDTH;
    group_tid_40791 = get_group_id(0);
    
    int32_t phys_tid_37746;
    
    phys_tid_37746 = global_tid_40789;
    
    int32_t phys_group_id_40794;
    
    phys_group_id_40794 = get_group_id(0);
    for (int32_t i_40795 = 0; i_40795 < sdiv_up32(required_groups_40788 -
                                                  phys_group_id_40794,
                                                  num_groups_38156);
         i_40795++) {
        int32_t virt_group_id_40796 = phys_group_id_40794 + i_40795 *
                num_groups_38156;
        int32_t flat_idx_40797 = virt_group_id_40796 *
                segscan_group_sizze_38155 + local_tid_40790;
        int32_t gtid_37735 = squot32(flat_idx_40797, pts_per_node_at_lev_32225);
        int32_t gtid_37745 = flat_idx_40797 - squot32(flat_idx_40797,
                                                      pts_per_node_at_lev_32225) *
                pts_per_node_at_lev_32225;
        int32_t orig_group_40798 = squot32(flat_idx_40797,
                                           segscan_group_sizze_38155 *
                                           sdiv_up32(nodes_this_lvl_32221 *
                                                     pts_per_node_at_lev_32225,
                                                     num_threads_40709));
        int32_t carry_in_flat_idx_40799 = orig_group_40798 *
                (segscan_group_sizze_38155 * sdiv_up32(nodes_this_lvl_32221 *
                                                       pts_per_node_at_lev_32225,
                                                       num_threads_40709)) - 1;
        
        if (slt32(gtid_37735, nodes_this_lvl_32221) && slt32(gtid_37745,
                                                             pts_per_node_at_lev_32225)) {
            if (!(orig_group_40798 == 0 || (flat_idx_40797 ==
                                            (orig_group_40798 + 1) *
                                            (segscan_group_sizze_38155 *
                                             sdiv_up32(nodes_this_lvl_32221 *
                                                       pts_per_node_at_lev_32225,
                                                       num_threads_40709)) -
                                            1 || slt32(srem32(flat_idx_40797,
                                                              pts_per_node_at_lev_32225),
                                                       flat_idx_40797 -
                                                       carry_in_flat_idx_40799)))) {
                int32_t x_38163;
                int32_t x_38164;
                int32_t x_38165;
                int32_t x_38166;
                int32_t x_38167;
                int32_t x_38168;
                int32_t x_38169;
                int32_t x_38170;
                
                x_38163 = ((__global
                            int32_t *) mem_39905)[sext_i32_i64(squot32(carry_in_flat_idx_40799,
                                                                       pts_per_node_at_lev_32225)) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(carry_in_flat_idx_40799 -
                                                  squot32(carry_in_flat_idx_40799,
                                                          pts_per_node_at_lev_32225) *
                                                  pts_per_node_at_lev_32225)];
                x_38164 = ((__global
                            int32_t *) mem_39910)[sext_i32_i64(squot32(carry_in_flat_idx_40799,
                                                                       pts_per_node_at_lev_32225)) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(carry_in_flat_idx_40799 -
                                                  squot32(carry_in_flat_idx_40799,
                                                          pts_per_node_at_lev_32225) *
                                                  pts_per_node_at_lev_32225)];
                x_38165 = ((__global
                            int32_t *) mem_39915)[sext_i32_i64(squot32(carry_in_flat_idx_40799,
                                                                       pts_per_node_at_lev_32225)) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(carry_in_flat_idx_40799 -
                                                  squot32(carry_in_flat_idx_40799,
                                                          pts_per_node_at_lev_32225) *
                                                  pts_per_node_at_lev_32225)];
                x_38166 = ((__global
                            int32_t *) mem_39920)[sext_i32_i64(squot32(carry_in_flat_idx_40799,
                                                                       pts_per_node_at_lev_32225)) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(carry_in_flat_idx_40799 -
                                                  squot32(carry_in_flat_idx_40799,
                                                          pts_per_node_at_lev_32225) *
                                                  pts_per_node_at_lev_32225)];
                x_38167 = ((__global
                            int32_t *) mem_39905)[sext_i32_i64(gtid_37735) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(gtid_37745)];
                x_38168 = ((__global
                            int32_t *) mem_39910)[sext_i32_i64(gtid_37735) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(gtid_37745)];
                x_38169 = ((__global
                            int32_t *) mem_39915)[sext_i32_i64(gtid_37735) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(gtid_37745)];
                x_38170 = ((__global
                            int32_t *) mem_39920)[sext_i32_i64(gtid_37735) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(gtid_37745)];
                
                int32_t res_38171;
                
                res_38171 = add32(x_38163, x_38167);
                
                int32_t res_38172 = add32(x_38164, x_38168);
                int32_t res_38173 = add32(x_38165, x_38169);
                int32_t res_38174 = add32(x_38166, x_38170);
                
                x_38163 = res_38171;
                x_38164 = res_38172;
                x_38165 = res_38173;
                x_38166 = res_38174;
                ((__global int32_t *) mem_39905)[sext_i32_i64(gtid_37735) *
                                                 sext_i32_i64(pts_per_node_at_lev_32225) +
                                                 sext_i32_i64(gtid_37745)] =
                    x_38163;
                ((__global int32_t *) mem_39910)[sext_i32_i64(gtid_37735) *
                                                 sext_i32_i64(pts_per_node_at_lev_32225) +
                                                 sext_i32_i64(gtid_37745)] =
                    x_38164;
                ((__global int32_t *) mem_39915)[sext_i32_i64(gtid_37735) *
                                                 sext_i32_i64(pts_per_node_at_lev_32225) +
                                                 sext_i32_i64(gtid_37745)] =
                    x_38165;
                ((__global int32_t *) mem_39920)[sext_i32_i64(gtid_37735) *
                                                 sext_i32_i64(pts_per_node_at_lev_32225) +
                                                 sext_i32_i64(gtid_37745)] =
                    x_38166;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_38155
}
__kernel void buildKDtreezisegmap_36314(__global int *global_failure,
                                        int32_t m_32135, int32_t d_32136,
                                        __global unsigned char *input_mem_39418,
                                        __global unsigned char *mem_39422)
{
    #define segmap_group_sizze_36330 (buildKDtreezisegmap_group_sizze_36317)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40315;
    int32_t local_tid_40316;
    int32_t group_sizze_40319;
    int32_t wave_sizze_40318;
    int32_t group_tid_40317;
    
    global_tid_40315 = get_global_id(0);
    local_tid_40316 = get_local_id(0);
    group_sizze_40319 = get_local_size(0);
    wave_sizze_40318 = LOCKSTEP_WIDTH;
    group_tid_40317 = get_group_id(0);
    
    int32_t phys_tid_36314;
    
    phys_tid_36314 = global_tid_40315;
    
    int32_t gtid_36313;
    
    gtid_36313 = sext_i64_i32(sext_i32_i64(group_tid_40317) *
        sext_i32_i64(segmap_group_sizze_36330) + sext_i32_i64(local_tid_40316));
    if (slt32(gtid_36313, d_32136)) {
        float res_36336;
        float redout_39353 = INFINITY;
        
        for (int32_t i_39354 = 0; i_39354 < m_32135; i_39354++) {
            float x_36340 = ((__global
                              float *) input_mem_39418)[sext_i32_i64(i_39354) *
                                                        sext_i32_i64(d_32136) +
                                                        sext_i32_i64(gtid_36313)];
            float res_36339 = fmin32(x_36340, redout_39353);
            float redout_tmp_40320 = res_36339;
            
            redout_39353 = redout_tmp_40320;
        }
        res_36336 = redout_39353;
        ((__global float *) mem_39422)[sext_i32_i64(gtid_36313)] = res_36336;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36330
}
__kernel void buildKDtreezisegmap_36375(__global int *global_failure,
                                        int32_t m_32135, int32_t d_32136,
                                        __global unsigned char *input_mem_39418,
                                        __global unsigned char *mem_39436)
{
    #define segmap_group_sizze_36391 (buildKDtreezisegmap_group_sizze_36378)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40382;
    int32_t local_tid_40383;
    int32_t group_sizze_40386;
    int32_t wave_sizze_40385;
    int32_t group_tid_40384;
    
    global_tid_40382 = get_global_id(0);
    local_tid_40383 = get_local_id(0);
    group_sizze_40386 = get_local_size(0);
    wave_sizze_40385 = LOCKSTEP_WIDTH;
    group_tid_40384 = get_group_id(0);
    
    int32_t phys_tid_36375;
    
    phys_tid_36375 = global_tid_40382;
    
    int32_t gtid_36374;
    
    gtid_36374 = sext_i64_i32(sext_i32_i64(group_tid_40384) *
        sext_i32_i64(segmap_group_sizze_36391) + sext_i32_i64(local_tid_40383));
    if (slt32(gtid_36374, d_32136)) {
        float res_36397;
        float redout_39355 = -INFINITY;
        
        for (int32_t i_39356 = 0; i_39356 < m_32135; i_39356++) {
            float x_36401 = ((__global
                              float *) input_mem_39418)[sext_i32_i64(i_39356) *
                                                        sext_i32_i64(d_32136) +
                                                        sext_i32_i64(gtid_36374)];
            float res_36400 = fmax32(x_36401, redout_39355);
            float redout_tmp_40387 = res_36400;
            
            redout_39355 = redout_tmp_40387;
        }
        res_36397 = redout_39355;
        ((__global float *) mem_39436)[sext_i32_i64(gtid_36374)] = res_36397;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36391
}
__kernel void buildKDtreezisegmap_36436(__global int *global_failure,
                                        int failure_is_an_option, __global
                                        int *global_failure_args,
                                        int32_t d_32136, int32_t res_32166,
                                        int32_t conc_tmp_32196, int32_t d_32210,
                                        int32_t lev_32216,
                                        int32_t nodes_this_lvl_32221,
                                        int32_t segmap_usable_groups_36536,
                                        __global unsigned char *mem_39449,
                                        __global unsigned char *mem_39455,
                                        __global unsigned char *mem_39458,
                                        __global unsigned char *mem_39493,
                                        __global unsigned char *mem_39508,
                                        __global unsigned char *mem_39511)
{
    #define segmap_group_sizze_36533 (buildKDtreezisegmap_group_sizze_36439)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40484;
    int32_t local_tid_40485;
    int32_t group_sizze_40488;
    int32_t wave_sizze_40487;
    int32_t group_tid_40486;
    
    global_tid_40484 = get_global_id(0);
    local_tid_40485 = get_local_id(0);
    group_sizze_40488 = get_local_size(0);
    wave_sizze_40487 = LOCKSTEP_WIDTH;
    group_tid_40486 = get_group_id(0);
    
    int32_t phys_tid_36436;
    
    phys_tid_36436 = global_tid_40484;
    
    int32_t gtid_36435;
    
    gtid_36435 = sext_i64_i32(sext_i32_i64(group_tid_40486) *
        sext_i32_i64(segmap_group_sizze_36533) + sext_i32_i64(local_tid_40485));
    if (slt32(gtid_36435, nodes_this_lvl_32221)) {
        int32_t x_36540 = add32(nodes_this_lvl_32221, gtid_36435);
        int32_t node_ind_36541 = sub32(x_36540, 1);
        
        for (int32_t i_40489 = 0; i_40489 < conc_tmp_32196; i_40489++) {
            ((__global float *) mem_39493)[sext_i32_i64(phys_tid_36436) +
                                           sext_i32_i64(i_40489) *
                                           sext_i32_i64(segmap_usable_groups_36536 *
                                           segmap_group_sizze_36533)] =
                ((__global float *) mem_39449)[sext_i32_i64(i_40489)];
        }
        
        int32_t res_36543;
        int32_t ancestor_36546 = 0;
        
        for (int32_t i_36545 = 0; i_36545 < lev_32216; i_36545++) {
            int32_t x_36548 = sub32(lev_32216, i_36545);
            int32_t k_36549 = sub32(x_36548, 1);
            int32_t tpk_36550 = 1 << k_36549;
            int32_t x_36551 = sub32(x_36540, tpk_36550);
            bool zzero_36552 = tpk_36550 == 0;
            bool nonzzero_36553 = !zzero_36552;
            bool nonzzero_cert_36554;
            
            if (!nonzzero_36553) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 0) ==
                        -1) {
                        ;
                    }
                    return;
                }
            }
            
            int32_t res_36555 = sdiv32(x_36551, tpk_36550);
            bool x_36556 = sle32(0, ancestor_36546);
            bool y_36557 = slt32(ancestor_36546, res_32166);
            bool bounds_check_36558 = x_36556 && y_36557;
            bool index_certs_36559;
            
            if (!bounds_check_36558) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 1) ==
                        -1) {
                        global_failure_args[0] = ancestor_36546;
                        global_failure_args[1] = res_32166;
                        ;
                    }
                    return;
                }
            }
            
            int32_t anc_dim_36560 = ((__global
                                      int32_t *) mem_39458)[sext_i32_i64(ancestor_36546)];
            int32_t x_36561 = 1 & res_36555;
            bool cond_36562 = x_36561 == 0;
            int32_t lub_ind_36563;
            
            if (cond_36562) {
                lub_ind_36563 = anc_dim_36560;
            } else {
                int32_t res_36564 = add32(d_32210, anc_dim_36560);
                
                lub_ind_36563 = res_36564;
            }
            
            float anc_med_36565 = ((__global
                                    float *) mem_39455)[sext_i32_i64(ancestor_36546)];
            bool res_36566;
            
            res_36566 = futrts_isinf32(anc_med_36565);
            
            bool cond_36567 = !res_36566;
            float lw_val_36568;
            
            if (cond_36567) {
                lw_val_36568 = anc_med_36565;
            } else {
                bool x_36569 = sle32(0, lub_ind_36563);
                bool y_36570 = slt32(lub_ind_36563, conc_tmp_32196);
                bool bounds_check_36571 = x_36569 && y_36570;
                bool index_certs_36572;
                
                if (!bounds_check_36571) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 2) ==
                            -1) {
                            global_failure_args[0] = lub_ind_36563;
                            global_failure_args[1] = conc_tmp_32196;
                            ;
                        }
                        return;
                    }
                }
                
                float res_36573 = ((__global
                                    float *) mem_39493)[sext_i32_i64(phys_tid_36436) +
                                                        sext_i32_i64(lub_ind_36563) *
                                                        sext_i32_i64(segmap_usable_groups_36536 *
                                                        segmap_group_sizze_36533)];
                
                lw_val_36568 = res_36573;
            }
            
            bool x_36574 = sle32(0, lub_ind_36563);
            bool y_36575 = slt32(lub_ind_36563, conc_tmp_32196);
            bool bounds_check_36576 = x_36574 && y_36575;
            bool index_certs_36577;
            
            if (!bounds_check_36576) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 3) ==
                        -1) {
                        global_failure_args[0] = lub_ind_36563;
                        global_failure_args[1] = conc_tmp_32196;
                        ;
                    }
                    return;
                }
            }
            ((__global float *) mem_39493)[sext_i32_i64(phys_tid_36436) +
                                           sext_i32_i64(lub_ind_36563) *
                                           sext_i32_i64(segmap_usable_groups_36536 *
                                           segmap_group_sizze_36533)] =
                lw_val_36568;
            
            int32_t ancestor_tmp_40490 = res_36555;
            
            ancestor_36546 = ancestor_tmp_40490;
        }
        res_36543 = ancestor_36546;
        
        int32_t res_36579;
        float res_36580;
        int32_t redout_39284;
        float redout_39285;
        
        redout_39284 = -1;
        redout_39285 = -INFINITY;
        for (int32_t i_39286 = 0; i_39286 < d_32136; i_39286++) {
            int32_t i_36589 = add32(d_32136, i_39286);
            bool x_36590 = sle32(0, i_36589);
            bool y_36591 = slt32(i_36589, conc_tmp_32196);
            bool bounds_check_36592 = x_36590 && y_36591;
            bool index_certs_36593;
            
            if (!bounds_check_36592) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 4) ==
                        -1) {
                        global_failure_args[0] = i_36589;
                        global_failure_args[1] = conc_tmp_32196;
                        ;
                    }
                    return;
                }
            }
            
            float x_36594 = ((__global
                              float *) mem_39493)[sext_i32_i64(phys_tid_36436) +
                                                  sext_i32_i64(i_36589) *
                                                  sext_i32_i64(segmap_usable_groups_36536 *
                                                  segmap_group_sizze_36533)];
            bool y_36596 = slt32(i_39286, conc_tmp_32196);
            bool index_certs_36598;
            
            if (!y_36596) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 5) ==
                        -1) {
                        global_failure_args[0] = i_39286;
                        global_failure_args[1] = conc_tmp_32196;
                        ;
                    }
                    return;
                }
            }
            
            float y_36599 = ((__global
                              float *) mem_39493)[sext_i32_i64(phys_tid_36436) +
                                                  sext_i32_i64(i_39286) *
                                                  sext_i32_i64(segmap_usable_groups_36536 *
                                                  segmap_group_sizze_36533)];
            float abs_arg_36600 = x_36594 - y_36599;
            float res_36601 = (float) fabs(abs_arg_36600);
            bool cond_36585 = res_36601 <= redout_39285;
            int32_t res_36586;
            
            if (cond_36585) {
                res_36586 = redout_39284;
            } else {
                res_36586 = i_39286;
            }
            
            float res_36587;
            
            if (cond_36585) {
                res_36587 = redout_39285;
            } else {
                res_36587 = res_36601;
            }
            
            int32_t redout_tmp_40492 = res_36586;
            float redout_tmp_40493 = res_36587;
            
            redout_39284 = redout_tmp_40492;
            redout_39285 = redout_tmp_40493;
        }
        res_36579 = redout_39284;
        res_36580 = redout_39285;
        
        bool cond_36602 = node_ind_36541 == 0;
        bool cond_36603 = !cond_36602;
        bool res_36604;
        int32_t res_36605;
        int32_t res_36606;
        bool loop_while_36607;
        int32_t cur_node_36608;
        int32_t res_ind_36609;
        
        loop_while_36607 = cond_36603;
        cur_node_36608 = node_ind_36541;
        res_ind_36609 = -1;
        while (loop_while_36607) {
            int32_t x_36610 = sub32(cur_node_36608, 1);
            int32_t res_36611 = sdiv32(x_36610, 2);
            bool x_36612 = sle32(0, res_36611);
            bool y_36613 = slt32(res_36611, res_32166);
            bool bounds_check_36614 = x_36612 && y_36613;
            bool index_certs_36615;
            
            if (!bounds_check_36614) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 6) ==
                        -1) {
                        global_failure_args[0] = res_36611;
                        global_failure_args[1] = res_32166;
                        ;
                    }
                    return;
                }
            }
            
            int32_t x_36616 = ((__global
                                int32_t *) mem_39458)[sext_i32_i64(res_36611)];
            bool cond_36617 = x_36616 == res_36579;
            int32_t res_ind_36618;
            
            if (cond_36617) {
                res_ind_36618 = res_36611;
            } else {
                res_ind_36618 = -1;
            }
            
            bool cond_36619 = res_36611 == 0;
            bool cond_36620 = !cond_36619;
            bool eq_x_y_36621 = -1 == res_36611;
            bool p_and_eq_x_y_36622 = cond_36617 && eq_x_y_36621;
            bool not_p_36623 = !cond_36617;
            bool res_36624 = p_and_eq_x_y_36622 || not_p_36623;
            bool x_36625 = cond_36620 && res_36624;
            bool loop_while_tmp_40494 = x_36625;
            int32_t cur_node_tmp_40495 = res_36611;
            int32_t res_ind_tmp_40496 = res_ind_36618;
            
            loop_while_36607 = loop_while_tmp_40494;
            cur_node_36608 = cur_node_tmp_40495;
            res_ind_36609 = res_ind_tmp_40496;
        }
        res_36604 = loop_while_36607;
        res_36605 = cur_node_36608;
        res_36606 = res_ind_36609;
        ((__global int32_t *) mem_39508)[sext_i32_i64(gtid_36435)] = res_36579;
        ((__global int32_t *) mem_39511)[sext_i32_i64(gtid_36435)] = res_36606;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36533
}
__kernel void buildKDtreezisegmap_36628(__global int *global_failure,
                                        int failure_is_an_option, __global
                                        int *global_failure_args,
                                        int32_t res_32166,
                                        int32_t nodes_this_lvl_32221, __global
                                        unsigned char *mem_39458, __global
                                        unsigned char *mem_39544, __global
                                        unsigned char *mem_39551)
{
    #define segmap_group_sizze_36876 (buildKDtreezisegmap_group_sizze_36631)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40583;
    int32_t local_tid_40584;
    int32_t group_sizze_40587;
    int32_t wave_sizze_40586;
    int32_t group_tid_40585;
    
    global_tid_40583 = get_global_id(0);
    local_tid_40584 = get_local_id(0);
    group_sizze_40587 = get_local_size(0);
    wave_sizze_40586 = LOCKSTEP_WIDTH;
    group_tid_40585 = get_group_id(0);
    
    int32_t phys_tid_36628;
    
    phys_tid_36628 = global_tid_40583;
    
    int32_t gtid_36627;
    
    gtid_36627 = sext_i64_i32(sext_i32_i64(group_tid_40585) *
        sext_i32_i64(segmap_group_sizze_36876) + sext_i32_i64(local_tid_40584));
    if (slt32(gtid_36627, nodes_this_lvl_32221)) {
        int32_t binop_x_39295 = add32(nodes_this_lvl_32221, gtid_36627);
        int32_t index_primexp_39296 = sub32(binop_x_39295, 1);
        int32_t res_36882 = ((__global
                              int32_t *) mem_39544)[sext_i32_i64(gtid_36627)];
        bool cond_36883 = index_primexp_39296 == 0;
        bool cond_36884 = !cond_36883;
        bool res_36885;
        int32_t res_36886;
        int32_t res_36887;
        bool loop_while_36888;
        int32_t cur_node_36889;
        int32_t res_ind_36890;
        
        loop_while_36888 = cond_36884;
        cur_node_36889 = index_primexp_39296;
        res_ind_36890 = -1;
        while (loop_while_36888) {
            int32_t x_36891 = sub32(cur_node_36889, 1);
            int32_t res_36892 = sdiv32(x_36891, 2);
            bool x_36893 = sle32(0, res_36892);
            bool y_36894 = slt32(res_36892, res_32166);
            bool bounds_check_36895 = x_36893 && y_36894;
            bool index_certs_36896;
            
            if (!bounds_check_36895) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 15) ==
                        -1) {
                        global_failure_args[0] = res_36892;
                        global_failure_args[1] = res_32166;
                        ;
                    }
                    return;
                }
            }
            
            int32_t x_36897 = ((__global
                                int32_t *) mem_39458)[sext_i32_i64(res_36892)];
            bool cond_36898 = x_36897 == res_36882;
            int32_t res_ind_36899;
            
            if (cond_36898) {
                res_ind_36899 = res_36892;
            } else {
                res_ind_36899 = -1;
            }
            
            bool cond_36900 = res_36892 == 0;
            bool cond_36901 = !cond_36900;
            bool eq_x_y_36902 = -1 == res_36892;
            bool p_and_eq_x_y_36903 = cond_36898 && eq_x_y_36902;
            bool not_p_36904 = !cond_36898;
            bool res_36905 = p_and_eq_x_y_36903 || not_p_36904;
            bool x_36906 = cond_36901 && res_36905;
            bool loop_while_tmp_40588 = x_36906;
            int32_t cur_node_tmp_40589 = res_36892;
            int32_t res_ind_tmp_40590 = res_ind_36899;
            
            loop_while_36888 = loop_while_tmp_40588;
            cur_node_36889 = cur_node_tmp_40589;
            res_ind_36890 = res_ind_tmp_40590;
        }
        res_36885 = loop_while_36888;
        res_36886 = cur_node_36889;
        res_36887 = res_ind_36890;
        ((__global int32_t *) mem_39551)[sext_i32_i64(gtid_36627)] = res_36887;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36876
}
__kernel void buildKDtreezisegmap_36748(__global int *global_failure,
                                        int failure_is_an_option, __global
                                        int *global_failure_args,
                                        int32_t res_32166,
                                        int32_t conc_tmp_32196, int32_t d_32210,
                                        int32_t lev_32216,
                                        int32_t nodes_this_lvl_32221,
                                        int32_t num_groups_36799, __global
                                        unsigned char *mem_39449, __global
                                        unsigned char *mem_39455, __global
                                        unsigned char *mem_39458, __global
                                        unsigned char *mem_39515, __global
                                        unsigned char *mem_39532, __global
                                        unsigned char *mem_39535)
{
    #define segmap_group_sizze_36798 (buildKDtreezisegmap_group_sizze_36751)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40497;
    int32_t local_tid_40498;
    int32_t group_sizze_40501;
    int32_t wave_sizze_40500;
    int32_t group_tid_40499;
    
    global_tid_40497 = get_global_id(0);
    local_tid_40498 = get_local_id(0);
    group_sizze_40501 = get_local_size(0);
    wave_sizze_40500 = LOCKSTEP_WIDTH;
    group_tid_40499 = get_group_id(0);
    
    int32_t phys_tid_36748;
    
    phys_tid_36748 = global_tid_40497;
    
    int32_t phys_group_id_40502;
    
    phys_group_id_40502 = get_group_id(0);
    for (int32_t i_40503 = 0; i_40503 <
         sdiv_up32(sdiv_up32(nodes_this_lvl_32221, segmap_group_sizze_36798) -
                   phys_group_id_40502, num_groups_36799); i_40503++) {
        int32_t virt_group_id_40504 = phys_group_id_40502 + i_40503 *
                num_groups_36799;
        int32_t gtid_36747 = sext_i64_i32(sext_i32_i64(virt_group_id_40504) *
                sext_i32_i64(segmap_group_sizze_36798) +
                sext_i32_i64(local_tid_40498));
        
        if (slt32(gtid_36747, nodes_this_lvl_32221)) {
            int32_t x_36804 = add32(nodes_this_lvl_32221, gtid_36747);
            int32_t node_ind_36805 = sub32(x_36804, 1);
            
            for (int32_t i_40505 = 0; i_40505 < conc_tmp_32196; i_40505++) {
                ((__global float *) mem_39515)[sext_i32_i64(phys_tid_36748) +
                                               sext_i32_i64(i_40505) *
                                               sext_i32_i64(num_groups_36799 *
                                               segmap_group_sizze_36798)] =
                    ((__global float *) mem_39449)[sext_i32_i64(i_40505)];
            }
            
            int32_t res_36807;
            int32_t ancestor_36810 = 0;
            
            for (int32_t i_36809 = 0; i_36809 < lev_32216; i_36809++) {
                int32_t x_36812 = sub32(lev_32216, i_36809);
                int32_t k_36813 = sub32(x_36812, 1);
                int32_t tpk_36814 = 1 << k_36813;
                int32_t x_36815 = sub32(x_36804, tpk_36814);
                bool zzero_36816 = tpk_36814 == 0;
                bool nonzzero_36817 = !zzero_36816;
                bool nonzzero_cert_36818;
                
                if (!nonzzero_36817) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 7) ==
                            -1) {
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                int32_t res_36819 = sdiv32(x_36815, tpk_36814);
                bool x_36820 = sle32(0, ancestor_36810);
                bool y_36821 = slt32(ancestor_36810, res_32166);
                bool bounds_check_36822 = x_36820 && y_36821;
                bool index_certs_36823;
                
                if (!bounds_check_36822) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 8) ==
                            -1) {
                            global_failure_args[0] = ancestor_36810;
                            global_failure_args[1] = res_32166;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                int32_t anc_dim_36824 = ((__global
                                          int32_t *) mem_39458)[sext_i32_i64(ancestor_36810)];
                int32_t x_36825 = 1 & res_36819;
                bool cond_36826 = x_36825 == 0;
                int32_t lub_ind_36827;
                
                if (cond_36826) {
                    lub_ind_36827 = anc_dim_36824;
                } else {
                    int32_t res_36828 = add32(d_32210, anc_dim_36824);
                    
                    lub_ind_36827 = res_36828;
                }
                
                float anc_med_36829 = ((__global
                                        float *) mem_39455)[sext_i32_i64(ancestor_36810)];
                bool res_36830;
                
                res_36830 = futrts_isinf32(anc_med_36829);
                
                bool cond_36831 = !res_36830;
                float lw_val_36832;
                
                if (cond_36831) {
                    lw_val_36832 = anc_med_36829;
                } else {
                    bool x_36833 = sle32(0, lub_ind_36827);
                    bool y_36834 = slt32(lub_ind_36827, conc_tmp_32196);
                    bool bounds_check_36835 = x_36833 && y_36834;
                    bool index_certs_36836;
                    
                    if (!bounds_check_36835) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          9) == -1) {
                                global_failure_args[0] = lub_ind_36827;
                                global_failure_args[1] = conc_tmp_32196;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float res_36837 = ((__global
                                        float *) mem_39515)[sext_i32_i64(phys_tid_36748) +
                                                            sext_i32_i64(lub_ind_36827) *
                                                            sext_i32_i64(num_groups_36799 *
                                                            segmap_group_sizze_36798)];
                    
                    lw_val_36832 = res_36837;
                }
                
                bool x_36838 = sle32(0, lub_ind_36827);
                bool y_36839 = slt32(lub_ind_36827, conc_tmp_32196);
                bool bounds_check_36840 = x_36838 && y_36839;
                bool index_certs_36841;
                
                if (!bounds_check_36840) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 10) ==
                            -1) {
                            global_failure_args[0] = lub_ind_36827;
                            global_failure_args[1] = conc_tmp_32196;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                ((__global float *) mem_39515)[sext_i32_i64(phys_tid_36748) +
                                               sext_i32_i64(lub_ind_36827) *
                                               sext_i32_i64(num_groups_36799 *
                                               segmap_group_sizze_36798)] =
                    lw_val_36832;
                
                int32_t ancestor_tmp_40506 = res_36819;
                
                ancestor_36810 = ancestor_tmp_40506;
            }
            res_36807 = ancestor_36810;
            for (int32_t i_40508 = 0; i_40508 < conc_tmp_32196; i_40508++) {
                ((__global float *) mem_39532)[sext_i32_i64(i_40508) *
                                               sext_i32_i64(nodes_this_lvl_32221) +
                                               sext_i32_i64(gtid_36747)] =
                    ((__global
                      float *) mem_39515)[sext_i32_i64(phys_tid_36748) +
                                          sext_i32_i64(i_40508) *
                                          sext_i32_i64(num_groups_36799 *
                                          segmap_group_sizze_36798)];
            }
            ((__global int32_t *) mem_39535)[sext_i32_i64(gtid_36747)] =
                node_ind_36805;
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36798
}
__kernel void buildKDtreezisegmap_36942(__global int *global_failure,
                                        int failure_is_an_option, __global
                                        int *global_failure_args,
                                        int32_t m_32135, int32_t d_32136,
                                        int32_t res_32168,
                                        int32_t nodes_this_lvl_32221,
                                        int32_t pts_per_node_at_lev_32225,
                                        int32_t iters_32322, int32_t i_32324,
                                        int32_t num_groups_37068, __global
                                        unsigned char *input_mem_39418, __global
                                        unsigned char *mem_param_39466, __global
                                        unsigned char *res_mem_39552, __global
                                        unsigned char *mem_39556, __global
                                        unsigned char *mem_39566, __global
                                        unsigned char *mem_39570, __global
                                        unsigned char *mem_39597, __global
                                        unsigned char *mem_39600, __global
                                        unsigned char *mem_39603, __global
                                        unsigned char *mem_39606, __global
                                        unsigned char *mem_39609, __global
                                        unsigned char *mem_39672, __global
                                        unsigned char *mem_39675, __global
                                        unsigned char *mem_39712, __global
                                        unsigned char *mem_39715, __global
                                        unsigned char *mem_39744, __global
                                        unsigned char *mem_39749, __global
                                        unsigned char *double_buffer_mem_40098,
                                        __global
                                        unsigned char *double_buffer_mem_40099)
{
    #define segmap_group_sizze_37067 (buildKDtreezisegmap_group_sizze_36945)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40599;
    int32_t local_tid_40600;
    int32_t group_sizze_40603;
    int32_t wave_sizze_40602;
    int32_t group_tid_40601;
    
    global_tid_40599 = get_global_id(0);
    local_tid_40600 = get_local_id(0);
    group_sizze_40603 = get_local_size(0);
    wave_sizze_40602 = LOCKSTEP_WIDTH;
    group_tid_40601 = get_group_id(0);
    
    int32_t phys_tid_36942;
    
    phys_tid_36942 = global_tid_40599;
    
    int32_t phys_group_id_40604;
    
    phys_group_id_40604 = get_group_id(0);
    for (int32_t i_40605 = 0; i_40605 <
         sdiv_up32(sdiv_up32(nodes_this_lvl_32221, segmap_group_sizze_37067) -
                   phys_group_id_40604, num_groups_37068); i_40605++) {
        int32_t virt_group_id_40606 = phys_group_id_40604 + i_40605 *
                num_groups_37068;
        int32_t gtid_36941 = sext_i64_i32(sext_i32_i64(virt_group_id_40606) *
                sext_i32_i64(segmap_group_sizze_37067) +
                sext_i32_i64(local_tid_40600));
        
        if (slt32(gtid_36941, nodes_this_lvl_32221)) {
            int32_t x_37073 = ((__global
                                int32_t *) res_mem_39552)[sext_i32_i64(gtid_36941)];
            bool x_37074 = sle32(0, x_37073);
            bool y_37075 = slt32(x_37073, d_32136);
            bool bounds_check_37076 = x_37074 && y_37075;
            
            for (int32_t i_39359 = 0; i_39359 < pts_per_node_at_lev_32225;
                 i_39359++) {
                int32_t x_37078 = ((__global
                                    int32_t *) mem_39566)[sext_i32_i64(i_39359) *
                                                          sext_i32_i64(nodes_this_lvl_32221) +
                                                          sext_i32_i64(gtid_36941)];
                bool x_37079 = sle32(0, x_37078);
                bool y_37080 = slt32(x_37078, res_32168);
                bool bounds_check_37081 = x_37079 && y_37080;
                bool index_ok_37082 = bounds_check_37076 && bounds_check_37081;
                bool index_certs_37083;
                
                if (!index_ok_37082) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 16) ==
                            -1) {
                            global_failure_args[0] = x_37078;
                            global_failure_args[1] = x_37073;
                            global_failure_args[2] = res_32168;
                            global_failure_args[3] = d_32136;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                bool index_concat_cmp_37084 = sle32(m_32135, x_37078);
                float index_concat_branch_37085;
                
                if (index_concat_cmp_37084) {
                    index_concat_branch_37085 = INFINITY;
                } else {
                    float index_concat_37086 = ((__global
                                                 float *) input_mem_39418)[sext_i32_i64(x_37078) *
                                                                           sext_i32_i64(d_32136) +
                                                                           sext_i32_i64(x_37073)];
                    
                    index_concat_branch_37085 = index_concat_37086;
                }
                ((__global float *) mem_39570)[sext_i32_i64(phys_tid_36942) +
                                               sext_i32_i64(i_39359) *
                                               sext_i32_i64(num_groups_37068 *
                                               segmap_group_sizze_37067)] =
                    index_concat_branch_37085;
            }
            for (int32_t i_40608 = 0; i_40608 < pts_per_node_at_lev_32225;
                 i_40608++) {
                ((__global
                  float *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_36942) +
                                                    sext_i32_i64(i_40608) *
                                                    sext_i32_i64(num_groups_37068 *
                                                    segmap_group_sizze_37067)] =
                    ((__global
                      float *) mem_39570)[sext_i32_i64(phys_tid_36942) +
                                          sext_i32_i64(i_40608) *
                                          sext_i32_i64(num_groups_37068 *
                                          segmap_group_sizze_37067)];
            }
            for (int32_t i_40609 = 0; i_40609 < pts_per_node_at_lev_32225;
                 i_40609++) {
                ((__global
                  int32_t *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_36942) +
                                                      sext_i32_i64(i_40609) *
                                                      sext_i32_i64(num_groups_37068 *
                                                      segmap_group_sizze_37067)] =
                    ((__global int32_t *) mem_39556)[sext_i32_i64(i_40609)];
            }
            for (int32_t i_37090 = 0; i_37090 < iters_32322; i_37090++) {
                int32_t lifted_2_radix_sort_step_arg_37093 = mul32(2, i_37090);
                int32_t lifted_0_get_bit_arg_37094 = add32(1,
                                                           lifted_2_radix_sort_step_arg_37093);
                bool res_37095 = lifted_0_get_bit_arg_37094 == 31;
                bool res_37096 = lifted_2_radix_sort_step_arg_37093 == 31;
                int32_t discard_39381;
                int32_t discard_39382;
                int32_t discard_39383;
                int32_t discard_39384;
                int32_t scanacc_39366;
                int32_t scanacc_39367;
                int32_t scanacc_39368;
                int32_t scanacc_39369;
                
                scanacc_39366 = 0;
                scanacc_39367 = 0;
                scanacc_39368 = 0;
                scanacc_39369 = 0;
                for (int32_t i_39375 = 0; i_39375 < pts_per_node_at_lev_32225;
                     i_39375++) {
                    float x_37114 = ((__global
                                      float *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_36942) +
                                                                        sext_i32_i64(i_39375) *
                                                                        sext_i32_i64(num_groups_37068 *
                                                                        segmap_group_sizze_37067)];
                    int32_t i32_arg_37115;
                    
                    i32_arg_37115 = futrts_to_bits32(x_37114);
                    
                    int32_t unsign_arg_37116 = ashr32(i32_arg_37115,
                                                      lifted_0_get_bit_arg_37094);
                    int32_t unsign_arg_37117 = 1 & unsign_arg_37116;
                    int32_t unsign_arg_37118 = ashr32(i32_arg_37115, 31);
                    int32_t unsign_arg_37119 = 1 & unsign_arg_37118;
                    bool cond_37120 = unsign_arg_37119 == 1;
                    bool x_37121 = !cond_37120;
                    bool y_37122 = res_37095 && x_37121;
                    bool cond_37123 = cond_37120 || y_37122;
                    int32_t res_37124;
                    
                    if (cond_37123) {
                        int32_t res_37125 = 1 ^ unsign_arg_37117;
                        
                        res_37124 = res_37125;
                    } else {
                        res_37124 = unsign_arg_37117;
                    }
                    
                    int32_t x_37126 = mul32(2, res_37124);
                    int32_t unsign_arg_37127 = ashr32(i32_arg_37115,
                                                      lifted_2_radix_sort_step_arg_37093);
                    int32_t unsign_arg_37128 = 1 & unsign_arg_37127;
                    bool y_37129 = res_37096 && x_37121;
                    bool cond_37130 = cond_37120 || y_37129;
                    int32_t res_37131;
                    
                    if (cond_37130) {
                        int32_t res_37132 = 1 ^ unsign_arg_37128;
                        
                        res_37131 = res_37132;
                    } else {
                        res_37131 = unsign_arg_37128;
                    }
                    
                    int32_t res_37133 = add32(x_37126, res_37131);
                    bool cond_37134 = res_37133 == 0;
                    int32_t res_37135 = btoi_bool_i32(cond_37134);
                    int32_t res_37136;
                    int32_t res_37137;
                    int32_t res_37138;
                    
                    if (cond_37134) {
                        res_37136 = 0;
                        res_37137 = 0;
                        res_37138 = 0;
                    } else {
                        bool cond_37139 = res_37133 == 1;
                        int32_t res_37140 = btoi_bool_i32(cond_37139);
                        int32_t res_37141;
                        int32_t res_37142;
                        
                        if (cond_37139) {
                            res_37141 = 0;
                            res_37142 = 0;
                        } else {
                            bool cond_37143 = res_37133 == 2;
                            int32_t res_37144 = btoi_bool_i32(cond_37143);
                            bool cond_neg_37145 = !cond_37143;
                            int32_t res_37146 = btoi_bool_i32(cond_neg_37145);
                            
                            res_37141 = res_37144;
                            res_37142 = res_37146;
                        }
                        res_37136 = res_37140;
                        res_37137 = res_37141;
                        res_37138 = res_37142;
                    }
                    
                    int32_t res_37110 = add32(res_37135, scanacc_39366);
                    int32_t res_37111 = add32(res_37136, scanacc_39367);
                    int32_t res_37112 = add32(res_37137, scanacc_39368);
                    int32_t res_37113 = add32(res_37138, scanacc_39369);
                    
                    ((__global
                      int32_t *) mem_39597)[sext_i32_i64(phys_tid_36942) +
                                            sext_i32_i64(i_39375) *
                                            sext_i32_i64(num_groups_37068 *
                                            segmap_group_sizze_37067)] =
                        res_37110;
                    ((__global
                      int32_t *) mem_39600)[sext_i32_i64(phys_tid_36942) +
                                            sext_i32_i64(i_39375) *
                                            sext_i32_i64(num_groups_37068 *
                                            segmap_group_sizze_37067)] =
                        res_37111;
                    ((__global
                      int32_t *) mem_39603)[sext_i32_i64(phys_tid_36942) +
                                            sext_i32_i64(i_39375) *
                                            sext_i32_i64(num_groups_37068 *
                                            segmap_group_sizze_37067)] =
                        res_37112;
                    ((__global
                      int32_t *) mem_39606)[sext_i32_i64(phys_tid_36942) +
                                            sext_i32_i64(i_39375) *
                                            sext_i32_i64(num_groups_37068 *
                                            segmap_group_sizze_37067)] =
                        res_37113;
                    ((__global
                      int32_t *) mem_39609)[sext_i32_i64(phys_tid_36942) +
                                            sext_i32_i64(i_39375) *
                                            sext_i32_i64(num_groups_37068 *
                                            segmap_group_sizze_37067)] =
                        res_37133;
                    
                    int32_t scanacc_tmp_40612 = res_37110;
                    int32_t scanacc_tmp_40613 = res_37111;
                    int32_t scanacc_tmp_40614 = res_37112;
                    int32_t scanacc_tmp_40615 = res_37113;
                    
                    scanacc_39366 = scanacc_tmp_40612;
                    scanacc_39367 = scanacc_tmp_40613;
                    scanacc_39368 = scanacc_tmp_40614;
                    scanacc_39369 = scanacc_tmp_40615;
                }
                discard_39381 = scanacc_39366;
                discard_39382 = scanacc_39367;
                discard_39383 = scanacc_39368;
                discard_39384 = scanacc_39369;
                
                int32_t res_37147 = ((__global
                                      int32_t *) mem_39597)[sext_i32_i64(phys_tid_36942) +
                                                            sext_i32_i64(i_32324) *
                                                            sext_i32_i64(num_groups_37068 *
                                                            segmap_group_sizze_37067)];
                int32_t res_37148 = ((__global
                                      int32_t *) mem_39600)[sext_i32_i64(phys_tid_36942) +
                                                            sext_i32_i64(i_32324) *
                                                            sext_i32_i64(num_groups_37068 *
                                                            segmap_group_sizze_37067)];
                int32_t res_37149 = ((__global
                                      int32_t *) mem_39603)[sext_i32_i64(phys_tid_36942) +
                                                            sext_i32_i64(i_32324) *
                                                            sext_i32_i64(num_groups_37068 *
                                                            segmap_group_sizze_37067)];
                
                for (int32_t i_40621 = 0; i_40621 < pts_per_node_at_lev_32225;
                     i_40621++) {
                    ((__global
                      float *) mem_39672)[sext_i32_i64(phys_tid_36942) +
                                          sext_i32_i64(i_40621) *
                                          sext_i32_i64(num_groups_37068 *
                                          segmap_group_sizze_37067)] =
                        ((__global
                          float *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_36942) +
                                                            sext_i32_i64(i_40621) *
                                                            sext_i32_i64(num_groups_37068 *
                                                            segmap_group_sizze_37067)];
                }
                for (int32_t i_40622 = 0; i_40622 < pts_per_node_at_lev_32225;
                     i_40622++) {
                    ((__global
                      int32_t *) mem_39675)[sext_i32_i64(phys_tid_36942) +
                                            sext_i32_i64(i_40622) *
                                            sext_i32_i64(num_groups_37068 *
                                            segmap_group_sizze_37067)] =
                        ((__global
                          int32_t *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_36942) +
                                                              sext_i32_i64(i_40622) *
                                                              sext_i32_i64(num_groups_37068 *
                                                              segmap_group_sizze_37067)];
                }
                for (int32_t write_iter_39385 = 0; write_iter_39385 <
                     pts_per_node_at_lev_32225; write_iter_39385++) {
                    int32_t write_iv_39388 = ((__global
                                               int32_t *) mem_39609)[sext_i32_i64(phys_tid_36942) +
                                                                     sext_i32_i64(write_iter_39385) *
                                                                     sext_i32_i64(num_groups_37068 *
                                                                     segmap_group_sizze_37067)];
                    bool match_lit_37161 = 0 == write_iv_39388;
                    int32_t res_37162;
                    
                    if (match_lit_37161) {
                        int32_t write_iv_39389 = ((__global
                                                   int32_t *) mem_39597)[sext_i32_i64(phys_tid_36942) +
                                                                         sext_i32_i64(write_iter_39385) *
                                                                         sext_i32_i64(num_groups_37068 *
                                                                         segmap_group_sizze_37067)];
                        int32_t res_37163 = sub32(write_iv_39389, 1);
                        
                        res_37162 = res_37163;
                    } else {
                        int32_t write_iv_39390 = ((__global
                                                   int32_t *) mem_39600)[sext_i32_i64(phys_tid_36942) +
                                                                         sext_i32_i64(write_iter_39385) *
                                                                         sext_i32_i64(num_groups_37068 *
                                                                         segmap_group_sizze_37067)];
                        int32_t write_iv_39391 = ((__global
                                                   int32_t *) mem_39603)[sext_i32_i64(phys_tid_36942) +
                                                                         sext_i32_i64(write_iter_39385) *
                                                                         sext_i32_i64(num_groups_37068 *
                                                                         segmap_group_sizze_37067)];
                        int32_t write_iv_39392 = ((__global
                                                   int32_t *) mem_39606)[sext_i32_i64(phys_tid_36942) +
                                                                         sext_i32_i64(write_iter_39385) *
                                                                         sext_i32_i64(num_groups_37068 *
                                                                         segmap_group_sizze_37067)];
                        bool match_lit_37164 = 1 == write_iv_39388;
                        int32_t x_37165;
                        
                        if (match_lit_37164) {
                            int32_t x_37166 = add32(res_37147, write_iv_39390);
                            int32_t res_37167 = sub32(x_37166, 1);
                            
                            x_37165 = res_37167;
                        } else {
                            bool match_lit_37168 = 2 == write_iv_39388;
                            int32_t x_37169;
                            
                            if (match_lit_37168) {
                                int32_t x_37170 = add32(res_37147, res_37148);
                                int32_t x_37171 = add32(x_37170,
                                                        write_iv_39391);
                                int32_t res_37172 = sub32(x_37171, 1);
                                
                                x_37169 = res_37172;
                            } else {
                                int32_t x_37173 = add32(res_37147, res_37148);
                                int32_t x_37174 = add32(res_37149, x_37173);
                                int32_t x_37175 = add32(x_37174,
                                                        write_iv_39392);
                                int32_t res_37176 = sub32(x_37175, 1);
                                
                                x_37169 = res_37176;
                            }
                            x_37165 = x_37169;
                        }
                        res_37162 = x_37165;
                    }
                    
                    bool less_than_zzero_39395 = slt32(res_37162, 0);
                    bool greater_than_sizze_39396 =
                         sle32(pts_per_node_at_lev_32225, res_37162);
                    bool outside_bounds_dim_39397 = less_than_zzero_39395 ||
                         greater_than_sizze_39396;
                    
                    if (!outside_bounds_dim_39397) {
                        for (int32_t i_40625 = 0; i_40625 < 1; i_40625++) {
                            ((__global
                              float *) mem_39672)[sext_i32_i64(phys_tid_36942) +
                                                  sext_i32_i64(res_37162 +
                                                  i_40625) *
                                                  sext_i32_i64(num_groups_37068 *
                                                  segmap_group_sizze_37067)] =
                                ((__global
                                  float *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_36942 +
                                                                    num_groups_37068 *
                                                                    segmap_group_sizze_37067 *
                                                                    write_iter_39385) +
                                                                    sext_i32_i64(i_40625) *
                                                                    sext_i32_i64(num_groups_37068 *
                                                                    segmap_group_sizze_37067)];
                        }
                    }
                    if (!outside_bounds_dim_39397) {
                        for (int32_t i_40626 = 0; i_40626 < 1; i_40626++) {
                            ((__global
                              int32_t *) mem_39675)[sext_i32_i64(phys_tid_36942) +
                                                    sext_i32_i64(res_37162 +
                                                    i_40626) *
                                                    sext_i32_i64(num_groups_37068 *
                                                    segmap_group_sizze_37067)] =
                                ((__global
                                  int32_t *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_36942 +
                                                                      num_groups_37068 *
                                                                      segmap_group_sizze_37067 *
                                                                      write_iter_39385) +
                                                                      sext_i32_i64(i_40626) *
                                                                      sext_i32_i64(num_groups_37068 *
                                                                      segmap_group_sizze_37067)];
                        }
                    }
                }
                for (int32_t i_40627 = 0; i_40627 < pts_per_node_at_lev_32225;
                     i_40627++) {
                    ((__global
                      float *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_36942) +
                                                        sext_i32_i64(i_40627) *
                                                        sext_i32_i64(num_groups_37068 *
                                                        segmap_group_sizze_37067)] =
                        ((__global
                          float *) mem_39672)[sext_i32_i64(phys_tid_36942) +
                                              sext_i32_i64(i_40627) *
                                              sext_i32_i64(num_groups_37068 *
                                              segmap_group_sizze_37067)];
                }
                for (int32_t i_40628 = 0; i_40628 < pts_per_node_at_lev_32225;
                     i_40628++) {
                    ((__global
                      int32_t *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_36942) +
                                                          sext_i32_i64(i_40628) *
                                                          sext_i32_i64(num_groups_37068 *
                                                          segmap_group_sizze_37067)] =
                        ((__global
                          int32_t *) mem_39675)[sext_i32_i64(phys_tid_36942) +
                                                sext_i32_i64(i_40628) *
                                                sext_i32_i64(num_groups_37068 *
                                                segmap_group_sizze_37067)];
                }
            }
            
            int32_t binop_x_39307 = pts_per_node_at_lev_32225 * gtid_36941;
            
            for (int32_t i_39411 = 0; i_39411 < pts_per_node_at_lev_32225;
                 i_39411++) {
                int32_t x_37179 = ((__global
                                    int32_t *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_36942) +
                                                                        sext_i32_i64(i_39411) *
                                                                        sext_i32_i64(num_groups_37068 *
                                                                        segmap_group_sizze_37067)];
                bool x_37180 = sle32(0, x_37179);
                bool y_37181 = slt32(x_37179, pts_per_node_at_lev_32225);
                bool bounds_check_37182 = x_37180 && y_37181;
                bool index_certs_37183;
                
                if (!bounds_check_37182) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 17) ==
                            -1) {
                            global_failure_args[0] = x_37179;
                            global_failure_args[1] = pts_per_node_at_lev_32225;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                bool index_certs_37185;
                
                if (!bounds_check_37182) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 18) ==
                            -1) {
                            global_failure_args[0] = x_37179;
                            global_failure_args[1] = pts_per_node_at_lev_32225;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                int32_t new_index_39308 = x_37179 + binop_x_39307;
                
                for (int32_t i_40631 = 0; i_40631 < 1; i_40631++) {
                    ((__global
                      int32_t *) mem_39712)[sext_i32_i64(phys_tid_36942) +
                                            sext_i32_i64(i_39411 + i_40631) *
                                            sext_i32_i64(num_groups_37068 *
                                            segmap_group_sizze_37067)] =
                        ((__global
                          int32_t *) mem_param_39466)[sext_i32_i64(new_index_39308) +
                                                      sext_i32_i64(i_40631)];
                }
                for (int32_t i_40632 = 0; i_40632 < 1; i_40632++) {
                    ((__global
                      float *) mem_39715)[sext_i32_i64(phys_tid_36942) +
                                          sext_i32_i64(i_39411 + i_40632) *
                                          sext_i32_i64(num_groups_37068 *
                                          segmap_group_sizze_37067)] =
                        ((__global
                          float *) mem_39570)[sext_i32_i64(phys_tid_36942 +
                                              num_groups_37068 *
                                              segmap_group_sizze_37067 *
                                              x_37179) + sext_i32_i64(i_40632) *
                                              sext_i32_i64(num_groups_37068 *
                                              segmap_group_sizze_37067)];
                }
            }
            for (int32_t i_40633 = 0; i_40633 < pts_per_node_at_lev_32225;
                 i_40633++) {
                ((__global int32_t *) mem_39744)[sext_i32_i64(i_40633) *
                                                 sext_i32_i64(nodes_this_lvl_32221) +
                                                 sext_i32_i64(gtid_36941)] =
                    ((__global
                      int32_t *) mem_39712)[sext_i32_i64(phys_tid_36942) +
                                            sext_i32_i64(i_40633) *
                                            sext_i32_i64(num_groups_37068 *
                                            segmap_group_sizze_37067)];
            }
            for (int32_t i_40634 = 0; i_40634 < pts_per_node_at_lev_32225;
                 i_40634++) {
                ((__global float *) mem_39749)[sext_i32_i64(i_40634) *
                                               sext_i32_i64(nodes_this_lvl_32221) +
                                               sext_i32_i64(gtid_36941)] =
                    ((__global
                      float *) mem_39715)[sext_i32_i64(phys_tid_36942) +
                                          sext_i32_i64(i_40634) *
                                          sext_i32_i64(num_groups_37068 *
                                          segmap_group_sizze_37067)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_37067
}
__kernel void buildKDtreezisegmap_37340(__global int *global_failure,
                                        int failure_is_an_option, __global
                                        int *global_failure_args,
                                        int32_t nodes_this_lvl_32221,
                                        int32_t pts_per_node_at_lev_32225,
                                        __global unsigned char *mem_param_39466,
                                        __global unsigned char *mem_39835,
                                        __global unsigned char *res_r_mem_39958,
                                        __global unsigned char *mem_39964,
                                        __global unsigned char *mem_39969)
{
    #define segmap_group_sizze_38288 (buildKDtreezisegmap_group_sizze_37345)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40806;
    int32_t local_tid_40807;
    int32_t group_sizze_40810;
    int32_t wave_sizze_40809;
    int32_t group_tid_40808;
    
    global_tid_40806 = get_global_id(0);
    local_tid_40807 = get_local_id(0);
    group_sizze_40810 = get_local_size(0);
    wave_sizze_40809 = LOCKSTEP_WIDTH;
    group_tid_40808 = get_group_id(0);
    
    int32_t phys_tid_37340;
    
    phys_tid_37340 = global_tid_40806;
    
    int32_t gtid_37338;
    
    gtid_37338 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40808) *
                                      sext_i32_i64(segmap_group_sizze_38288) +
                                      sext_i32_i64(local_tid_40807),
                                      sext_i32_i64(pts_per_node_at_lev_32225)));
    
    int32_t gtid_37339;
    
    gtid_37339 = sext_i64_i32(sext_i32_i64(group_tid_40808) *
        sext_i32_i64(segmap_group_sizze_38288) + sext_i32_i64(local_tid_40807) -
        squot64(sext_i32_i64(group_tid_40808) *
                sext_i32_i64(segmap_group_sizze_38288) +
                sext_i32_i64(local_tid_40807),
                sext_i32_i64(pts_per_node_at_lev_32225)) *
        sext_i32_i64(pts_per_node_at_lev_32225));
    if (slt32(gtid_37338, nodes_this_lvl_32221) && slt32(gtid_37339,
                                                         pts_per_node_at_lev_32225)) {
        int32_t x_38296 = ((__global
                            int32_t *) res_r_mem_39958)[sext_i32_i64(gtid_37338) *
                                                        sext_i32_i64(pts_per_node_at_lev_32225) +
                                                        sext_i32_i64(gtid_37339)];
        bool x_38297 = sle32(0, x_38296);
        bool y_38298 = slt32(x_38296, pts_per_node_at_lev_32225);
        bool bounds_check_38299 = x_38297 && y_38298;
        bool index_certs_38300;
        
        if (!bounds_check_38299) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 23) == -1) {
                    global_failure_args[0] = x_38296;
                    global_failure_args[1] = pts_per_node_at_lev_32225;
                    ;
                }
                return;
            }
        }
        
        float res_38301 = ((__global
                            float *) mem_39835)[sext_i32_i64(gtid_37338) *
                                                sext_i32_i64(pts_per_node_at_lev_32225) +
                                                sext_i32_i64(x_38296)];
        bool index_certs_38302;
        
        if (!bounds_check_38299) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 24) == -1) {
                    global_failure_args[0] = x_38296;
                    global_failure_args[1] = pts_per_node_at_lev_32225;
                    ;
                }
                return;
            }
        }
        
        int32_t binop_x_39314 = pts_per_node_at_lev_32225 * gtid_37338;
        int32_t new_index_39315 = x_38296 + binop_x_39314;
        int32_t res_38303 = ((__global
                              int32_t *) mem_param_39466)[sext_i32_i64(new_index_39315)];
        
        ((__global int32_t *) mem_39964)[sext_i32_i64(gtid_37338) *
                                         sext_i32_i64(pts_per_node_at_lev_32225) +
                                         sext_i32_i64(gtid_37339)] = res_38303;
        ((__global float *) mem_39969)[sext_i32_i64(gtid_37338) *
                                       sext_i32_i64(pts_per_node_at_lev_32225) +
                                       sext_i32_i64(gtid_37339)] = res_38301;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_38288
}
__kernel void buildKDtreezisegmap_37675(__global int *global_failure,
                                        int32_t nodes_this_lvl_32221,
                                        int32_t pts_per_node_at_lev_32225,
                                        int32_t i_32324, __global
                                        unsigned char *mem_param_39843, __global
                                        unsigned char *mem_param_39851, __global
                                        unsigned char *mem_39905, __global
                                        unsigned char *mem_39910, __global
                                        unsigned char *mem_39915, __global
                                        unsigned char *mem_39920, __global
                                        unsigned char *mem_39925, __global
                                        unsigned char *mem_39930, __global
                                        unsigned char *mem_39935)
{
    #define segmap_group_sizze_38230 (buildKDtreezisegmap_group_sizze_37680)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40800;
    int32_t local_tid_40801;
    int32_t group_sizze_40804;
    int32_t wave_sizze_40803;
    int32_t group_tid_40802;
    
    global_tid_40800 = get_global_id(0);
    local_tid_40801 = get_local_id(0);
    group_sizze_40804 = get_local_size(0);
    wave_sizze_40803 = LOCKSTEP_WIDTH;
    group_tid_40802 = get_group_id(0);
    
    int32_t phys_tid_37675;
    
    phys_tid_37675 = global_tid_40800;
    
    int32_t gtid_37673;
    
    gtid_37673 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40802) *
                                      sext_i32_i64(segmap_group_sizze_38230) +
                                      sext_i32_i64(local_tid_40801),
                                      sext_i32_i64(pts_per_node_at_lev_32225)));
    
    int32_t gtid_37674;
    
    gtid_37674 = sext_i64_i32(sext_i32_i64(group_tid_40802) *
        sext_i32_i64(segmap_group_sizze_38230) + sext_i32_i64(local_tid_40801) -
        squot64(sext_i32_i64(group_tid_40802) *
                sext_i32_i64(segmap_group_sizze_38230) +
                sext_i32_i64(local_tid_40801),
                sext_i32_i64(pts_per_node_at_lev_32225)) *
        sext_i32_i64(pts_per_node_at_lev_32225));
    if (slt32(gtid_37673, nodes_this_lvl_32221) && slt32(gtid_37674,
                                                         pts_per_node_at_lev_32225)) {
        int32_t x_38248 = ((__global
                            int32_t *) mem_39925)[sext_i32_i64(gtid_37673) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(gtid_37674)];
        float write_value_38253 = ((__global
                                    float *) mem_param_39843)[sext_i32_i64(gtid_37673) *
                                                              sext_i32_i64(pts_per_node_at_lev_32225) +
                                                              sext_i32_i64(gtid_37674)];
        int32_t write_value_38254 = ((__global
                                      int32_t *) mem_param_39851)[sext_i32_i64(gtid_37673) *
                                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                                  sext_i32_i64(gtid_37674)];
        bool match_lit_38255 = 0 == x_38248;
        int32_t res_38256;
        
        if (match_lit_38255) {
            int32_t x_38249 = ((__global
                                int32_t *) mem_39905)[sext_i32_i64(gtid_37673) *
                                                      sext_i32_i64(pts_per_node_at_lev_32225) +
                                                      sext_i32_i64(gtid_37674)];
            int32_t res_38257 = sub32(x_38249, 1);
            
            res_38256 = res_38257;
        } else {
            int32_t res_38243 = ((__global
                                  int32_t *) mem_39905)[sext_i32_i64(gtid_37673) *
                                                        sext_i32_i64(pts_per_node_at_lev_32225) +
                                                        sext_i32_i64(i_32324)];
            int32_t res_38244 = ((__global
                                  int32_t *) mem_39910)[sext_i32_i64(gtid_37673) *
                                                        sext_i32_i64(pts_per_node_at_lev_32225) +
                                                        sext_i32_i64(i_32324)];
            int32_t res_38245 = ((__global
                                  int32_t *) mem_39915)[sext_i32_i64(gtid_37673) *
                                                        sext_i32_i64(pts_per_node_at_lev_32225) +
                                                        sext_i32_i64(i_32324)];
            int32_t x_38250 = ((__global
                                int32_t *) mem_39910)[sext_i32_i64(gtid_37673) *
                                                      sext_i32_i64(pts_per_node_at_lev_32225) +
                                                      sext_i32_i64(gtid_37674)];
            int32_t x_38251 = ((__global
                                int32_t *) mem_39915)[sext_i32_i64(gtid_37673) *
                                                      sext_i32_i64(pts_per_node_at_lev_32225) +
                                                      sext_i32_i64(gtid_37674)];
            int32_t x_38252 = ((__global
                                int32_t *) mem_39920)[sext_i32_i64(gtid_37673) *
                                                      sext_i32_i64(pts_per_node_at_lev_32225) +
                                                      sext_i32_i64(gtid_37674)];
            bool match_lit_38258 = 1 == x_38248;
            int32_t x_38259;
            
            if (match_lit_38258) {
                int32_t x_38260 = add32(res_38243, x_38250);
                int32_t res_38261 = sub32(x_38260, 1);
                
                x_38259 = res_38261;
            } else {
                bool match_lit_38262 = 2 == x_38248;
                int32_t x_38263;
                
                if (match_lit_38262) {
                    int32_t x_38264 = add32(res_38243, res_38244);
                    int32_t x_38265 = add32(x_38251, x_38264);
                    int32_t res_38266 = sub32(x_38265, 1);
                    
                    x_38263 = res_38266;
                } else {
                    int32_t x_38267 = add32(res_38243, res_38244);
                    int32_t x_38268 = add32(res_38245, x_38267);
                    int32_t x_38269 = add32(x_38252, x_38268);
                    int32_t res_38270 = sub32(x_38269, 1);
                    
                    x_38263 = res_38270;
                }
                x_38259 = x_38263;
            }
            res_38256 = x_38259;
        }
        if ((sle32(0, gtid_37673) && slt32(gtid_37673, nodes_this_lvl_32221)) &&
            (sle32(0, res_38256) && slt32(res_38256,
                                          pts_per_node_at_lev_32225))) {
            ((__global float *) mem_39930)[sext_i32_i64(gtid_37673) *
                                           sext_i32_i64(pts_per_node_at_lev_32225) +
                                           sext_i32_i64(res_38256)] =
                write_value_38253;
        }
        if ((sle32(0, gtid_37673) && slt32(gtid_37673, nodes_this_lvl_32221)) &&
            (sle32(0, res_38256) && slt32(res_38256,
                                          pts_per_node_at_lev_32225))) {
            ((__global int32_t *) mem_39935)[sext_i32_i64(gtid_37673) *
                                             sext_i32_i64(pts_per_node_at_lev_32225) +
                                             sext_i32_i64(res_38256)] =
                write_value_38254;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_38230
}
__kernel void buildKDtreezisegmap_37946(__global int *global_failure,
                                        int failure_is_an_option, __global
                                        int *global_failure_args,
                                        int32_t m_32135, int32_t d_32136,
                                        int32_t res_32168,
                                        int32_t nodes_this_lvl_32221,
                                        int32_t pts_per_node_at_lev_32225,
                                        __global unsigned char *input_mem_39418,
                                        __global unsigned char *mem_param_39466,
                                        __global unsigned char *res_mem_39552,
                                        __global unsigned char *mem_39829,
                                        __global unsigned char *mem_39835)
{
    #define segmap_group_sizze_38028 (buildKDtreezisegmap_group_sizze_37951)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40670;
    int32_t local_tid_40671;
    int32_t group_sizze_40674;
    int32_t wave_sizze_40673;
    int32_t group_tid_40672;
    
    global_tid_40670 = get_global_id(0);
    local_tid_40671 = get_local_id(0);
    group_sizze_40674 = get_local_size(0);
    wave_sizze_40673 = LOCKSTEP_WIDTH;
    group_tid_40672 = get_group_id(0);
    
    int32_t phys_tid_37946;
    
    phys_tid_37946 = global_tid_40670;
    
    int32_t gtid_37944;
    
    gtid_37944 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40672) *
                                      sext_i32_i64(segmap_group_sizze_38028) +
                                      sext_i32_i64(local_tid_40671),
                                      sext_i32_i64(pts_per_node_at_lev_32225)));
    
    int32_t gtid_37945;
    
    gtid_37945 = sext_i64_i32(sext_i32_i64(group_tid_40672) *
        sext_i32_i64(segmap_group_sizze_38028) + sext_i32_i64(local_tid_40671) -
        squot64(sext_i32_i64(group_tid_40672) *
                sext_i32_i64(segmap_group_sizze_38028) +
                sext_i32_i64(local_tid_40671),
                sext_i32_i64(pts_per_node_at_lev_32225)) *
        sext_i32_i64(pts_per_node_at_lev_32225));
    if (slt32(gtid_37944, nodes_this_lvl_32221) && slt32(gtid_37945,
                                                         pts_per_node_at_lev_32225)) {
        int32_t x_38033 = ((__global
                            int32_t *) res_mem_39552)[sext_i32_i64(gtid_37944)];
        bool bounds_check_38034 = ((__global
                                    bool *) mem_39829)[sext_i32_i64(gtid_37944)];
        int32_t binop_x_39297 = pts_per_node_at_lev_32225 * gtid_37944;
        int32_t new_index_39298 = gtid_37945 + binop_x_39297;
        int32_t x_38035 = ((__global
                            int32_t *) mem_param_39466)[sext_i32_i64(new_index_39298)];
        bool x_38036 = sle32(0, x_38035);
        bool y_38037 = slt32(x_38035, res_32168);
        bool bounds_check_38038 = x_38036 && y_38037;
        bool index_ok_38039 = bounds_check_38034 && bounds_check_38038;
        bool index_certs_38040;
        
        if (!index_ok_38039) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 22) == -1) {
                    global_failure_args[0] = x_38035;
                    global_failure_args[1] = x_38033;
                    global_failure_args[2] = res_32168;
                    global_failure_args[3] = d_32136;
                    ;
                }
                return;
            }
        }
        
        bool index_concat_cmp_38041 = sle32(m_32135, x_38035);
        float index_concat_branch_38042;
        
        if (index_concat_cmp_38041) {
            index_concat_branch_38042 = INFINITY;
        } else {
            float index_concat_38043 = ((__global
                                         float *) input_mem_39418)[sext_i32_i64(x_38035) *
                                                                   sext_i32_i64(d_32136) +
                                                                   sext_i32_i64(x_38033)];
            
            index_concat_branch_38042 = index_concat_38043;
        }
        
        float res_38044 = index_concat_branch_38042;
        
        ((__global float *) mem_39835)[sext_i32_i64(gtid_37944) *
                                       sext_i32_i64(pts_per_node_at_lev_32225) +
                                       sext_i32_i64(gtid_37945)] = res_38044;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_38028
}
__kernel void buildKDtreezisegmap_37990(__global int *global_failure,
                                        int32_t d_32136,
                                        int32_t nodes_this_lvl_32221, __global
                                        unsigned char *res_mem_39552, __global
                                        unsigned char *mem_39829)
{
    #define segmap_group_sizze_38004 (buildKDtreezisegmap_group_sizze_37993)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40665;
    int32_t local_tid_40666;
    int32_t group_sizze_40669;
    int32_t wave_sizze_40668;
    int32_t group_tid_40667;
    
    global_tid_40665 = get_global_id(0);
    local_tid_40666 = get_local_id(0);
    group_sizze_40669 = get_local_size(0);
    wave_sizze_40668 = LOCKSTEP_WIDTH;
    group_tid_40667 = get_group_id(0);
    
    int32_t phys_tid_37990;
    
    phys_tid_37990 = global_tid_40665;
    
    int32_t gtid_37989;
    
    gtid_37989 = sext_i64_i32(sext_i32_i64(group_tid_40667) *
        sext_i32_i64(segmap_group_sizze_38004) + sext_i32_i64(local_tid_40666));
    if (slt32(gtid_37989, nodes_this_lvl_32221)) {
        int32_t x_38009 = ((__global
                            int32_t *) res_mem_39552)[sext_i32_i64(gtid_37989)];
        bool x_38010 = sle32(0, x_38009);
        bool y_38011 = slt32(x_38009, d_32136);
        bool bounds_check_38012 = x_38010 && y_38011;
        
        ((__global bool *) mem_39829)[sext_i32_i64(gtid_37989)] =
            bounds_check_38012;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_38004
}
__kernel void buildKDtreezisegmap_38311(__global int *global_failure,
                                        int32_t res_32166,
                                        int32_t nodes_this_lvl_32221,
                                        int32_t pts_per_node_at_lev_32225,
                                        int32_t mi_32473, int32_t i_32478,
                                        int32_t y_32643, __global
                                        unsigned char *mem_39455, __global
                                        unsigned char *mem_39458, __global
                                        unsigned char *mem_39461, __global
                                        unsigned char *res_mem_39552, __global
                                        unsigned char *res_mem_39553, __global
                                        unsigned char *res_mem_39985)
{
    #define segmap_group_sizze_38315 (buildKDtreezisegmap_group_sizze_38314)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40813;
    int32_t local_tid_40814;
    int32_t group_sizze_40817;
    int32_t wave_sizze_40816;
    int32_t group_tid_40815;
    
    global_tid_40813 = get_global_id(0);
    local_tid_40814 = get_local_id(0);
    group_sizze_40817 = get_local_size(0);
    wave_sizze_40816 = LOCKSTEP_WIDTH;
    group_tid_40815 = get_group_id(0);
    
    int32_t phys_tid_38311;
    
    phys_tid_38311 = global_tid_40813;
    
    int32_t write_i_38310;
    
    write_i_38310 = sext_i64_i32(sext_i32_i64(group_tid_40815) *
        sext_i32_i64(segmap_group_sizze_38315) + sext_i32_i64(local_tid_40814));
    if (slt32(write_i_38310, nodes_this_lvl_32221)) {
        int32_t write_value_32649 = ((__global
                                      int32_t *) res_mem_39552)[sext_i32_i64(write_i_38310)];
        int32_t write_value_32650 = ((__global
                                      int32_t *) res_mem_39553)[sext_i32_i64(write_i_38310)];
        float x_32651 = ((__global
                          float *) res_mem_39985)[sext_i32_i64(write_i_38310) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(mi_32473)];
        float y_32652 = ((__global
                          float *) res_mem_39985)[sext_i32_i64(write_i_38310) *
                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(i_32478)];
        float x_32653 = x_32651 + y_32652;
        float res_32654 = x_32653 / 2.0F;
        int32_t res_32655 = add32(y_32643, write_i_38310);
        
        if (sle32(0, res_32655) && slt32(res_32655, res_32166)) {
            ((__global int32_t *) mem_39461)[sext_i32_i64(res_32655)] =
                write_value_32650;
        }
        if (sle32(0, res_32655) && slt32(res_32655, res_32166)) {
            ((__global float *) mem_39455)[sext_i32_i64(res_32655)] = res_32654;
        }
        if (sle32(0, res_32655) && slt32(res_32655, res_32166)) {
            ((__global int32_t *) mem_39458)[sext_i32_i64(res_32655)] =
                write_value_32649;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_38315
}
__kernel void buildKDtreezisegmap_38398(__global int *global_failure,
                                        int failure_is_an_option, __global
                                        int *global_failure_args,
                                        int32_t d_32136, int32_t res_32168,
                                        __global unsigned char *input_mem_39418,
                                        __global unsigned char *res_mem_39993,
                                        __global unsigned char *mem_40011,
                                        __global unsigned char *mem_40013,
                                        __global unsigned char *mem_40019)
{
    #define segmap_group_sizze_38483 (buildKDtreezisegmap_group_sizze_38403)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40823;
    int32_t local_tid_40824;
    int32_t group_sizze_40827;
    int32_t wave_sizze_40826;
    int32_t group_tid_40825;
    
    global_tid_40823 = get_global_id(0);
    local_tid_40824 = get_local_id(0);
    group_sizze_40827 = get_local_size(0);
    wave_sizze_40826 = LOCKSTEP_WIDTH;
    group_tid_40825 = get_group_id(0);
    
    int32_t phys_tid_38398;
    
    phys_tid_38398 = global_tid_40823;
    
    int32_t gtid_38396;
    
    gtid_38396 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40825) *
                                      sext_i32_i64(segmap_group_sizze_38483) +
                                      sext_i32_i64(local_tid_40824),
                                      sext_i32_i64(d_32136)));
    
    int32_t gtid_38397;
    
    gtid_38397 = sext_i64_i32(sext_i32_i64(group_tid_40825) *
        sext_i32_i64(segmap_group_sizze_38483) + sext_i32_i64(local_tid_40824) -
        squot64(sext_i32_i64(group_tid_40825) *
                sext_i32_i64(segmap_group_sizze_38483) +
                sext_i32_i64(local_tid_40824), sext_i32_i64(d_32136)) *
        sext_i32_i64(d_32136));
    if (slt32(gtid_38396, res_32168) && slt32(gtid_38397, d_32136)) {
        int32_t x_38488 = ((__global
                            int32_t *) res_mem_39993)[sext_i32_i64(gtid_38396)];
        bool bounds_check_38489 = ((__global
                                    bool *) mem_40011)[sext_i32_i64(gtid_38396)];
        bool index_concat_cmp_38490 = ((__global
                                        bool *) mem_40013)[sext_i32_i64(gtid_38396)];
        bool index_certs_38496;
        
        if (!bounds_check_38489) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 25) == -1) {
                    global_failure_args[0] = x_38488;
                    global_failure_args[1] = gtid_38397;
                    global_failure_args[2] = res_32168;
                    global_failure_args[3] = d_32136;
                    ;
                }
                return;
            }
        }
        
        float index_concat_branch_38497;
        
        if (index_concat_cmp_38490) {
            index_concat_branch_38497 = INFINITY;
        } else {
            float index_concat_38498 = ((__global
                                         float *) input_mem_39418)[sext_i32_i64(x_38488) *
                                                                   sext_i32_i64(d_32136) +
                                                                   sext_i32_i64(gtid_38397)];
            
            index_concat_branch_38497 = index_concat_38498;
        }
        
        float res_38499 = index_concat_branch_38497;
        
        ((__global float *) mem_40019)[sext_i32_i64(gtid_38396) *
                                       sext_i32_i64(d_32136) +
                                       sext_i32_i64(gtid_38397)] = res_38499;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_38483
}
__kernel void buildKDtreezisegmap_38442(__global int *global_failure,
                                        int32_t m_32135, int32_t res_32168,
                                        __global unsigned char *res_mem_39993,
                                        __global unsigned char *mem_40011,
                                        __global unsigned char *mem_40013)
{
    #define segmap_group_sizze_38457 (buildKDtreezisegmap_group_sizze_38445)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40818;
    int32_t local_tid_40819;
    int32_t group_sizze_40822;
    int32_t wave_sizze_40821;
    int32_t group_tid_40820;
    
    global_tid_40818 = get_global_id(0);
    local_tid_40819 = get_local_id(0);
    group_sizze_40822 = get_local_size(0);
    wave_sizze_40821 = LOCKSTEP_WIDTH;
    group_tid_40820 = get_group_id(0);
    
    int32_t phys_tid_38442;
    
    phys_tid_38442 = global_tid_40818;
    
    int32_t gtid_38441;
    
    gtid_38441 = sext_i64_i32(sext_i32_i64(group_tid_40820) *
        sext_i32_i64(segmap_group_sizze_38457) + sext_i32_i64(local_tid_40819));
    if (slt32(gtid_38441, res_32168)) {
        int32_t x_38463 = ((__global
                            int32_t *) res_mem_39993)[sext_i32_i64(gtid_38441)];
        bool x_38464 = sle32(0, x_38463);
        bool y_38465 = slt32(x_38463, res_32168);
        bool bounds_check_38466 = x_38464 && y_38465;
        bool index_concat_cmp_38467 = sle32(m_32135, x_38463);
        
        ((__global bool *) mem_40011)[sext_i32_i64(gtid_38441)] =
            bounds_check_38466;
        ((__global bool *) mem_40013)[sext_i32_i64(gtid_38441)] =
            index_concat_cmp_38467;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_38457
}
__kernel void buildKDtreezisegmap_38502(__global int *global_failure,
                                        int32_t res_32167, int32_t res_32168,
                                        __global unsigned char *res_mem_39993,
                                        __global unsigned char *mem_40022)
{
    #define segmap_group_sizze_38506 (buildKDtreezisegmap_group_sizze_38505)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40828;
    int32_t local_tid_40829;
    int32_t group_sizze_40832;
    int32_t wave_sizze_40831;
    int32_t group_tid_40830;
    
    global_tid_40828 = get_global_id(0);
    local_tid_40829 = get_local_id(0);
    group_sizze_40832 = get_local_size(0);
    wave_sizze_40831 = LOCKSTEP_WIDTH;
    group_tid_40830 = get_group_id(0);
    
    int32_t phys_tid_38502;
    
    phys_tid_38502 = global_tid_40828;
    
    int32_t write_i_38501;
    
    write_i_38501 = sext_i64_i32(sext_i32_i64(group_tid_40830) *
        sext_i32_i64(segmap_group_sizze_38506) + sext_i32_i64(local_tid_40829));
    if (slt32(write_i_38501, res_32168)) {
        int32_t write_index_32681 = ((__global
                                      int32_t *) res_mem_39993)[sext_i32_i64(write_i_38501)];
        int32_t res_32682 = sdiv32(write_i_38501, res_32167);
        
        if (sle32(0, write_index_32681) && slt32(write_index_32681,
                                                 res_32168)) {
            ((__global int32_t *) mem_40022)[sext_i32_i64(write_index_32681)] =
                res_32682;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_38506
}
__kernel void buildKDtreezisegmap_intragroup_36940(__global int *global_failure,
                                                   int failure_is_an_option,
                                                   __global
                                                   int *global_failure_args,
                                                   uint mem_39816_backing_offset_0,
                                                   uint mem_39813_backing_offset_1,
                                                   uint mem_39787_backing_offset_2,
                                                   uint double_buffer_mem_40108_backing_offset_3,
                                                   uint mem_39790_backing_offset_4,
                                                   uint mem_39784_backing_offset_5,
                                                   uint mem_39781_backing_offset_6,
                                                   uint mem_39778_backing_offset_7,
                                                   uint mem_39775_backing_offset_8,
                                                   uint mem_39772_backing_offset_9,
                                                   uint mem_39754_backing_offset_10,
                                                   int32_t m_32135,
                                                   int32_t d_32136,
                                                   int32_t res_32168,
                                                   int32_t pts_per_node_at_lev_32225,
                                                   int32_t iters_32322,
                                                   int32_t i_32324, __global
                                                   unsigned char *input_mem_39418,
                                                   __global
                                                   unsigned char *mem_param_39466,
                                                   __global
                                                   unsigned char *res_mem_39552,
                                                   __global
                                                   unsigned char *mem_39556,
                                                   __global
                                                   unsigned char *mem_39821,
                                                   __global
                                                   unsigned char *mem_39826,
                                                   __global
                                                   unsigned char *double_buffer_mem_40109)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *mem_39816_backing_10 =
                  &shared_mem[mem_39816_backing_offset_0];
    volatile char *mem_39813_backing_9 =
                  &shared_mem[mem_39813_backing_offset_1];
    volatile char *mem_39787_backing_8 =
                  &shared_mem[mem_39787_backing_offset_2];
    volatile char *double_buffer_mem_40108_backing_7 =
                  &shared_mem[double_buffer_mem_40108_backing_offset_3];
    volatile char *mem_39790_backing_6 =
                  &shared_mem[mem_39790_backing_offset_4];
    volatile char *mem_39784_backing_5 =
                  &shared_mem[mem_39784_backing_offset_5];
    volatile char *mem_39781_backing_4 =
                  &shared_mem[mem_39781_backing_offset_6];
    volatile char *mem_39778_backing_3 =
                  &shared_mem[mem_39778_backing_offset_7];
    volatile char *mem_39775_backing_2 =
                  &shared_mem[mem_39775_backing_offset_8];
    volatile char *mem_39772_backing_1 =
                  &shared_mem[mem_39772_backing_offset_9];
    volatile char *mem_39754_backing_0 =
                  &shared_mem[mem_39754_backing_offset_10];
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40635;
    int32_t local_tid_40636;
    int32_t group_sizze_40639;
    int32_t wave_sizze_40638;
    int32_t group_tid_40637;
    
    global_tid_40635 = get_global_id(0);
    local_tid_40636 = get_local_id(0);
    group_sizze_40639 = get_local_size(0);
    wave_sizze_40638 = LOCKSTEP_WIDTH;
    group_tid_40637 = get_group_id(0);
    
    int32_t phys_tid_36940;
    
    phys_tid_36940 = group_tid_40637;
    
    int32_t ltid_pre_40640;
    
    ltid_pre_40640 = local_tid_40636;
    
    int32_t gtid_36911;
    
    gtid_36911 = group_tid_40637;
    
    int32_t x_37195;
    
    x_37195 = ((__global int32_t *) res_mem_39552)[sext_i32_i64(gtid_36911)];
    
    bool x_37196 = sle32(0, x_37195);
    bool y_37197 = slt32(x_37195, d_32136);
    bool bounds_check_37198 = x_37196 && y_37197;
    int32_t binop_x_39309 = pts_per_node_at_lev_32225 * gtid_36911;
    __local char *mem_39754;
    
    mem_39754 = (__local char *) mem_39754_backing_0;
    
    int32_t gtid_36914 = ltid_pre_40640;
    int32_t phys_tid_36915 = local_tid_40636;
    
    if (slt32(gtid_36914, pts_per_node_at_lev_32225)) {
        int32_t new_index_39310 = gtid_36914 + binop_x_39309;
        int32_t x_37200 = ((__global
                            int32_t *) mem_param_39466)[sext_i32_i64(new_index_39310)];
        bool x_37201 = sle32(0, x_37200);
        bool y_37202 = slt32(x_37200, res_32168);
        bool bounds_check_37203 = x_37201 && y_37202;
        bool index_ok_37204 = bounds_check_37198 && bounds_check_37203;
        bool index_certs_37205;
        
        if (!index_ok_37204) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 19) == -1) {
                    global_failure_args[0] = x_37200;
                    global_failure_args[1] = x_37195;
                    global_failure_args[2] = res_32168;
                    global_failure_args[3] = d_32136;
                    ;
                }
                local_failure = true;
                goto error_0;
            }
        }
        
        bool index_concat_cmp_37206 = sle32(m_32135, x_37200);
        float index_concat_branch_37207;
        
        if (index_concat_cmp_37206) {
            index_concat_branch_37207 = INFINITY;
        } else {
            float index_concat_37208 = ((__global
                                         float *) input_mem_39418)[sext_i32_i64(x_37200) *
                                                                   sext_i32_i64(d_32136) +
                                                                   sext_i32_i64(x_37195)];
            
            index_concat_branch_37207 = index_concat_37208;
        }
        
        float res_37209 = index_concat_branch_37207;
        
        ((__local float *) mem_39754)[sext_i32_i64(gtid_36914)] = res_37209;
    }
    
  error_0:
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_failure)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_39772;
    
    mem_39772 = (__local char *) mem_39772_backing_1;
    
    __local char *mem_39775;
    
    mem_39775 = (__local char *) mem_39775_backing_2;
    
    __local char *mem_39778;
    
    mem_39778 = (__local char *) mem_39778_backing_3;
    
    __local char *mem_39781;
    
    mem_39781 = (__local char *) mem_39781_backing_4;
    
    __local char *mem_39784;
    
    mem_39784 = (__local char *) mem_39784_backing_5;
    
    __local char *mem_39790;
    
    mem_39790 = (__local char *) mem_39790_backing_6;
    
    __local char *double_buffer_mem_40108;
    
    double_buffer_mem_40108 = (__local
                               char *) double_buffer_mem_40108_backing_7;
    ((__local float *) double_buffer_mem_40108)[sext_i32_i64(local_tid_40636)] =
        ((__local float *) mem_39754)[sext_i32_i64(local_tid_40636)];
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global int32_t *) double_buffer_mem_40109)[sext_i32_i64(phys_tid_36940 *
                                                   pts_per_node_at_lev_32225) +
                                                   sext_i32_i64(local_tid_40636)] =
        ((__global int32_t *) mem_39556)[sext_i32_i64(local_tid_40636)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_39787;
    
    mem_39787 = (__local char *) mem_39787_backing_8;
    for (int32_t i_37212 = 0; i_37212 < iters_32322; i_37212++) {
        int32_t lifted_2_radix_sort_step_arg_37215 = mul32(2, i_37212);
        int32_t lifted_0_get_bit_arg_37216 = add32(1,
                                                   lifted_2_radix_sort_step_arg_37215);
        bool res_37217 = lifted_0_get_bit_arg_37216 == 31;
        bool res_37218 = lifted_2_radix_sort_step_arg_37215 == 31;
        int32_t gtid_36926 = ltid_pre_40640;
        int32_t phys_tid_36927 = local_tid_40636;
        
        if (slt32(gtid_36926, pts_per_node_at_lev_32225)) {
            float x_37236 = ((__local
                              float *) double_buffer_mem_40108)[sext_i32_i64(gtid_36926)];
            int32_t i32_arg_37237;
            
            i32_arg_37237 = futrts_to_bits32(x_37236);
            
            int32_t unsign_arg_37238 = ashr32(i32_arg_37237,
                                              lifted_0_get_bit_arg_37216);
            int32_t unsign_arg_37239 = 1 & unsign_arg_37238;
            int32_t unsign_arg_37240 = ashr32(i32_arg_37237, 31);
            int32_t unsign_arg_37241 = 1 & unsign_arg_37240;
            bool cond_37242 = unsign_arg_37241 == 1;
            bool x_37243 = !cond_37242;
            bool y_37244 = res_37217 && x_37243;
            bool cond_37245 = cond_37242 || y_37244;
            int32_t res_37246;
            
            if (cond_37245) {
                int32_t res_37247 = 1 ^ unsign_arg_37239;
                
                res_37246 = res_37247;
            } else {
                res_37246 = unsign_arg_37239;
            }
            
            int32_t x_37248 = mul32(2, res_37246);
            int32_t unsign_arg_37249 = ashr32(i32_arg_37237,
                                              lifted_2_radix_sort_step_arg_37215);
            int32_t unsign_arg_37250 = 1 & unsign_arg_37249;
            bool y_37251 = res_37218 && x_37243;
            bool cond_37252 = cond_37242 || y_37251;
            int32_t res_37253;
            
            if (cond_37252) {
                int32_t res_37254 = 1 ^ unsign_arg_37250;
                
                res_37253 = res_37254;
            } else {
                res_37253 = unsign_arg_37250;
            }
            
            int32_t res_37255 = add32(x_37248, res_37253);
            bool cond_37256 = res_37255 == 0;
            int32_t res_37257 = btoi_bool_i32(cond_37256);
            int32_t res_37258;
            int32_t res_37259;
            int32_t res_37260;
            
            if (cond_37256) {
                res_37258 = 0;
                res_37259 = 0;
                res_37260 = 0;
            } else {
                bool cond_37261 = res_37255 == 1;
                int32_t res_37262 = btoi_bool_i32(cond_37261);
                int32_t res_37263;
                int32_t res_37264;
                
                if (cond_37261) {
                    res_37263 = 0;
                    res_37264 = 0;
                } else {
                    bool cond_37265 = res_37255 == 2;
                    int32_t res_37266 = btoi_bool_i32(cond_37265);
                    bool cond_neg_37267 = !cond_37265;
                    int32_t res_37268 = btoi_bool_i32(cond_neg_37267);
                    
                    res_37263 = res_37266;
                    res_37264 = res_37268;
                }
                res_37258 = res_37262;
                res_37259 = res_37263;
                res_37260 = res_37264;
            }
            ((__local int32_t *) mem_39772)[sext_i32_i64(gtid_36926)] =
                res_37257;
            ((__local int32_t *) mem_39775)[sext_i32_i64(gtid_36926)] =
                res_37258;
            ((__local int32_t *) mem_39778)[sext_i32_i64(gtid_36926)] =
                res_37259;
            ((__local int32_t *) mem_39781)[sext_i32_i64(gtid_36926)] =
                res_37260;
            ((__local int32_t *) mem_39784)[sext_i32_i64(gtid_36926)] =
                res_37255;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t dims_flat_40643;
        
        dims_flat_40643 = pts_per_node_at_lev_32225;
        
        int32_t x_37224;
        int32_t x_37225;
        int32_t x_37226;
        int32_t x_37227;
        int32_t x_37228;
        int32_t x_37229;
        int32_t x_37230;
        int32_t x_37231;
        int32_t x_40648;
        int32_t x_40649;
        int32_t x_40650;
        int32_t x_40651;
        int32_t x_40652;
        int32_t x_40653;
        int32_t x_40654;
        int32_t x_40655;
        int32_t skip_threads_40660;
        
        // read input for in-block scan
        {
            if (slt32(local_tid_40636, pts_per_node_at_lev_32225)) {
                x_37228 = ((volatile __local
                            int32_t *) mem_39772)[sext_i32_i64(local_tid_40636)];
                x_37229 = ((volatile __local
                            int32_t *) mem_39775)[sext_i32_i64(local_tid_40636)];
                x_37230 = ((volatile __local
                            int32_t *) mem_39778)[sext_i32_i64(local_tid_40636)];
                x_37231 = ((volatile __local
                            int32_t *) mem_39781)[sext_i32_i64(local_tid_40636)];
                if ((local_tid_40636 - squot32(local_tid_40636, 32) * 32) ==
                    0) {
                    x_37224 = x_37228;
                    x_37225 = x_37229;
                    x_37226 = x_37230;
                    x_37227 = x_37231;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_40660 = 1;
            while (slt32(skip_threads_40660, 32)) {
                if (sle32(skip_threads_40660, local_tid_40636 -
                          squot32(local_tid_40636, 32) * 32) &&
                    slt32(local_tid_40636, pts_per_node_at_lev_32225)) {
                    // read operands
                    {
                        x_37224 = ((volatile __local
                                    int32_t *) mem_39772)[sext_i32_i64(local_tid_40636 -
                                                          skip_threads_40660)];
                        x_37225 = ((volatile __local
                                    int32_t *) mem_39775)[sext_i32_i64(local_tid_40636 -
                                                          skip_threads_40660)];
                        x_37226 = ((volatile __local
                                    int32_t *) mem_39778)[sext_i32_i64(local_tid_40636 -
                                                          skip_threads_40660)];
                        x_37227 = ((volatile __local
                                    int32_t *) mem_39781)[sext_i32_i64(local_tid_40636 -
                                                          skip_threads_40660)];
                    }
                    // perform operation
                    {
                        bool inactive_40661 = slt32(srem32(local_tid_40636,
                                                           pts_per_node_at_lev_32225),
                                                    local_tid_40636 -
                                                    (local_tid_40636 -
                                                     skip_threads_40660));
                        
                        if (inactive_40661) {
                            x_37224 = x_37228;
                            x_37225 = x_37229;
                            x_37226 = x_37230;
                            x_37227 = x_37231;
                        }
                        if (!inactive_40661) {
                            int32_t res_37232 = add32(x_37224, x_37228);
                            int32_t res_37233 = add32(x_37225, x_37229);
                            int32_t res_37234 = add32(x_37226, x_37230);
                            int32_t res_37235 = add32(x_37227, x_37231);
                            
                            x_37224 = res_37232;
                            x_37225 = res_37233;
                            x_37226 = res_37234;
                            x_37227 = res_37235;
                        }
                    }
                }
                if (sle32(wave_sizze_40638, skip_threads_40660)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_40660, local_tid_40636 -
                          squot32(local_tid_40636, 32) * 32) &&
                    slt32(local_tid_40636, pts_per_node_at_lev_32225)) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) mem_39772)[sext_i32_i64(local_tid_40636)] =
                            x_37224;
                        x_37228 = x_37224;
                        ((volatile __local
                          int32_t *) mem_39775)[sext_i32_i64(local_tid_40636)] =
                            x_37225;
                        x_37229 = x_37225;
                        ((volatile __local
                          int32_t *) mem_39778)[sext_i32_i64(local_tid_40636)] =
                            x_37226;
                        x_37230 = x_37226;
                        ((volatile __local
                          int32_t *) mem_39781)[sext_i32_i64(local_tid_40636)] =
                            x_37227;
                        x_37231 = x_37227;
                    }
                }
                if (sle32(wave_sizze_40638, skip_threads_40660)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_40660 *= 2;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // last thread of block 'i' writes its result to offset 'i'
        {
            if ((local_tid_40636 - squot32(local_tid_40636, 32) * 32) == 31 &&
                slt32(local_tid_40636, pts_per_node_at_lev_32225)) {
                ((volatile __local
                  int32_t *) mem_39772)[sext_i32_i64(squot32(local_tid_40636,
                                                             32))] = x_37224;
                ((volatile __local
                  int32_t *) mem_39775)[sext_i32_i64(squot32(local_tid_40636,
                                                             32))] = x_37225;
                ((volatile __local
                  int32_t *) mem_39778)[sext_i32_i64(squot32(local_tid_40636,
                                                             32))] = x_37226;
                ((volatile __local
                  int32_t *) mem_39781)[sext_i32_i64(squot32(local_tid_40636,
                                                             32))] = x_37227;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
        {
            int32_t skip_threads_40662;
            
            // read input for in-block scan
            {
                if (squot32(local_tid_40636, 32) == 0 && slt32(local_tid_40636,
                                                               pts_per_node_at_lev_32225)) {
                    x_40652 = ((volatile __local
                                int32_t *) mem_39772)[sext_i32_i64(local_tid_40636)];
                    x_40653 = ((volatile __local
                                int32_t *) mem_39775)[sext_i32_i64(local_tid_40636)];
                    x_40654 = ((volatile __local
                                int32_t *) mem_39778)[sext_i32_i64(local_tid_40636)];
                    x_40655 = ((volatile __local
                                int32_t *) mem_39781)[sext_i32_i64(local_tid_40636)];
                    if ((local_tid_40636 - squot32(local_tid_40636, 32) * 32) ==
                        0) {
                        x_40648 = x_40652;
                        x_40649 = x_40653;
                        x_40650 = x_40654;
                        x_40651 = x_40655;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_40662 = 1;
                while (slt32(skip_threads_40662, 32)) {
                    if (sle32(skip_threads_40662, local_tid_40636 -
                              squot32(local_tid_40636, 32) * 32) &&
                        (squot32(local_tid_40636, 32) == 0 &&
                         slt32(local_tid_40636, pts_per_node_at_lev_32225))) {
                        // read operands
                        {
                            x_40648 = ((volatile __local
                                        int32_t *) mem_39772)[sext_i32_i64(local_tid_40636 -
                                                              skip_threads_40662)];
                            x_40649 = ((volatile __local
                                        int32_t *) mem_39775)[sext_i32_i64(local_tid_40636 -
                                                              skip_threads_40662)];
                            x_40650 = ((volatile __local
                                        int32_t *) mem_39778)[sext_i32_i64(local_tid_40636 -
                                                              skip_threads_40662)];
                            x_40651 = ((volatile __local
                                        int32_t *) mem_39781)[sext_i32_i64(local_tid_40636 -
                                                              skip_threads_40662)];
                        }
                        // perform operation
                        {
                            bool inactive_40663 = slt32(srem32(local_tid_40636 *
                                                               32 + 32 - 1,
                                                               pts_per_node_at_lev_32225),
                                                        local_tid_40636 * 32 +
                                                        32 - 1 -
                                                        ((local_tid_40636 -
                                                          skip_threads_40662) *
                                                         32 + 32 - 1));
                            
                            if (inactive_40663) {
                                x_40648 = x_40652;
                                x_40649 = x_40653;
                                x_40650 = x_40654;
                                x_40651 = x_40655;
                            }
                            if (!inactive_40663) {
                                int32_t res_40656 = add32(x_40648, x_40652);
                                int32_t res_40657 = add32(x_40649, x_40653);
                                int32_t res_40658 = add32(x_40650, x_40654);
                                int32_t res_40659 = add32(x_40651, x_40655);
                                
                                x_40648 = res_40656;
                                x_40649 = res_40657;
                                x_40650 = res_40658;
                                x_40651 = res_40659;
                            }
                        }
                    }
                    if (sle32(wave_sizze_40638, skip_threads_40662)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_40662, local_tid_40636 -
                              squot32(local_tid_40636, 32) * 32) &&
                        (squot32(local_tid_40636, 32) == 0 &&
                         slt32(local_tid_40636, pts_per_node_at_lev_32225))) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) mem_39772)[sext_i32_i64(local_tid_40636)] =
                                x_40648;
                            x_40652 = x_40648;
                            ((volatile __local
                              int32_t *) mem_39775)[sext_i32_i64(local_tid_40636)] =
                                x_40649;
                            x_40653 = x_40649;
                            ((volatile __local
                              int32_t *) mem_39778)[sext_i32_i64(local_tid_40636)] =
                                x_40650;
                            x_40654 = x_40650;
                            ((volatile __local
                              int32_t *) mem_39781)[sext_i32_i64(local_tid_40636)] =
                                x_40651;
                            x_40655 = x_40651;
                        }
                    }
                    if (sle32(wave_sizze_40638, skip_threads_40662)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_40662 *= 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // carry-in for every block except the first
        {
            if (!(squot32(local_tid_40636, 32) == 0 || !slt32(local_tid_40636,
                                                              pts_per_node_at_lev_32225))) {
                // read operands
                {
                    x_37228 = x_37224;
                    x_37229 = x_37225;
                    x_37230 = x_37226;
                    x_37231 = x_37227;
                    x_37224 = ((__local
                                int32_t *) mem_39772)[sext_i32_i64(squot32(local_tid_40636,
                                                                           32) -
                                                      1)];
                    x_37225 = ((__local
                                int32_t *) mem_39775)[sext_i32_i64(squot32(local_tid_40636,
                                                                           32) -
                                                      1)];
                    x_37226 = ((__local
                                int32_t *) mem_39778)[sext_i32_i64(squot32(local_tid_40636,
                                                                           32) -
                                                      1)];
                    x_37227 = ((__local
                                int32_t *) mem_39781)[sext_i32_i64(squot32(local_tid_40636,
                                                                           32) -
                                                      1)];
                }
                // perform operation
                {
                    bool inactive_40664 = slt32(srem32(local_tid_40636,
                                                       pts_per_node_at_lev_32225),
                                                local_tid_40636 -
                                                (squot32(local_tid_40636, 32) *
                                                 32 - 1));
                    
                    if (inactive_40664) {
                        x_37224 = x_37228;
                        x_37225 = x_37229;
                        x_37226 = x_37230;
                        x_37227 = x_37231;
                    }
                    if (!inactive_40664) {
                        int32_t res_37232 = add32(x_37224, x_37228);
                        int32_t res_37233 = add32(x_37225, x_37229);
                        int32_t res_37234 = add32(x_37226, x_37230);
                        int32_t res_37235 = add32(x_37227, x_37231);
                        
                        x_37224 = res_37232;
                        x_37225 = res_37233;
                        x_37226 = res_37234;
                        x_37227 = res_37235;
                    }
                }
                // write final result
                {
                    ((__local
                      int32_t *) mem_39772)[sext_i32_i64(local_tid_40636)] =
                        x_37224;
                    ((__local
                      int32_t *) mem_39775)[sext_i32_i64(local_tid_40636)] =
                        x_37225;
                    ((__local
                      int32_t *) mem_39778)[sext_i32_i64(local_tid_40636)] =
                        x_37226;
                    ((__local
                      int32_t *) mem_39781)[sext_i32_i64(local_tid_40636)] =
                        x_37227;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // restore correct values for first block
        {
            if (squot32(local_tid_40636, 32) == 0) {
                ((__local int32_t *) mem_39772)[sext_i32_i64(local_tid_40636)] =
                    x_37228;
                ((__local int32_t *) mem_39775)[sext_i32_i64(local_tid_40636)] =
                    x_37229;
                ((__local int32_t *) mem_39778)[sext_i32_i64(local_tid_40636)] =
                    x_37230;
                ((__local int32_t *) mem_39781)[sext_i32_i64(local_tid_40636)] =
                    x_37231;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t res_37269 = ((__local
                              int32_t *) mem_39772)[sext_i32_i64(i_32324)];
        int32_t res_37270 = ((__local
                              int32_t *) mem_39775)[sext_i32_i64(i_32324)];
        int32_t res_37271 = ((__local
                              int32_t *) mem_39778)[sext_i32_i64(i_32324)];
        
        ((__local float *) mem_39787)[sext_i32_i64(local_tid_40636)] = ((__local
                                                                         float *) double_buffer_mem_40108)[sext_i32_i64(local_tid_40636)];
        barrier(CLK_LOCAL_MEM_FENCE);
        ((__local int32_t *) mem_39790)[sext_i32_i64(local_tid_40636)] =
            ((__global
              int32_t *) double_buffer_mem_40109)[sext_i32_i64(phys_tid_36940 *
                                                  pts_per_node_at_lev_32225) +
                                                  sext_i32_i64(local_tid_40636)];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t write_i_36928 = ltid_pre_40640;
        int32_t phys_tid_36929 = local_tid_40636;
        
        if (slt32(write_i_36928, pts_per_node_at_lev_32225)) {
            int32_t x_37276 = ((__local
                                int32_t *) mem_39784)[sext_i32_i64(write_i_36928)];
            float write_value_37281 = ((__local
                                        float *) double_buffer_mem_40108)[sext_i32_i64(write_i_36928)];
            int32_t write_value_37282 = ((__global
                                          int32_t *) double_buffer_mem_40109)[sext_i32_i64(phys_tid_36940 *
                                                                              pts_per_node_at_lev_32225) +
                                                                              sext_i32_i64(write_i_36928)];
            bool match_lit_37283 = 0 == x_37276;
            int32_t res_37284;
            
            if (match_lit_37283) {
                int32_t x_37277 = ((__local
                                    int32_t *) mem_39772)[sext_i32_i64(write_i_36928)];
                int32_t res_37285 = sub32(x_37277, 1);
                
                res_37284 = res_37285;
            } else {
                int32_t x_37278 = ((__local
                                    int32_t *) mem_39775)[sext_i32_i64(write_i_36928)];
                int32_t x_37279 = ((__local
                                    int32_t *) mem_39778)[sext_i32_i64(write_i_36928)];
                int32_t x_37280 = ((__local
                                    int32_t *) mem_39781)[sext_i32_i64(write_i_36928)];
                bool match_lit_37286 = 1 == x_37276;
                int32_t x_37287;
                
                if (match_lit_37286) {
                    int32_t x_37288 = add32(res_37269, x_37278);
                    int32_t res_37289 = sub32(x_37288, 1);
                    
                    x_37287 = res_37289;
                } else {
                    bool match_lit_37290 = 2 == x_37276;
                    int32_t x_37291;
                    
                    if (match_lit_37290) {
                        int32_t x_37292 = add32(res_37269, res_37270);
                        int32_t x_37293 = add32(x_37279, x_37292);
                        int32_t res_37294 = sub32(x_37293, 1);
                        
                        x_37291 = res_37294;
                    } else {
                        int32_t x_37295 = add32(res_37269, res_37270);
                        int32_t x_37296 = add32(res_37271, x_37295);
                        int32_t x_37297 = add32(x_37280, x_37296);
                        int32_t res_37298 = sub32(x_37297, 1);
                        
                        x_37291 = res_37298;
                    }
                    x_37287 = x_37291;
                }
                res_37284 = x_37287;
            }
            if (sle32(0, res_37284) && slt32(res_37284,
                                             pts_per_node_at_lev_32225)) {
                ((__local float *) mem_39787)[sext_i32_i64(res_37284)] =
                    write_value_37281;
            }
            if (sle32(0, res_37284) && slt32(res_37284,
                                             pts_per_node_at_lev_32225)) {
                ((__local int32_t *) mem_39790)[sext_i32_i64(res_37284)] =
                    write_value_37282;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        ((__local
          float *) double_buffer_mem_40108)[sext_i32_i64(local_tid_40636)] =
            ((__local float *) mem_39787)[sext_i32_i64(local_tid_40636)];
        barrier(CLK_LOCAL_MEM_FENCE);
        ((__global
          int32_t *) double_buffer_mem_40109)[sext_i32_i64(phys_tid_36940 *
                                              pts_per_node_at_lev_32225) +
                                              sext_i32_i64(local_tid_40636)] =
            ((__local int32_t *) mem_39790)[sext_i32_i64(local_tid_40636)];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_39813;
    
    mem_39813 = (__local char *) mem_39813_backing_9;
    
    __local char *mem_39816;
    
    mem_39816 = (__local char *) mem_39816_backing_10;
    
    int32_t gtid_36930 = ltid_pre_40640;
    int32_t phys_tid_36931 = local_tid_40636;
    
    if (slt32(gtid_36930, pts_per_node_at_lev_32225)) {
        int32_t x_37301 = ((__global
                            int32_t *) double_buffer_mem_40109)[sext_i32_i64(phys_tid_36940 *
                                                                pts_per_node_at_lev_32225) +
                                                                sext_i32_i64(gtid_36930)];
        bool x_37302 = sle32(0, x_37301);
        bool y_37303 = slt32(x_37301, pts_per_node_at_lev_32225);
        bool bounds_check_37304 = x_37302 && y_37303;
        bool index_certs_37305;
        
        if (!bounds_check_37304) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 20) == -1) {
                    global_failure_args[0] = x_37301;
                    global_failure_args[1] = pts_per_node_at_lev_32225;
                    ;
                }
                local_failure = true;
                goto error_3;
            }
        }
        
        float res_37306 = ((__local float *) mem_39754)[sext_i32_i64(x_37301)];
        bool index_certs_37307;
        
        if (!bounds_check_37304) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 21) == -1) {
                    global_failure_args[0] = x_37301;
                    global_failure_args[1] = pts_per_node_at_lev_32225;
                    ;
                }
                local_failure = true;
                goto error_3;
            }
        }
        
        int32_t new_index_39312 = x_37301 + binop_x_39309;
        int32_t res_37308 = ((__global
                              int32_t *) mem_param_39466)[sext_i32_i64(new_index_39312)];
        
        ((__local int32_t *) mem_39813)[sext_i32_i64(gtid_36930)] = res_37308;
        ((__local float *) mem_39816)[sext_i32_i64(gtid_36930)] = res_37306;
    }
    
  error_3:
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_failure)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global int32_t *) mem_39821)[sext_i32_i64(gtid_36911) *
                                     sext_i32_i64(pts_per_node_at_lev_32225) +
                                     sext_i32_i64(local_tid_40636)] = ((__local
                                                                        int32_t *) mem_39813)[sext_i32_i64(local_tid_40636)];
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global float *) mem_39826)[sext_i32_i64(gtid_36911) *
                                   sext_i32_i64(pts_per_node_at_lev_32225) +
                                   sext_i32_i64(local_tid_40636)] = ((__local
                                                                      float *) mem_39816)[sext_i32_i64(local_tid_40636)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_4:
    return;
}
__kernel void buildKDtreezisegmap_intragroup_37393(__global int *global_failure,
                                                   uint mem_39888_backing_offset_0,
                                                   uint mem_39885_backing_offset_1,
                                                   uint mem_39882_backing_offset_2,
                                                   uint mem_39879_backing_offset_3,
                                                   uint mem_39876_backing_offset_4,
                                                   uint mem_39873_backing_offset_5,
                                                   uint mem_39870_backing_offset_6,
                                                   int32_t nodes_this_lvl_32221,
                                                   int32_t pts_per_node_at_lev_32225,
                                                   int32_t i_32324,
                                                   int32_t lifted_2_radix_sort_step_arg_38052,
                                                   int32_t lifted_0_get_bit_arg_38053,
                                                   unsigned char res_38054,
                                                   unsigned char res_38055,
                                                   __global
                                                   unsigned char *mem_param_39843,
                                                   __global
                                                   unsigned char *mem_param_39851,
                                                   __global
                                                   unsigned char *mem_39860,
                                                   __global
                                                   unsigned char *mem_39865,
                                                   __global
                                                   unsigned char *mem_39894,
                                                   __global
                                                   unsigned char *mem_39899)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *mem_39888_backing_6 =
                  &shared_mem[mem_39888_backing_offset_0];
    volatile char *mem_39885_backing_5 =
                  &shared_mem[mem_39885_backing_offset_1];
    volatile char *mem_39882_backing_4 =
                  &shared_mem[mem_39882_backing_offset_2];
    volatile char *mem_39879_backing_3 =
                  &shared_mem[mem_39879_backing_offset_3];
    volatile char *mem_39876_backing_2 =
                  &shared_mem[mem_39876_backing_offset_4];
    volatile char *mem_39873_backing_1 =
                  &shared_mem[mem_39873_backing_offset_5];
    volatile char *mem_39870_backing_0 =
                  &shared_mem[mem_39870_backing_offset_6];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40679;
    int32_t local_tid_40680;
    int32_t group_sizze_40683;
    int32_t wave_sizze_40682;
    int32_t group_tid_40681;
    
    global_tid_40679 = get_global_id(0);
    local_tid_40680 = get_local_id(0);
    group_sizze_40683 = get_local_size(0);
    wave_sizze_40682 = LOCKSTEP_WIDTH;
    group_tid_40681 = get_group_id(0);
    
    int32_t phys_tid_37393;
    
    phys_tid_37393 = group_tid_40681;
    
    int32_t ltid_pre_40684;
    
    ltid_pre_40684 = local_tid_40680;
    
    int32_t gtid_37386;
    
    gtid_37386 = group_tid_40681;
    
    __local char *mem_39870;
    
    mem_39870 = (__local char *) mem_39870_backing_0;
    
    __local char *mem_39873;
    
    mem_39873 = (__local char *) mem_39873_backing_1;
    
    __local char *mem_39876;
    
    mem_39876 = (__local char *) mem_39876_backing_2;
    
    __local char *mem_39879;
    
    mem_39879 = (__local char *) mem_39879_backing_3;
    
    __local char *mem_39882;
    
    mem_39882 = (__local char *) mem_39882_backing_4;
    
    int32_t gtid_37389 = ltid_pre_40684;
    int32_t phys_tid_37390 = local_tid_40680;
    
    if (slt32(gtid_37389, pts_per_node_at_lev_32225)) {
        float x_38088 = ((__global
                          float *) mem_param_39843)[sext_i32_i64(gtid_37386) *
                                                    sext_i32_i64(pts_per_node_at_lev_32225) +
                                                    sext_i32_i64(gtid_37389)];
        int32_t i32_arg_38089;
        
        i32_arg_38089 = futrts_to_bits32(x_38088);
        
        int32_t unsign_arg_38090 = ashr32(i32_arg_38089,
                                          lifted_0_get_bit_arg_38053);
        int32_t unsign_arg_38091 = 1 & unsign_arg_38090;
        int32_t unsign_arg_38092 = ashr32(i32_arg_38089, 31);
        int32_t unsign_arg_38093 = 1 & unsign_arg_38092;
        bool cond_38094 = unsign_arg_38093 == 1;
        bool x_38095 = !cond_38094;
        bool y_38096 = res_38054 && x_38095;
        bool cond_38097 = cond_38094 || y_38096;
        int32_t res_38098;
        
        if (cond_38097) {
            int32_t res_38099 = 1 ^ unsign_arg_38091;
            
            res_38098 = res_38099;
        } else {
            res_38098 = unsign_arg_38091;
        }
        
        int32_t x_38100 = mul32(2, res_38098);
        int32_t unsign_arg_38101 = ashr32(i32_arg_38089,
                                          lifted_2_radix_sort_step_arg_38052);
        int32_t unsign_arg_38102 = 1 & unsign_arg_38101;
        bool y_38103 = res_38055 && x_38095;
        bool cond_38104 = cond_38094 || y_38103;
        int32_t res_38105;
        
        if (cond_38104) {
            int32_t res_38106 = 1 ^ unsign_arg_38102;
            
            res_38105 = res_38106;
        } else {
            res_38105 = unsign_arg_38102;
        }
        
        int32_t res_38107 = add32(x_38100, res_38105);
        bool cond_38108 = res_38107 == 0;
        int32_t res_38109 = btoi_bool_i32(cond_38108);
        int32_t res_38110;
        int32_t res_38111;
        int32_t res_38112;
        
        if (cond_38108) {
            res_38110 = 0;
            res_38111 = 0;
            res_38112 = 0;
        } else {
            bool cond_38113 = res_38107 == 1;
            int32_t res_38114 = btoi_bool_i32(cond_38113);
            int32_t res_38115;
            int32_t res_38116;
            
            if (cond_38113) {
                res_38115 = 0;
                res_38116 = 0;
            } else {
                bool cond_38117 = res_38107 == 2;
                int32_t res_38118 = btoi_bool_i32(cond_38117);
                bool cond_neg_38119 = !cond_38117;
                int32_t res_38120 = btoi_bool_i32(cond_neg_38119);
                
                res_38115 = res_38118;
                res_38116 = res_38120;
            }
            res_38110 = res_38114;
            res_38111 = res_38115;
            res_38112 = res_38116;
        }
        ((__local int32_t *) mem_39870)[sext_i32_i64(gtid_37389)] = res_38109;
        ((__local int32_t *) mem_39873)[sext_i32_i64(gtid_37389)] = res_38110;
        ((__local int32_t *) mem_39876)[sext_i32_i64(gtid_37389)] = res_38111;
        ((__local int32_t *) mem_39879)[sext_i32_i64(gtid_37389)] = res_38112;
        ((__local int32_t *) mem_39882)[sext_i32_i64(gtid_37389)] = res_38107;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_40685;
    
    dims_flat_40685 = pts_per_node_at_lev_32225;
    
    int32_t x_38076;
    int32_t x_38077;
    int32_t x_38078;
    int32_t x_38079;
    int32_t x_38080;
    int32_t x_38081;
    int32_t x_38082;
    int32_t x_38083;
    int32_t x_40690;
    int32_t x_40691;
    int32_t x_40692;
    int32_t x_40693;
    int32_t x_40694;
    int32_t x_40695;
    int32_t x_40696;
    int32_t x_40697;
    int32_t skip_threads_40702;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_40680, pts_per_node_at_lev_32225)) {
            x_38080 = ((volatile __local
                        int32_t *) mem_39870)[sext_i32_i64(local_tid_40680)];
            x_38081 = ((volatile __local
                        int32_t *) mem_39873)[sext_i32_i64(local_tid_40680)];
            x_38082 = ((volatile __local
                        int32_t *) mem_39876)[sext_i32_i64(local_tid_40680)];
            x_38083 = ((volatile __local
                        int32_t *) mem_39879)[sext_i32_i64(local_tid_40680)];
            if ((local_tid_40680 - squot32(local_tid_40680, 32) * 32) == 0) {
                x_38076 = x_38080;
                x_38077 = x_38081;
                x_38078 = x_38082;
                x_38079 = x_38083;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_40702 = 1;
        while (slt32(skip_threads_40702, 32)) {
            if (sle32(skip_threads_40702, local_tid_40680 -
                      squot32(local_tid_40680, 32) * 32) &&
                slt32(local_tid_40680, pts_per_node_at_lev_32225)) {
                // read operands
                {
                    x_38076 = ((volatile __local
                                int32_t *) mem_39870)[sext_i32_i64(local_tid_40680 -
                                                      skip_threads_40702)];
                    x_38077 = ((volatile __local
                                int32_t *) mem_39873)[sext_i32_i64(local_tid_40680 -
                                                      skip_threads_40702)];
                    x_38078 = ((volatile __local
                                int32_t *) mem_39876)[sext_i32_i64(local_tid_40680 -
                                                      skip_threads_40702)];
                    x_38079 = ((volatile __local
                                int32_t *) mem_39879)[sext_i32_i64(local_tid_40680 -
                                                      skip_threads_40702)];
                }
                // perform operation
                {
                    bool inactive_40703 = slt32(srem32(local_tid_40680,
                                                       pts_per_node_at_lev_32225),
                                                local_tid_40680 -
                                                (local_tid_40680 -
                                                 skip_threads_40702));
                    
                    if (inactive_40703) {
                        x_38076 = x_38080;
                        x_38077 = x_38081;
                        x_38078 = x_38082;
                        x_38079 = x_38083;
                    }
                    if (!inactive_40703) {
                        int32_t res_38084 = add32(x_38076, x_38080);
                        int32_t res_38085 = add32(x_38077, x_38081);
                        int32_t res_38086 = add32(x_38078, x_38082);
                        int32_t res_38087 = add32(x_38079, x_38083);
                        
                        x_38076 = res_38084;
                        x_38077 = res_38085;
                        x_38078 = res_38086;
                        x_38079 = res_38087;
                    }
                }
            }
            if (sle32(wave_sizze_40682, skip_threads_40702)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_40702, local_tid_40680 -
                      squot32(local_tid_40680, 32) * 32) &&
                slt32(local_tid_40680, pts_per_node_at_lev_32225)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) mem_39870)[sext_i32_i64(local_tid_40680)] =
                        x_38076;
                    x_38080 = x_38076;
                    ((volatile __local
                      int32_t *) mem_39873)[sext_i32_i64(local_tid_40680)] =
                        x_38077;
                    x_38081 = x_38077;
                    ((volatile __local
                      int32_t *) mem_39876)[sext_i32_i64(local_tid_40680)] =
                        x_38078;
                    x_38082 = x_38078;
                    ((volatile __local
                      int32_t *) mem_39879)[sext_i32_i64(local_tid_40680)] =
                        x_38079;
                    x_38083 = x_38079;
                }
            }
            if (sle32(wave_sizze_40682, skip_threads_40702)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_40702 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_40680 - squot32(local_tid_40680, 32) * 32) == 31 &&
            slt32(local_tid_40680, pts_per_node_at_lev_32225)) {
            ((volatile __local
              int32_t *) mem_39870)[sext_i32_i64(squot32(local_tid_40680,
                                                         32))] = x_38076;
            ((volatile __local
              int32_t *) mem_39873)[sext_i32_i64(squot32(local_tid_40680,
                                                         32))] = x_38077;
            ((volatile __local
              int32_t *) mem_39876)[sext_i32_i64(squot32(local_tid_40680,
                                                         32))] = x_38078;
            ((volatile __local
              int32_t *) mem_39879)[sext_i32_i64(squot32(local_tid_40680,
                                                         32))] = x_38079;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_40704;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_40680, 32) == 0 && slt32(local_tid_40680,
                                                           pts_per_node_at_lev_32225)) {
                x_40694 = ((volatile __local
                            int32_t *) mem_39870)[sext_i32_i64(local_tid_40680)];
                x_40695 = ((volatile __local
                            int32_t *) mem_39873)[sext_i32_i64(local_tid_40680)];
                x_40696 = ((volatile __local
                            int32_t *) mem_39876)[sext_i32_i64(local_tid_40680)];
                x_40697 = ((volatile __local
                            int32_t *) mem_39879)[sext_i32_i64(local_tid_40680)];
                if ((local_tid_40680 - squot32(local_tid_40680, 32) * 32) ==
                    0) {
                    x_40690 = x_40694;
                    x_40691 = x_40695;
                    x_40692 = x_40696;
                    x_40693 = x_40697;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_40704 = 1;
            while (slt32(skip_threads_40704, 32)) {
                if (sle32(skip_threads_40704, local_tid_40680 -
                          squot32(local_tid_40680, 32) * 32) &&
                    (squot32(local_tid_40680, 32) == 0 && slt32(local_tid_40680,
                                                                pts_per_node_at_lev_32225))) {
                    // read operands
                    {
                        x_40690 = ((volatile __local
                                    int32_t *) mem_39870)[sext_i32_i64(local_tid_40680 -
                                                          skip_threads_40704)];
                        x_40691 = ((volatile __local
                                    int32_t *) mem_39873)[sext_i32_i64(local_tid_40680 -
                                                          skip_threads_40704)];
                        x_40692 = ((volatile __local
                                    int32_t *) mem_39876)[sext_i32_i64(local_tid_40680 -
                                                          skip_threads_40704)];
                        x_40693 = ((volatile __local
                                    int32_t *) mem_39879)[sext_i32_i64(local_tid_40680 -
                                                          skip_threads_40704)];
                    }
                    // perform operation
                    {
                        bool inactive_40705 = slt32(srem32(local_tid_40680 *
                                                           32 + 32 - 1,
                                                           pts_per_node_at_lev_32225),
                                                    local_tid_40680 * 32 + 32 -
                                                    1 - ((local_tid_40680 -
                                                          skip_threads_40704) *
                                                         32 + 32 - 1));
                        
                        if (inactive_40705) {
                            x_40690 = x_40694;
                            x_40691 = x_40695;
                            x_40692 = x_40696;
                            x_40693 = x_40697;
                        }
                        if (!inactive_40705) {
                            int32_t res_40698 = add32(x_40690, x_40694);
                            int32_t res_40699 = add32(x_40691, x_40695);
                            int32_t res_40700 = add32(x_40692, x_40696);
                            int32_t res_40701 = add32(x_40693, x_40697);
                            
                            x_40690 = res_40698;
                            x_40691 = res_40699;
                            x_40692 = res_40700;
                            x_40693 = res_40701;
                        }
                    }
                }
                if (sle32(wave_sizze_40682, skip_threads_40704)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_40704, local_tid_40680 -
                          squot32(local_tid_40680, 32) * 32) &&
                    (squot32(local_tid_40680, 32) == 0 && slt32(local_tid_40680,
                                                                pts_per_node_at_lev_32225))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) mem_39870)[sext_i32_i64(local_tid_40680)] =
                            x_40690;
                        x_40694 = x_40690;
                        ((volatile __local
                          int32_t *) mem_39873)[sext_i32_i64(local_tid_40680)] =
                            x_40691;
                        x_40695 = x_40691;
                        ((volatile __local
                          int32_t *) mem_39876)[sext_i32_i64(local_tid_40680)] =
                            x_40692;
                        x_40696 = x_40692;
                        ((volatile __local
                          int32_t *) mem_39879)[sext_i32_i64(local_tid_40680)] =
                            x_40693;
                        x_40697 = x_40693;
                    }
                }
                if (sle32(wave_sizze_40682, skip_threads_40704)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_40704 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_40680, 32) == 0 || !slt32(local_tid_40680,
                                                          pts_per_node_at_lev_32225))) {
            // read operands
            {
                x_38080 = x_38076;
                x_38081 = x_38077;
                x_38082 = x_38078;
                x_38083 = x_38079;
                x_38076 = ((__local
                            int32_t *) mem_39870)[sext_i32_i64(squot32(local_tid_40680,
                                                                       32) -
                                                  1)];
                x_38077 = ((__local
                            int32_t *) mem_39873)[sext_i32_i64(squot32(local_tid_40680,
                                                                       32) -
                                                  1)];
                x_38078 = ((__local
                            int32_t *) mem_39876)[sext_i32_i64(squot32(local_tid_40680,
                                                                       32) -
                                                  1)];
                x_38079 = ((__local
                            int32_t *) mem_39879)[sext_i32_i64(squot32(local_tid_40680,
                                                                       32) -
                                                  1)];
            }
            // perform operation
            {
                bool inactive_40706 = slt32(srem32(local_tid_40680,
                                                   pts_per_node_at_lev_32225),
                                            local_tid_40680 -
                                            (squot32(local_tid_40680, 32) * 32 -
                                             1));
                
                if (inactive_40706) {
                    x_38076 = x_38080;
                    x_38077 = x_38081;
                    x_38078 = x_38082;
                    x_38079 = x_38083;
                }
                if (!inactive_40706) {
                    int32_t res_38084 = add32(x_38076, x_38080);
                    int32_t res_38085 = add32(x_38077, x_38081);
                    int32_t res_38086 = add32(x_38078, x_38082);
                    int32_t res_38087 = add32(x_38079, x_38083);
                    
                    x_38076 = res_38084;
                    x_38077 = res_38085;
                    x_38078 = res_38086;
                    x_38079 = res_38087;
                }
            }
            // write final result
            {
                ((__local int32_t *) mem_39870)[sext_i32_i64(local_tid_40680)] =
                    x_38076;
                ((__local int32_t *) mem_39873)[sext_i32_i64(local_tid_40680)] =
                    x_38077;
                ((__local int32_t *) mem_39876)[sext_i32_i64(local_tid_40680)] =
                    x_38078;
                ((__local int32_t *) mem_39879)[sext_i32_i64(local_tid_40680)] =
                    x_38079;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_40680, 32) == 0) {
            ((__local int32_t *) mem_39870)[sext_i32_i64(local_tid_40680)] =
                x_38080;
            ((__local int32_t *) mem_39873)[sext_i32_i64(local_tid_40680)] =
                x_38081;
            ((__local int32_t *) mem_39876)[sext_i32_i64(local_tid_40680)] =
                x_38082;
            ((__local int32_t *) mem_39879)[sext_i32_i64(local_tid_40680)] =
                x_38083;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t res_38121 = ((__local int32_t *) mem_39870)[sext_i32_i64(i_32324)];
    int32_t res_38122 = ((__local int32_t *) mem_39873)[sext_i32_i64(i_32324)];
    int32_t res_38123 = ((__local int32_t *) mem_39876)[sext_i32_i64(i_32324)];
    __local char *mem_39885;
    
    mem_39885 = (__local char *) mem_39885_backing_5;
    ((__local float *) mem_39885)[sext_i32_i64(local_tid_40680)] = ((__global
                                                                     float *) mem_39860)[sext_i32_i64(gtid_37386) +
                                                                                         sext_i32_i64(local_tid_40680) *
                                                                                         sext_i32_i64(nodes_this_lvl_32221)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_39888;
    
    mem_39888 = (__local char *) mem_39888_backing_6;
    ((__local int32_t *) mem_39888)[sext_i32_i64(local_tid_40680)] = ((__global
                                                                       int32_t *) mem_39865)[sext_i32_i64(gtid_37386) +
                                                                                             sext_i32_i64(local_tid_40680) *
                                                                                             sext_i32_i64(nodes_this_lvl_32221)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t write_i_37391 = ltid_pre_40684;
    int32_t phys_tid_37392 = local_tid_40680;
    
    if (slt32(write_i_37391, pts_per_node_at_lev_32225)) {
        int32_t x_38128 = ((__local
                            int32_t *) mem_39882)[sext_i32_i64(write_i_37391)];
        float write_value_38133 = ((__global
                                    float *) mem_param_39843)[sext_i32_i64(gtid_37386) *
                                                              sext_i32_i64(pts_per_node_at_lev_32225) +
                                                              sext_i32_i64(write_i_37391)];
        int32_t write_value_38134 = ((__global
                                      int32_t *) mem_param_39851)[sext_i32_i64(gtid_37386) *
                                                                  sext_i32_i64(pts_per_node_at_lev_32225) +
                                                                  sext_i32_i64(write_i_37391)];
        bool match_lit_38135 = 0 == x_38128;
        int32_t res_38136;
        
        if (match_lit_38135) {
            int32_t x_38129 = ((__local
                                int32_t *) mem_39870)[sext_i32_i64(write_i_37391)];
            int32_t res_38137 = sub32(x_38129, 1);
            
            res_38136 = res_38137;
        } else {
            int32_t x_38130 = ((__local
                                int32_t *) mem_39873)[sext_i32_i64(write_i_37391)];
            int32_t x_38131 = ((__local
                                int32_t *) mem_39876)[sext_i32_i64(write_i_37391)];
            int32_t x_38132 = ((__local
                                int32_t *) mem_39879)[sext_i32_i64(write_i_37391)];
            bool match_lit_38138 = 1 == x_38128;
            int32_t x_38139;
            
            if (match_lit_38138) {
                int32_t x_38140 = add32(res_38121, x_38130);
                int32_t res_38141 = sub32(x_38140, 1);
                
                x_38139 = res_38141;
            } else {
                bool match_lit_38142 = 2 == x_38128;
                int32_t x_38143;
                
                if (match_lit_38142) {
                    int32_t x_38144 = add32(res_38121, res_38122);
                    int32_t x_38145 = add32(x_38131, x_38144);
                    int32_t res_38146 = sub32(x_38145, 1);
                    
                    x_38143 = res_38146;
                } else {
                    int32_t x_38147 = add32(res_38121, res_38122);
                    int32_t x_38148 = add32(res_38123, x_38147);
                    int32_t x_38149 = add32(x_38132, x_38148);
                    int32_t res_38150 = sub32(x_38149, 1);
                    
                    x_38143 = res_38150;
                }
                x_38139 = x_38143;
            }
            res_38136 = x_38139;
        }
        if (sle32(0, res_38136) && slt32(res_38136,
                                         pts_per_node_at_lev_32225)) {
            ((__local float *) mem_39885)[sext_i32_i64(res_38136)] =
                write_value_38133;
        }
        if (sle32(0, res_38136) && slt32(res_38136,
                                         pts_per_node_at_lev_32225)) {
            ((__local int32_t *) mem_39888)[sext_i32_i64(res_38136)] =
                write_value_38134;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global float *) mem_39894)[sext_i32_i64(gtid_37386) *
                                   sext_i32_i64(pts_per_node_at_lev_32225) +
                                   sext_i32_i64(local_tid_40680)] = ((__local
                                                                      float *) mem_39885)[sext_i32_i64(local_tid_40680)];
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global int32_t *) mem_39899)[sext_i32_i64(gtid_37386) *
                                     sext_i32_i64(pts_per_node_at_lev_32225) +
                                     sext_i32_i64(local_tid_40680)] = ((__local
                                                                        int32_t *) mem_39888)[sext_i32_i64(local_tid_40680)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_2:
    return;
}
__kernel void buildKDtreezisegred_large_36352(__global int *global_failure,
                                              uint sync_arr_mem_40357_backing_offset_0,
                                              uint red_arr_mem_40355_backing_offset_1,
                                              int32_t m_32135,
                                              int32_t num_groups_36363, __global
                                              unsigned char *mem_39427, __global
                                              unsigned char *mem_39431,
                                              int32_t groups_per_segment_40341,
                                              int32_t elements_per_thread_40342,
                                              int32_t virt_num_groups_40343,
                                              int32_t threads_per_segment_40345,
                                              __global
                                              unsigned char *group_res_arr_mem_40346,
                                              __global
                                              unsigned char *buildKDtreezicounter_mem_40348)
{
    #define segred_group_sizze_36362 (buildKDtreezisegred_group_sizze_36346)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *sync_arr_mem_40357_backing_1 =
                  &shared_mem[sync_arr_mem_40357_backing_offset_0];
    volatile char *red_arr_mem_40355_backing_0 =
                  &shared_mem[red_arr_mem_40355_backing_offset_1];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40350;
    int32_t local_tid_40351;
    int32_t group_sizze_40354;
    int32_t wave_sizze_40353;
    int32_t group_tid_40352;
    
    global_tid_40350 = get_global_id(0);
    local_tid_40351 = get_local_id(0);
    group_sizze_40354 = get_local_size(0);
    wave_sizze_40353 = LOCKSTEP_WIDTH;
    group_tid_40352 = get_group_id(0);
    
    int32_t phys_tid_36352;
    
    phys_tid_36352 = global_tid_40350;
    
    __local char *red_arr_mem_40355;
    
    red_arr_mem_40355 = (__local char *) red_arr_mem_40355_backing_0;
    
    __local char *sync_arr_mem_40357;
    
    sync_arr_mem_40357 = (__local char *) sync_arr_mem_40357_backing_1;
    
    int32_t phys_group_id_40359;
    
    phys_group_id_40359 = get_group_id(0);
    for (int32_t i_40360 = 0; i_40360 < sdiv_up32(virt_num_groups_40343 -
                                                  phys_group_id_40359,
                                                  num_groups_36363);
         i_40360++) {
        int32_t virt_group_id_40361 = phys_group_id_40359 + i_40360 *
                num_groups_36363;
        int32_t flat_segment_id_40362 = squot32(virt_group_id_40361,
                                                groups_per_segment_40341);
        int32_t global_tid_40363 = srem32(virt_group_id_40361 *
                                          segred_group_sizze_36362 +
                                          local_tid_40351,
                                          segred_group_sizze_36362 *
                                          groups_per_segment_40341);
        int32_t gtid_36341 = flat_segment_id_40362;
        int32_t gtid_36351;
        float x_acc_40364;
        int32_t chunk_sizze_40365;
        
        chunk_sizze_40365 = smin32(elements_per_thread_40342,
                                   sdiv_up32(m_32135 - global_tid_40363,
                                             threads_per_segment_40345));
        
        float x_36366;
        float x_36367;
        
        // neutral-initialise the accumulators
        {
            x_acc_40364 = INFINITY;
        }
        for (int32_t i_40369 = 0; i_40369 < chunk_sizze_40365; i_40369++) {
            gtid_36351 = global_tid_40363 + threads_per_segment_40345 * i_40369;
            // apply map function
            {
                float x_36370 = ((__global
                                  float *) mem_39427)[sext_i32_i64(gtid_36341) *
                                                      sext_i32_i64(m_32135) +
                                                      sext_i32_i64(gtid_36351)];
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_36366 = x_acc_40364;
                }
                // load new values
                {
                    x_36367 = x_36370;
                }
                // apply reduction operator
                {
                    float res_36368 = fmin32(x_36366, x_36367);
                    
                    // store in accumulator
                    {
                        x_acc_40364 = res_36368;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_36366 = x_acc_40364;
            ((__local
              float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351)] =
                x_36366;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_40370;
        int32_t skip_waves_40371;
        float x_40366;
        float x_40367;
        
        offset_40370 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_40351, segred_group_sizze_36362)) {
                x_40366 = ((__local
                            float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351 +
                                                        offset_40370)];
            }
        }
        offset_40370 = 1;
        while (slt32(offset_40370, wave_sizze_40353)) {
            if (slt32(local_tid_40351 + offset_40370,
                      segred_group_sizze_36362) && ((local_tid_40351 -
                                                     squot32(local_tid_40351,
                                                             wave_sizze_40353) *
                                                     wave_sizze_40353) & (2 *
                                                                          offset_40370 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_40367 = ((volatile __local
                                float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351 +
                                                            offset_40370)];
                }
                // apply reduction operation
                {
                    float res_40368 = fmin32(x_40366, x_40367);
                    
                    x_40366 = res_40368;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351)] =
                        x_40366;
                }
            }
            offset_40370 *= 2;
        }
        skip_waves_40371 = 1;
        while (slt32(skip_waves_40371, squot32(segred_group_sizze_36362 +
                                               wave_sizze_40353 - 1,
                                               wave_sizze_40353))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_40370 = skip_waves_40371 * wave_sizze_40353;
            if (slt32(local_tid_40351 + offset_40370,
                      segred_group_sizze_36362) && ((local_tid_40351 -
                                                     squot32(local_tid_40351,
                                                             wave_sizze_40353) *
                                                     wave_sizze_40353) == 0 &&
                                                    (squot32(local_tid_40351,
                                                             wave_sizze_40353) &
                                                     (2 * skip_waves_40371 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_40367 = ((__local
                                float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351 +
                                                            offset_40370)];
                }
                // apply reduction operation
                {
                    float res_40368 = fmin32(x_40366, x_40367);
                    
                    x_40366 = res_40368;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351)] =
                        x_40366;
                }
            }
            skip_waves_40371 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_40351 == 0) {
                x_acc_40364 = x_40366;
            }
        }
        if (groups_per_segment_40341 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_40351 == 0) {
                    ((__global float *) mem_39431)[sext_i32_i64(gtid_36341)] =
                        x_acc_40364;
                }
            }
        } else {
            int32_t old_counter_40372;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_40351 == 0) {
                    ((__global
                      float *) group_res_arr_mem_40346)[sext_i32_i64(virt_group_id_40361) *
                                                        sext_i32_i64(segred_group_sizze_36362)] =
                        x_acc_40364;
                    mem_fence_global();
                    old_counter_40372 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) buildKDtreezicounter_mem_40348)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_40362,
                                                                                                                         10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_40357)[0] =
                        old_counter_40372 == groups_per_segment_40341 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_40373;
            
            is_last_group_40373 = ((__local bool *) sync_arr_mem_40357)[0];
            if (is_last_group_40373) {
                if (local_tid_40351 == 0) {
                    old_counter_40372 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) buildKDtreezicounter_mem_40348)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_40362,
                                                                                                                         10240)))],
                                              (int) (0 -
                                                     groups_per_segment_40341));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_40374 =
                            sdiv_up32(groups_per_segment_40341,
                                      segred_group_sizze_36362);
                    
                    x_36366 = INFINITY;
                    for (int32_t i_40375 = 0; i_40375 < read_per_thread_40374;
                         i_40375++) {
                        int32_t group_res_id_40376 = local_tid_40351 *
                                read_per_thread_40374 + i_40375;
                        int32_t index_of_group_res_40377 =
                                flat_segment_id_40362 *
                                groups_per_segment_40341 + group_res_id_40376;
                        
                        if (slt32(group_res_id_40376,
                                  groups_per_segment_40341)) {
                            x_36367 = ((__global
                                        float *) group_res_arr_mem_40346)[sext_i32_i64(index_of_group_res_40377) *
                                                                          sext_i32_i64(segred_group_sizze_36362)];
                            
                            float res_36368;
                            
                            res_36368 = fmin32(x_36366, x_36367);
                            x_36366 = res_36368;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351)] =
                    x_36366;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_40378;
                    int32_t skip_waves_40379;
                    float x_40366;
                    float x_40367;
                    
                    offset_40378 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_40351, segred_group_sizze_36362)) {
                            x_40366 = ((__local
                                        float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351 +
                                                                    offset_40378)];
                        }
                    }
                    offset_40378 = 1;
                    while (slt32(offset_40378, wave_sizze_40353)) {
                        if (slt32(local_tid_40351 + offset_40378,
                                  segred_group_sizze_36362) &&
                            ((local_tid_40351 - squot32(local_tid_40351,
                                                        wave_sizze_40353) *
                              wave_sizze_40353) & (2 * offset_40378 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_40367 = ((volatile __local
                                            float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351 +
                                                                        offset_40378)];
                            }
                            // apply reduction operation
                            {
                                float res_40368 = fmin32(x_40366, x_40367);
                                
                                x_40366 = res_40368;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351)] =
                                    x_40366;
                            }
                        }
                        offset_40378 *= 2;
                    }
                    skip_waves_40379 = 1;
                    while (slt32(skip_waves_40379,
                                 squot32(segred_group_sizze_36362 +
                                         wave_sizze_40353 - 1,
                                         wave_sizze_40353))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_40378 = skip_waves_40379 * wave_sizze_40353;
                        if (slt32(local_tid_40351 + offset_40378,
                                  segred_group_sizze_36362) &&
                            ((local_tid_40351 - squot32(local_tid_40351,
                                                        wave_sizze_40353) *
                              wave_sizze_40353) == 0 &&
                             (squot32(local_tid_40351, wave_sizze_40353) & (2 *
                                                                            skip_waves_40379 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_40367 = ((__local
                                            float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351 +
                                                                        offset_40378)];
                            }
                            // apply reduction operation
                            {
                                float res_40368 = fmin32(x_40366, x_40367);
                                
                                x_40366 = res_40368;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_40355)[sext_i32_i64(local_tid_40351)] =
                                    x_40366;
                            }
                        }
                        skip_waves_40379 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_40351 == 0) {
                            ((__global
                              float *) mem_39431)[sext_i32_i64(gtid_36341)] =
                                x_40366;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_36362
}
__kernel void buildKDtreezisegred_large_36413(__global int *global_failure,
                                              uint sync_arr_mem_40424_backing_offset_0,
                                              uint red_arr_mem_40422_backing_offset_1,
                                              int32_t m_32135,
                                              int32_t num_groups_36424, __global
                                              unsigned char *mem_39441, __global
                                              unsigned char *mem_39445,
                                              int32_t groups_per_segment_40408,
                                              int32_t elements_per_thread_40409,
                                              int32_t virt_num_groups_40410,
                                              int32_t threads_per_segment_40412,
                                              __global
                                              unsigned char *group_res_arr_mem_40413,
                                              __global
                                              unsigned char *buildKDtreezicounter_mem_40415)
{
    #define segred_group_sizze_36423 (buildKDtreezisegred_group_sizze_36407)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *sync_arr_mem_40424_backing_1 =
                  &shared_mem[sync_arr_mem_40424_backing_offset_0];
    volatile char *red_arr_mem_40422_backing_0 =
                  &shared_mem[red_arr_mem_40422_backing_offset_1];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40417;
    int32_t local_tid_40418;
    int32_t group_sizze_40421;
    int32_t wave_sizze_40420;
    int32_t group_tid_40419;
    
    global_tid_40417 = get_global_id(0);
    local_tid_40418 = get_local_id(0);
    group_sizze_40421 = get_local_size(0);
    wave_sizze_40420 = LOCKSTEP_WIDTH;
    group_tid_40419 = get_group_id(0);
    
    int32_t phys_tid_36413;
    
    phys_tid_36413 = global_tid_40417;
    
    __local char *red_arr_mem_40422;
    
    red_arr_mem_40422 = (__local char *) red_arr_mem_40422_backing_0;
    
    __local char *sync_arr_mem_40424;
    
    sync_arr_mem_40424 = (__local char *) sync_arr_mem_40424_backing_1;
    
    int32_t phys_group_id_40426;
    
    phys_group_id_40426 = get_group_id(0);
    for (int32_t i_40427 = 0; i_40427 < sdiv_up32(virt_num_groups_40410 -
                                                  phys_group_id_40426,
                                                  num_groups_36424);
         i_40427++) {
        int32_t virt_group_id_40428 = phys_group_id_40426 + i_40427 *
                num_groups_36424;
        int32_t flat_segment_id_40429 = squot32(virt_group_id_40428,
                                                groups_per_segment_40408);
        int32_t global_tid_40430 = srem32(virt_group_id_40428 *
                                          segred_group_sizze_36423 +
                                          local_tid_40418,
                                          segred_group_sizze_36423 *
                                          groups_per_segment_40408);
        int32_t gtid_36402 = flat_segment_id_40429;
        int32_t gtid_36412;
        float x_acc_40431;
        int32_t chunk_sizze_40432;
        
        chunk_sizze_40432 = smin32(elements_per_thread_40409,
                                   sdiv_up32(m_32135 - global_tid_40430,
                                             threads_per_segment_40412));
        
        float x_36427;
        float x_36428;
        
        // neutral-initialise the accumulators
        {
            x_acc_40431 = -INFINITY;
        }
        for (int32_t i_40436 = 0; i_40436 < chunk_sizze_40432; i_40436++) {
            gtid_36412 = global_tid_40430 + threads_per_segment_40412 * i_40436;
            // apply map function
            {
                float x_36431 = ((__global
                                  float *) mem_39441)[sext_i32_i64(gtid_36402) *
                                                      sext_i32_i64(m_32135) +
                                                      sext_i32_i64(gtid_36412)];
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_36427 = x_acc_40431;
                }
                // load new values
                {
                    x_36428 = x_36431;
                }
                // apply reduction operator
                {
                    float res_36429 = fmax32(x_36427, x_36428);
                    
                    // store in accumulator
                    {
                        x_acc_40431 = res_36429;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_36427 = x_acc_40431;
            ((__local
              float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418)] =
                x_36427;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_40437;
        int32_t skip_waves_40438;
        float x_40433;
        float x_40434;
        
        offset_40437 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_40418, segred_group_sizze_36423)) {
                x_40433 = ((__local
                            float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418 +
                                                        offset_40437)];
            }
        }
        offset_40437 = 1;
        while (slt32(offset_40437, wave_sizze_40420)) {
            if (slt32(local_tid_40418 + offset_40437,
                      segred_group_sizze_36423) && ((local_tid_40418 -
                                                     squot32(local_tid_40418,
                                                             wave_sizze_40420) *
                                                     wave_sizze_40420) & (2 *
                                                                          offset_40437 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_40434 = ((volatile __local
                                float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418 +
                                                            offset_40437)];
                }
                // apply reduction operation
                {
                    float res_40435 = fmax32(x_40433, x_40434);
                    
                    x_40433 = res_40435;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418)] =
                        x_40433;
                }
            }
            offset_40437 *= 2;
        }
        skip_waves_40438 = 1;
        while (slt32(skip_waves_40438, squot32(segred_group_sizze_36423 +
                                               wave_sizze_40420 - 1,
                                               wave_sizze_40420))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_40437 = skip_waves_40438 * wave_sizze_40420;
            if (slt32(local_tid_40418 + offset_40437,
                      segred_group_sizze_36423) && ((local_tid_40418 -
                                                     squot32(local_tid_40418,
                                                             wave_sizze_40420) *
                                                     wave_sizze_40420) == 0 &&
                                                    (squot32(local_tid_40418,
                                                             wave_sizze_40420) &
                                                     (2 * skip_waves_40438 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_40434 = ((__local
                                float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418 +
                                                            offset_40437)];
                }
                // apply reduction operation
                {
                    float res_40435 = fmax32(x_40433, x_40434);
                    
                    x_40433 = res_40435;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418)] =
                        x_40433;
                }
            }
            skip_waves_40438 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_40418 == 0) {
                x_acc_40431 = x_40433;
            }
        }
        if (groups_per_segment_40408 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_40418 == 0) {
                    ((__global float *) mem_39445)[sext_i32_i64(gtid_36402)] =
                        x_acc_40431;
                }
            }
        } else {
            int32_t old_counter_40439;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_40418 == 0) {
                    ((__global
                      float *) group_res_arr_mem_40413)[sext_i32_i64(virt_group_id_40428) *
                                                        sext_i32_i64(segred_group_sizze_36423)] =
                        x_acc_40431;
                    mem_fence_global();
                    old_counter_40439 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) buildKDtreezicounter_mem_40415)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_40429,
                                                                                                                         10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_40424)[0] =
                        old_counter_40439 == groups_per_segment_40408 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_40440;
            
            is_last_group_40440 = ((__local bool *) sync_arr_mem_40424)[0];
            if (is_last_group_40440) {
                if (local_tid_40418 == 0) {
                    old_counter_40439 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) buildKDtreezicounter_mem_40415)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_40429,
                                                                                                                         10240)))],
                                              (int) (0 -
                                                     groups_per_segment_40408));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_40441 =
                            sdiv_up32(groups_per_segment_40408,
                                      segred_group_sizze_36423);
                    
                    x_36427 = -INFINITY;
                    for (int32_t i_40442 = 0; i_40442 < read_per_thread_40441;
                         i_40442++) {
                        int32_t group_res_id_40443 = local_tid_40418 *
                                read_per_thread_40441 + i_40442;
                        int32_t index_of_group_res_40444 =
                                flat_segment_id_40429 *
                                groups_per_segment_40408 + group_res_id_40443;
                        
                        if (slt32(group_res_id_40443,
                                  groups_per_segment_40408)) {
                            x_36428 = ((__global
                                        float *) group_res_arr_mem_40413)[sext_i32_i64(index_of_group_res_40444) *
                                                                          sext_i32_i64(segred_group_sizze_36423)];
                            
                            float res_36429;
                            
                            res_36429 = fmax32(x_36427, x_36428);
                            x_36427 = res_36429;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418)] =
                    x_36427;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_40445;
                    int32_t skip_waves_40446;
                    float x_40433;
                    float x_40434;
                    
                    offset_40445 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_40418, segred_group_sizze_36423)) {
                            x_40433 = ((__local
                                        float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418 +
                                                                    offset_40445)];
                        }
                    }
                    offset_40445 = 1;
                    while (slt32(offset_40445, wave_sizze_40420)) {
                        if (slt32(local_tid_40418 + offset_40445,
                                  segred_group_sizze_36423) &&
                            ((local_tid_40418 - squot32(local_tid_40418,
                                                        wave_sizze_40420) *
                              wave_sizze_40420) & (2 * offset_40445 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_40434 = ((volatile __local
                                            float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418 +
                                                                        offset_40445)];
                            }
                            // apply reduction operation
                            {
                                float res_40435 = fmax32(x_40433, x_40434);
                                
                                x_40433 = res_40435;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418)] =
                                    x_40433;
                            }
                        }
                        offset_40445 *= 2;
                    }
                    skip_waves_40446 = 1;
                    while (slt32(skip_waves_40446,
                                 squot32(segred_group_sizze_36423 +
                                         wave_sizze_40420 - 1,
                                         wave_sizze_40420))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_40445 = skip_waves_40446 * wave_sizze_40420;
                        if (slt32(local_tid_40418 + offset_40445,
                                  segred_group_sizze_36423) &&
                            ((local_tid_40418 - squot32(local_tid_40418,
                                                        wave_sizze_40420) *
                              wave_sizze_40420) == 0 &&
                             (squot32(local_tid_40418, wave_sizze_40420) & (2 *
                                                                            skip_waves_40446 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_40434 = ((__local
                                            float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418 +
                                                                        offset_40445)];
                            }
                            // apply reduction operation
                            {
                                float res_40435 = fmax32(x_40433, x_40434);
                                
                                x_40433 = res_40435;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_40422)[sext_i32_i64(local_tid_40418)] =
                                    x_40433;
                            }
                        }
                        skip_waves_40446 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_40418 == 0) {
                            ((__global
                              float *) mem_39445)[sext_i32_i64(gtid_36402)] =
                                x_40433;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_36423
}
__kernel void buildKDtreezisegred_large_36675(__global int *global_failure,
                                              int failure_is_an_option, __global
                                              int *global_failure_args,
                                              uint sync_arr_mem_40555_backing_offset_0,
                                              uint red_arr_mem_40553_backing_offset_1,
                                              uint red_arr_mem_40551_backing_offset_2,
                                              int32_t d_32136,
                                              int32_t conc_tmp_32196,
                                              int32_t nodes_this_lvl_32221,
                                              int32_t num_groups_36848, __global
                                              unsigned char *mem_39532, __global
                                              unsigned char *mem_39540, __global
                                              unsigned char *mem_39544, __global
                                              unsigned char *mem_39547,
                                              int32_t groups_per_segment_40535,
                                              int32_t elements_per_thread_40536,
                                              int32_t virt_num_groups_40537,
                                              int32_t threads_per_segment_40539,
                                              __global
                                              unsigned char *group_res_arr_mem_40540,
                                              __global
                                              unsigned char *group_res_arr_mem_40542,
                                              __global
                                              unsigned char *buildKDtreezicounter_mem_40544)
{
    #define segred_group_sizze_36847 (buildKDtreezisegred_group_sizze_36669)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *sync_arr_mem_40555_backing_2 =
                  &shared_mem[sync_arr_mem_40555_backing_offset_0];
    volatile char *red_arr_mem_40553_backing_1 =
                  &shared_mem[red_arr_mem_40553_backing_offset_1];
    volatile char *red_arr_mem_40551_backing_0 =
                  &shared_mem[red_arr_mem_40551_backing_offset_2];
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40546;
    int32_t local_tid_40547;
    int32_t group_sizze_40550;
    int32_t wave_sizze_40549;
    int32_t group_tid_40548;
    
    global_tid_40546 = get_global_id(0);
    local_tid_40547 = get_local_id(0);
    group_sizze_40550 = get_local_size(0);
    wave_sizze_40549 = LOCKSTEP_WIDTH;
    group_tid_40548 = get_group_id(0);
    
    int32_t phys_tid_36675;
    
    phys_tid_36675 = global_tid_40546;
    
    __local char *red_arr_mem_40551;
    
    red_arr_mem_40551 = (__local char *) red_arr_mem_40551_backing_0;
    
    __local char *red_arr_mem_40553;
    
    red_arr_mem_40553 = (__local char *) red_arr_mem_40553_backing_1;
    
    __local char *sync_arr_mem_40555;
    
    sync_arr_mem_40555 = (__local char *) sync_arr_mem_40555_backing_2;
    
    int32_t phys_group_id_40557;
    
    phys_group_id_40557 = get_group_id(0);
    for (int32_t i_40558 = 0; i_40558 < sdiv_up32(virt_num_groups_40537 -
                                                  phys_group_id_40557,
                                                  num_groups_36848);
         i_40558++) {
        int32_t virt_group_id_40559 = phys_group_id_40557 + i_40558 *
                num_groups_36848;
        int32_t flat_segment_id_40560 = squot32(virt_group_id_40559,
                                                groups_per_segment_40535);
        int32_t global_tid_40561 = srem32(virt_group_id_40559 *
                                          segred_group_sizze_36847 +
                                          local_tid_40547,
                                          segred_group_sizze_36847 *
                                          groups_per_segment_40535);
        int32_t gtid_36664 = flat_segment_id_40560;
        int32_t gtid_36674;
        int32_t x_acc_40562;
        float x_acc_40563;
        int32_t chunk_sizze_40564;
        
        chunk_sizze_40564 = smin32(elements_per_thread_40536,
                                   sdiv_up32(d_32136 - global_tid_40561,
                                             threads_per_segment_40539));
        
        int32_t x_36852;
        float x_36853;
        int32_t x_36854;
        float x_36855;
        
        // neutral-initialise the accumulators
        {
            x_acc_40562 = -1;
            x_acc_40563 = -INFINITY;
        }
        for (int32_t i_40572 = 0; i_40572 < chunk_sizze_40564; i_40572++) {
            gtid_36674 = global_tid_40561 + threads_per_segment_40539 * i_40572;
            // apply map function
            {
                int32_t i_36861 = add32(d_32136, gtid_36674);
                bool x_36862 = sle32(0, i_36861);
                bool y_36863 = slt32(i_36861, conc_tmp_32196);
                bool bounds_check_36864 = x_36862 && y_36863;
                bool index_certs_36865;
                
                if (!bounds_check_36864) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 13) ==
                            -1) {
                            global_failure_args[0] = i_36861;
                            global_failure_args[1] = conc_tmp_32196;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float x_36866 = ((__global
                                  float *) mem_39540)[sext_i32_i64(gtid_36664) *
                                                      sext_i32_i64(conc_tmp_32196) +
                                                      sext_i32_i64(i_36861)];
                bool y_36868 = slt32(gtid_36674, conc_tmp_32196);
                bool index_certs_36870;
                
                if (!y_36868) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 14) ==
                            -1) {
                            global_failure_args[0] = gtid_36674;
                            global_failure_args[1] = conc_tmp_32196;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float y_36871 = ((__global
                                  float *) mem_39532)[sext_i32_i64(gtid_36674) *
                                                      sext_i32_i64(nodes_this_lvl_32221) +
                                                      sext_i32_i64(gtid_36664)];
                float abs_arg_36872 = x_36866 - y_36871;
                float res_36873 = (float) fabs(abs_arg_36872);
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_36852 = x_acc_40562;
                    x_36853 = x_acc_40563;
                }
                // load new values
                {
                    x_36854 = gtid_36674;
                    x_36855 = res_36873;
                }
                // apply reduction operator
                {
                    bool cond_36856 = x_36855 <= x_36853;
                    int32_t res_36857;
                    
                    if (cond_36856) {
                        res_36857 = x_36852;
                    } else {
                        res_36857 = x_36854;
                    }
                    
                    float res_36858;
                    
                    if (cond_36856) {
                        res_36858 = x_36853;
                    } else {
                        res_36858 = x_36855;
                    }
                    // store in accumulator
                    {
                        x_acc_40562 = res_36857;
                        x_acc_40563 = res_36858;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_36852 = x_acc_40562;
            x_36853 = x_acc_40563;
            ((__local
              int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547)] =
                x_36852;
            ((__local
              float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547)] =
                x_36853;
        }
        
      error_0:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_40573;
        int32_t skip_waves_40574;
        int32_t x_40565;
        float x_40566;
        int32_t x_40567;
        float x_40568;
        
        offset_40573 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_40547, segred_group_sizze_36847)) {
                x_40565 = ((__local
                            int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547 +
                                                          offset_40573)];
                x_40566 = ((__local
                            float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547 +
                                                        offset_40573)];
            }
        }
        offset_40573 = 1;
        while (slt32(offset_40573, wave_sizze_40549)) {
            if (slt32(local_tid_40547 + offset_40573,
                      segred_group_sizze_36847) && ((local_tid_40547 -
                                                     squot32(local_tid_40547,
                                                             wave_sizze_40549) *
                                                     wave_sizze_40549) & (2 *
                                                                          offset_40573 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_40567 = ((volatile __local
                                int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547 +
                                                              offset_40573)];
                    x_40568 = ((volatile __local
                                float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547 +
                                                            offset_40573)];
                }
                // apply reduction operation
                {
                    bool cond_40569 = x_40568 <= x_40566;
                    int32_t res_40570;
                    
                    if (cond_40569) {
                        res_40570 = x_40565;
                    } else {
                        res_40570 = x_40567;
                    }
                    
                    float res_40571;
                    
                    if (cond_40569) {
                        res_40571 = x_40566;
                    } else {
                        res_40571 = x_40568;
                    }
                    x_40565 = res_40570;
                    x_40566 = res_40571;
                }
                // write result of operation
                {
                    ((volatile __local
                      int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547)] =
                        x_40565;
                    ((volatile __local
                      float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547)] =
                        x_40566;
                }
            }
            offset_40573 *= 2;
        }
        skip_waves_40574 = 1;
        while (slt32(skip_waves_40574, squot32(segred_group_sizze_36847 +
                                               wave_sizze_40549 - 1,
                                               wave_sizze_40549))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_40573 = skip_waves_40574 * wave_sizze_40549;
            if (slt32(local_tid_40547 + offset_40573,
                      segred_group_sizze_36847) && ((local_tid_40547 -
                                                     squot32(local_tid_40547,
                                                             wave_sizze_40549) *
                                                     wave_sizze_40549) == 0 &&
                                                    (squot32(local_tid_40547,
                                                             wave_sizze_40549) &
                                                     (2 * skip_waves_40574 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_40567 = ((__local
                                int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547 +
                                                              offset_40573)];
                    x_40568 = ((__local
                                float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547 +
                                                            offset_40573)];
                }
                // apply reduction operation
                {
                    bool cond_40569 = x_40568 <= x_40566;
                    int32_t res_40570;
                    
                    if (cond_40569) {
                        res_40570 = x_40565;
                    } else {
                        res_40570 = x_40567;
                    }
                    
                    float res_40571;
                    
                    if (cond_40569) {
                        res_40571 = x_40566;
                    } else {
                        res_40571 = x_40568;
                    }
                    x_40565 = res_40570;
                    x_40566 = res_40571;
                }
                // write result of operation
                {
                    ((__local
                      int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547)] =
                        x_40565;
                    ((__local
                      float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547)] =
                        x_40566;
                }
            }
            skip_waves_40574 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_40547 == 0) {
                x_acc_40562 = x_40565;
                x_acc_40563 = x_40566;
            }
        }
        if (groups_per_segment_40535 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_40547 == 0) {
                    ((__global int32_t *) mem_39544)[sext_i32_i64(gtid_36664)] =
                        x_acc_40562;
                    ((__global float *) mem_39547)[sext_i32_i64(gtid_36664)] =
                        x_acc_40563;
                }
            }
        } else {
            int32_t old_counter_40575;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_40547 == 0) {
                    ((__global
                      int32_t *) group_res_arr_mem_40540)[sext_i32_i64(virt_group_id_40559) *
                                                          sext_i32_i64(segred_group_sizze_36847)] =
                        x_acc_40562;
                    ((__global
                      float *) group_res_arr_mem_40542)[sext_i32_i64(virt_group_id_40559) *
                                                        sext_i32_i64(segred_group_sizze_36847)] =
                        x_acc_40563;
                    mem_fence_global();
                    old_counter_40575 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) buildKDtreezicounter_mem_40544)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_40560,
                                                                                                                         10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_40555)[0] =
                        old_counter_40575 == groups_per_segment_40535 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_40576;
            
            is_last_group_40576 = ((__local bool *) sync_arr_mem_40555)[0];
            if (is_last_group_40576) {
                if (local_tid_40547 == 0) {
                    old_counter_40575 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) buildKDtreezicounter_mem_40544)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_40560,
                                                                                                                         10240)))],
                                              (int) (0 -
                                                     groups_per_segment_40535));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_40577 =
                            sdiv_up32(groups_per_segment_40535,
                                      segred_group_sizze_36847);
                    
                    x_36852 = -1;
                    x_36853 = -INFINITY;
                    for (int32_t i_40578 = 0; i_40578 < read_per_thread_40577;
                         i_40578++) {
                        int32_t group_res_id_40579 = local_tid_40547 *
                                read_per_thread_40577 + i_40578;
                        int32_t index_of_group_res_40580 =
                                flat_segment_id_40560 *
                                groups_per_segment_40535 + group_res_id_40579;
                        
                        if (slt32(group_res_id_40579,
                                  groups_per_segment_40535)) {
                            x_36854 = ((__global
                                        int32_t *) group_res_arr_mem_40540)[sext_i32_i64(index_of_group_res_40580) *
                                                                            sext_i32_i64(segred_group_sizze_36847)];
                            x_36855 = ((__global
                                        float *) group_res_arr_mem_40542)[sext_i32_i64(index_of_group_res_40580) *
                                                                          sext_i32_i64(segred_group_sizze_36847)];
                            
                            bool cond_36856;
                            
                            cond_36856 = x_36855 <= x_36853;
                            
                            int32_t res_36857;
                            
                            if (cond_36856) {
                                res_36857 = x_36852;
                            } else {
                                res_36857 = x_36854;
                            }
                            
                            float res_36858;
                            
                            if (cond_36856) {
                                res_36858 = x_36853;
                            } else {
                                res_36858 = x_36855;
                            }
                            x_36852 = res_36857;
                            x_36853 = res_36858;
                        }
                    }
                }
                ((__local
                  int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547)] =
                    x_36852;
                ((__local
                  float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547)] =
                    x_36853;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_40581;
                    int32_t skip_waves_40582;
                    int32_t x_40565;
                    float x_40566;
                    int32_t x_40567;
                    float x_40568;
                    
                    offset_40581 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_40547, segred_group_sizze_36847)) {
                            x_40565 = ((__local
                                        int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547 +
                                                                      offset_40581)];
                            x_40566 = ((__local
                                        float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547 +
                                                                    offset_40581)];
                        }
                    }
                    offset_40581 = 1;
                    while (slt32(offset_40581, wave_sizze_40549)) {
                        if (slt32(local_tid_40547 + offset_40581,
                                  segred_group_sizze_36847) &&
                            ((local_tid_40547 - squot32(local_tid_40547,
                                                        wave_sizze_40549) *
                              wave_sizze_40549) & (2 * offset_40581 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_40567 = ((volatile __local
                                            int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547 +
                                                                          offset_40581)];
                                x_40568 = ((volatile __local
                                            float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547 +
                                                                        offset_40581)];
                            }
                            // apply reduction operation
                            {
                                bool cond_40569 = x_40568 <= x_40566;
                                int32_t res_40570;
                                
                                if (cond_40569) {
                                    res_40570 = x_40565;
                                } else {
                                    res_40570 = x_40567;
                                }
                                
                                float res_40571;
                                
                                if (cond_40569) {
                                    res_40571 = x_40566;
                                } else {
                                    res_40571 = x_40568;
                                }
                                x_40565 = res_40570;
                                x_40566 = res_40571;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547)] =
                                    x_40565;
                                ((volatile __local
                                  float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547)] =
                                    x_40566;
                            }
                        }
                        offset_40581 *= 2;
                    }
                    skip_waves_40582 = 1;
                    while (slt32(skip_waves_40582,
                                 squot32(segred_group_sizze_36847 +
                                         wave_sizze_40549 - 1,
                                         wave_sizze_40549))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_40581 = skip_waves_40582 * wave_sizze_40549;
                        if (slt32(local_tid_40547 + offset_40581,
                                  segred_group_sizze_36847) &&
                            ((local_tid_40547 - squot32(local_tid_40547,
                                                        wave_sizze_40549) *
                              wave_sizze_40549) == 0 &&
                             (squot32(local_tid_40547, wave_sizze_40549) & (2 *
                                                                            skip_waves_40582 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_40567 = ((__local
                                            int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547 +
                                                                          offset_40581)];
                                x_40568 = ((__local
                                            float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547 +
                                                                        offset_40581)];
                            }
                            // apply reduction operation
                            {
                                bool cond_40569 = x_40568 <= x_40566;
                                int32_t res_40570;
                                
                                if (cond_40569) {
                                    res_40570 = x_40565;
                                } else {
                                    res_40570 = x_40567;
                                }
                                
                                float res_40571;
                                
                                if (cond_40569) {
                                    res_40571 = x_40566;
                                } else {
                                    res_40571 = x_40568;
                                }
                                x_40565 = res_40570;
                                x_40566 = res_40571;
                            }
                            // write result of operation
                            {
                                ((__local
                                  int32_t *) red_arr_mem_40551)[sext_i32_i64(local_tid_40547)] =
                                    x_40565;
                                ((__local
                                  float *) red_arr_mem_40553)[sext_i32_i64(local_tid_40547)] =
                                    x_40566;
                            }
                        }
                        skip_waves_40582 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_40547 == 0) {
                            ((__global
                              int32_t *) mem_39544)[sext_i32_i64(gtid_36664)] =
                                x_40565;
                            ((__global
                              float *) mem_39547)[sext_i32_i64(gtid_36664)] =
                                x_40566;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_36847
}
__kernel void buildKDtreezisegred_small_36352(__global int *global_failure,
                                              uint red_arr_mem_40328_backing_offset_0,
                                              int32_t m_32135, int32_t d_32136,
                                              int32_t num_groups_36363, __global
                                              unsigned char *mem_39427, __global
                                              unsigned char *mem_39431,
                                              int32_t segment_sizze_nonzzero_40321)
{
    #define segred_group_sizze_36362 (buildKDtreezisegred_group_sizze_36346)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40328_backing_0 =
                  &shared_mem[red_arr_mem_40328_backing_offset_0];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40323;
    int32_t local_tid_40324;
    int32_t group_sizze_40327;
    int32_t wave_sizze_40326;
    int32_t group_tid_40325;
    
    global_tid_40323 = get_global_id(0);
    local_tid_40324 = get_local_id(0);
    group_sizze_40327 = get_local_size(0);
    wave_sizze_40326 = LOCKSTEP_WIDTH;
    group_tid_40325 = get_group_id(0);
    
    int32_t phys_tid_36352;
    
    phys_tid_36352 = global_tid_40323;
    
    __local char *red_arr_mem_40328;
    
    red_arr_mem_40328 = (__local char *) red_arr_mem_40328_backing_0;
    
    int32_t phys_group_id_40330;
    
    phys_group_id_40330 = get_group_id(0);
    for (int32_t i_40331 = 0; i_40331 < sdiv_up32(sdiv_up32(d_32136,
                                                            squot32(segred_group_sizze_36362,
                                                                    segment_sizze_nonzzero_40321)) -
                                                  phys_group_id_40330,
                                                  num_groups_36363);
         i_40331++) {
        int32_t virt_group_id_40332 = phys_group_id_40330 + i_40331 *
                num_groups_36363;
        int32_t gtid_36341 = squot32(local_tid_40324,
                                     segment_sizze_nonzzero_40321) +
                virt_group_id_40332 * squot32(segred_group_sizze_36362,
                                              segment_sizze_nonzzero_40321);
        int32_t gtid_36351 = srem32(local_tid_40324, m_32135);
        
        // apply map function if in bounds
        {
            if (slt32(0, m_32135) && (slt32(gtid_36341, d_32136) &&
                                      slt32(local_tid_40324, m_32135 *
                                            squot32(segred_group_sizze_36362,
                                                    segment_sizze_nonzzero_40321)))) {
                float x_36370 = ((__global
                                  float *) mem_39427)[sext_i32_i64(gtid_36341) *
                                                      sext_i32_i64(m_32135) +
                                                      sext_i32_i64(gtid_36351)];
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324)] =
                        x_36370;
                }
            } else {
                ((__local
                  float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324)] =
                    INFINITY;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, m_32135)) {
            // perform segmented scan to imitate reduction
            {
                float x_36366;
                float x_36367;
                float x_40333;
                float x_40334;
                int32_t skip_threads_40336;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_40324, m_32135 *
                              squot32(segred_group_sizze_36362,
                                      segment_sizze_nonzzero_40321))) {
                        x_36367 = ((volatile __local
                                    float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324)];
                        if ((local_tid_40324 - squot32(local_tid_40324, 32) *
                             32) == 0) {
                            x_36366 = x_36367;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_40336 = 1;
                    while (slt32(skip_threads_40336, 32)) {
                        if (sle32(skip_threads_40336, local_tid_40324 -
                                  squot32(local_tid_40324, 32) * 32) &&
                            slt32(local_tid_40324, m_32135 *
                                  squot32(segred_group_sizze_36362,
                                          segment_sizze_nonzzero_40321))) {
                            // read operands
                            {
                                x_36366 = ((volatile __local
                                            float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324 -
                                                                        skip_threads_40336)];
                            }
                            // perform operation
                            {
                                bool inactive_40337 =
                                     slt32(srem32(local_tid_40324, m_32135),
                                           local_tid_40324 - (local_tid_40324 -
                                                              skip_threads_40336));
                                
                                if (inactive_40337) {
                                    x_36366 = x_36367;
                                }
                                if (!inactive_40337) {
                                    float res_36368 = fmin32(x_36366, x_36367);
                                    
                                    x_36366 = res_36368;
                                }
                            }
                        }
                        if (sle32(wave_sizze_40326, skip_threads_40336)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_40336, local_tid_40324 -
                                  squot32(local_tid_40324, 32) * 32) &&
                            slt32(local_tid_40324, m_32135 *
                                  squot32(segred_group_sizze_36362,
                                          segment_sizze_nonzzero_40321))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324)] =
                                    x_36366;
                                x_36367 = x_36366;
                            }
                        }
                        if (sle32(wave_sizze_40326, skip_threads_40336)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_40336 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_40324 - squot32(local_tid_40324, 32) * 32) ==
                        31 && slt32(local_tid_40324, m_32135 *
                                    squot32(segred_group_sizze_36362,
                                            segment_sizze_nonzzero_40321))) {
                        ((volatile __local
                          float *) red_arr_mem_40328)[sext_i32_i64(squot32(local_tid_40324,
                                                                           32))] =
                            x_36366;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_40338;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_40324, 32) == 0 &&
                            slt32(local_tid_40324, m_32135 *
                                  squot32(segred_group_sizze_36362,
                                          segment_sizze_nonzzero_40321))) {
                            x_40334 = ((volatile __local
                                        float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324)];
                            if ((local_tid_40324 - squot32(local_tid_40324,
                                                           32) * 32) == 0) {
                                x_40333 = x_40334;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_40338 = 1;
                        while (slt32(skip_threads_40338, 32)) {
                            if (sle32(skip_threads_40338, local_tid_40324 -
                                      squot32(local_tid_40324, 32) * 32) &&
                                (squot32(local_tid_40324, 32) == 0 &&
                                 slt32(local_tid_40324, m_32135 *
                                       squot32(segred_group_sizze_36362,
                                               segment_sizze_nonzzero_40321)))) {
                                // read operands
                                {
                                    x_40333 = ((volatile __local
                                                float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324 -
                                                                            skip_threads_40338)];
                                }
                                // perform operation
                                {
                                    bool inactive_40339 =
                                         slt32(srem32(local_tid_40324 * 32 +
                                                      32 - 1, m_32135),
                                               local_tid_40324 * 32 + 32 - 1 -
                                               ((local_tid_40324 -
                                                 skip_threads_40338) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_40339) {
                                        x_40333 = x_40334;
                                    }
                                    if (!inactive_40339) {
                                        float res_40335 = fmin32(x_40333,
                                                                 x_40334);
                                        
                                        x_40333 = res_40335;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_40326, skip_threads_40338)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_40338, local_tid_40324 -
                                      squot32(local_tid_40324, 32) * 32) &&
                                (squot32(local_tid_40324, 32) == 0 &&
                                 slt32(local_tid_40324, m_32135 *
                                       squot32(segred_group_sizze_36362,
                                               segment_sizze_nonzzero_40321)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324)] =
                                        x_40333;
                                    x_40334 = x_40333;
                                }
                            }
                            if (sle32(wave_sizze_40326, skip_threads_40338)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_40338 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_40324, 32) == 0 ||
                          !slt32(local_tid_40324, m_32135 *
                                 squot32(segred_group_sizze_36362,
                                         segment_sizze_nonzzero_40321)))) {
                        // read operands
                        {
                            x_36367 = x_36366;
                            x_36366 = ((__local
                                        float *) red_arr_mem_40328)[sext_i32_i64(squot32(local_tid_40324,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_40340 = slt32(srem32(local_tid_40324,
                                                               m_32135),
                                                        local_tid_40324 -
                                                        (squot32(local_tid_40324,
                                                                 32) * 32 - 1));
                            
                            if (inactive_40340) {
                                x_36366 = x_36367;
                            }
                            if (!inactive_40340) {
                                float res_36368 = fmin32(x_36366, x_36367);
                                
                                x_36366 = res_36368;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324)] =
                                x_36366;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_40324, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_40328)[sext_i32_i64(local_tid_40324)] =
                            x_36367;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_40332 * squot32(segred_group_sizze_36362,
                                                    segment_sizze_nonzzero_40321) +
                      local_tid_40324, d_32136) && slt32(local_tid_40324,
                                                         squot32(segred_group_sizze_36362,
                                                                 segment_sizze_nonzzero_40321))) {
                ((__global
                  float *) mem_39431)[sext_i32_i64(virt_group_id_40332 *
                                      squot32(segred_group_sizze_36362,
                                              segment_sizze_nonzzero_40321) +
                                      local_tid_40324)] = ((__local
                                                            float *) red_arr_mem_40328)[sext_i32_i64((local_tid_40324 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_40321 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_36362
}
__kernel void buildKDtreezisegred_small_36413(__global int *global_failure,
                                              uint red_arr_mem_40395_backing_offset_0,
                                              int32_t m_32135, int32_t d_32136,
                                              int32_t num_groups_36424, __global
                                              unsigned char *mem_39441, __global
                                              unsigned char *mem_39445,
                                              int32_t segment_sizze_nonzzero_40388)
{
    #define segred_group_sizze_36423 (buildKDtreezisegred_group_sizze_36407)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40395_backing_0 =
                  &shared_mem[red_arr_mem_40395_backing_offset_0];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40390;
    int32_t local_tid_40391;
    int32_t group_sizze_40394;
    int32_t wave_sizze_40393;
    int32_t group_tid_40392;
    
    global_tid_40390 = get_global_id(0);
    local_tid_40391 = get_local_id(0);
    group_sizze_40394 = get_local_size(0);
    wave_sizze_40393 = LOCKSTEP_WIDTH;
    group_tid_40392 = get_group_id(0);
    
    int32_t phys_tid_36413;
    
    phys_tid_36413 = global_tid_40390;
    
    __local char *red_arr_mem_40395;
    
    red_arr_mem_40395 = (__local char *) red_arr_mem_40395_backing_0;
    
    int32_t phys_group_id_40397;
    
    phys_group_id_40397 = get_group_id(0);
    for (int32_t i_40398 = 0; i_40398 < sdiv_up32(sdiv_up32(d_32136,
                                                            squot32(segred_group_sizze_36423,
                                                                    segment_sizze_nonzzero_40388)) -
                                                  phys_group_id_40397,
                                                  num_groups_36424);
         i_40398++) {
        int32_t virt_group_id_40399 = phys_group_id_40397 + i_40398 *
                num_groups_36424;
        int32_t gtid_36402 = squot32(local_tid_40391,
                                     segment_sizze_nonzzero_40388) +
                virt_group_id_40399 * squot32(segred_group_sizze_36423,
                                              segment_sizze_nonzzero_40388);
        int32_t gtid_36412 = srem32(local_tid_40391, m_32135);
        
        // apply map function if in bounds
        {
            if (slt32(0, m_32135) && (slt32(gtid_36402, d_32136) &&
                                      slt32(local_tid_40391, m_32135 *
                                            squot32(segred_group_sizze_36423,
                                                    segment_sizze_nonzzero_40388)))) {
                float x_36431 = ((__global
                                  float *) mem_39441)[sext_i32_i64(gtid_36402) *
                                                      sext_i32_i64(m_32135) +
                                                      sext_i32_i64(gtid_36412)];
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391)] =
                        x_36431;
                }
            } else {
                ((__local
                  float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391)] =
                    -INFINITY;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, m_32135)) {
            // perform segmented scan to imitate reduction
            {
                float x_36427;
                float x_36428;
                float x_40400;
                float x_40401;
                int32_t skip_threads_40403;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_40391, m_32135 *
                              squot32(segred_group_sizze_36423,
                                      segment_sizze_nonzzero_40388))) {
                        x_36428 = ((volatile __local
                                    float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391)];
                        if ((local_tid_40391 - squot32(local_tid_40391, 32) *
                             32) == 0) {
                            x_36427 = x_36428;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_40403 = 1;
                    while (slt32(skip_threads_40403, 32)) {
                        if (sle32(skip_threads_40403, local_tid_40391 -
                                  squot32(local_tid_40391, 32) * 32) &&
                            slt32(local_tid_40391, m_32135 *
                                  squot32(segred_group_sizze_36423,
                                          segment_sizze_nonzzero_40388))) {
                            // read operands
                            {
                                x_36427 = ((volatile __local
                                            float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391 -
                                                                        skip_threads_40403)];
                            }
                            // perform operation
                            {
                                bool inactive_40404 =
                                     slt32(srem32(local_tid_40391, m_32135),
                                           local_tid_40391 - (local_tid_40391 -
                                                              skip_threads_40403));
                                
                                if (inactive_40404) {
                                    x_36427 = x_36428;
                                }
                                if (!inactive_40404) {
                                    float res_36429 = fmax32(x_36427, x_36428);
                                    
                                    x_36427 = res_36429;
                                }
                            }
                        }
                        if (sle32(wave_sizze_40393, skip_threads_40403)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_40403, local_tid_40391 -
                                  squot32(local_tid_40391, 32) * 32) &&
                            slt32(local_tid_40391, m_32135 *
                                  squot32(segred_group_sizze_36423,
                                          segment_sizze_nonzzero_40388))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391)] =
                                    x_36427;
                                x_36428 = x_36427;
                            }
                        }
                        if (sle32(wave_sizze_40393, skip_threads_40403)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_40403 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_40391 - squot32(local_tid_40391, 32) * 32) ==
                        31 && slt32(local_tid_40391, m_32135 *
                                    squot32(segred_group_sizze_36423,
                                            segment_sizze_nonzzero_40388))) {
                        ((volatile __local
                          float *) red_arr_mem_40395)[sext_i32_i64(squot32(local_tid_40391,
                                                                           32))] =
                            x_36427;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_40405;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_40391, 32) == 0 &&
                            slt32(local_tid_40391, m_32135 *
                                  squot32(segred_group_sizze_36423,
                                          segment_sizze_nonzzero_40388))) {
                            x_40401 = ((volatile __local
                                        float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391)];
                            if ((local_tid_40391 - squot32(local_tid_40391,
                                                           32) * 32) == 0) {
                                x_40400 = x_40401;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_40405 = 1;
                        while (slt32(skip_threads_40405, 32)) {
                            if (sle32(skip_threads_40405, local_tid_40391 -
                                      squot32(local_tid_40391, 32) * 32) &&
                                (squot32(local_tid_40391, 32) == 0 &&
                                 slt32(local_tid_40391, m_32135 *
                                       squot32(segred_group_sizze_36423,
                                               segment_sizze_nonzzero_40388)))) {
                                // read operands
                                {
                                    x_40400 = ((volatile __local
                                                float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391 -
                                                                            skip_threads_40405)];
                                }
                                // perform operation
                                {
                                    bool inactive_40406 =
                                         slt32(srem32(local_tid_40391 * 32 +
                                                      32 - 1, m_32135),
                                               local_tid_40391 * 32 + 32 - 1 -
                                               ((local_tid_40391 -
                                                 skip_threads_40405) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_40406) {
                                        x_40400 = x_40401;
                                    }
                                    if (!inactive_40406) {
                                        float res_40402 = fmax32(x_40400,
                                                                 x_40401);
                                        
                                        x_40400 = res_40402;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_40393, skip_threads_40405)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_40405, local_tid_40391 -
                                      squot32(local_tid_40391, 32) * 32) &&
                                (squot32(local_tid_40391, 32) == 0 &&
                                 slt32(local_tid_40391, m_32135 *
                                       squot32(segred_group_sizze_36423,
                                               segment_sizze_nonzzero_40388)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391)] =
                                        x_40400;
                                    x_40401 = x_40400;
                                }
                            }
                            if (sle32(wave_sizze_40393, skip_threads_40405)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_40405 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_40391, 32) == 0 ||
                          !slt32(local_tid_40391, m_32135 *
                                 squot32(segred_group_sizze_36423,
                                         segment_sizze_nonzzero_40388)))) {
                        // read operands
                        {
                            x_36428 = x_36427;
                            x_36427 = ((__local
                                        float *) red_arr_mem_40395)[sext_i32_i64(squot32(local_tid_40391,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_40407 = slt32(srem32(local_tid_40391,
                                                               m_32135),
                                                        local_tid_40391 -
                                                        (squot32(local_tid_40391,
                                                                 32) * 32 - 1));
                            
                            if (inactive_40407) {
                                x_36427 = x_36428;
                            }
                            if (!inactive_40407) {
                                float res_36429 = fmax32(x_36427, x_36428);
                                
                                x_36427 = res_36429;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391)] =
                                x_36427;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_40391, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_40395)[sext_i32_i64(local_tid_40391)] =
                            x_36428;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_40399 * squot32(segred_group_sizze_36423,
                                                    segment_sizze_nonzzero_40388) +
                      local_tid_40391, d_32136) && slt32(local_tid_40391,
                                                         squot32(segred_group_sizze_36423,
                                                                 segment_sizze_nonzzero_40388))) {
                ((__global
                  float *) mem_39445)[sext_i32_i64(virt_group_id_40399 *
                                      squot32(segred_group_sizze_36423,
                                              segment_sizze_nonzzero_40388) +
                                      local_tid_40391)] = ((__local
                                                            float *) red_arr_mem_40395)[sext_i32_i64((local_tid_40391 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_40388 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_36423
}
__kernel void buildKDtreezisegred_small_36675(__global int *global_failure,
                                              int failure_is_an_option, __global
                                              int *global_failure_args,
                                              uint red_arr_mem_40518_backing_offset_0,
                                              uint red_arr_mem_40516_backing_offset_1,
                                              int32_t d_32136,
                                              int32_t conc_tmp_32196,
                                              int32_t nodes_this_lvl_32221,
                                              int32_t num_groups_36848, __global
                                              unsigned char *mem_39532, __global
                                              unsigned char *mem_39540, __global
                                              unsigned char *mem_39544, __global
                                              unsigned char *mem_39547,
                                              int32_t segment_sizze_nonzzero_40509)
{
    #define segred_group_sizze_36847 (buildKDtreezisegred_group_sizze_36669)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40518_backing_1 =
                  &shared_mem[red_arr_mem_40518_backing_offset_0];
    volatile char *red_arr_mem_40516_backing_0 =
                  &shared_mem[red_arr_mem_40516_backing_offset_1];
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40511;
    int32_t local_tid_40512;
    int32_t group_sizze_40515;
    int32_t wave_sizze_40514;
    int32_t group_tid_40513;
    
    global_tid_40511 = get_global_id(0);
    local_tid_40512 = get_local_id(0);
    group_sizze_40515 = get_local_size(0);
    wave_sizze_40514 = LOCKSTEP_WIDTH;
    group_tid_40513 = get_group_id(0);
    
    int32_t phys_tid_36675;
    
    phys_tid_36675 = global_tid_40511;
    
    __local char *red_arr_mem_40516;
    
    red_arr_mem_40516 = (__local char *) red_arr_mem_40516_backing_0;
    
    __local char *red_arr_mem_40518;
    
    red_arr_mem_40518 = (__local char *) red_arr_mem_40518_backing_1;
    
    int32_t phys_group_id_40520;
    
    phys_group_id_40520 = get_group_id(0);
    for (int32_t i_40521 = 0; i_40521 <
         sdiv_up32(sdiv_up32(nodes_this_lvl_32221,
                             squot32(segred_group_sizze_36847,
                                     segment_sizze_nonzzero_40509)) -
                   phys_group_id_40520, num_groups_36848); i_40521++) {
        int32_t virt_group_id_40522 = phys_group_id_40520 + i_40521 *
                num_groups_36848;
        int32_t gtid_36664 = squot32(local_tid_40512,
                                     segment_sizze_nonzzero_40509) +
                virt_group_id_40522 * squot32(segred_group_sizze_36847,
                                              segment_sizze_nonzzero_40509);
        int32_t gtid_36674 = srem32(local_tid_40512, d_32136);
        
        // apply map function if in bounds
        {
            if (slt32(0, d_32136) && (slt32(gtid_36664, nodes_this_lvl_32221) &&
                                      slt32(local_tid_40512, d_32136 *
                                            squot32(segred_group_sizze_36847,
                                                    segment_sizze_nonzzero_40509)))) {
                int32_t i_36861 = add32(d_32136, gtid_36674);
                bool x_36862 = sle32(0, i_36861);
                bool y_36863 = slt32(i_36861, conc_tmp_32196);
                bool bounds_check_36864 = x_36862 && y_36863;
                bool index_certs_36865;
                
                if (!bounds_check_36864) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 11) ==
                            -1) {
                            global_failure_args[0] = i_36861;
                            global_failure_args[1] = conc_tmp_32196;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float x_36866 = ((__global
                                  float *) mem_39540)[sext_i32_i64(gtid_36664) *
                                                      sext_i32_i64(conc_tmp_32196) +
                                                      sext_i32_i64(i_36861)];
                bool y_36868 = slt32(gtid_36674, conc_tmp_32196);
                bool index_certs_36870;
                
                if (!y_36868) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 12) ==
                            -1) {
                            global_failure_args[0] = gtid_36674;
                            global_failure_args[1] = conc_tmp_32196;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                float y_36871 = ((__global
                                  float *) mem_39532)[sext_i32_i64(gtid_36674) *
                                                      sext_i32_i64(nodes_this_lvl_32221) +
                                                      sext_i32_i64(gtid_36664)];
                float abs_arg_36872 = x_36866 - y_36871;
                float res_36873 = (float) fabs(abs_arg_36872);
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512)] =
                        gtid_36674;
                    ((__local
                      float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512)] =
                        res_36873;
                }
            } else {
                ((__local
                  int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512)] =
                    -1;
                ((__local
                  float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512)] =
                    -INFINITY;
            }
        }
        
      error_0:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, d_32136)) {
            // perform segmented scan to imitate reduction
            {
                int32_t x_36852;
                float x_36853;
                int32_t x_36854;
                float x_36855;
                int32_t x_40523;
                float x_40524;
                int32_t x_40525;
                float x_40526;
                int32_t skip_threads_40530;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_40512, d_32136 *
                              squot32(segred_group_sizze_36847,
                                      segment_sizze_nonzzero_40509))) {
                        x_36854 = ((volatile __local
                                    int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512)];
                        x_36855 = ((volatile __local
                                    float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512)];
                        if ((local_tid_40512 - squot32(local_tid_40512, 32) *
                             32) == 0) {
                            x_36852 = x_36854;
                            x_36853 = x_36855;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_40530 = 1;
                    while (slt32(skip_threads_40530, 32)) {
                        if (sle32(skip_threads_40530, local_tid_40512 -
                                  squot32(local_tid_40512, 32) * 32) &&
                            slt32(local_tid_40512, d_32136 *
                                  squot32(segred_group_sizze_36847,
                                          segment_sizze_nonzzero_40509))) {
                            // read operands
                            {
                                x_36852 = ((volatile __local
                                            int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512 -
                                                                          skip_threads_40530)];
                                x_36853 = ((volatile __local
                                            float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512 -
                                                                        skip_threads_40530)];
                            }
                            // perform operation
                            {
                                bool inactive_40531 =
                                     slt32(srem32(local_tid_40512, d_32136),
                                           local_tid_40512 - (local_tid_40512 -
                                                              skip_threads_40530));
                                
                                if (inactive_40531) {
                                    x_36852 = x_36854;
                                    x_36853 = x_36855;
                                }
                                if (!inactive_40531) {
                                    bool cond_36856 = x_36855 <= x_36853;
                                    int32_t res_36857;
                                    
                                    if (cond_36856) {
                                        res_36857 = x_36852;
                                    } else {
                                        res_36857 = x_36854;
                                    }
                                    
                                    float res_36858;
                                    
                                    if (cond_36856) {
                                        res_36858 = x_36853;
                                    } else {
                                        res_36858 = x_36855;
                                    }
                                    x_36852 = res_36857;
                                    x_36853 = res_36858;
                                }
                            }
                        }
                        if (sle32(wave_sizze_40514, skip_threads_40530)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_40530, local_tid_40512 -
                                  squot32(local_tid_40512, 32) * 32) &&
                            slt32(local_tid_40512, d_32136 *
                                  squot32(segred_group_sizze_36847,
                                          segment_sizze_nonzzero_40509))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512)] =
                                    x_36852;
                                x_36854 = x_36852;
                                ((volatile __local
                                  float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512)] =
                                    x_36853;
                                x_36855 = x_36853;
                            }
                        }
                        if (sle32(wave_sizze_40514, skip_threads_40530)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_40530 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_40512 - squot32(local_tid_40512, 32) * 32) ==
                        31 && slt32(local_tid_40512, d_32136 *
                                    squot32(segred_group_sizze_36847,
                                            segment_sizze_nonzzero_40509))) {
                        ((volatile __local
                          int32_t *) red_arr_mem_40516)[sext_i32_i64(squot32(local_tid_40512,
                                                                             32))] =
                            x_36852;
                        ((volatile __local
                          float *) red_arr_mem_40518)[sext_i32_i64(squot32(local_tid_40512,
                                                                           32))] =
                            x_36853;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_40532;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_40512, 32) == 0 &&
                            slt32(local_tid_40512, d_32136 *
                                  squot32(segred_group_sizze_36847,
                                          segment_sizze_nonzzero_40509))) {
                            x_40525 = ((volatile __local
                                        int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512)];
                            x_40526 = ((volatile __local
                                        float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512)];
                            if ((local_tid_40512 - squot32(local_tid_40512,
                                                           32) * 32) == 0) {
                                x_40523 = x_40525;
                                x_40524 = x_40526;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_40532 = 1;
                        while (slt32(skip_threads_40532, 32)) {
                            if (sle32(skip_threads_40532, local_tid_40512 -
                                      squot32(local_tid_40512, 32) * 32) &&
                                (squot32(local_tid_40512, 32) == 0 &&
                                 slt32(local_tid_40512, d_32136 *
                                       squot32(segred_group_sizze_36847,
                                               segment_sizze_nonzzero_40509)))) {
                                // read operands
                                {
                                    x_40523 = ((volatile __local
                                                int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512 -
                                                                              skip_threads_40532)];
                                    x_40524 = ((volatile __local
                                                float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512 -
                                                                            skip_threads_40532)];
                                }
                                // perform operation
                                {
                                    bool inactive_40533 =
                                         slt32(srem32(local_tid_40512 * 32 +
                                                      32 - 1, d_32136),
                                               local_tid_40512 * 32 + 32 - 1 -
                                               ((local_tid_40512 -
                                                 skip_threads_40532) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_40533) {
                                        x_40523 = x_40525;
                                        x_40524 = x_40526;
                                    }
                                    if (!inactive_40533) {
                                        bool cond_40527 = x_40526 <= x_40524;
                                        int32_t res_40528;
                                        
                                        if (cond_40527) {
                                            res_40528 = x_40523;
                                        } else {
                                            res_40528 = x_40525;
                                        }
                                        
                                        float res_40529;
                                        
                                        if (cond_40527) {
                                            res_40529 = x_40524;
                                        } else {
                                            res_40529 = x_40526;
                                        }
                                        x_40523 = res_40528;
                                        x_40524 = res_40529;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_40514, skip_threads_40532)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_40532, local_tid_40512 -
                                      squot32(local_tid_40512, 32) * 32) &&
                                (squot32(local_tid_40512, 32) == 0 &&
                                 slt32(local_tid_40512, d_32136 *
                                       squot32(segred_group_sizze_36847,
                                               segment_sizze_nonzzero_40509)))) {
                                // write result
                                {
                                    ((volatile __local
                                      int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512)] =
                                        x_40523;
                                    x_40525 = x_40523;
                                    ((volatile __local
                                      float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512)] =
                                        x_40524;
                                    x_40526 = x_40524;
                                }
                            }
                            if (sle32(wave_sizze_40514, skip_threads_40532)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_40532 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_40512, 32) == 0 ||
                          !slt32(local_tid_40512, d_32136 *
                                 squot32(segred_group_sizze_36847,
                                         segment_sizze_nonzzero_40509)))) {
                        // read operands
                        {
                            x_36854 = x_36852;
                            x_36855 = x_36853;
                            x_36852 = ((__local
                                        int32_t *) red_arr_mem_40516)[sext_i32_i64(squot32(local_tid_40512,
                                                                                           32) -
                                                                      1)];
                            x_36853 = ((__local
                                        float *) red_arr_mem_40518)[sext_i32_i64(squot32(local_tid_40512,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_40534 = slt32(srem32(local_tid_40512,
                                                               d_32136),
                                                        local_tid_40512 -
                                                        (squot32(local_tid_40512,
                                                                 32) * 32 - 1));
                            
                            if (inactive_40534) {
                                x_36852 = x_36854;
                                x_36853 = x_36855;
                            }
                            if (!inactive_40534) {
                                bool cond_36856 = x_36855 <= x_36853;
                                int32_t res_36857;
                                
                                if (cond_36856) {
                                    res_36857 = x_36852;
                                } else {
                                    res_36857 = x_36854;
                                }
                                
                                float res_36858;
                                
                                if (cond_36856) {
                                    res_36858 = x_36853;
                                } else {
                                    res_36858 = x_36855;
                                }
                                x_36852 = res_36857;
                                x_36853 = res_36858;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512)] =
                                x_36852;
                            ((__local
                              float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512)] =
                                x_36853;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_40512, 32) == 0) {
                        ((__local
                          int32_t *) red_arr_mem_40516)[sext_i32_i64(local_tid_40512)] =
                            x_36854;
                        ((__local
                          float *) red_arr_mem_40518)[sext_i32_i64(local_tid_40512)] =
                            x_36855;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_40522 * squot32(segred_group_sizze_36847,
                                                    segment_sizze_nonzzero_40509) +
                      local_tid_40512, nodes_this_lvl_32221) &&
                slt32(local_tid_40512, squot32(segred_group_sizze_36847,
                                               segment_sizze_nonzzero_40509))) {
                ((__global
                  int32_t *) mem_39544)[sext_i32_i64(virt_group_id_40522 *
                                        squot32(segred_group_sizze_36847,
                                                segment_sizze_nonzzero_40509) +
                                        local_tid_40512)] = ((__local
                                                              int32_t *) red_arr_mem_40516)[sext_i32_i64((local_tid_40512 +
                                                                                                          1) *
                                                                                            segment_sizze_nonzzero_40509 -
                                                                                            1)];
                ((__global
                  float *) mem_39547)[sext_i32_i64(virt_group_id_40522 *
                                      squot32(segred_group_sizze_36847,
                                              segment_sizze_nonzzero_40509) +
                                      local_tid_40512)] = ((__local
                                                            float *) red_arr_mem_40518)[sext_i32_i64((local_tid_40512 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_40509 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_36847
}
__kernel void builtinzhiota_i32ziiota_i32_40454(__global
                                                unsigned char *mem_40449,
                                                int32_t n_40450,
                                                int32_t x_40451,
                                                int32_t s_40452)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t iota_gtid_40454;
    int32_t iota_ltid_40455;
    int32_t iota_gid_40456;
    
    iota_gtid_40454 = get_global_id(0);
    iota_ltid_40455 = get_local_id(0);
    iota_gid_40456 = get_group_id(0);
    if (slt64(iota_gtid_40454, sext_i32_i64(n_40450))) {
        ((__global int32_t *) mem_40449)[sext_i32_i64(iota_gtid_40454)] =
            add32(mul32(iota_gtid_40454, s_40452), x_40451);
    }
    
  error_0:
    return;
}
__kernel void builtinzhreplicate_f32zireplicate_40463(__global
                                                      unsigned char *mem_40459,
                                                      int32_t num_elems_40460,
                                                      float val_40461)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_40463;
    int32_t replicate_ltid_40464;
    int32_t replicate_gid_40465;
    
    replicate_gtid_40463 = get_global_id(0);
    replicate_ltid_40464 = get_local_id(0);
    replicate_gid_40465 = get_group_id(0);
    if (slt64(replicate_gtid_40463, sext_i32_i64(num_elems_40460))) {
        ((__global float *) mem_40459)[sext_i32_i64(replicate_gtid_40463)] =
            val_40461;
    }
    
  error_0:
    return;
}
__kernel void builtinzhreplicate_i32zireplicate_40472(__global
                                                      unsigned char *mem_40468,
                                                      int32_t num_elems_40469,
                                                      int32_t val_40470)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_40472;
    int32_t replicate_ltid_40473;
    int32_t replicate_gid_40474;
    
    replicate_gtid_40472 = get_global_id(0);
    replicate_ltid_40473 = get_local_id(0);
    replicate_gid_40474 = get_group_id(0);
    if (slt64(replicate_gtid_40472, sext_i32_i64(num_elems_40469))) {
        ((__global int32_t *) mem_40468)[sext_i32_i64(replicate_gtid_40472)] =
            val_40470;
    }
    
  error_0:
    return;
}
__kernel void exactKnnFixKzicopy_40302(int32_t kk_31030, int32_t s_31037,
                                       __global unsigned char *knn_is_mem_39424,
                                       __global unsigned char *mem_39429)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40302;
    int32_t copy_ltid_40303;
    int32_t copy_gid_40304;
    
    copy_gtid_40302 = get_global_id(0);
    copy_ltid_40303 = get_local_id(0);
    copy_gid_40304 = get_group_id(0);
    if (slt32(copy_gtid_40302, sext_i64_i32(sext_i32_i64(s_31037) * 8))) {
        ((__global int32_t *) mem_39429)[sext_i32_i64(squot32(copy_gtid_40302,
                                                              8)) * 8 +
                                         sext_i32_i64(copy_gtid_40302 -
                                         squot32(copy_gtid_40302, 8) * 8)] =
            ((__global
              int32_t *) knn_is_mem_39424)[sext_i32_i64(squot32(copy_gtid_40302,
                                                                8)) *
                                           sext_i32_i64(kk_31030) +
                                           sext_i32_i64(copy_gtid_40302 -
                                           squot32(copy_gtid_40302, 8) * 8)];
    }
    
  error_0:
    return;
}
__kernel void exactKnnFixKzicopy_40307(int32_t kk_31032, int32_t s_31037,
                                       __global unsigned char *knn_vs_mem_39425,
                                       __global unsigned char *mem_39433)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40307;
    int32_t copy_ltid_40308;
    int32_t copy_gid_40309;
    
    copy_gtid_40307 = get_global_id(0);
    copy_ltid_40308 = get_local_id(0);
    copy_gid_40309 = get_group_id(0);
    if (slt32(copy_gtid_40307, sext_i64_i32(sext_i32_i64(s_31037) * 8))) {
        ((__global float *) mem_39433)[sext_i32_i64(squot32(copy_gtid_40307,
                                                            8)) * 8 +
                                       sext_i32_i64(copy_gtid_40307 -
                                       squot32(copy_gtid_40307, 8) * 8)] =
            ((__global
              float *) knn_vs_mem_39425)[sext_i32_i64(squot32(copy_gtid_40307,
                                                              8)) *
                                         sext_i32_i64(kk_31032) +
                                         sext_i32_i64(copy_gtid_40307 -
                                         squot32(copy_gtid_40307, 8) * 8)];
    }
    
  error_0:
    return;
}
__kernel void exactKnnFixKzisegmap_33962(__global int *global_failure,
                                         int failure_is_an_option, __global
                                         int *global_failure_args,
                                         int32_t d_31022, int32_t n_31026,
                                         int32_t s_31037,
                                         int32_t num_leaves_31075,
                                         int32_t ppl_31090,
                                         int32_t num_groups_34084, __global
                                         unsigned char *mem_39444, __global
                                         unsigned char *mem_39451, __global
                                         unsigned char *mem_39504, __global
                                         unsigned char *mem_39514, __global
                                         unsigned char *mem_39518, __global
                                         unsigned char *mem_39556, __global
                                         unsigned char *mem_39666, __global
                                         unsigned char *mem_39670, __global
                                         unsigned char *double_buffer_mem_40098,
                                         __global
                                         unsigned char *double_buffer_mem_40099)
{
    #define segmap_group_sizze_34083 (exactKnnFixKzisegmap_group_sizze_33965)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40395;
    int32_t local_tid_40396;
    int32_t group_sizze_40399;
    int32_t wave_sizze_40398;
    int32_t group_tid_40397;
    
    global_tid_40395 = get_global_id(0);
    local_tid_40396 = get_local_id(0);
    group_sizze_40399 = get_local_size(0);
    wave_sizze_40398 = LOCKSTEP_WIDTH;
    group_tid_40397 = get_group_id(0);
    
    int32_t phys_tid_33962;
    
    phys_tid_33962 = global_tid_40395;
    
    int32_t phys_group_id_40400;
    
    phys_group_id_40400 = get_group_id(0);
    for (int32_t i_40401 = 0; i_40401 < sdiv_up32(sdiv_up32(s_31037,
                                                            segmap_group_sizze_34083) -
                                                  phys_group_id_40400,
                                                  num_groups_34084);
         i_40401++) {
        int32_t virt_group_id_40402 = phys_group_id_40400 + i_40401 *
                num_groups_34084;
        int32_t gtid_33961 = sext_i64_i32(sext_i32_i64(virt_group_id_40402) *
                sext_i32_i64(segmap_group_sizze_34083) +
                sext_i32_i64(local_tid_40396));
        
        if (slt32(gtid_33961, s_31037)) {
            int32_t x_34091 = ((__global
                                int32_t *) mem_39504)[sext_i32_i64(gtid_33961)];
            bool cond_34092 = slt32(x_34091, num_leaves_31075);
            int32_t count_34093 = btoi_bool_i32(cond_34092);
            bool loop_nonempty_34094 = slt32(0, count_34093);
            int32_t bruteForcePar_arg_34095 = mul32(ppl_31090, x_34091);
            bool x_34096 = sle32(0, x_34091);
            bool bounds_check_34097 = cond_34092 && x_34096;
            bool loop_not_taken_34098 = !loop_nonempty_34094;
            bool protect_assert_disj_34099 = bounds_check_34097 ||
                 loop_not_taken_34098;
            bool index_certs_34100;
            
            if (!protect_assert_disj_34099) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 32) ==
                        -1) {
                        global_failure_args[0] = x_34091;
                        global_failure_args[1] = num_leaves_31075;
                        ;
                    }
                    local_failure = true;
                    goto error_0;
                }
            }
            
            int32_t mem_39551[8];
            float mem_39553[8];
            int32_t mem_39608[8];
            float mem_39610[8];
            
            for (int32_t i_40403 = 0; i_40403 < 8; i_40403++) {
                ((__global
                  int32_t *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_33962) +
                                                      sext_i32_i64(i_40403) *
                                                      sext_i32_i64(num_groups_34084 *
                                                      segmap_group_sizze_34083)] =
                    ((__global int32_t *) mem_39514)[sext_i32_i64(gtid_33961) +
                                                     sext_i32_i64(i_40403) *
                                                     sext_i32_i64(s_31037)];
            }
            for (int32_t i_40404 = 0; i_40404 < 8; i_40404++) {
                ((__global
                  float *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_33962) +
                                                    sext_i32_i64(i_40404) *
                                                    sext_i32_i64(num_groups_34084 *
                                                    segmap_group_sizze_34083)] =
                    ((__global float *) mem_39518)[sext_i32_i64(gtid_33961) +
                                                   sext_i32_i64(i_40404) *
                                                   sext_i32_i64(s_31037)];
            }
            for (int32_t _j_34103 = 0; _j_34103 < count_34093; _j_34103++) {
                for (int32_t i_40407 = 0; i_40407 < 8; i_40407++) {
                    mem_39551[sext_i32_i64(i_40407)] = ((__global
                                                         int32_t *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_33962) +
                                                                                             sext_i32_i64(i_40407) *
                                                                                             sext_i32_i64(num_groups_34084 *
                                                                                             segmap_group_sizze_34083)];
                }
                for (int32_t i_40408 = 0; i_40408 < 8; i_40408++) {
                    mem_39553[sext_i32_i64(i_40408)] = ((__global
                                                         float *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_33962) +
                                                                                           sext_i32_i64(i_40408) *
                                                                                           sext_i32_i64(num_groups_34084 *
                                                                                           segmap_group_sizze_34083)];
                }
                for (int32_t i_39355 = 0; i_39355 < ppl_31090; i_39355++) {
                    float res_34111;
                    float res_34113 = 0.0F;
                    float x_34114;
                    float y_34115;
                    
                    for (int32_t i_34112 = 0; i_34112 < d_31022; i_34112++) {
                        x_34114 = ((__global
                                    float *) mem_39444)[sext_i32_i64(gtid_33961) +
                                                        sext_i32_i64(i_34112) *
                                                        sext_i32_i64(n_31026)];
                        y_34115 = ((__global
                                    float *) mem_39451)[sext_i32_i64(i_39355 *
                                                        (num_leaves_31075 *
                                                         d_31022) + x_34091) +
                                                        sext_i32_i64(i_34112) *
                                                        sext_i32_i64(num_leaves_31075)];
                        
                        float zz_34116;
                        
                        zz_34116 = x_34114 - y_34115;
                        
                        float y_34117 = zz_34116 * zz_34116;
                        float loopres_34118 = res_34113 + y_34117;
                        float res_tmp_40410 = loopres_34118;
                        
                        res_34113 = res_tmp_40410;
                    }
                    res_34111 = res_34113;
                    ((__global
                      float *) mem_39556)[sext_i32_i64(phys_tid_33962) +
                                          sext_i32_i64(i_39355) *
                                          sext_i32_i64(num_groups_34084 *
                                          segmap_group_sizze_34083)] =
                        res_34111;
                }
                
                bool knn_34119;
                int32_t knn_34123;
                bool loop_while_34124;
                int32_t j_34128;
                
                loop_while_34124 = 1;
                j_34128 = 0;
                while (loop_while_34124) {
                    int32_t res_34129;
                    float res_34130;
                    int32_t redout_39357;
                    float redout_39358;
                    
                    redout_39357 = ppl_31090;
                    redout_39358 = INFINITY;
                    for (int32_t i_39359 = 0; i_39359 < ppl_31090; i_39359++) {
                        float x_34144 = ((__global
                                          float *) mem_39556)[sext_i32_i64(phys_tid_33962) +
                                                              sext_i32_i64(i_39359) *
                                                              sext_i32_i64(num_groups_34084 *
                                                              segmap_group_sizze_34083)];
                        bool cond_34135 = redout_39358 < x_34144;
                        int32_t res_34136;
                        float res_34137;
                        
                        if (cond_34135) {
                            res_34136 = redout_39357;
                            res_34137 = redout_39358;
                        } else {
                            bool cond_34138 = x_34144 < redout_39358;
                            float res_34139;
                            
                            if (cond_34138) {
                                res_34139 = x_34144;
                            } else {
                                res_34139 = redout_39358;
                            }
                            
                            int32_t res_34140;
                            
                            if (cond_34138) {
                                res_34140 = i_39359;
                            } else {
                                bool cond_34141 = sle32(redout_39357, i_39359);
                                int32_t res_34142;
                                
                                if (cond_34141) {
                                    res_34142 = redout_39357;
                                } else {
                                    res_34142 = i_39359;
                                }
                                res_34140 = res_34142;
                            }
                            res_34136 = res_34140;
                            res_34137 = res_34139;
                        }
                        
                        int32_t redout_tmp_40416 = res_34136;
                        float redout_tmp_40417 = res_34137;
                        
                        redout_39357 = redout_tmp_40416;
                        redout_39358 = redout_tmp_40417;
                    }
                    res_34129 = redout_39357;
                    res_34130 = redout_39358;
                    
                    int32_t i_34145 = sub32(7, j_34128);
                    bool x_34146 = sle32(0, i_34145);
                    bool y_34147 = slt32(i_34145, 8);
                    bool bounds_check_34148 = x_34146 && y_34147;
                    bool index_certs_34149;
                    
                    if (!bounds_check_34148) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          33) == -1) {
                                global_failure_args[0] = i_34145;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_34150 = mem_39553[sext_i32_i64(i_34145)];
                    bool cond_34151 = res_34130 < y_34150;
                    int32_t loopres_34155;
                    
                    if (cond_34151) {
                        bool x_34156 = sle32(0, res_34129);
                        bool y_34157 = slt32(res_34129, ppl_31090);
                        bool bounds_check_34158 = x_34156 && y_34157;
                        bool index_certs_34159;
                        
                        if (!bounds_check_34158) {
                            {
                                if (atomic_cmpxchg_i32_global(global_failure,
                                                              -1, 34) == -1) {
                                    global_failure_args[0] = res_34129;
                                    global_failure_args[1] = ppl_31090;
                                    ;
                                }
                                local_failure = true;
                                goto error_0;
                            }
                        }
                        ((__global
                          float *) mem_39556)[sext_i32_i64(phys_tid_33962) +
                                              sext_i32_i64(res_34129) *
                                              sext_i32_i64(num_groups_34084 *
                                              segmap_group_sizze_34083)] =
                            INFINITY;
                        
                        int32_t lw_val_34161 = add32(bruteForcePar_arg_34095,
                                                     res_34129);
                        
                        mem_39551[sext_i32_i64(i_34145)] = lw_val_34161;
                        mem_39553[sext_i32_i64(i_34145)] = res_34130;
                        
                        int32_t res_34164 = add32(1, j_34128);
                        
                        loopres_34155 = res_34164;
                    } else {
                        loopres_34155 = j_34128;
                    }
                    
                    bool res_34165 = slt32(loopres_34155, 8);
                    bool x_34166 = cond_34151 && res_34165;
                    bool loop_while_tmp_40411 = x_34166;
                    int32_t j_tmp_40415 = loopres_34155;
                    
                    loop_while_34124 = loop_while_tmp_40411;
                    j_34128 = j_tmp_40415;
                }
                knn_34119 = loop_while_34124;
                knn_34123 = j_34128;
                for (int32_t i_40418 = 0; i_40418 < 8; i_40418++) {
                    mem_39608[sext_i32_i64(i_40418)] = -1;
                }
                for (int32_t i_40419 = 0; i_40419 < 8; i_40419++) {
                    mem_39610[sext_i32_i64(i_40419)] = INFINITY;
                }
                
                int32_t res_34171;
                int32_t res_34172;
                int32_t beg_34176;
                int32_t end_34177;
                
                beg_34176 = 0;
                end_34177 = 7;
                for (int32_t i_34173 = 0; i_34173 < 8; i_34173++) {
                    bool x_34178 = sle32(0, beg_34176);
                    bool y_34179 = slt32(beg_34176, 8);
                    bool bounds_check_34180 = x_34178 && y_34179;
                    bool index_certs_34181;
                    
                    if (!bounds_check_34180) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          35) == -1) {
                                global_failure_args[0] = beg_34176;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float x_34182 = mem_39553[sext_i32_i64(beg_34176)];
                    bool x_34183 = sle32(0, end_34177);
                    bool y_34184 = slt32(end_34177, 8);
                    bool bounds_check_34185 = x_34183 && y_34184;
                    bool index_certs_34186;
                    
                    if (!bounds_check_34185) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          36) == -1) {
                                global_failure_args[0] = end_34177;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_34187 = mem_39553[sext_i32_i64(end_34177)];
                    bool cond_34188 = x_34182 < y_34187;
                    float loopres_34189;
                    
                    if (cond_34188) {
                        loopres_34189 = x_34182;
                    } else {
                        loopres_34189 = y_34187;
                    }
                    
                    int32_t loopres_34190;
                    int32_t loopres_34191;
                    int32_t loopres_34192;
                    
                    if (cond_34188) {
                        int32_t res_34193 = mem_39551[sext_i32_i64(beg_34176)];
                        int32_t res_34194 = add32(1, beg_34176);
                        
                        loopres_34190 = res_34193;
                        loopres_34191 = res_34194;
                        loopres_34192 = end_34177;
                    } else {
                        int32_t res_34195 = mem_39551[sext_i32_i64(end_34177)];
                        int32_t res_34196 = sub32(end_34177, 1);
                        
                        loopres_34190 = res_34195;
                        loopres_34191 = beg_34176;
                        loopres_34192 = res_34196;
                    }
                    mem_39608[sext_i32_i64(i_34173)] = loopres_34190;
                    mem_39610[sext_i32_i64(i_34173)] = loopres_34189;
                    
                    int32_t beg_tmp_40422 = loopres_34191;
                    int32_t end_tmp_40423 = loopres_34192;
                    
                    beg_34176 = beg_tmp_40422;
                    end_34177 = end_tmp_40423;
                }
                res_34171 = beg_34176;
                res_34172 = end_34177;
                for (int32_t i_40424 = 0; i_40424 < 8; i_40424++) {
                    ((__global
                      int32_t *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_33962) +
                                                          sext_i32_i64(i_40424) *
                                                          sext_i32_i64(num_groups_34084 *
                                                          segmap_group_sizze_34083)] =
                        mem_39608[sext_i32_i64(i_40424)];
                }
                for (int32_t i_40425 = 0; i_40425 < 8; i_40425++) {
                    ((__global
                      float *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_33962) +
                                                        sext_i32_i64(i_40425) *
                                                        sext_i32_i64(num_groups_34084 *
                                                        segmap_group_sizze_34083)] =
                        mem_39610[sext_i32_i64(i_40425)];
                }
            }
            for (int32_t i_40426 = 0; i_40426 < 8; i_40426++) {
                ((__global int32_t *) mem_39666)[sext_i32_i64(i_40426) *
                                                 sext_i32_i64(s_31037) +
                                                 sext_i32_i64(gtid_33961)] =
                    ((__global
                      int32_t *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_33962) +
                                                          sext_i32_i64(i_40426) *
                                                          sext_i32_i64(num_groups_34084 *
                                                          segmap_group_sizze_34083)];
            }
            for (int32_t i_40427 = 0; i_40427 < 8; i_40427++) {
                ((__global float *) mem_39670)[sext_i32_i64(i_40427) *
                                               sext_i32_i64(s_31037) +
                                               sext_i32_i64(gtid_33961)] =
                    ((__global
                      float *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_33962) +
                                                        sext_i32_i64(i_40427) *
                                                        sext_i32_i64(num_groups_34084 *
                                                        segmap_group_sizze_34083)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_34083
}
__kernel void exactKnnFixKzisegmap_34321(__global int *global_failure,
                                         int failure_is_an_option, __global
                                         int *global_failure_args,
                                         int32_t d_31022, int32_t n_31026,
                                         int32_t s_31037,
                                         int32_t num_leaves_31075,
                                         int32_t ppl_31090,
                                         int32_t num_groups_34443, __global
                                         unsigned char *mem_39444, __global
                                         unsigned char *mem_39451, __global
                                         unsigned char *mem_39504, __global
                                         unsigned char *mem_39826, __global
                                         unsigned char *mem_39830, __global
                                         unsigned char *mem_39868, __global
                                         unsigned char *mem_39978, __global
                                         unsigned char *mem_39982, __global
                                         unsigned char *double_buffer_mem_40110,
                                         __global
                                         unsigned char *double_buffer_mem_40111)
{
    #define segmap_group_sizze_34442 (exactKnnFixKzisegmap_group_sizze_34324)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40462;
    int32_t local_tid_40463;
    int32_t group_sizze_40466;
    int32_t wave_sizze_40465;
    int32_t group_tid_40464;
    
    global_tid_40462 = get_global_id(0);
    local_tid_40463 = get_local_id(0);
    group_sizze_40466 = get_local_size(0);
    wave_sizze_40465 = LOCKSTEP_WIDTH;
    group_tid_40464 = get_group_id(0);
    
    int32_t phys_tid_34321;
    
    phys_tid_34321 = global_tid_40462;
    
    int32_t phys_group_id_40467;
    
    phys_group_id_40467 = get_group_id(0);
    for (int32_t i_40468 = 0; i_40468 < sdiv_up32(sdiv_up32(s_31037,
                                                            segmap_group_sizze_34442) -
                                                  phys_group_id_40467,
                                                  num_groups_34443);
         i_40468++) {
        int32_t virt_group_id_40469 = phys_group_id_40467 + i_40468 *
                num_groups_34443;
        int32_t gtid_34320 = sext_i64_i32(sext_i32_i64(virt_group_id_40469) *
                sext_i32_i64(segmap_group_sizze_34442) +
                sext_i32_i64(local_tid_40463));
        
        if (slt32(gtid_34320, s_31037)) {
            int32_t x_34450 = ((__global
                                int32_t *) mem_39504)[sext_i32_i64(gtid_34320)];
            bool cond_34451 = slt32(x_34450, num_leaves_31075);
            int32_t count_34452 = btoi_bool_i32(cond_34451);
            bool loop_nonempty_34453 = slt32(0, count_34452);
            int32_t bruteForcePar_arg_34454 = mul32(ppl_31090, x_34450);
            bool x_34455 = sle32(0, x_34450);
            bool bounds_check_34456 = cond_34451 && x_34455;
            bool loop_not_taken_34457 = !loop_nonempty_34453;
            bool protect_assert_disj_34458 = bounds_check_34456 ||
                 loop_not_taken_34457;
            bool index_certs_34459;
            
            if (!protect_assert_disj_34458) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 42) ==
                        -1) {
                        global_failure_args[0] = x_34450;
                        global_failure_args[1] = num_leaves_31075;
                        ;
                    }
                    local_failure = true;
                    goto error_0;
                }
            }
            
            int32_t mem_39863[8];
            float mem_39865[8];
            int32_t mem_39920[8];
            float mem_39922[8];
            
            for (int32_t i_40470 = 0; i_40470 < 8; i_40470++) {
                ((__global
                  int32_t *) double_buffer_mem_40110)[sext_i32_i64(phys_tid_34321) +
                                                      sext_i32_i64(i_40470) *
                                                      sext_i32_i64(num_groups_34443 *
                                                      segmap_group_sizze_34442)] =
                    ((__global int32_t *) mem_39826)[sext_i32_i64(gtid_34320) +
                                                     sext_i32_i64(i_40470) *
                                                     sext_i32_i64(s_31037)];
            }
            for (int32_t i_40471 = 0; i_40471 < 8; i_40471++) {
                ((__global
                  float *) double_buffer_mem_40111)[sext_i32_i64(phys_tid_34321) +
                                                    sext_i32_i64(i_40471) *
                                                    sext_i32_i64(num_groups_34443 *
                                                    segmap_group_sizze_34442)] =
                    ((__global float *) mem_39830)[sext_i32_i64(gtid_34320) +
                                                   sext_i32_i64(i_40471) *
                                                   sext_i32_i64(s_31037)];
            }
            for (int32_t _j_34462 = 0; _j_34462 < count_34452; _j_34462++) {
                for (int32_t i_40474 = 0; i_40474 < 8; i_40474++) {
                    mem_39863[sext_i32_i64(i_40474)] = ((__global
                                                         int32_t *) double_buffer_mem_40110)[sext_i32_i64(phys_tid_34321) +
                                                                                             sext_i32_i64(i_40474) *
                                                                                             sext_i32_i64(num_groups_34443 *
                                                                                             segmap_group_sizze_34442)];
                }
                for (int32_t i_40475 = 0; i_40475 < 8; i_40475++) {
                    mem_39865[sext_i32_i64(i_40475)] = ((__global
                                                         float *) double_buffer_mem_40111)[sext_i32_i64(phys_tid_34321) +
                                                                                           sext_i32_i64(i_40475) *
                                                                                           sext_i32_i64(num_groups_34443 *
                                                                                           segmap_group_sizze_34442)];
                }
                for (int32_t i_39362 = 0; i_39362 < ppl_31090; i_39362++) {
                    float res_34470;
                    float res_34472 = 0.0F;
                    float x_34473;
                    float y_34474;
                    
                    for (int32_t i_34471 = 0; i_34471 < d_31022; i_34471++) {
                        x_34473 = ((__global
                                    float *) mem_39444)[sext_i32_i64(gtid_34320) +
                                                        sext_i32_i64(i_34471) *
                                                        sext_i32_i64(n_31026)];
                        y_34474 = ((__global
                                    float *) mem_39451)[sext_i32_i64(i_39362 *
                                                        (num_leaves_31075 *
                                                         d_31022) + x_34450) +
                                                        sext_i32_i64(i_34471) *
                                                        sext_i32_i64(num_leaves_31075)];
                        
                        float zz_34475;
                        
                        zz_34475 = x_34473 - y_34474;
                        
                        float y_34476 = zz_34475 * zz_34475;
                        float loopres_34477 = res_34472 + y_34476;
                        float res_tmp_40477 = loopres_34477;
                        
                        res_34472 = res_tmp_40477;
                    }
                    res_34470 = res_34472;
                    ((__global
                      float *) mem_39868)[sext_i32_i64(phys_tid_34321) +
                                          sext_i32_i64(i_39362) *
                                          sext_i32_i64(num_groups_34443 *
                                          segmap_group_sizze_34442)] =
                        res_34470;
                }
                
                bool knn_34478;
                int32_t knn_34482;
                bool loop_while_34483;
                int32_t j_34487;
                
                loop_while_34483 = 1;
                j_34487 = 0;
                while (loop_while_34483) {
                    int32_t res_34488;
                    float res_34489;
                    int32_t redout_39364;
                    float redout_39365;
                    
                    redout_39364 = ppl_31090;
                    redout_39365 = INFINITY;
                    for (int32_t i_39366 = 0; i_39366 < ppl_31090; i_39366++) {
                        float x_34503 = ((__global
                                          float *) mem_39868)[sext_i32_i64(phys_tid_34321) +
                                                              sext_i32_i64(i_39366) *
                                                              sext_i32_i64(num_groups_34443 *
                                                              segmap_group_sizze_34442)];
                        bool cond_34494 = redout_39365 < x_34503;
                        int32_t res_34495;
                        float res_34496;
                        
                        if (cond_34494) {
                            res_34495 = redout_39364;
                            res_34496 = redout_39365;
                        } else {
                            bool cond_34497 = x_34503 < redout_39365;
                            float res_34498;
                            
                            if (cond_34497) {
                                res_34498 = x_34503;
                            } else {
                                res_34498 = redout_39365;
                            }
                            
                            int32_t res_34499;
                            
                            if (cond_34497) {
                                res_34499 = i_39366;
                            } else {
                                bool cond_34500 = sle32(redout_39364, i_39366);
                                int32_t res_34501;
                                
                                if (cond_34500) {
                                    res_34501 = redout_39364;
                                } else {
                                    res_34501 = i_39366;
                                }
                                res_34499 = res_34501;
                            }
                            res_34495 = res_34499;
                            res_34496 = res_34498;
                        }
                        
                        int32_t redout_tmp_40483 = res_34495;
                        float redout_tmp_40484 = res_34496;
                        
                        redout_39364 = redout_tmp_40483;
                        redout_39365 = redout_tmp_40484;
                    }
                    res_34488 = redout_39364;
                    res_34489 = redout_39365;
                    
                    int32_t i_34504 = sub32(7, j_34487);
                    bool x_34505 = sle32(0, i_34504);
                    bool y_34506 = slt32(i_34504, 8);
                    bool bounds_check_34507 = x_34505 && y_34506;
                    bool index_certs_34508;
                    
                    if (!bounds_check_34507) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          43) == -1) {
                                global_failure_args[0] = i_34504;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_34509 = mem_39865[sext_i32_i64(i_34504)];
                    bool cond_34510 = res_34489 < y_34509;
                    int32_t loopres_34514;
                    
                    if (cond_34510) {
                        bool x_34515 = sle32(0, res_34488);
                        bool y_34516 = slt32(res_34488, ppl_31090);
                        bool bounds_check_34517 = x_34515 && y_34516;
                        bool index_certs_34518;
                        
                        if (!bounds_check_34517) {
                            {
                                if (atomic_cmpxchg_i32_global(global_failure,
                                                              -1, 44) == -1) {
                                    global_failure_args[0] = res_34488;
                                    global_failure_args[1] = ppl_31090;
                                    ;
                                }
                                local_failure = true;
                                goto error_0;
                            }
                        }
                        ((__global
                          float *) mem_39868)[sext_i32_i64(phys_tid_34321) +
                                              sext_i32_i64(res_34488) *
                                              sext_i32_i64(num_groups_34443 *
                                              segmap_group_sizze_34442)] =
                            INFINITY;
                        
                        int32_t lw_val_34520 = add32(bruteForcePar_arg_34454,
                                                     res_34488);
                        
                        mem_39863[sext_i32_i64(i_34504)] = lw_val_34520;
                        mem_39865[sext_i32_i64(i_34504)] = res_34489;
                        
                        int32_t res_34523 = add32(1, j_34487);
                        
                        loopres_34514 = res_34523;
                    } else {
                        loopres_34514 = j_34487;
                    }
                    
                    bool res_34524 = slt32(loopres_34514, 8);
                    bool x_34525 = cond_34510 && res_34524;
                    bool loop_while_tmp_40478 = x_34525;
                    int32_t j_tmp_40482 = loopres_34514;
                    
                    loop_while_34483 = loop_while_tmp_40478;
                    j_34487 = j_tmp_40482;
                }
                knn_34478 = loop_while_34483;
                knn_34482 = j_34487;
                for (int32_t i_40485 = 0; i_40485 < 8; i_40485++) {
                    mem_39920[sext_i32_i64(i_40485)] = -1;
                }
                for (int32_t i_40486 = 0; i_40486 < 8; i_40486++) {
                    mem_39922[sext_i32_i64(i_40486)] = INFINITY;
                }
                
                int32_t res_34530;
                int32_t res_34531;
                int32_t beg_34535;
                int32_t end_34536;
                
                beg_34535 = 0;
                end_34536 = 7;
                for (int32_t i_34532 = 0; i_34532 < 8; i_34532++) {
                    bool x_34537 = sle32(0, beg_34535);
                    bool y_34538 = slt32(beg_34535, 8);
                    bool bounds_check_34539 = x_34537 && y_34538;
                    bool index_certs_34540;
                    
                    if (!bounds_check_34539) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          45) == -1) {
                                global_failure_args[0] = beg_34535;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float x_34541 = mem_39865[sext_i32_i64(beg_34535)];
                    bool x_34542 = sle32(0, end_34536);
                    bool y_34543 = slt32(end_34536, 8);
                    bool bounds_check_34544 = x_34542 && y_34543;
                    bool index_certs_34545;
                    
                    if (!bounds_check_34544) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          46) == -1) {
                                global_failure_args[0] = end_34536;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_34546 = mem_39865[sext_i32_i64(end_34536)];
                    bool cond_34547 = x_34541 < y_34546;
                    float loopres_34548;
                    
                    if (cond_34547) {
                        loopres_34548 = x_34541;
                    } else {
                        loopres_34548 = y_34546;
                    }
                    
                    int32_t loopres_34549;
                    int32_t loopres_34550;
                    int32_t loopres_34551;
                    
                    if (cond_34547) {
                        int32_t res_34552 = mem_39863[sext_i32_i64(beg_34535)];
                        int32_t res_34553 = add32(1, beg_34535);
                        
                        loopres_34549 = res_34552;
                        loopres_34550 = res_34553;
                        loopres_34551 = end_34536;
                    } else {
                        int32_t res_34554 = mem_39863[sext_i32_i64(end_34536)];
                        int32_t res_34555 = sub32(end_34536, 1);
                        
                        loopres_34549 = res_34554;
                        loopres_34550 = beg_34535;
                        loopres_34551 = res_34555;
                    }
                    mem_39920[sext_i32_i64(i_34532)] = loopres_34549;
                    mem_39922[sext_i32_i64(i_34532)] = loopres_34548;
                    
                    int32_t beg_tmp_40489 = loopres_34550;
                    int32_t end_tmp_40490 = loopres_34551;
                    
                    beg_34535 = beg_tmp_40489;
                    end_34536 = end_tmp_40490;
                }
                res_34530 = beg_34535;
                res_34531 = end_34536;
                for (int32_t i_40491 = 0; i_40491 < 8; i_40491++) {
                    ((__global
                      int32_t *) double_buffer_mem_40110)[sext_i32_i64(phys_tid_34321) +
                                                          sext_i32_i64(i_40491) *
                                                          sext_i32_i64(num_groups_34443 *
                                                          segmap_group_sizze_34442)] =
                        mem_39920[sext_i32_i64(i_40491)];
                }
                for (int32_t i_40492 = 0; i_40492 < 8; i_40492++) {
                    ((__global
                      float *) double_buffer_mem_40111)[sext_i32_i64(phys_tid_34321) +
                                                        sext_i32_i64(i_40492) *
                                                        sext_i32_i64(num_groups_34443 *
                                                        segmap_group_sizze_34442)] =
                        mem_39922[sext_i32_i64(i_40492)];
                }
            }
            for (int32_t i_40493 = 0; i_40493 < 8; i_40493++) {
                ((__global int32_t *) mem_39978)[sext_i32_i64(i_40493) *
                                                 sext_i32_i64(s_31037) +
                                                 sext_i32_i64(gtid_34320)] =
                    ((__global
                      int32_t *) double_buffer_mem_40110)[sext_i32_i64(phys_tid_34321) +
                                                          sext_i32_i64(i_40493) *
                                                          sext_i32_i64(num_groups_34443 *
                                                          segmap_group_sizze_34442)];
            }
            for (int32_t i_40494 = 0; i_40494 < 8; i_40494++) {
                ((__global float *) mem_39982)[sext_i32_i64(i_40494) *
                                               sext_i32_i64(s_31037) +
                                               sext_i32_i64(gtid_34320)] =
                    ((__global
                      float *) double_buffer_mem_40111)[sext_i32_i64(phys_tid_34321) +
                                                        sext_i32_i64(i_40494) *
                                                        sext_i32_i64(num_groups_34443 *
                                                        segmap_group_sizze_34442)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_34442
}
__kernel void exactKnnFixKzisegmap_intragroup_33960(__global
                                                    int *global_failure,
                                                    int failure_is_an_option,
                                                    __global
                                                    int *global_failure_args,
                                                    uint red_arr_mem_40448_backing_offset_0,
                                                    uint red_arr_mem_40446_backing_offset_1,
                                                    uint mem_39762_backing_offset_2,
                                                    uint mem_39760_backing_offset_3,
                                                    uint mem_39717_backing_offset_4,
                                                    uint mem_39713_backing_offset_5,
                                                    uint mem_39711_backing_offset_6,
                                                    int32_t d_31022,
                                                    int32_t n_31026,
                                                    int32_t s_31037,
                                                    int32_t num_leaves_31075,
                                                    int32_t ppl_31090, __global
                                                    unsigned char *ref_pts_mem_39418,
                                                    __global
                                                    unsigned char *mem_39444,
                                                    __global
                                                    unsigned char *mem_39504,
                                                    __global
                                                    unsigned char *mem_39674,
                                                    __global
                                                    unsigned char *mem_39678,
                                                    __global
                                                    unsigned char *mem_39818,
                                                    __global
                                                    unsigned char *mem_39822,
                                                    __global
                                                    unsigned char *double_buffer_mem_40104,
                                                    __global
                                                    unsigned char *double_buffer_mem_40105)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40448_backing_6 =
                  &shared_mem[red_arr_mem_40448_backing_offset_0];
    volatile char *red_arr_mem_40446_backing_5 =
                  &shared_mem[red_arr_mem_40446_backing_offset_1];
    volatile char *mem_39762_backing_4 =
                  &shared_mem[mem_39762_backing_offset_2];
    volatile char *mem_39760_backing_3 =
                  &shared_mem[mem_39760_backing_offset_3];
    volatile char *mem_39717_backing_2 =
                  &shared_mem[mem_39717_backing_offset_4];
    volatile char *mem_39713_backing_1 =
                  &shared_mem[mem_39713_backing_offset_5];
    volatile char *mem_39711_backing_0 =
                  &shared_mem[mem_39711_backing_offset_6];
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40428;
    int32_t local_tid_40429;
    int32_t group_sizze_40432;
    int32_t wave_sizze_40431;
    int32_t group_tid_40430;
    
    global_tid_40428 = get_global_id(0);
    local_tid_40429 = get_local_id(0);
    group_sizze_40432 = get_local_size(0);
    wave_sizze_40431 = LOCKSTEP_WIDTH;
    group_tid_40430 = get_group_id(0);
    
    int32_t phys_tid_33960;
    
    phys_tid_33960 = group_tid_40430;
    
    int32_t ltid_pre_40433;
    
    ltid_pre_40433 = local_tid_40429;
    
    int32_t gtid_33944;
    
    gtid_33944 = group_tid_40430;
    
    int32_t x_34209 = ((__global
                        int32_t *) mem_39504)[sext_i32_i64(gtid_33944)];
    bool cond_34210 = slt32(x_34209, num_leaves_31075);
    int32_t count_34211 = btoi_bool_i32(cond_34210);
    bool loop_nonempty_34212 = slt32(0, count_34211);
    int32_t bruteForcePar_arg_34213 = mul32(ppl_31090, x_34209);
    bool x_34214 = sle32(0, x_34209);
    bool bounds_check_34215 = cond_34210 && x_34214;
    bool loop_not_taken_34216 = !loop_nonempty_34212;
    bool protect_assert_disj_34217 = bounds_check_34215 || loop_not_taken_34216;
    bool index_certs_34218;
    
    if (!protect_assert_disj_34217) {
        {
            if (atomic_cmpxchg_i32_global(global_failure, -1, 37) == -1) {
                global_failure_args[0] = x_34209;
                global_failure_args[1] = num_leaves_31075;
                ;
            }
            local_failure = true;
            goto error_0;
        }
    }
    
    __local char *mem_39711;
    
    mem_39711 = (__local char *) mem_39711_backing_0;
    
    __local char *mem_39713;
    
    mem_39713 = (__local char *) mem_39713_backing_1;
    
    __local char *mem_39717;
    
    mem_39717 = (__local char *) mem_39717_backing_2;
    
    __local char *mem_39760;
    
    mem_39760 = (__local char *) mem_39760_backing_3;
    
    __local char *mem_39762;
    
    mem_39762 = (__local char *) mem_39762_backing_4;
    for (int32_t i_40434 = 0; i_40434 < sdiv_up32(8 - local_tid_40429,
                                                  ppl_31090); i_40434++) {
        ((__global
          int32_t *) double_buffer_mem_40104)[sext_i32_i64(phys_tid_33960 * 8) +
                                              sext_i32_i64(i_40434 * ppl_31090 +
                                              local_tid_40429)] = ((__global
                                                                    int32_t *) mem_39674)[sext_i32_i64(gtid_33944) +
                                                                                          sext_i32_i64(i_40434 *
                                                                                          ppl_31090 +
                                                                                          local_tid_40429) *
                                                                                          sext_i32_i64(s_31037)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_40435 = 0; i_40435 < sdiv_up32(8 - local_tid_40429,
                                                  ppl_31090); i_40435++) {
        ((__global
          float *) double_buffer_mem_40105)[sext_i32_i64(phys_tid_33960 * 8) +
                                            sext_i32_i64(i_40435 * ppl_31090 +
                                            local_tid_40429)] = ((__global
                                                                  float *) mem_39678)[sext_i32_i64(gtid_33944) +
                                                                                      sext_i32_i64(i_40435 *
                                                                                      ppl_31090 +
                                                                                      local_tid_40429) *
                                                                                      sext_i32_i64(s_31037)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t _j_34221 = 0; _j_34221 < count_34211; _j_34221++) {
        for (int32_t i_40438 = 0; i_40438 < sdiv_up32(8 - local_tid_40429,
                                                      ppl_31090); i_40438++) {
            ((__local int32_t *) mem_39711)[sext_i32_i64(i_40438 * ppl_31090 +
                                            local_tid_40429)] = ((__global
                                                                  int32_t *) double_buffer_mem_40104)[sext_i32_i64(phys_tid_33960 *
                                                                                                      8) +
                                                                                                      sext_i32_i64(i_40438 *
                                                                                                      ppl_31090 +
                                                                                                      local_tid_40429)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_40439 = 0; i_40439 < sdiv_up32(8 - local_tid_40429,
                                                      ppl_31090); i_40439++) {
            ((__local float *) mem_39713)[sext_i32_i64(i_40439 * ppl_31090 +
                                          local_tid_40429)] = ((__global
                                                                float *) double_buffer_mem_40105)[sext_i32_i64(phys_tid_33960 *
                                                                                                  8) +
                                                                                                  sext_i32_i64(i_40439 *
                                                                                                  ppl_31090 +
                                                                                                  local_tid_40429)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t gtid_33947 = ltid_pre_40433;
        int32_t phys_tid_33948 = local_tid_40429;
        
        if (slt32(gtid_33947, ppl_31090)) {
            float res_34229;
            float res_34231 = 0.0F;
            float x_34232;
            float y_34233;
            
            for (int32_t i_34230 = 0; i_34230 < d_31022; i_34230++) {
                x_34232 = ((__global
                            float *) mem_39444)[sext_i32_i64(gtid_33944) +
                                                sext_i32_i64(i_34230) *
                                                sext_i32_i64(n_31026)];
                y_34233 = ((__global
                            float *) ref_pts_mem_39418)[sext_i32_i64(x_34209 *
                                                        (d_31022 * ppl_31090) +
                                                        gtid_33947 * d_31022) +
                                                        sext_i32_i64(i_34230)];
                
                float zz_34234;
                
                zz_34234 = x_34232 - y_34233;
                
                float y_34235 = zz_34234 * zz_34234;
                float loopres_34236 = res_34231 + y_34235;
                float res_tmp_40440 = loopres_34236;
                
                res_34231 = res_tmp_40440;
            }
            res_34229 = res_34231;
            ((__local float *) mem_39717)[sext_i32_i64(gtid_33947)] = res_34229;
        }
        
      error_0:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool knn_34237;
        int32_t knn_34241;
        bool loop_while_34242;
        int32_t j_34246;
        
        loop_while_34242 = 1;
        j_34246 = 0;
        while (loop_while_34242) {
            int32_t res_34247;
            float res_34248;
            int32_t gtid_33958 = ltid_pre_40433;
            int32_t phys_tid_33959 = local_tid_40429;
            __local char *red_arr_mem_40446;
            
            red_arr_mem_40446 = (__local char *) red_arr_mem_40446_backing_5;
            
            __local char *red_arr_mem_40448;
            
            red_arr_mem_40448 = (__local char *) red_arr_mem_40448_backing_6;
            if (slt32(gtid_33958, ppl_31090)) {
                float x_34262 = ((__local
                                  float *) mem_39717)[sext_i32_i64(gtid_33958)];
                
                ((__local
                  int32_t *) red_arr_mem_40446)[sext_i32_i64(gtid_33958)] =
                    gtid_33958;
                ((__local
                  float *) red_arr_mem_40448)[sext_i32_i64(gtid_33958)] =
                    x_34262;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t offset_40450;
            int32_t skip_waves_40451;
            int32_t x_34249;
            float x_34250;
            int32_t x_34251;
            float x_34252;
            
            offset_40450 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_40429, ppl_31090)) {
                    x_34249 = ((__local
                                int32_t *) red_arr_mem_40446)[sext_i32_i64(local_tid_40429 +
                                                              offset_40450)];
                    x_34250 = ((__local
                                float *) red_arr_mem_40448)[sext_i32_i64(local_tid_40429 +
                                                            offset_40450)];
                }
            }
            offset_40450 = 1;
            while (slt32(offset_40450, wave_sizze_40431)) {
                if (slt32(local_tid_40429 + offset_40450, ppl_31090) &&
                    ((local_tid_40429 - squot32(local_tid_40429,
                                                wave_sizze_40431) *
                      wave_sizze_40431) & (2 * offset_40450 - 1)) == 0) {
                    // read array element
                    {
                        x_34251 = ((volatile __local
                                    int32_t *) red_arr_mem_40446)[sext_i32_i64(local_tid_40429 +
                                                                  offset_40450)];
                        x_34252 = ((volatile __local
                                    float *) red_arr_mem_40448)[sext_i32_i64(local_tid_40429 +
                                                                offset_40450)];
                    }
                    // apply reduction operation
                    {
                        bool cond_34253 = x_34250 < x_34252;
                        int32_t res_34254;
                        float res_34255;
                        
                        if (cond_34253) {
                            res_34254 = x_34249;
                            res_34255 = x_34250;
                        } else {
                            bool cond_34256 = x_34252 < x_34250;
                            float res_34257;
                            
                            if (cond_34256) {
                                res_34257 = x_34252;
                            } else {
                                res_34257 = x_34250;
                            }
                            
                            int32_t res_34258;
                            
                            if (cond_34256) {
                                res_34258 = x_34251;
                            } else {
                                bool cond_34259 = sle32(x_34249, x_34251);
                                int32_t res_34260;
                                
                                if (cond_34259) {
                                    res_34260 = x_34249;
                                } else {
                                    res_34260 = x_34251;
                                }
                                res_34258 = res_34260;
                            }
                            res_34254 = res_34258;
                            res_34255 = res_34257;
                        }
                        x_34249 = res_34254;
                        x_34250 = res_34255;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int32_t *) red_arr_mem_40446)[sext_i32_i64(local_tid_40429)] =
                            x_34249;
                        ((volatile __local
                          float *) red_arr_mem_40448)[sext_i32_i64(local_tid_40429)] =
                            x_34250;
                    }
                }
                offset_40450 *= 2;
            }
            skip_waves_40451 = 1;
            while (slt32(skip_waves_40451, squot32(ppl_31090 +
                                                   wave_sizze_40431 - 1,
                                                   wave_sizze_40431))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_40450 = skip_waves_40451 * wave_sizze_40431;
                if (slt32(local_tid_40429 + offset_40450, ppl_31090) &&
                    ((local_tid_40429 - squot32(local_tid_40429,
                                                wave_sizze_40431) *
                      wave_sizze_40431) == 0 && (squot32(local_tid_40429,
                                                         wave_sizze_40431) &
                                                 (2 * skip_waves_40451 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_34251 = ((__local
                                    int32_t *) red_arr_mem_40446)[sext_i32_i64(local_tid_40429 +
                                                                  offset_40450)];
                        x_34252 = ((__local
                                    float *) red_arr_mem_40448)[sext_i32_i64(local_tid_40429 +
                                                                offset_40450)];
                    }
                    // apply reduction operation
                    {
                        bool cond_34253 = x_34250 < x_34252;
                        int32_t res_34254;
                        float res_34255;
                        
                        if (cond_34253) {
                            res_34254 = x_34249;
                            res_34255 = x_34250;
                        } else {
                            bool cond_34256 = x_34252 < x_34250;
                            float res_34257;
                            
                            if (cond_34256) {
                                res_34257 = x_34252;
                            } else {
                                res_34257 = x_34250;
                            }
                            
                            int32_t res_34258;
                            
                            if (cond_34256) {
                                res_34258 = x_34251;
                            } else {
                                bool cond_34259 = sle32(x_34249, x_34251);
                                int32_t res_34260;
                                
                                if (cond_34259) {
                                    res_34260 = x_34249;
                                } else {
                                    res_34260 = x_34251;
                                }
                                res_34258 = res_34260;
                            }
                            res_34254 = res_34258;
                            res_34255 = res_34257;
                        }
                        x_34249 = res_34254;
                        x_34250 = res_34255;
                    }
                    // write result of operation
                    {
                        ((__local
                          int32_t *) red_arr_mem_40446)[sext_i32_i64(local_tid_40429)] =
                            x_34249;
                        ((__local
                          float *) red_arr_mem_40448)[sext_i32_i64(local_tid_40429)] =
                            x_34250;
                    }
                }
                skip_waves_40451 *= 2;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            res_34247 = ((__local int32_t *) red_arr_mem_40446)[0];
            res_34248 = ((__local float *) red_arr_mem_40448)[0];
            
            int32_t i_34263 = sub32(7, j_34246);
            bool x_34264 = sle32(0, i_34263);
            bool y_34265 = slt32(i_34263, 8);
            bool bounds_check_34266 = x_34264 && y_34265;
            bool index_certs_34267;
            
            if (!bounds_check_34266) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 38) ==
                        -1) {
                        global_failure_args[0] = i_34263;
                        global_failure_args[1] = 8;
                        ;
                    }
                    local_failure = true;
                    goto error_3;
                }
            }
            
            float y_34268 = ((__local
                              float *) mem_39713)[sext_i32_i64(i_34263)];
            bool cond_34269 = res_34248 < y_34268;
            int32_t loopres_34273;

            barrier(CLK_LOCAL_MEM_FENCE); // COSMIN 2
            barrier(CLK_GLOBAL_MEM_FENCE);
            
            if (cond_34269) {
                bool x_34274 = sle32(0, res_34247);
                bool y_34275 = slt32(res_34247, ppl_31090);
                bool bounds_check_34276 = x_34274 && y_34275;
                bool index_certs_34277;
                
                if (!bounds_check_34276) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 39) ==
                            -1) {
                            global_failure_args[0] = res_34247;
                            global_failure_args[1] = ppl_31090;
                            ;
                        }
                        local_failure = true;
                        goto error_3;
                    }
                }
                //barrier(CLK_LOCAL_MEM_FENCE);
                if (local_tid_40429 == 0) {
                    ((__local float *) mem_39717)[sext_i32_i64(res_34247)] =
                        INFINITY;
                }
                //barrier(CLK_LOCAL_MEM_FENCE);
                
                int32_t lw_val_34279 = add32(bruteForcePar_arg_34213,
                                             res_34247);
                
                //barrier(CLK_LOCAL_MEM_FENCE);
                if (local_tid_40429 == 0) {
                    ((__local int32_t *) mem_39711)[sext_i32_i64(i_34263)] =
                        lw_val_34279;
                }
                //barrier(CLK_LOCAL_MEM_FENCE);
                //barrier(CLK_LOCAL_MEM_FENCE);
                if (local_tid_40429 == 0) {
                    ((__local float *) mem_39713)[sext_i32_i64(i_34263)] =
                        res_34248;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                int32_t res_34282 = add32(1, j_34246);
                
                loopres_34273 = res_34282;
            } else {
                loopres_34273 = j_34246;
            }
            
            bool res_34283 = slt32(loopres_34273, 8);
            bool x_34284 = cond_34269 && res_34283;
            bool loop_while_tmp_40441 = x_34284;
            int32_t j_tmp_40445 = loopres_34273;
            
            loop_while_34242 = loop_while_tmp_40441;
            j_34246 = j_tmp_40445;
        }
        
        knn_34237 = loop_while_34242;
        knn_34241 = j_34246;
        for (int32_t i_40452 = 0; i_40452 < sdiv_up32(8 - local_tid_40429,
                                                      ppl_31090); i_40452++) {
            ((__local int32_t *) mem_39760)[sext_i32_i64(i_40452 * ppl_31090 +
                                            local_tid_40429)] = -1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_40453 = 0; i_40453 < sdiv_up32(8 - local_tid_40429,
                                                      ppl_31090); i_40453++) {
            ((__local float *) mem_39762)[sext_i32_i64(i_40453 * ppl_31090 +
                                          local_tid_40429)] = INFINITY;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t res_34289;
        int32_t res_34290;
        int32_t beg_34294;
        int32_t end_34295;
        
        beg_34294 = 0;
        end_34295 = 7;
        for (int32_t i_34291 = 0; i_34291 < 8; i_34291++) {
            bool x_34296 = sle32(0, beg_34294);
            bool y_34297 = slt32(beg_34294, 8);
            bool bounds_check_34298 = x_34296 && y_34297;
            bool index_certs_34299;
            
            if (!bounds_check_34298) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 40) ==
                        -1) {
                        global_failure_args[0] = beg_34294;
                        global_failure_args[1] = 8;
                        ;
                    }
                    local_failure = true;
                    goto error_3;
                }
            }
            
            float x_34300 = ((__local
                              float *) mem_39713)[sext_i32_i64(beg_34294)];
            bool x_34301 = sle32(0, end_34295);
            bool y_34302 = slt32(end_34295, 8);
            bool bounds_check_34303 = x_34301 && y_34302;
            bool index_certs_34304;
            
            if (!bounds_check_34303) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 41) ==
                        -1) {
                        global_failure_args[0] = end_34295;
                        global_failure_args[1] = 8;
                        ;
                    }
                    local_failure = true;
                    goto error_3;
                }
            }
            
            float y_34305 = ((__local
                              float *) mem_39713)[sext_i32_i64(end_34295)];
            bool cond_34306 = x_34300 < y_34305;
            float loopres_34307;
            
            if (cond_34306) {
                loopres_34307 = x_34300;
            } else {
                loopres_34307 = y_34305;
            }
            
            int32_t loopres_34308;
            int32_t loopres_34309;
            int32_t loopres_34310;
            
            if (cond_34306) {
                int32_t res_34311 = ((__local
                                      int32_t *) mem_39711)[sext_i32_i64(beg_34294)];
                int32_t res_34312 = add32(1, beg_34294);
                
                loopres_34308 = res_34311;
                loopres_34309 = res_34312;
                loopres_34310 = end_34295;
            } else {
                int32_t res_34313 = ((__local
                                      int32_t *) mem_39711)[sext_i32_i64(end_34295)];
                int32_t res_34314 = sub32(end_34295, 1);
                
                loopres_34308 = res_34313;
                loopres_34309 = beg_34294;
                loopres_34310 = res_34314;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_tid_40429 == 0) {
                ((__local int32_t *) mem_39760)[sext_i32_i64(i_34291)] =
                    loopres_34308;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_tid_40429 == 0) {
                ((__local float *) mem_39762)[sext_i32_i64(i_34291)] =
                    loopres_34307;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t beg_tmp_40456 = loopres_34309;
            int32_t end_tmp_40457 = loopres_34310;
            
            beg_34294 = beg_tmp_40456;
            end_34295 = end_tmp_40457;
        }
        res_34289 = beg_34294;
        res_34290 = end_34295;
        for (int32_t i_40458 = 0; i_40458 < sdiv_up32(8 - local_tid_40429,
                                                      ppl_31090); i_40458++) {
            ((__global
              int32_t *) double_buffer_mem_40104)[sext_i32_i64(phys_tid_33960 *
                                                  8) + sext_i32_i64(i_40458 *
                                                  ppl_31090 +
                                                  local_tid_40429)] = ((__local
                                                                        int32_t *) mem_39760)[sext_i32_i64(i_40458 *
                                                                                              ppl_31090 +
                                                                                              local_tid_40429)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_40459 = 0; i_40459 < sdiv_up32(8 - local_tid_40429,
                                                      ppl_31090); i_40459++) {
            ((__global
              float *) double_buffer_mem_40105)[sext_i32_i64(phys_tid_33960 *
                                                8) + sext_i32_i64(i_40459 *
                                                ppl_31090 + local_tid_40429)] =
                ((__local float *) mem_39762)[sext_i32_i64(i_40459 * ppl_31090 +
                                              local_tid_40429)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_tid_40429 == 0) {
        for (int32_t i_40460 = 0; i_40460 < 8; i_40460++) {
            ((__global int32_t *) mem_39818)[sext_i32_i64(gtid_33944) * 8 +
                                             sext_i32_i64(i_40460)] = ((__global
                                                                        int32_t *) double_buffer_mem_40104)[sext_i32_i64(phys_tid_33960 *
                                                                                                            8) +
                                                                                                            sext_i32_i64(i_40460)];
        }
    }
    if (local_tid_40429 == 0) {
        for (int32_t i_40461 = 0; i_40461 < 8; i_40461++) {
            ((__global float *) mem_39822)[sext_i32_i64(gtid_33944) * 8 +
                                           sext_i32_i64(i_40461)] = ((__global
                                                                      float *) double_buffer_mem_40105)[sext_i32_i64(phys_tid_33960 *
                                                                                                        8) +
                                                                                                        sext_i32_i64(i_40461)];
        }
    }
    
  error_3:
    return;
}
__kernel void exactKnnFixKzisegred_nonseg_33941(__global int *global_failure,
                                                int failure_is_an_option,
                                                __global
                                                int *global_failure_args,
                                                uint red_arr_mem_40361_backing_offset_0,
                                                uint sync_arr_mem_40359_backing_offset_1,
                                                int32_t d_31022,
                                                int32_t q_31023,
                                                int32_t n_31026,
                                                int32_t s_31037,
                                                int32_t num_leaves_31075,
                                                int32_t res_31079,
                                                int32_t h_31086,
                                                int32_t num_groups_33933,
                                                __global
                                                unsigned char *median_dims_mem_39419,
                                                __global
                                                unsigned char *median_vals_mem_39420,
                                                __global
                                                unsigned char *prev_eqdims_mem_39421,
                                                __global
                                                unsigned char *mem_39444,
                                                __global
                                                unsigned char *mem_param_39467,
                                                __global
                                                unsigned char *mem_param_39472,
                                                __global
                                                unsigned char *mem_param_39477,
                                                __global
                                                unsigned char *mem_param_39482,
                                                __global
                                                unsigned char *mem_39501,
                                                __global
                                                unsigned char *mem_39504,
                                                __global
                                                unsigned char *mem_39507,
                                                __global
                                                unsigned char *mem_39510,
                                                __global
                                                unsigned char *exactKnnFixKzicounter_mem_40349,
                                                __global
                                                unsigned char *group_res_arr_mem_40351,
                                                int32_t num_threads_40353)
{
    #define segred_group_sizze_33931 (exactKnnFixKzisegred_group_sizze_33930)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40361_backing_1 =
                  &shared_mem[red_arr_mem_40361_backing_offset_0];
    volatile char *sync_arr_mem_40359_backing_0 =
                  &shared_mem[sync_arr_mem_40359_backing_offset_1];
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40354;
    int32_t local_tid_40355;
    int32_t group_sizze_40358;
    int32_t wave_sizze_40357;
    int32_t group_tid_40356;
    
    global_tid_40354 = get_global_id(0);
    local_tid_40355 = get_local_id(0);
    group_sizze_40358 = get_local_size(0);
    wave_sizze_40357 = LOCKSTEP_WIDTH;
    group_tid_40356 = get_group_id(0);
    
    int32_t phys_tid_33941;
    
    phys_tid_33941 = global_tid_40354;
    
    __local char *sync_arr_mem_40359;
    
    sync_arr_mem_40359 = (__local char *) sync_arr_mem_40359_backing_0;
    
    __local char *red_arr_mem_40361;
    
    red_arr_mem_40361 = (__local char *) red_arr_mem_40361_backing_1;
    
    int32_t dummy_33939;
    
    dummy_33939 = 0;
    
    int32_t gtid_33940;
    
    gtid_33940 = 0;
    
    int32_t x_acc_40363;
    int32_t chunk_sizze_40364;
    
    chunk_sizze_40364 = smin32(sdiv_up32(s_31037, segred_group_sizze_33931 *
                                         num_groups_33933), sdiv_up32(s_31037 -
                                                                      phys_tid_33941,
                                                                      num_threads_40353));
    
    int32_t x_31283;
    int32_t x_31284;
    
    // neutral-initialise the accumulators
    {
        x_acc_40363 = 0;
    }
    for (int32_t i_40368 = 0; i_40368 < chunk_sizze_40364; i_40368++) {
        gtid_33940 = phys_tid_33941 + num_threads_40353 * i_40368;
        // apply map function
        {
            int32_t x_31288 = ((__global
                                int32_t *) mem_param_39472)[sext_i32_i64(gtid_33940)];
            int32_t x_31289 = ((__global
                                int32_t *) mem_param_39477)[sext_i32_i64(gtid_33940)];
            float x_31290 = ((__global
                              float *) mem_param_39482)[sext_i32_i64(gtid_33940)];
            bool cond_31291 = sle32(num_leaves_31075, x_31288);
            int32_t res_31292;
            int32_t res_31293;
            float res_31294;
            
            if (cond_31291) {
                res_31292 = num_leaves_31075;
                res_31293 = x_31289;
                res_31294 = x_31290;
            } else {
                float x_31287 = ((__global
                                  float *) mem_param_39467)[sext_i32_i64(gtid_33940) *
                                                            8 + 7];
                int32_t last_leaf_31295 = add32(q_31023, x_31288);
                int32_t x_31296 = mul32(2, q_31023);
                int32_t no_leaf_31297 = add32(1, x_31296);
                bool cond_31298 = last_leaf_31295 == 0;
                bool cond_31299 = !cond_31298;
                bool res_31300;
                int32_t res_31301;
                int32_t res_31302;
                int32_t res_31303;
                float res_31304;
                int32_t res_31305;
                bool loop_while_31306;
                int32_t node_index_31307;
                int32_t stack_31308;
                int32_t count_31309;
                float dist_31310;
                int32_t rec_node_31311;
                
                loop_while_31306 = cond_31299;
                node_index_31307 = last_leaf_31295;
                stack_31308 = x_31289;
                count_31309 = h_31086;
                dist_31310 = x_31290;
                rec_node_31311 = -1;
                while (loop_while_31306) {
                    int32_t x_31312 = sub32(node_index_31307, 1);
                    int32_t res_31313 = sdiv32(x_31312, 2);
                    int32_t y_31314 = 1 << count_31309;
                    int32_t b_31315 = stack_31308 & y_31314;
                    bool res_31316 = b_31315 == 0;
                    bool res_31317 = !res_31316;
                    bool x_31318 = sle32(0, res_31313);
                    bool y_31319 = slt32(res_31313, q_31023);
                    bool bounds_check_31320 = x_31318 && y_31319;
                    bool index_certs_31321;
                    
                    if (!bounds_check_31320) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          26) == -1) {
                                global_failure_args[0] = res_31313;
                                global_failure_args[1] = q_31023;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    int32_t i_31322 = ((__global
                                        int32_t *) median_dims_mem_39419)[sext_i32_i64(res_31313)];
                    bool x_31323 = sle32(0, i_31322);
                    bool y_31324 = slt32(i_31322, d_31022);
                    bool bounds_check_31325 = x_31323 && y_31324;
                    bool index_certs_31326;
                    
                    if (!bounds_check_31325) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          27) == -1) {
                                global_failure_args[0] = i_31322;
                                global_failure_args[1] = d_31022;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float q_m_d_31327 = ((__global
                                          float *) mem_39444)[sext_i32_i64(i_31322) *
                                                              sext_i32_i64(n_31026) +
                                                              sext_i32_i64(gtid_33940)];
                    float x_31328 = ((__global
                                      float *) median_vals_mem_39420)[sext_i32_i64(res_31313)];
                    float cur_med_dst_31329 = x_31328 - q_m_d_31327;
                    float cur_med_sqr_31330 = cur_med_dst_31329 *
                          cur_med_dst_31329;
                    bool res_31331;
                    int32_t res_31332;
                    float res_31333;
                    bool loop_while_31334;
                    int32_t idx_31335;
                    float res_31336;
                    
                    loop_while_31334 = x_31318;
                    idx_31335 = res_31313;
                    res_31336 = 0.0F;
                    while (loop_while_31334) {
                        bool x_31337 = sle32(0, idx_31335);
                        bool y_31338 = slt32(idx_31335, q_31023);
                        bool bounds_check_31339 = x_31337 && y_31338;
                        bool index_certs_31340;
                        
                        if (!bounds_check_31339) {
                            {
                                if (atomic_cmpxchg_i32_global(global_failure,
                                                              -1, 28) == -1) {
                                    global_failure_args[0] = idx_31335;
                                    global_failure_args[1] = q_31023;
                                    ;
                                }
                                local_failure = true;
                                goto error_0;
                            }
                        }
                        
                        int32_t anc_31341 = ((__global
                                              int32_t *) prev_eqdims_mem_39421)[sext_i32_i64(idx_31335)];
                        bool cond_31342 = anc_31341 == -1;
                        int32_t loopres_31343;
                        float loopres_31344;
                        
                        if (cond_31342) {
                            loopres_31343 = -1;
                            loopres_31344 = 0.0F;
                        } else {
                            int32_t log2_arg_31345 = add32(1, anc_31341);
                            bool loop_cond_31346 = slt32(1, log2_arg_31345);
                            bool res_31347;
                            int32_t res_31348;
                            int32_t res_31349;
                            bool loop_while_31350;
                            int32_t y_31351;
                            int32_t c_31352;
                            
                            loop_while_31350 = loop_cond_31346;
                            y_31351 = log2_arg_31345;
                            c_31352 = 0;
                            while (loop_while_31350) {
                                int32_t loopres_31353 = ashr32(y_31351, 1);
                                int32_t loopres_31354 = add32(1, c_31352);
                                bool loop_cond_31355 = slt32(1, loopres_31353);
                                bool loop_while_tmp_40378 = loop_cond_31355;
                                int32_t y_tmp_40379 = loopres_31353;
                                int32_t c_tmp_40380 = loopres_31354;
                                
                                loop_while_31350 = loop_while_tmp_40378;
                                y_31351 = y_tmp_40379;
                                c_31352 = c_tmp_40380;
                            }
                            res_31347 = loop_while_31350;
                            res_31348 = y_31351;
                            res_31349 = c_31352;
                            
                            int32_t y_31356 = 1 << res_31349;
                            int32_t b_31357 = stack_31308 & y_31356;
                            bool res_31358 = b_31357 == 0;
                            int32_t res_31359;
                            
                            if (res_31358) {
                                res_31359 = anc_31341;
                            } else {
                                res_31359 = -1;
                            }
                            
                            float res_31360;
                            
                            if (res_31358) {
                                res_31360 = res_31336;
                            } else {
                                bool x_31361 = sle32(0, anc_31341);
                                bool y_31362 = slt32(anc_31341, q_31023);
                                bool bounds_check_31363 = x_31361 && y_31362;
                                bool index_certs_31364;
                                
                                if (!bounds_check_31363) {
                                    {
                                        if (atomic_cmpxchg_i32_global(global_failure,
                                                                      -1, 29) ==
                                            -1) {
                                            global_failure_args[0] = anc_31341;
                                            global_failure_args[1] = q_31023;
                                            ;
                                        }
                                        local_failure = true;
                                        goto error_0;
                                    }
                                }
                                
                                float x_31365 = ((__global
                                                  float *) median_vals_mem_39420)[sext_i32_i64(anc_31341)];
                                float res_31366 = x_31365 - q_m_d_31327;
                                
                                res_31360 = res_31366;
                            }
                            loopres_31343 = res_31359;
                            loopres_31344 = res_31360;
                        }
                        
                        bool loop_cond_31367 = sle32(0, loopres_31343);
                        bool loop_while_tmp_40375 = loop_cond_31367;
                        int32_t idx_tmp_40376 = loopres_31343;
                        float res_tmp_40377 = loopres_31344;
                        
                        loop_while_31334 = loop_while_tmp_40375;
                        idx_31335 = idx_tmp_40376;
                        res_31336 = res_tmp_40377;
                    }
                    res_31331 = loop_while_31334;
                    res_31332 = idx_31335;
                    res_31333 = res_31336;
                    
                    float prv_med_sqr_31368 = res_31333 * res_31333;
                    float x_31369 = dist_31310 - cur_med_sqr_31330;
                    float abs_arg_31370 = prv_med_sqr_31368 + x_31369;
                    float res_31371 = (float) fabs(abs_arg_31370);
                    float x_31372 = dist_31310 - prv_med_sqr_31368;
                    float abs_arg_31373 = cur_med_sqr_31330 + x_31372;
                    float res_31374 = (float) fabs(abs_arg_31373);
                    int32_t loopres_31375;
                    int32_t loopres_31376;
                    float loopres_31377;
                    int32_t loopres_31378;
                    
                    if (res_31317) {
                        int32_t res_31379 = sub32(count_31309, 1);
                        
                        loopres_31375 = stack_31308;
                        loopres_31376 = res_31379;
                        loopres_31377 = res_31371;
                        loopres_31378 = -1;
                    } else {
                        bool to_visit_31380 = res_31374 < x_31287;
                        bool cond_31381 = !to_visit_31380;
                        float res_31382;
                        
                        if (cond_31381) {
                            res_31382 = dist_31310;
                        } else {
                            res_31382 = res_31374;
                        }
                        
                        int32_t res_31383;
                        int32_t res_31384;
                        int32_t res_31385;
                        
                        if (cond_31381) {
                            int32_t res_31386 = sub32(count_31309, 1);
                            
                            res_31383 = stack_31308;
                            res_31384 = res_31386;
                            res_31385 = -1;
                        } else {
                            int32_t x_31387 = smod32(node_index_31307, 2);
                            bool cond_31388 = x_31387 == 0;
                            int32_t snd_node_31389;
                            
                            if (cond_31388) {
                                snd_node_31389 = x_31312;
                            } else {
                                int32_t res_31390 = add32(1, node_index_31307);
                                
                                snd_node_31389 = res_31390;
                            }
                            
                            int32_t y_31391 = sub32(y_31314, 1);
                            int32_t fst_31392 = stack_31308 & y_31391;
                            int32_t y_31393 = add32(1, count_31309);
                            int32_t x_31394 = ashr32(stack_31308, y_31393);
                            int32_t snd_31395 = x_31394 << y_31393;
                            int32_t x_31396 = fst_31392 | snd_31395;
                            int32_t res_31397 = y_31314 | x_31396;
                            
                            res_31383 = res_31397;
                            res_31384 = count_31309;
                            res_31385 = snd_node_31389;
                        }
                        loopres_31375 = res_31383;
                        loopres_31376 = res_31384;
                        loopres_31377 = res_31382;
                        loopres_31378 = res_31385;
                    }
                    
                    bool cond_31398 = res_31313 == 0;
                    bool cond_31399 = !cond_31398;
                    bool res_31400 = slt32(loopres_31378, 0);
                    bool x_31401 = cond_31399 && res_31400;
                    bool loop_while_tmp_40369 = x_31401;
                    int32_t node_index_tmp_40370 = res_31313;
                    int32_t stack_tmp_40371 = loopres_31375;
                    int32_t count_tmp_40372 = loopres_31376;
                    float dist_tmp_40373 = loopres_31377;
                    int32_t rec_node_tmp_40374 = loopres_31378;
                    
                    loop_while_31306 = loop_while_tmp_40369;
                    node_index_31307 = node_index_tmp_40370;
                    stack_31308 = stack_tmp_40371;
                    count_31309 = count_tmp_40372;
                    dist_31310 = dist_tmp_40373;
                    rec_node_31311 = rec_node_tmp_40374;
                }
                res_31300 = loop_while_31306;
                res_31301 = node_index_31307;
                res_31302 = stack_31308;
                res_31303 = count_31309;
                res_31304 = dist_31310;
                res_31305 = rec_node_31311;
                
                bool cond_31402 = res_31301 == 0;
                bool res_31403 = res_31305 == -1;
                bool x_31404 = cond_31402 && res_31403;
                int32_t res_31405;
                int32_t res_31406;
                
                if (x_31404) {
                    res_31405 = no_leaf_31297;
                    res_31406 = res_31302;
                } else {
                    int32_t x_31407 = 1 << res_31079;
                    int32_t y_31408 = sub32(x_31407, 1);
                    bool res_31409 = sle32(y_31408, res_31305);
                    bool loop_cond_31410 = !res_31409;
                    bool res_31411;
                    int32_t res_31412;
                    int32_t res_31413;
                    int32_t res_31414;
                    bool loop_while_31415;
                    int32_t node_index_31416;
                    int32_t stack_31417;
                    int32_t count_31418;
                    
                    loop_while_31415 = loop_cond_31410;
                    node_index_31416 = res_31305;
                    stack_31417 = res_31302;
                    count_31418 = res_31303;
                    while (loop_while_31415) {
                        int32_t count_31419 = add32(1, count_31418);
                        int32_t x_31420 = 1 << count_31419;
                        int32_t y_31421 = sub32(x_31420, 1);
                        int32_t fst_31422 = stack_31417 & y_31421;
                        int32_t y_31423 = add32(1, count_31419);
                        int32_t x_31424 = ashr32(stack_31417, y_31423);
                        int32_t snd_31425 = x_31424 << y_31423;
                        int32_t x_31426 = fst_31422 | snd_31425;
                        bool x_31427 = sle32(0, node_index_31416);
                        bool y_31428 = slt32(node_index_31416, q_31023);
                        bool bounds_check_31429 = x_31427 && y_31428;
                        bool index_certs_31430;
                        
                        if (!bounds_check_31429) {
                            {
                                if (atomic_cmpxchg_i32_global(global_failure,
                                                              -1, 30) == -1) {
                                    global_failure_args[0] = node_index_31416;
                                    global_failure_args[1] = q_31023;
                                    ;
                                }
                                local_failure = true;
                                goto error_0;
                            }
                        }
                        
                        int32_t i_31431 = ((__global
                                            int32_t *) median_dims_mem_39419)[sext_i32_i64(node_index_31416)];
                        bool x_31432 = sle32(0, i_31431);
                        bool y_31433 = slt32(i_31431, d_31022);
                        bool bounds_check_31434 = x_31432 && y_31433;
                        bool index_certs_31435;
                        
                        if (!bounds_check_31434) {
                            {
                                if (atomic_cmpxchg_i32_global(global_failure,
                                                              -1, 31) == -1) {
                                    global_failure_args[0] = i_31431;
                                    global_failure_args[1] = d_31022;
                                    ;
                                }
                                local_failure = true;
                                goto error_0;
                            }
                        }
                        
                        float x_31436 = ((__global
                                          float *) mem_39444)[sext_i32_i64(i_31431) *
                                                              sext_i32_i64(n_31026) +
                                                              sext_i32_i64(gtid_33940)];
                        float y_31437 = ((__global
                                          float *) median_vals_mem_39420)[sext_i32_i64(node_index_31416)];
                        bool cond_31438 = x_31436 <= y_31437;
                        int32_t node_index_31439;
                        
                        if (cond_31438) {
                            int32_t x_31440 = add32(1, node_index_31416);
                            int32_t x_31441 = mul32(2, x_31440);
                            int32_t res_31442 = sub32(x_31441, 1);
                            
                            node_index_31439 = res_31442;
                        } else {
                            int32_t x_31443 = add32(1, node_index_31416);
                            int32_t res_31444 = mul32(2, x_31443);
                            
                            node_index_31439 = res_31444;
                        }
                        
                        bool res_31445 = sle32(y_31408, node_index_31439);
                        bool loop_cond_31446 = !res_31445;
                        bool loop_while_tmp_40381 = loop_cond_31446;
                        int32_t node_index_tmp_40382 = node_index_31439;
                        int32_t stack_tmp_40383 = x_31426;
                        int32_t count_tmp_40384 = count_31419;
                        
                        loop_while_31415 = loop_while_tmp_40381;
                        node_index_31416 = node_index_tmp_40382;
                        stack_31417 = stack_tmp_40383;
                        count_31418 = count_tmp_40384;
                    }
                    res_31411 = loop_while_31415;
                    res_31412 = node_index_31416;
                    res_31413 = stack_31417;
                    res_31414 = count_31418;
                    res_31405 = res_31412;
                    res_31406 = res_31413;
                }
                
                int32_t res_31447 = sub32(res_31405, q_31023);
                
                res_31292 = res_31447;
                res_31293 = res_31406;
                res_31294 = res_31304;
            }
            
            bool cond_31448 = slt32(res_31292, num_leaves_31075);
            int32_t res_31449 = btoi_bool_i32(cond_31448);
            
            // save map-out results
            {
                ((__global int32_t *) mem_39504)[sext_i32_i64(dummy_33939) *
                                                 sext_i32_i64(s_31037) +
                                                 sext_i32_i64(gtid_33940)] =
                    res_31292;
                ((__global int32_t *) mem_39507)[sext_i32_i64(dummy_33939) *
                                                 sext_i32_i64(s_31037) +
                                                 sext_i32_i64(gtid_33940)] =
                    res_31293;
                ((__global float *) mem_39510)[sext_i32_i64(dummy_33939) *
                                               sext_i32_i64(s_31037) +
                                               sext_i32_i64(gtid_33940)] =
                    res_31294;
            }
            // load accumulator
            {
                x_31283 = x_acc_40363;
            }
            // load new values
            {
                x_31284 = res_31449;
            }
            // apply reduction operator
            {
                int32_t res_31285 = add32(x_31283, x_31284);
                
                // store in accumulator
                {
                    x_acc_40363 = res_31285;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_31283 = x_acc_40363;
        ((__local int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355)] =
            x_31283;
    }
    
  error_0:
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_failure)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_40385;
    int32_t skip_waves_40386;
    int32_t x_40365;
    int32_t x_40366;
    
    offset_40385 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_40355, segred_group_sizze_33931)) {
            x_40365 = ((__local
                        int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355 +
                                                      offset_40385)];
        }
    }
    offset_40385 = 1;
    while (slt32(offset_40385, wave_sizze_40357)) {
        if (slt32(local_tid_40355 + offset_40385, segred_group_sizze_33931) &&
            ((local_tid_40355 - squot32(local_tid_40355, wave_sizze_40357) *
              wave_sizze_40357) & (2 * offset_40385 - 1)) == 0) {
            // read array element
            {
                x_40366 = ((volatile __local
                            int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355 +
                                                          offset_40385)];
            }
            // apply reduction operation
            {
                int32_t res_40367 = add32(x_40365, x_40366);
                
                x_40365 = res_40367;
            }
            // write result of operation
            {
                ((volatile __local
                  int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355)] =
                    x_40365;
            }
        }
        offset_40385 *= 2;
    }
    skip_waves_40386 = 1;
    while (slt32(skip_waves_40386, squot32(segred_group_sizze_33931 +
                                           wave_sizze_40357 - 1,
                                           wave_sizze_40357))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_40385 = skip_waves_40386 * wave_sizze_40357;
        if (slt32(local_tid_40355 + offset_40385, segred_group_sizze_33931) &&
            ((local_tid_40355 - squot32(local_tid_40355, wave_sizze_40357) *
              wave_sizze_40357) == 0 && (squot32(local_tid_40355,
                                                 wave_sizze_40357) & (2 *
                                                                      skip_waves_40386 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_40366 = ((__local
                            int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355 +
                                                          offset_40385)];
            }
            // apply reduction operation
            {
                int32_t res_40367 = add32(x_40365, x_40366);
                
                x_40365 = res_40367;
            }
            // write result of operation
            {
                ((__local
                  int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355)] =
                    x_40365;
            }
        }
        skip_waves_40386 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (local_tid_40355 == 0) {
            x_acc_40363 = x_40365;
        }
    }
    
    int32_t old_counter_40387;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_40355 == 0) {
            ((__global
              int32_t *) group_res_arr_mem_40351)[sext_i32_i64(group_tid_40356) *
                                                  sext_i32_i64(segred_group_sizze_33931)] =
                x_acc_40363;
            mem_fence_global();
            old_counter_40387 = atomic_add_i32_global(&((volatile __global
                                                         int *) exactKnnFixKzicounter_mem_40349)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_40359)[0] = old_counter_40387 ==
                num_groups_33933 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_40388;
    
    is_last_group_40388 = ((__local bool *) sync_arr_mem_40359)[0];
    if (is_last_group_40388) {
        if (local_tid_40355 == 0) {
            old_counter_40387 = atomic_add_i32_global(&((volatile __global
                                                         int *) exactKnnFixKzicounter_mem_40349)[0],
                                                      (int) (0 -
                                                             num_groups_33933));
        }
        // read in the per-group-results
        {
            int32_t read_per_thread_40389 = sdiv_up32(num_groups_33933,
                                                      segred_group_sizze_33931);
            
            x_31283 = 0;
            for (int32_t i_40390 = 0; i_40390 < read_per_thread_40389;
                 i_40390++) {
                int32_t group_res_id_40391 = local_tid_40355 *
                        read_per_thread_40389 + i_40390;
                int32_t index_of_group_res_40392 = group_res_id_40391;
                
                if (slt32(group_res_id_40391, num_groups_33933)) {
                    x_31284 = ((__global
                                int32_t *) group_res_arr_mem_40351)[sext_i32_i64(index_of_group_res_40392) *
                                                                    sext_i32_i64(segred_group_sizze_33931)];
                    
                    int32_t res_31285;
                    
                    res_31285 = add32(x_31283, x_31284);
                    x_31283 = res_31285;
                }
            }
        }
        ((__local int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355)] =
            x_31283;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_40393;
            int32_t skip_waves_40394;
            int32_t x_40365;
            int32_t x_40366;
            
            offset_40393 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_40355, segred_group_sizze_33931)) {
                    x_40365 = ((__local
                                int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355 +
                                                              offset_40393)];
                }
            }
            offset_40393 = 1;
            while (slt32(offset_40393, wave_sizze_40357)) {
                if (slt32(local_tid_40355 + offset_40393,
                          segred_group_sizze_33931) && ((local_tid_40355 -
                                                         squot32(local_tid_40355,
                                                                 wave_sizze_40357) *
                                                         wave_sizze_40357) &
                                                        (2 * offset_40393 -
                                                         1)) == 0) {
                    // read array element
                    {
                        x_40366 = ((volatile __local
                                    int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355 +
                                                                  offset_40393)];
                    }
                    // apply reduction operation
                    {
                        int32_t res_40367 = add32(x_40365, x_40366);
                        
                        x_40365 = res_40367;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355)] =
                            x_40365;
                    }
                }
                offset_40393 *= 2;
            }
            skip_waves_40394 = 1;
            while (slt32(skip_waves_40394, squot32(segred_group_sizze_33931 +
                                                   wave_sizze_40357 - 1,
                                                   wave_sizze_40357))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_40393 = skip_waves_40394 * wave_sizze_40357;
                if (slt32(local_tid_40355 + offset_40393,
                          segred_group_sizze_33931) && ((local_tid_40355 -
                                                         squot32(local_tid_40355,
                                                                 wave_sizze_40357) *
                                                         wave_sizze_40357) ==
                                                        0 &&
                                                        (squot32(local_tid_40355,
                                                                 wave_sizze_40357) &
                                                         (2 * skip_waves_40394 -
                                                          1)) == 0)) {
                    // read array element
                    {
                        x_40366 = ((__local
                                    int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355 +
                                                                  offset_40393)];
                    }
                    // apply reduction operation
                    {
                        int32_t res_40367 = add32(x_40365, x_40366);
                        
                        x_40365 = res_40367;
                    }
                    // write result of operation
                    {
                        ((__local
                          int32_t *) red_arr_mem_40361)[sext_i32_i64(local_tid_40355)] =
                            x_40365;
                    }
                }
                skip_waves_40394 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_40355 == 0) {
                    ((__global int32_t *) mem_39501)[0] = x_40365;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_33931
}
__kernel void findNaturalLeavesFixKziscan_stage1_35797(__global
                                                       int *global_failure,
                                                       uint scan_arr_mem_40342_backing_offset_0,
                                                       uint scan_arr_mem_40340_backing_offset_1,
                                                       uint scan_arr_mem_40338_backing_offset_2,
                                                       uint scan_arr_mem_40336_backing_offset_3,
                                                       int32_t n_31866,
                                                       int32_t lifted_2_radix_sort_step_arg_31946,
                                                       int32_t lifted_0_get_bit_arg_31947,
                                                       __global
                                                       unsigned char *mem_param_39438,
                                                       __global
                                                       unsigned char *mem_39451,
                                                       __global
                                                       unsigned char *mem_39454,
                                                       __global
                                                       unsigned char *mem_39457,
                                                       __global
                                                       unsigned char *mem_39460,
                                                       __global
                                                       unsigned char *mem_39463,
                                                       int32_t num_threads_40330)
{
    #define segscan_group_sizze_35792 (findNaturalLeavesFixKzisegscan_group_sizze_35791)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *scan_arr_mem_40342_backing_3 =
                  &shared_mem[scan_arr_mem_40342_backing_offset_0];
    volatile char *scan_arr_mem_40340_backing_2 =
                  &shared_mem[scan_arr_mem_40340_backing_offset_1];
    volatile char *scan_arr_mem_40338_backing_1 =
                  &shared_mem[scan_arr_mem_40338_backing_offset_2];
    volatile char *scan_arr_mem_40336_backing_0 =
                  &shared_mem[scan_arr_mem_40336_backing_offset_3];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40331;
    int32_t local_tid_40332;
    int32_t group_sizze_40335;
    int32_t wave_sizze_40334;
    int32_t group_tid_40333;
    
    global_tid_40331 = get_global_id(0);
    local_tid_40332 = get_local_id(0);
    group_sizze_40335 = get_local_size(0);
    wave_sizze_40334 = LOCKSTEP_WIDTH;
    group_tid_40333 = get_group_id(0);
    
    int32_t phys_tid_35797;
    
    phys_tid_35797 = global_tid_40331;
    
    __local char *scan_arr_mem_40336;
    __local char *scan_arr_mem_40338;
    __local char *scan_arr_mem_40340;
    __local char *scan_arr_mem_40342;
    
    scan_arr_mem_40336 = (__local char *) scan_arr_mem_40336_backing_0;
    scan_arr_mem_40338 = (__local char *) scan_arr_mem_40338_backing_1;
    scan_arr_mem_40340 = (__local char *) scan_arr_mem_40340_backing_2;
    scan_arr_mem_40342 = (__local char *) scan_arr_mem_40342_backing_3;
    
    int32_t x_31961;
    int32_t x_31962;
    int32_t x_31963;
    int32_t x_31964;
    int32_t x_31965;
    int32_t x_31966;
    int32_t x_31967;
    int32_t x_31968;
    
    x_31961 = 0;
    x_31962 = 0;
    x_31963 = 0;
    x_31964 = 0;
    for (int32_t j_40344 = 0; j_40344 < sdiv_up32(n_31866, num_threads_40330);
         j_40344++) {
        int32_t chunk_offset_40345 = segscan_group_sizze_35792 * j_40344 +
                group_tid_40333 * (segscan_group_sizze_35792 *
                                   sdiv_up32(n_31866, num_threads_40330));
        int32_t flat_idx_40346 = chunk_offset_40345 + local_tid_40332;
        int32_t gtid_35796 = flat_idx_40346;
        
        // threads in bounds read input
        {
            if (slt32(gtid_35796, n_31866)) {
                int32_t x_31973 = ((__global
                                    int32_t *) mem_param_39438)[sext_i32_i64(gtid_35796)];
                int32_t res_31974 = ashr32(x_31973, lifted_0_get_bit_arg_31947);
                int32_t res_31975 = 1 & res_31974;
                int32_t x_31976 = mul32(2, res_31975);
                int32_t res_31977 = ashr32(x_31973,
                                           lifted_2_radix_sort_step_arg_31946);
                int32_t res_31978 = 1 & res_31977;
                int32_t res_31979 = add32(x_31976, res_31978);
                bool cond_31980 = res_31979 == 0;
                int32_t res_31981 = btoi_bool_i32(cond_31980);
                int32_t res_31982;
                int32_t res_31983;
                int32_t res_31984;
                
                if (cond_31980) {
                    res_31982 = 0;
                    res_31983 = 0;
                    res_31984 = 0;
                } else {
                    bool cond_31985 = res_31979 == 1;
                    int32_t res_31986 = btoi_bool_i32(cond_31985);
                    int32_t res_31987;
                    int32_t res_31988;
                    
                    if (cond_31985) {
                        res_31987 = 0;
                        res_31988 = 0;
                    } else {
                        bool cond_31989 = res_31979 == 2;
                        int32_t res_31990 = btoi_bool_i32(cond_31989);
                        bool cond_neg_31991 = !cond_31989;
                        int32_t res_31992 = btoi_bool_i32(cond_neg_31991);
                        
                        res_31987 = res_31990;
                        res_31988 = res_31992;
                    }
                    res_31982 = res_31986;
                    res_31983 = res_31987;
                    res_31984 = res_31988;
                }
                // write to-scan values to parameters
                {
                    x_31965 = res_31981;
                    x_31966 = res_31982;
                    x_31967 = res_31983;
                    x_31968 = res_31984;
                }
                // write mapped values results to global memory
                {
                    ((__global int32_t *) mem_39463)[sext_i32_i64(gtid_35796)] =
                        res_31979;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!slt32(gtid_35796, n_31866)) {
                    x_31965 = 0;
                    x_31966 = 0;
                    x_31967 = 0;
                    x_31968 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t res_31969 = add32(x_31961, x_31965);
                int32_t res_31970 = add32(x_31962, x_31966);
                int32_t res_31971 = add32(x_31963, x_31967);
                int32_t res_31972 = add32(x_31964, x_31968);
                
                ((__local
                  int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332)] =
                    res_31969;
                ((__local
                  int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332)] =
                    res_31970;
                ((__local
                  int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332)] =
                    res_31971;
                ((__local
                  int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332)] =
                    res_31972;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_40347;
            int32_t x_40348;
            int32_t x_40349;
            int32_t x_40350;
            int32_t x_40351;
            int32_t x_40352;
            int32_t x_40353;
            int32_t x_40354;
            int32_t x_40359;
            int32_t x_40360;
            int32_t x_40361;
            int32_t x_40362;
            int32_t x_40363;
            int32_t x_40364;
            int32_t x_40365;
            int32_t x_40366;
            int32_t skip_threads_40371;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_40332, segscan_group_sizze_35792)) {
                    x_40351 = ((volatile __local
                                int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332)];
                    x_40352 = ((volatile __local
                                int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332)];
                    x_40353 = ((volatile __local
                                int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332)];
                    x_40354 = ((volatile __local
                                int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332)];
                    if ((local_tid_40332 - squot32(local_tid_40332, 32) * 32) ==
                        0) {
                        x_40347 = x_40351;
                        x_40348 = x_40352;
                        x_40349 = x_40353;
                        x_40350 = x_40354;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_40371 = 1;
                while (slt32(skip_threads_40371, 32)) {
                    if (sle32(skip_threads_40371, local_tid_40332 -
                              squot32(local_tid_40332, 32) * 32) &&
                        slt32(local_tid_40332, segscan_group_sizze_35792)) {
                        // read operands
                        {
                            x_40347 = ((volatile __local
                                        int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332 -
                                                                       skip_threads_40371)];
                            x_40348 = ((volatile __local
                                        int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332 -
                                                                       skip_threads_40371)];
                            x_40349 = ((volatile __local
                                        int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332 -
                                                                       skip_threads_40371)];
                            x_40350 = ((volatile __local
                                        int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332 -
                                                                       skip_threads_40371)];
                        }
                        // perform operation
                        {
                            int32_t res_40355 = add32(x_40347, x_40351);
                            int32_t res_40356 = add32(x_40348, x_40352);
                            int32_t res_40357 = add32(x_40349, x_40353);
                            int32_t res_40358 = add32(x_40350, x_40354);
                            
                            x_40347 = res_40355;
                            x_40348 = res_40356;
                            x_40349 = res_40357;
                            x_40350 = res_40358;
                        }
                    }
                    if (sle32(wave_sizze_40334, skip_threads_40371)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_40371, local_tid_40332 -
                              squot32(local_tid_40332, 32) * 32) &&
                        slt32(local_tid_40332, segscan_group_sizze_35792)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332)] =
                                x_40347;
                            x_40351 = x_40347;
                            ((volatile __local
                              int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332)] =
                                x_40348;
                            x_40352 = x_40348;
                            ((volatile __local
                              int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332)] =
                                x_40349;
                            x_40353 = x_40349;
                            ((volatile __local
                              int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332)] =
                                x_40350;
                            x_40354 = x_40350;
                        }
                    }
                    if (sle32(wave_sizze_40334, skip_threads_40371)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_40371 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_40332 - squot32(local_tid_40332, 32) * 32) ==
                    31 && slt32(local_tid_40332, segscan_group_sizze_35792)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_40336)[sext_i32_i64(squot32(local_tid_40332,
                                                                          32))] =
                        x_40347;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40338)[sext_i32_i64(squot32(local_tid_40332,
                                                                          32))] =
                        x_40348;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40340)[sext_i32_i64(squot32(local_tid_40332,
                                                                          32))] =
                        x_40349;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40342)[sext_i32_i64(squot32(local_tid_40332,
                                                                          32))] =
                        x_40350;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_40372;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_40332, 32) == 0 &&
                        slt32(local_tid_40332, segscan_group_sizze_35792)) {
                        x_40363 = ((volatile __local
                                    int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332)];
                        x_40364 = ((volatile __local
                                    int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332)];
                        x_40365 = ((volatile __local
                                    int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332)];
                        x_40366 = ((volatile __local
                                    int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332)];
                        if ((local_tid_40332 - squot32(local_tid_40332, 32) *
                             32) == 0) {
                            x_40359 = x_40363;
                            x_40360 = x_40364;
                            x_40361 = x_40365;
                            x_40362 = x_40366;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_40372 = 1;
                    while (slt32(skip_threads_40372, 32)) {
                        if (sle32(skip_threads_40372, local_tid_40332 -
                                  squot32(local_tid_40332, 32) * 32) &&
                            (squot32(local_tid_40332, 32) == 0 &&
                             slt32(local_tid_40332,
                                   segscan_group_sizze_35792))) {
                            // read operands
                            {
                                x_40359 = ((volatile __local
                                            int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332 -
                                                                           skip_threads_40372)];
                                x_40360 = ((volatile __local
                                            int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332 -
                                                                           skip_threads_40372)];
                                x_40361 = ((volatile __local
                                            int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332 -
                                                                           skip_threads_40372)];
                                x_40362 = ((volatile __local
                                            int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332 -
                                                                           skip_threads_40372)];
                            }
                            // perform operation
                            {
                                int32_t res_40367 = add32(x_40359, x_40363);
                                int32_t res_40368 = add32(x_40360, x_40364);
                                int32_t res_40369 = add32(x_40361, x_40365);
                                int32_t res_40370 = add32(x_40362, x_40366);
                                
                                x_40359 = res_40367;
                                x_40360 = res_40368;
                                x_40361 = res_40369;
                                x_40362 = res_40370;
                            }
                        }
                        if (sle32(wave_sizze_40334, skip_threads_40372)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_40372, local_tid_40332 -
                                  squot32(local_tid_40332, 32) * 32) &&
                            (squot32(local_tid_40332, 32) == 0 &&
                             slt32(local_tid_40332,
                                   segscan_group_sizze_35792))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332)] =
                                    x_40359;
                                x_40363 = x_40359;
                                ((volatile __local
                                  int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332)] =
                                    x_40360;
                                x_40364 = x_40360;
                                ((volatile __local
                                  int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332)] =
                                    x_40361;
                                x_40365 = x_40361;
                                ((volatile __local
                                  int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332)] =
                                    x_40362;
                                x_40366 = x_40362;
                            }
                        }
                        if (sle32(wave_sizze_40334, skip_threads_40372)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_40372 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_40332, 32) == 0 ||
                      !slt32(local_tid_40332, segscan_group_sizze_35792))) {
                    // read operands
                    {
                        x_40351 = x_40347;
                        x_40352 = x_40348;
                        x_40353 = x_40349;
                        x_40354 = x_40350;
                        x_40347 = ((__local
                                    int32_t *) scan_arr_mem_40336)[sext_i32_i64(squot32(local_tid_40332,
                                                                                        32) -
                                                                   1)];
                        x_40348 = ((__local
                                    int32_t *) scan_arr_mem_40338)[sext_i32_i64(squot32(local_tid_40332,
                                                                                        32) -
                                                                   1)];
                        x_40349 = ((__local
                                    int32_t *) scan_arr_mem_40340)[sext_i32_i64(squot32(local_tid_40332,
                                                                                        32) -
                                                                   1)];
                        x_40350 = ((__local
                                    int32_t *) scan_arr_mem_40342)[sext_i32_i64(squot32(local_tid_40332,
                                                                                        32) -
                                                                   1)];
                    }
                    // perform operation
                    {
                        int32_t res_40355 = add32(x_40347, x_40351);
                        int32_t res_40356 = add32(x_40348, x_40352);
                        int32_t res_40357 = add32(x_40349, x_40353);
                        int32_t res_40358 = add32(x_40350, x_40354);
                        
                        x_40347 = res_40355;
                        x_40348 = res_40356;
                        x_40349 = res_40357;
                        x_40350 = res_40358;
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332)] =
                            x_40347;
                        ((__local
                          int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332)] =
                            x_40348;
                        ((__local
                          int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332)] =
                            x_40349;
                        ((__local
                          int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332)] =
                            x_40350;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_40332, 32) == 0) {
                    ((__local
                      int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332)] =
                        x_40351;
                    ((__local
                      int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332)] =
                        x_40352;
                    ((__local
                      int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332)] =
                        x_40353;
                    ((__local
                      int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332)] =
                        x_40354;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_35796, n_31866)) {
                    ((__global int32_t *) mem_39451)[sext_i32_i64(gtid_35796)] =
                        ((__local
                          int32_t *) scan_arr_mem_40336)[sext_i32_i64(local_tid_40332)];
                    ((__global int32_t *) mem_39454)[sext_i32_i64(gtid_35796)] =
                        ((__local
                          int32_t *) scan_arr_mem_40338)[sext_i32_i64(local_tid_40332)];
                    ((__global int32_t *) mem_39457)[sext_i32_i64(gtid_35796)] =
                        ((__local
                          int32_t *) scan_arr_mem_40340)[sext_i32_i64(local_tid_40332)];
                    ((__global int32_t *) mem_39460)[sext_i32_i64(gtid_35796)] =
                        ((__local
                          int32_t *) scan_arr_mem_40342)[sext_i32_i64(local_tid_40332)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_40373 = 0;
                bool should_load_carry_40374 = local_tid_40332 == 0 &&
                     !crosses_segment_40373;
                
                if (should_load_carry_40374) {
                    x_31961 = ((__local
                                int32_t *) scan_arr_mem_40336)[sext_i32_i64(segscan_group_sizze_35792 -
                                                               1)];
                    x_31962 = ((__local
                                int32_t *) scan_arr_mem_40338)[sext_i32_i64(segscan_group_sizze_35792 -
                                                               1)];
                    x_31963 = ((__local
                                int32_t *) scan_arr_mem_40340)[sext_i32_i64(segscan_group_sizze_35792 -
                                                               1)];
                    x_31964 = ((__local
                                int32_t *) scan_arr_mem_40342)[sext_i32_i64(segscan_group_sizze_35792 -
                                                               1)];
                }
                if (!should_load_carry_40374) {
                    x_31961 = 0;
                    x_31962 = 0;
                    x_31963 = 0;
                    x_31964 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_35792
}
__kernel void findNaturalLeavesFixKziscan_stage2_35797(__global
                                                       int *global_failure,
                                                       uint scan_arr_mem_40386_backing_offset_0,
                                                       uint scan_arr_mem_40384_backing_offset_1,
                                                       uint scan_arr_mem_40382_backing_offset_2,
                                                       uint scan_arr_mem_40380_backing_offset_3,
                                                       int32_t n_31866, __global
                                                       unsigned char *mem_39451,
                                                       __global
                                                       unsigned char *mem_39454,
                                                       __global
                                                       unsigned char *mem_39457,
                                                       __global
                                                       unsigned char *mem_39460,
                                                       int32_t stage1_num_groups_40329,
                                                       int32_t num_threads_40330)
{
    #define segscan_group_sizze_35792 (findNaturalLeavesFixKzisegscan_group_sizze_35791)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *scan_arr_mem_40386_backing_3 =
                  &shared_mem[scan_arr_mem_40386_backing_offset_0];
    volatile char *scan_arr_mem_40384_backing_2 =
                  &shared_mem[scan_arr_mem_40384_backing_offset_1];
    volatile char *scan_arr_mem_40382_backing_1 =
                  &shared_mem[scan_arr_mem_40382_backing_offset_2];
    volatile char *scan_arr_mem_40380_backing_0 =
                  &shared_mem[scan_arr_mem_40380_backing_offset_3];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40375;
    int32_t local_tid_40376;
    int32_t group_sizze_40379;
    int32_t wave_sizze_40378;
    int32_t group_tid_40377;
    
    global_tid_40375 = get_global_id(0);
    local_tid_40376 = get_local_id(0);
    group_sizze_40379 = get_local_size(0);
    wave_sizze_40378 = LOCKSTEP_WIDTH;
    group_tid_40377 = get_group_id(0);
    
    int32_t phys_tid_35797;
    
    phys_tid_35797 = global_tid_40375;
    
    __local char *scan_arr_mem_40380;
    __local char *scan_arr_mem_40382;
    __local char *scan_arr_mem_40384;
    __local char *scan_arr_mem_40386;
    
    scan_arr_mem_40380 = (__local char *) scan_arr_mem_40380_backing_0;
    scan_arr_mem_40382 = (__local char *) scan_arr_mem_40382_backing_1;
    scan_arr_mem_40384 = (__local char *) scan_arr_mem_40384_backing_2;
    scan_arr_mem_40386 = (__local char *) scan_arr_mem_40386_backing_3;
    
    int32_t flat_idx_40388;
    
    flat_idx_40388 = (local_tid_40376 + 1) * (segscan_group_sizze_35792 *
                                              sdiv_up32(n_31866,
                                                        num_threads_40330)) - 1;
    
    int32_t gtid_35796;
    
    gtid_35796 = flat_idx_40388;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_35796, n_31866)) {
            ((__local
              int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)] =
                ((__global int32_t *) mem_39451)[sext_i32_i64(gtid_35796)];
            ((__local
              int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)] =
                ((__global int32_t *) mem_39454)[sext_i32_i64(gtid_35796)];
            ((__local
              int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)] =
                ((__global int32_t *) mem_39457)[sext_i32_i64(gtid_35796)];
            ((__local
              int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)] =
                ((__global int32_t *) mem_39460)[sext_i32_i64(gtid_35796)];
        } else {
            ((__local
              int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)] = 0;
            ((__local
              int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)] = 0;
            ((__local
              int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)] = 0;
            ((__local
              int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_31961;
    int32_t x_31962;
    int32_t x_31963;
    int32_t x_31964;
    int32_t x_31965;
    int32_t x_31966;
    int32_t x_31967;
    int32_t x_31968;
    int32_t x_40389;
    int32_t x_40390;
    int32_t x_40391;
    int32_t x_40392;
    int32_t x_40393;
    int32_t x_40394;
    int32_t x_40395;
    int32_t x_40396;
    int32_t skip_threads_40401;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_40376, stage1_num_groups_40329)) {
            x_31965 = ((volatile __local
                        int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)];
            x_31966 = ((volatile __local
                        int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)];
            x_31967 = ((volatile __local
                        int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)];
            x_31968 = ((volatile __local
                        int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)];
            if ((local_tid_40376 - squot32(local_tid_40376, 32) * 32) == 0) {
                x_31961 = x_31965;
                x_31962 = x_31966;
                x_31963 = x_31967;
                x_31964 = x_31968;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_40401 = 1;
        while (slt32(skip_threads_40401, 32)) {
            if (sle32(skip_threads_40401, local_tid_40376 -
                      squot32(local_tid_40376, 32) * 32) &&
                slt32(local_tid_40376, stage1_num_groups_40329)) {
                // read operands
                {
                    x_31961 = ((volatile __local
                                int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376 -
                                                               skip_threads_40401)];
                    x_31962 = ((volatile __local
                                int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376 -
                                                               skip_threads_40401)];
                    x_31963 = ((volatile __local
                                int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376 -
                                                               skip_threads_40401)];
                    x_31964 = ((volatile __local
                                int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376 -
                                                               skip_threads_40401)];
                }
                // perform operation
                {
                    int32_t res_31969 = add32(x_31961, x_31965);
                    int32_t res_31970 = add32(x_31962, x_31966);
                    int32_t res_31971 = add32(x_31963, x_31967);
                    int32_t res_31972 = add32(x_31964, x_31968);
                    
                    x_31961 = res_31969;
                    x_31962 = res_31970;
                    x_31963 = res_31971;
                    x_31964 = res_31972;
                }
            }
            if (sle32(wave_sizze_40378, skip_threads_40401)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_40401, local_tid_40376 -
                      squot32(local_tid_40376, 32) * 32) &&
                slt32(local_tid_40376, stage1_num_groups_40329)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)] =
                        x_31961;
                    x_31965 = x_31961;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)] =
                        x_31962;
                    x_31966 = x_31962;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)] =
                        x_31963;
                    x_31967 = x_31963;
                    ((volatile __local
                      int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)] =
                        x_31964;
                    x_31968 = x_31964;
                }
            }
            if (sle32(wave_sizze_40378, skip_threads_40401)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_40401 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_40376 - squot32(local_tid_40376, 32) * 32) == 31 &&
            slt32(local_tid_40376, stage1_num_groups_40329)) {
            ((volatile __local
              int32_t *) scan_arr_mem_40380)[sext_i32_i64(squot32(local_tid_40376,
                                                                  32))] =
                x_31961;
            ((volatile __local
              int32_t *) scan_arr_mem_40382)[sext_i32_i64(squot32(local_tid_40376,
                                                                  32))] =
                x_31962;
            ((volatile __local
              int32_t *) scan_arr_mem_40384)[sext_i32_i64(squot32(local_tid_40376,
                                                                  32))] =
                x_31963;
            ((volatile __local
              int32_t *) scan_arr_mem_40386)[sext_i32_i64(squot32(local_tid_40376,
                                                                  32))] =
                x_31964;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_40402;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_40376, 32) == 0 && slt32(local_tid_40376,
                                                           stage1_num_groups_40329)) {
                x_40393 = ((volatile __local
                            int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)];
                x_40394 = ((volatile __local
                            int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)];
                x_40395 = ((volatile __local
                            int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)];
                x_40396 = ((volatile __local
                            int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)];
                if ((local_tid_40376 - squot32(local_tid_40376, 32) * 32) ==
                    0) {
                    x_40389 = x_40393;
                    x_40390 = x_40394;
                    x_40391 = x_40395;
                    x_40392 = x_40396;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_40402 = 1;
            while (slt32(skip_threads_40402, 32)) {
                if (sle32(skip_threads_40402, local_tid_40376 -
                          squot32(local_tid_40376, 32) * 32) &&
                    (squot32(local_tid_40376, 32) == 0 && slt32(local_tid_40376,
                                                                stage1_num_groups_40329))) {
                    // read operands
                    {
                        x_40389 = ((volatile __local
                                    int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376 -
                                                                   skip_threads_40402)];
                        x_40390 = ((volatile __local
                                    int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376 -
                                                                   skip_threads_40402)];
                        x_40391 = ((volatile __local
                                    int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376 -
                                                                   skip_threads_40402)];
                        x_40392 = ((volatile __local
                                    int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376 -
                                                                   skip_threads_40402)];
                    }
                    // perform operation
                    {
                        int32_t res_40397 = add32(x_40389, x_40393);
                        int32_t res_40398 = add32(x_40390, x_40394);
                        int32_t res_40399 = add32(x_40391, x_40395);
                        int32_t res_40400 = add32(x_40392, x_40396);
                        
                        x_40389 = res_40397;
                        x_40390 = res_40398;
                        x_40391 = res_40399;
                        x_40392 = res_40400;
                    }
                }
                if (sle32(wave_sizze_40378, skip_threads_40402)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_40402, local_tid_40376 -
                          squot32(local_tid_40376, 32) * 32) &&
                    (squot32(local_tid_40376, 32) == 0 && slt32(local_tid_40376,
                                                                stage1_num_groups_40329))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)] =
                            x_40389;
                        x_40393 = x_40389;
                        ((volatile __local
                          int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)] =
                            x_40390;
                        x_40394 = x_40390;
                        ((volatile __local
                          int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)] =
                            x_40391;
                        x_40395 = x_40391;
                        ((volatile __local
                          int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)] =
                            x_40392;
                        x_40396 = x_40392;
                    }
                }
                if (sle32(wave_sizze_40378, skip_threads_40402)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_40402 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_40376, 32) == 0 || !slt32(local_tid_40376,
                                                          stage1_num_groups_40329))) {
            // read operands
            {
                x_31965 = x_31961;
                x_31966 = x_31962;
                x_31967 = x_31963;
                x_31968 = x_31964;
                x_31961 = ((__local
                            int32_t *) scan_arr_mem_40380)[sext_i32_i64(squot32(local_tid_40376,
                                                                                32) -
                                                           1)];
                x_31962 = ((__local
                            int32_t *) scan_arr_mem_40382)[sext_i32_i64(squot32(local_tid_40376,
                                                                                32) -
                                                           1)];
                x_31963 = ((__local
                            int32_t *) scan_arr_mem_40384)[sext_i32_i64(squot32(local_tid_40376,
                                                                                32) -
                                                           1)];
                x_31964 = ((__local
                            int32_t *) scan_arr_mem_40386)[sext_i32_i64(squot32(local_tid_40376,
                                                                                32) -
                                                           1)];
            }
            // perform operation
            {
                int32_t res_31969 = add32(x_31961, x_31965);
                int32_t res_31970 = add32(x_31962, x_31966);
                int32_t res_31971 = add32(x_31963, x_31967);
                int32_t res_31972 = add32(x_31964, x_31968);
                
                x_31961 = res_31969;
                x_31962 = res_31970;
                x_31963 = res_31971;
                x_31964 = res_31972;
            }
            // write final result
            {
                ((__local
                  int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)] =
                    x_31961;
                ((__local
                  int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)] =
                    x_31962;
                ((__local
                  int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)] =
                    x_31963;
                ((__local
                  int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)] =
                    x_31964;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_40376, 32) == 0) {
            ((__local
              int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)] =
                x_31965;
            ((__local
              int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)] =
                x_31966;
            ((__local
              int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)] =
                x_31967;
            ((__local
              int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)] =
                x_31968;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_35796, n_31866)) {
            ((__global int32_t *) mem_39451)[sext_i32_i64(gtid_35796)] =
                ((__local
                  int32_t *) scan_arr_mem_40380)[sext_i32_i64(local_tid_40376)];
            ((__global int32_t *) mem_39454)[sext_i32_i64(gtid_35796)] =
                ((__local
                  int32_t *) scan_arr_mem_40382)[sext_i32_i64(local_tid_40376)];
            ((__global int32_t *) mem_39457)[sext_i32_i64(gtid_35796)] =
                ((__local
                  int32_t *) scan_arr_mem_40384)[sext_i32_i64(local_tid_40376)];
            ((__global int32_t *) mem_39460)[sext_i32_i64(gtid_35796)] =
                ((__local
                  int32_t *) scan_arr_mem_40386)[sext_i32_i64(local_tid_40376)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_35792
}
__kernel void findNaturalLeavesFixKziscan_stage3_35797(__global
                                                       int *global_failure,
                                                       int32_t n_31866,
                                                       int32_t num_groups_35794,
                                                       __global
                                                       unsigned char *mem_39451,
                                                       __global
                                                       unsigned char *mem_39454,
                                                       __global
                                                       unsigned char *mem_39457,
                                                       __global
                                                       unsigned char *mem_39460,
                                                       int32_t num_threads_40330,
                                                       int32_t required_groups_40403)
{
    #define segscan_group_sizze_35792 (findNaturalLeavesFixKzisegscan_group_sizze_35791)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40404;
    int32_t local_tid_40405;
    int32_t group_sizze_40408;
    int32_t wave_sizze_40407;
    int32_t group_tid_40406;
    
    global_tid_40404 = get_global_id(0);
    local_tid_40405 = get_local_id(0);
    group_sizze_40408 = get_local_size(0);
    wave_sizze_40407 = LOCKSTEP_WIDTH;
    group_tid_40406 = get_group_id(0);
    
    int32_t phys_tid_35797;
    
    phys_tid_35797 = global_tid_40404;
    
    int32_t phys_group_id_40409;
    
    phys_group_id_40409 = get_group_id(0);
    for (int32_t i_40410 = 0; i_40410 < sdiv_up32(required_groups_40403 -
                                                  phys_group_id_40409,
                                                  num_groups_35794);
         i_40410++) {
        int32_t virt_group_id_40411 = phys_group_id_40409 + i_40410 *
                num_groups_35794;
        int32_t flat_idx_40412 = virt_group_id_40411 *
                segscan_group_sizze_35792 + local_tid_40405;
        int32_t gtid_35796 = flat_idx_40412;
        int32_t orig_group_40413 = squot32(flat_idx_40412,
                                           segscan_group_sizze_35792 *
                                           sdiv_up32(n_31866,
                                                     num_threads_40330));
        int32_t carry_in_flat_idx_40414 = orig_group_40413 *
                (segscan_group_sizze_35792 * sdiv_up32(n_31866,
                                                       num_threads_40330)) - 1;
        
        if (slt32(gtid_35796, n_31866)) {
            if (!(orig_group_40413 == 0 || flat_idx_40412 == (orig_group_40413 +
                                                              1) *
                  (segscan_group_sizze_35792 * sdiv_up32(n_31866,
                                                         num_threads_40330)) -
                  1)) {
                int32_t x_31961;
                int32_t x_31962;
                int32_t x_31963;
                int32_t x_31964;
                int32_t x_31965;
                int32_t x_31966;
                int32_t x_31967;
                int32_t x_31968;
                
                x_31961 = ((__global
                            int32_t *) mem_39451)[sext_i32_i64(carry_in_flat_idx_40414)];
                x_31962 = ((__global
                            int32_t *) mem_39454)[sext_i32_i64(carry_in_flat_idx_40414)];
                x_31963 = ((__global
                            int32_t *) mem_39457)[sext_i32_i64(carry_in_flat_idx_40414)];
                x_31964 = ((__global
                            int32_t *) mem_39460)[sext_i32_i64(carry_in_flat_idx_40414)];
                x_31965 = ((__global
                            int32_t *) mem_39451)[sext_i32_i64(gtid_35796)];
                x_31966 = ((__global
                            int32_t *) mem_39454)[sext_i32_i64(gtid_35796)];
                x_31967 = ((__global
                            int32_t *) mem_39457)[sext_i32_i64(gtid_35796)];
                x_31968 = ((__global
                            int32_t *) mem_39460)[sext_i32_i64(gtid_35796)];
                
                int32_t res_31969;
                
                res_31969 = add32(x_31961, x_31965);
                
                int32_t res_31970 = add32(x_31962, x_31966);
                int32_t res_31971 = add32(x_31963, x_31967);
                int32_t res_31972 = add32(x_31964, x_31968);
                
                x_31961 = res_31969;
                x_31962 = res_31970;
                x_31963 = res_31971;
                x_31964 = res_31972;
                ((__global int32_t *) mem_39451)[sext_i32_i64(gtid_35796)] =
                    x_31961;
                ((__global int32_t *) mem_39454)[sext_i32_i64(gtid_35796)] =
                    x_31962;
                ((__global int32_t *) mem_39457)[sext_i32_i64(gtid_35796)] =
                    x_31963;
                ((__global int32_t *) mem_39460)[sext_i32_i64(gtid_35796)] =
                    x_31964;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_35792
}
__kernel void findNaturalLeavesFixKzisegmap_35722(__global int *global_failure,
                                                  int failure_is_an_option,
                                                  __global
                                                  int *global_failure_args,
                                                  int32_t d_31863,
                                                  int32_t q_31864,
                                                  int32_t n_31866,
                                                  int32_t y_31897,
                                                  unsigned char loop_cond_31899,
                                                  __global
                                                  unsigned char *median_dims_mem_39419,
                                                  __global
                                                  unsigned char *median_vals_mem_39420,
                                                  __global
                                                  unsigned char *mem_39426,
                                                  __global
                                                  unsigned char *mem_39430)
{
    #define segmap_group_sizze_35758 (findNaturalLeavesFixKzisegmap_group_sizze_35725)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40306;
    int32_t local_tid_40307;
    int32_t group_sizze_40310;
    int32_t wave_sizze_40309;
    int32_t group_tid_40308;
    
    global_tid_40306 = get_global_id(0);
    local_tid_40307 = get_local_id(0);
    group_sizze_40310 = get_local_size(0);
    wave_sizze_40309 = LOCKSTEP_WIDTH;
    group_tid_40308 = get_group_id(0);
    
    int32_t phys_tid_35722;
    
    phys_tid_35722 = global_tid_40306;
    
    int32_t gtid_35721;
    
    gtid_35721 = sext_i64_i32(sext_i32_i64(group_tid_40308) *
        sext_i32_i64(segmap_group_sizze_35758) + sext_i32_i64(local_tid_40307));
    if (slt32(gtid_35721, n_31866)) {
        bool leaf_35764;
        int32_t leaf_35765;
        bool loop_while_35766;
        int32_t node_index_35767;
        
        loop_while_35766 = loop_cond_31899;
        node_index_35767 = 0;
        while (loop_while_35766) {
            bool x_35768 = sle32(0, node_index_35767);
            bool y_35769 = slt32(node_index_35767, q_31864);
            bool bounds_check_35770 = x_35768 && y_35769;
            bool index_certs_35771;
            
            if (!bounds_check_35770) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 47) ==
                        -1) {
                        global_failure_args[0] = node_index_35767;
                        global_failure_args[1] = q_31864;
                        ;
                    }
                    return;
                }
            }
            
            int32_t i_35772 = ((__global
                                int32_t *) median_dims_mem_39419)[sext_i32_i64(node_index_35767)];
            bool x_35773 = sle32(0, i_35772);
            bool y_35774 = slt32(i_35772, d_31863);
            bool bounds_check_35775 = x_35773 && y_35774;
            bool index_certs_35776;
            
            if (!bounds_check_35775) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 48) ==
                        -1) {
                        global_failure_args[0] = i_35772;
                        global_failure_args[1] = d_31863;
                        ;
                    }
                    return;
                }
            }
            
            float x_35777 = ((__global
                              float *) mem_39426)[sext_i32_i64(i_35772) *
                                                  sext_i32_i64(n_31866) +
                                                  sext_i32_i64(gtid_35721)];
            float y_35778 = ((__global
                              float *) median_vals_mem_39420)[sext_i32_i64(node_index_35767)];
            bool cond_35779 = x_35777 <= y_35778;
            int32_t loopres_35780;
            
            if (cond_35779) {
                int32_t x_35781 = add32(1, node_index_35767);
                int32_t x_35782 = mul32(2, x_35781);
                int32_t res_35783 = sub32(x_35782, 1);
                
                loopres_35780 = res_35783;
            } else {
                int32_t x_35784 = add32(1, node_index_35767);
                int32_t res_35785 = mul32(2, x_35784);
                
                loopres_35780 = res_35785;
            }
            
            bool res_35786 = sle32(y_31897, loopres_35780);
            bool loop_cond_35787 = !res_35786;
            bool loop_while_tmp_40311 = loop_cond_35787;
            int32_t node_index_tmp_40312 = loopres_35780;
            
            loop_while_35766 = loop_while_tmp_40311;
            node_index_35767 = node_index_tmp_40312;
        }
        leaf_35764 = loop_while_35766;
        leaf_35765 = node_index_35767;
        
        int32_t res_35788 = sub32(leaf_35765, q_31864);
        
        ((__global int32_t *) mem_39430)[sext_i32_i64(gtid_35721)] = res_35788;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_35758
}
__kernel void findNaturalLeavesFixKzisegmap_35799(__global int *global_failure,
                                                  int32_t n_31866,
                                                  int32_t res_31998,
                                                  int32_t res_31999,
                                                  int32_t res_32000, __global
                                                  unsigned char *mem_param_39438,
                                                  __global
                                                  unsigned char *mem_param_39443,
                                                  __global
                                                  unsigned char *mem_39451,
                                                  __global
                                                  unsigned char *mem_39454,
                                                  __global
                                                  unsigned char *mem_39457,
                                                  __global
                                                  unsigned char *mem_39460,
                                                  __global
                                                  unsigned char *mem_39463,
                                                  __global
                                                  unsigned char *mem_39466,
                                                  __global
                                                  unsigned char *mem_39469)
{
    #define segmap_group_sizze_35803 (findNaturalLeavesFixKzisegmap_group_sizze_35802)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40415;
    int32_t local_tid_40416;
    int32_t group_sizze_40419;
    int32_t wave_sizze_40418;
    int32_t group_tid_40417;
    
    global_tid_40415 = get_global_id(0);
    local_tid_40416 = get_local_id(0);
    group_sizze_40419 = get_local_size(0);
    wave_sizze_40418 = LOCKSTEP_WIDTH;
    group_tid_40417 = get_group_id(0);
    
    int32_t phys_tid_35799;
    
    phys_tid_35799 = global_tid_40415;
    
    int32_t write_i_35798;
    
    write_i_35798 = sext_i64_i32(sext_i32_i64(group_tid_40417) *
        sext_i32_i64(segmap_group_sizze_35803) + sext_i32_i64(local_tid_40416));
    if (slt32(write_i_35798, n_31866)) {
        int32_t x_32005 = ((__global
                            int32_t *) mem_39463)[sext_i32_i64(write_i_35798)];
        int32_t write_value_32010 = ((__global
                                      int32_t *) mem_param_39438)[sext_i32_i64(write_i_35798)];
        int32_t write_value_32011 = ((__global
                                      int32_t *) mem_param_39443)[sext_i32_i64(write_i_35798)];
        bool match_lit_32012 = 0 == x_32005;
        int32_t res_32013;
        
        if (match_lit_32012) {
            int32_t x_32006 = ((__global
                                int32_t *) mem_39451)[sext_i32_i64(write_i_35798)];
            int32_t res_32014 = sub32(x_32006, 1);
            
            res_32013 = res_32014;
        } else {
            int32_t x_32007 = ((__global
                                int32_t *) mem_39454)[sext_i32_i64(write_i_35798)];
            int32_t x_32008 = ((__global
                                int32_t *) mem_39457)[sext_i32_i64(write_i_35798)];
            int32_t x_32009 = ((__global
                                int32_t *) mem_39460)[sext_i32_i64(write_i_35798)];
            bool match_lit_32015 = 1 == x_32005;
            int32_t x_32016;
            
            if (match_lit_32015) {
                int32_t x_32017 = add32(res_31998, x_32007);
                int32_t res_32018 = sub32(x_32017, 1);
                
                x_32016 = res_32018;
            } else {
                bool match_lit_32019 = 2 == x_32005;
                int32_t x_32020;
                
                if (match_lit_32019) {
                    int32_t x_32021 = add32(res_31998, res_31999);
                    int32_t x_32022 = add32(x_32008, x_32021);
                    int32_t res_32023 = sub32(x_32022, 1);
                    
                    x_32020 = res_32023;
                } else {
                    int32_t x_32024 = add32(res_31998, res_31999);
                    int32_t x_32025 = add32(res_32000, x_32024);
                    int32_t x_32026 = add32(x_32009, x_32025);
                    int32_t res_32027 = sub32(x_32026, 1);
                    
                    x_32020 = res_32027;
                }
                x_32016 = x_32020;
            }
            res_32013 = x_32016;
        }
        if (sle32(0, res_32013) && slt32(res_32013, n_31866)) {
            ((__global int32_t *) mem_39466)[sext_i32_i64(res_32013)] =
                write_value_32010;
        }
        if (sle32(0, res_32013) && slt32(res_32013, n_31866)) {
            ((__global int32_t *) mem_39469)[sext_i32_i64(res_32013)] =
                write_value_32011;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_35803
}
__kernel void findNaturalLeavesFixKzisegmap_36062(__global int *global_failure,
                                                  int failure_is_an_option,
                                                  __global
                                                  int *global_failure_args,
                                                  int32_t d_31863,
                                                  int32_t n_31866,
                                                  int32_t num_leaves_31877,
                                                  int32_t ppl_31891,
                                                  int32_t num_groups_36252,
                                                  __global
                                                  unsigned char *mem_39486,
                                                  __global
                                                  unsigned char *mem_39488,
                                                  __global
                                                  unsigned char *mem_39496,
                                                  __global
                                                  unsigned char *mem_39510,
                                                  __global
                                                  unsigned char *mem_39517,
                                                  __global
                                                  unsigned char *mem_39584,
                                                  __global
                                                  unsigned char *mem_39588)
{
    #define segmap_group_sizze_36251 (findNaturalLeavesFixKzisegmap_group_sizze_36065)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40449;
    int32_t local_tid_40450;
    int32_t group_sizze_40453;
    int32_t wave_sizze_40452;
    int32_t group_tid_40451;
    
    global_tid_40449 = get_global_id(0);
    local_tid_40450 = get_local_id(0);
    group_sizze_40453 = get_local_size(0);
    wave_sizze_40452 = LOCKSTEP_WIDTH;
    group_tid_40451 = get_group_id(0);
    
    int32_t phys_tid_36062;
    
    phys_tid_36062 = global_tid_40449;
    
    int32_t phys_group_id_40454;
    
    phys_group_id_40454 = get_group_id(0);
    for (int32_t i_40455 = 0; i_40455 < sdiv_up32(sdiv_up32(n_31866,
                                                            segmap_group_sizze_36251) -
                                                  phys_group_id_40454,
                                                  num_groups_36252);
         i_40455++) {
        int32_t virt_group_id_40456 = phys_group_id_40454 + i_40455 *
                num_groups_36252;
        int32_t gtid_36061 = sext_i64_i32(sext_i32_i64(virt_group_id_40456) *
                sext_i32_i64(segmap_group_sizze_36251) +
                sext_i32_i64(local_tid_40450));
        
        if (slt32(gtid_36061, n_31866)) {
            int32_t res_36256 = ((__global
                                  int32_t *) mem_39496)[sext_i32_i64(gtid_36061)];
            int32_t bruteForce_arg_36258 = mul32(ppl_31891, res_36256);
            bool x_36259 = sle32(0, res_36256);
            bool y_36260 = slt32(res_36256, num_leaves_31877);
            bool bounds_check_36261 = x_36259 && y_36260;
            bool index_certs_36262;
            
            if (!bounds_check_36261) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 51) ==
                        -1) {
                        global_failure_args[0] = res_36256;
                        global_failure_args[1] = num_leaves_31877;
                        ;
                    }
                    local_failure = true;
                    goto error_0;
                }
            }
            
            int32_t double_buffer_mem_40098[8];
            float double_buffer_mem_40099[8];
            
            for (int32_t i_40457 = 0; i_40457 < 8; i_40457++) {
                double_buffer_mem_40098[sext_i32_i64(i_40457)] = ((__global
                                                                   int32_t *) mem_39486)[sext_i32_i64(i_40457)];
            }
            for (int32_t i_40458 = 0; i_40458 < 8; i_40458++) {
                double_buffer_mem_40099[sext_i32_i64(i_40458)] = ((__global
                                                                   float *) mem_39488)[sext_i32_i64(i_40458)];
            }
            
            float mem_40084[8];
            int32_t mem_40087[8];
            
            for (int32_t i_36267 = 0; i_36267 < ppl_31891; i_36267++) {
                float res_36271;
                float res_36273 = 0.0F;
                float x_36274;
                float y_36275;
                
                for (int32_t i_36272 = 0; i_36272 < d_31863; i_36272++) {
                    x_36274 = ((__global
                                float *) mem_39510)[sext_i32_i64(gtid_36061) +
                                                    sext_i32_i64(i_36272) *
                                                    sext_i32_i64(n_31866)];
                    y_36275 = ((__global
                                float *) mem_39517)[sext_i32_i64(res_36256 *
                                                    ppl_31891 + i_36267) +
                                                    sext_i32_i64(i_36272) *
                                                    sext_i32_i64(ppl_31891 *
                                                    num_leaves_31877)];
                    
                    float zz_36276;
                    
                    zz_36276 = x_36274 - y_36275;
                    
                    float y_36277 = zz_36276 * zz_36276;
                    float loopres_36278 = res_36273 + y_36277;
                    float res_tmp_40461 = loopres_36278;
                    
                    res_36273 = res_tmp_40461;
                }
                res_36271 = res_36273;
                
                float y_36279 = double_buffer_mem_40099[7];
                bool cond_36280 = y_36279 < res_36271;
                
                if (cond_36280) {
                    for (int32_t i_40462 = 0; i_40462 < 8; i_40462++) {
                        mem_40087[sext_i32_i64(i_40462)] =
                            double_buffer_mem_40098[sext_i32_i64(i_40462)];
                    }
                    for (int32_t i_40463 = 0; i_40463 < 8; i_40463++) {
                        mem_40084[sext_i32_i64(i_40463)] =
                            double_buffer_mem_40099[sext_i32_i64(i_40463)];
                    }
                } else {
                    int32_t ref_ind_36283 = add32(bruteForce_arg_36258,
                                                  i_36267);
                    float knnszq_36284;
                    int32_t knnszq_36285;
                    float dist_36289;
                    int32_t ref_ind_36290;
                    
                    dist_36289 = res_36271;
                    ref_ind_36290 = ref_ind_36283;
                    for (int32_t j_36288 = 0; j_36288 < 8; j_36288++) {
                        float cur_nn_36293 =
                              double_buffer_mem_40099[sext_i32_i64(j_36288)];
                        bool cond_36294 = cur_nn_36293 <= dist_36289;
                        float loopres_36295;
                        
                        if (cond_36294) {
                            loopres_36295 = dist_36289;
                        } else {
                            loopres_36295 = cur_nn_36293;
                        }
                        
                        int32_t loopres_36296;
                        
                        if (cond_36294) {
                            loopres_36296 = ref_ind_36290;
                        } else {
                            int32_t tmp_ind_36299 =
                                    double_buffer_mem_40098[sext_i32_i64(j_36288)];
                            
                            double_buffer_mem_40098[sext_i32_i64(j_36288)] =
                                ref_ind_36290;
                            double_buffer_mem_40099[sext_i32_i64(j_36288)] =
                                dist_36289;
                            loopres_36296 = tmp_ind_36299;
                        }
                        
                        float dist_tmp_40464 = loopres_36295;
                        int32_t ref_ind_tmp_40465 = loopres_36296;
                        
                        dist_36289 = dist_tmp_40464;
                        ref_ind_36290 = ref_ind_tmp_40465;
                    }
                    knnszq_36284 = dist_36289;
                    knnszq_36285 = ref_ind_36290;
                    for (int32_t i_40468 = 0; i_40468 < 8; i_40468++) {
                        mem_40087[sext_i32_i64(i_40468)] =
                            double_buffer_mem_40098[sext_i32_i64(i_40468)];
                    }
                    for (int32_t i_40469 = 0; i_40469 < 8; i_40469++) {
                        mem_40084[sext_i32_i64(i_40469)] =
                            double_buffer_mem_40099[sext_i32_i64(i_40469)];
                    }
                }
                for (int32_t i_40470 = 0; i_40470 < 8; i_40470++) {
                    double_buffer_mem_40098[sext_i32_i64(i_40470)] =
                        mem_40087[sext_i32_i64(i_40470)];
                }
                for (int32_t i_40471 = 0; i_40471 < 8; i_40471++) {
                    double_buffer_mem_40099[sext_i32_i64(i_40471)] =
                        mem_40084[sext_i32_i64(i_40471)];
                }
            }
            for (int32_t i_40472 = 0; i_40472 < 8; i_40472++) {
                ((__global int32_t *) mem_39584)[sext_i32_i64(i_40472) *
                                                 sext_i32_i64(n_31866) +
                                                 sext_i32_i64(gtid_36061)] =
                    double_buffer_mem_40098[sext_i32_i64(i_40472)];
            }
            for (int32_t i_40473 = 0; i_40473 < 8; i_40473++) {
                ((__global float *) mem_39588)[sext_i32_i64(i_40473) *
                                               sext_i32_i64(n_31866) +
                                               sext_i32_i64(gtid_36061)] =
                    double_buffer_mem_40099[sext_i32_i64(i_40473)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36251
}
__kernel void findNaturalLeavesFixKzisegmap_36146(__global int *global_failure,
                                                  int failure_is_an_option,
                                                  __global
                                                  int *global_failure_args,
                                                  int32_t d_31863,
                                                  int32_t n_31866,
                                                  int32_t d_31867, __global
                                                  unsigned char *queries_mem_39421,
                                                  __global
                                                  unsigned char *res_mem_39484,
                                                  __global
                                                  unsigned char *mem_39491,
                                                  __global
                                                  unsigned char *mem_39493,
                                                  __global
                                                  unsigned char *mem_39505)
{
    #define segmap_group_sizze_36233 (findNaturalLeavesFixKzisegmap_group_sizze_36151)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40443;
    int32_t local_tid_40444;
    int32_t group_sizze_40447;
    int32_t wave_sizze_40446;
    int32_t group_tid_40445;
    
    global_tid_40443 = get_global_id(0);
    local_tid_40444 = get_local_id(0);
    group_sizze_40447 = get_local_size(0);
    wave_sizze_40446 = LOCKSTEP_WIDTH;
    group_tid_40445 = get_group_id(0);
    
    int32_t phys_tid_36146;
    
    phys_tid_36146 = global_tid_40443;
    
    int32_t gtid_36144;
    
    gtid_36144 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40445) *
                                      sext_i32_i64(segmap_group_sizze_36233) +
                                      sext_i32_i64(local_tid_40444),
                                      sext_i32_i64(d_31863)));
    
    int32_t gtid_36145;
    
    gtid_36145 = sext_i64_i32(sext_i32_i64(group_tid_40445) *
        sext_i32_i64(segmap_group_sizze_36233) + sext_i32_i64(local_tid_40444) -
        squot64(sext_i32_i64(group_tid_40445) *
                sext_i32_i64(segmap_group_sizze_36233) +
                sext_i32_i64(local_tid_40444), sext_i32_i64(d_31863)) *
        sext_i32_i64(d_31863));
    if (slt32(gtid_36144, n_31866) && slt32(gtid_36145, d_31863)) {
        int32_t x_36238 = ((__global
                            int32_t *) res_mem_39484)[sext_i32_i64(gtid_36144)];
        bool bounds_check_36239 = ((__global
                                    bool *) mem_39491)[sext_i32_i64(gtid_36144)];
        bool index_certs_36240 = ((__global
                                   bool *) mem_39493)[sext_i32_i64(gtid_36144)];
        bool index_certs_36246;
        
        if (!bounds_check_36239) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 50) == -1) {
                    global_failure_args[0] = x_36238;
                    global_failure_args[1] = gtid_36145;
                    global_failure_args[2] = n_31866;
                    global_failure_args[3] = d_31863;
                    ;
                }
                return;
            }
        }
        
        float res_36247 = ((__global
                            float *) queries_mem_39421)[sext_i32_i64(x_36238) *
                                                        sext_i32_i64(d_31867) +
                                                        sext_i32_i64(gtid_36145)];
        
        ((__global float *) mem_39505)[sext_i32_i64(gtid_36144) *
                                       sext_i32_i64(d_31863) +
                                       sext_i32_i64(gtid_36145)] = res_36247;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36233
}
__kernel void findNaturalLeavesFixKzisegmap_36186(__global int *global_failure,
                                                  int failure_is_an_option,
                                                  __global
                                                  int *global_failure_args,
                                                  int32_t n_31866, __global
                                                  unsigned char *mem_39430,
                                                  __global
                                                  unsigned char *res_mem_39484,
                                                  __global
                                                  unsigned char *mem_39491,
                                                  __global
                                                  unsigned char *mem_39493,
                                                  __global
                                                  unsigned char *mem_39496,
                                                  __global
                                                  unsigned char *mem_39499)
{
    #define segmap_group_sizze_36203 (findNaturalLeavesFixKzisegmap_group_sizze_36189)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40438;
    int32_t local_tid_40439;
    int32_t group_sizze_40442;
    int32_t wave_sizze_40441;
    int32_t group_tid_40440;
    
    global_tid_40438 = get_global_id(0);
    local_tid_40439 = get_local_id(0);
    group_sizze_40442 = get_local_size(0);
    wave_sizze_40441 = LOCKSTEP_WIDTH;
    group_tid_40440 = get_group_id(0);
    
    int32_t phys_tid_36186;
    
    phys_tid_36186 = global_tid_40438;
    
    int32_t gtid_36185;
    
    gtid_36185 = sext_i64_i32(sext_i32_i64(group_tid_40440) *
        sext_i32_i64(segmap_group_sizze_36203) + sext_i32_i64(local_tid_40439));
    if (slt32(gtid_36185, n_31866)) {
        int32_t x_36211 = ((__global
                            int32_t *) res_mem_39484)[sext_i32_i64(gtid_36185)];
        bool x_36212 = sle32(0, x_36211);
        bool y_36213 = slt32(x_36211, n_31866);
        bool bounds_check_36214 = x_36212 && y_36213;
        bool index_certs_36215;
        
        if (!bounds_check_36214) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 49) == -1) {
                    global_failure_args[0] = x_36211;
                    global_failure_args[1] = n_31866;
                    ;
                }
                return;
            }
        }
        
        int32_t res_36216 = ((__global
                              int32_t *) mem_39430)[sext_i32_i64(x_36211)];
        int32_t x_36217 = x_36211;
        
        ((__global bool *) mem_39491)[sext_i32_i64(gtid_36185)] =
            bounds_check_36214;
        ((__global bool *) mem_39493)[sext_i32_i64(gtid_36185)] =
            index_certs_36215;
        ((__global int32_t *) mem_39496)[sext_i32_i64(gtid_36185)] = res_36216;
        ((__global int32_t *) mem_39499)[sext_i32_i64(gtid_36185)] = x_36217;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36203
}
__kernel void findNaturalLeavesFixKzisegmap_36303(__global int *global_failure,
                                                  int failure_is_an_option,
                                                  __global
                                                  int *global_failure_args,
                                                  int32_t n_31866,
                                                  int32_t nk_32108, __global
                                                  unsigned char *mem_39499,
                                                  __global
                                                  unsigned char *mem_39591,
                                                  __global
                                                  unsigned char *mem_39594,
                                                  __global
                                                  unsigned char *mem_39598,
                                                  __global
                                                  unsigned char *mem_39602)
{
    #define segmap_group_sizze_36307 (findNaturalLeavesFixKzisegmap_group_sizze_36306)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40474;
    int32_t local_tid_40475;
    int32_t group_sizze_40478;
    int32_t wave_sizze_40477;
    int32_t group_tid_40476;
    
    global_tid_40474 = get_global_id(0);
    local_tid_40475 = get_local_id(0);
    group_sizze_40478 = get_local_size(0);
    wave_sizze_40477 = LOCKSTEP_WIDTH;
    group_tid_40476 = get_group_id(0);
    
    int32_t phys_tid_36303;
    
    phys_tid_36303 = global_tid_40474;
    
    int32_t write_i_36302;
    
    write_i_36302 = sext_i64_i32(sext_i32_i64(group_tid_40476) *
        sext_i32_i64(segmap_group_sizze_36307) + sext_i32_i64(local_tid_40475));
    if (slt32(write_i_36302, nk_32108)) {
        int32_t new_index_39287 = squot32(write_i_36302, 8);
        int32_t binop_y_39289 = 8 * new_index_39287;
        int32_t new_index_39290 = write_i_36302 - binop_y_39289;
        int32_t write_value_32120 = ((__global
                                      int32_t *) mem_39598)[sext_i32_i64(new_index_39287) *
                                                            8 +
                                                            sext_i32_i64(new_index_39290)];
        float write_value_32121 = ((__global
                                    float *) mem_39602)[sext_i32_i64(new_index_39287) *
                                                        8 +
                                                        sext_i32_i64(new_index_39290)];
        int32_t res_32122 = sdiv32(write_i_36302, 8);
        int32_t res_32123 = smod32(write_i_36302, 8);
        bool x_32124 = sle32(0, res_32122);
        bool y_32125 = slt32(res_32122, n_31866);
        bool bounds_check_32126 = x_32124 && y_32125;
        bool index_certs_32127;
        
        if (!bounds_check_32126) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 52) == -1) {
                    global_failure_args[0] = res_32122;
                    global_failure_args[1] = n_31866;
                    ;
                }
                return;
            }
        }
        
        int32_t x_32128 = ((__global
                            int32_t *) mem_39499)[sext_i32_i64(res_32122)];
        int32_t x_32129 = mul32(8, x_32128);
        int32_t res_32130 = add32(res_32123, x_32129);
        
        if (sle32(0, res_32130) && slt32(res_32130, nk_32108)) {
            ((__global int32_t *) mem_39591)[sext_i32_i64(res_32130)] =
                write_value_32120;
        }
        if (sle32(0, res_32130) && slt32(res_32130, nk_32108)) {
            ((__global float *) mem_39594)[sext_i32_i64(res_32130)] =
                write_value_32121;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_36307
}
__kernel void gpu_map_transpose_f32(const int block_dim0, const int block_dim1,
                                    const int block_dim2,
                                    uint block_9_backing_offset_0,
                                    int32_t destoffset_1, int32_t srcoffset_3,
                                    int32_t num_arrays_4, int32_t x_elems_5,
                                    int32_t y_elems_6, int32_t mulx_7,
                                    int32_t muly_8, __global
                                    unsigned char *destmem_0, __global
                                    unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {
                ((__local float *) block_9)[sext_i32_i64((get_local_id_1_39 +
                                                          j_43 * 8) * 33 +
                                            get_local_id_0_38)] = ((__global
                                                                    float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                       index_in_35)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {
                ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                               index_out_36)] = ((__local
                                                                  float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                                    33 +
                                                                                    get_local_id_1_39 +
                                                                                    j_43 *
                                                                                    8)];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_low_height(const int block_dim0, const
                                               int block_dim1, const
                                               int block_dim2,
                                               uint block_9_backing_offset_0,
                                               int32_t destoffset_1,
                                               int32_t srcoffset_3,
                                               int32_t num_arrays_4,
                                               int32_t x_elems_5,
                                               int32_t y_elems_6,
                                               int32_t mulx_7, int32_t muly_8,
                                               __global
                                               unsigned char *destmem_0,
                                               __global unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_7) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_7);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local float *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                    get_local_id_0_38)] = ((__global
                                                            float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);
    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_7) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__local
                                                          float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                            17 +
                                                                            get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_low_width(const int block_dim0, const
                                              int block_dim1, const
                                              int block_dim2,
                                              uint block_9_backing_offset_0,
                                              int32_t destoffset_1,
                                              int32_t srcoffset_3,
                                              int32_t num_arrays_4,
                                              int32_t x_elems_5,
                                              int32_t y_elems_6, int32_t mulx_7,
                                              int32_t muly_8, __global
                                              unsigned char *destmem_0, __global
                                              unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_8);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_8) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local float *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                    get_local_id_0_38)] = ((__global
                                                            float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_8) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_8);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__local
                                                          float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                            17 +
                                                                            get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_small(uint block_9_backing_offset_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t mulx_7, int32_t muly_8,
                                          __global unsigned char *destmem_0,
                                          __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__global
                                                          float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                             index_in_35)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_i32(const int block_dim0, const int block_dim1,
                                    const int block_dim2,
                                    uint block_9_backing_offset_0,
                                    int32_t destoffset_1, int32_t srcoffset_3,
                                    int32_t num_arrays_4, int32_t x_elems_5,
                                    int32_t y_elems_6, int32_t mulx_7,
                                    int32_t muly_8, __global
                                    unsigned char *destmem_0, __global
                                    unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {
                ((__local int32_t *) block_9)[sext_i32_i64((get_local_id_1_39 +
                                                            j_43 * 8) * 33 +
                                              get_local_id_0_38)] = ((__global
                                                                      int32_t *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                           index_in_35)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {
                ((__global int32_t *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                                 index_out_36)] = ((__local
                                                                    int32_t *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                                        33 +
                                                                                        get_local_id_1_39 +
                                                                                        j_43 *
                                                                                        8)];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_i32_low_height(const int block_dim0, const
                                               int block_dim1, const
                                               int block_dim2,
                                               uint block_9_backing_offset_0,
                                               int32_t destoffset_1,
                                               int32_t srcoffset_3,
                                               int32_t num_arrays_4,
                                               int32_t x_elems_5,
                                               int32_t y_elems_6,
                                               int32_t mulx_7, int32_t muly_8,
                                               __global
                                               unsigned char *destmem_0,
                                               __global unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_7) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_7);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local int32_t *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                      get_local_id_0_38)] = ((__global
                                                              int32_t *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                   index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);
    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_7) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global int32_t *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                         index_out_36)] = ((__local
                                                            int32_t *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                                17 +
                                                                                get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_i32_low_width(const int block_dim0, const
                                              int block_dim1, const
                                              int block_dim2,
                                              uint block_9_backing_offset_0,
                                              int32_t destoffset_1,
                                              int32_t srcoffset_3,
                                              int32_t num_arrays_4,
                                              int32_t x_elems_5,
                                              int32_t y_elems_6, int32_t mulx_7,
                                              int32_t muly_8, __global
                                              unsigned char *destmem_0, __global
                                              unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_8);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_8) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local int32_t *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                      get_local_id_0_38)] = ((__global
                                                              int32_t *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                   index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_8) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_8);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global int32_t *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                         index_out_36)] = ((__local
                                                            int32_t *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                                17 +
                                                                                get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_i32_small(uint block_9_backing_offset_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t mulx_7, int32_t muly_8,
                                          __global unsigned char *destmem_0,
                                          __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {
        ((__global int32_t *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                         index_out_36)] = ((__global
                                                            int32_t *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                 index_in_35)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_i8(const int block_dim0, const int block_dim1,
                                   const int block_dim2,
                                   uint block_9_backing_offset_0,
                                   int32_t destoffset_1, int32_t srcoffset_3,
                                   int32_t num_arrays_4, int32_t x_elems_5,
                                   int32_t y_elems_6, int32_t mulx_7,
                                   int32_t muly_8, __global
                                   unsigned char *destmem_0, __global
                                   unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = destoffset_1 + our_array_offset_30;
    int32_t idata_offset_34 = srcoffset_3 + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {
                ((__local int8_t *) block_9)[sext_i32_i64((get_local_id_1_39 +
                                                           j_43 * 8) * 33 +
                                             get_local_id_0_38)] = ((__global
                                                                     int8_t *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                         index_in_35)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {
                ((__global int8_t *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                                index_out_36)] = ((__local
                                                                   int8_t *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                                      33 +
                                                                                      get_local_id_1_39 +
                                                                                      j_43 *
                                                                                      8)];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_i8_low_height(const int block_dim0, const
                                              int block_dim1, const
                                              int block_dim2,
                                              uint block_9_backing_offset_0,
                                              int32_t destoffset_1,
                                              int32_t srcoffset_3,
                                              int32_t num_arrays_4,
                                              int32_t x_elems_5,
                                              int32_t y_elems_6, int32_t mulx_7,
                                              int32_t muly_8, __global
                                              unsigned char *destmem_0, __global
                                              unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = destoffset_1 + our_array_offset_30;
    int32_t idata_offset_34 = srcoffset_3 + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_7) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_7);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local int8_t *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                     get_local_id_0_38)] = ((__global
                                                             int8_t *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                 index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);
    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_7) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global int8_t *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                        index_out_36)] = ((__local
                                                           int8_t *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                              17 +
                                                                              get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_i8_low_width(const int block_dim0, const
                                             int block_dim1, const
                                             int block_dim2,
                                             uint block_9_backing_offset_0,
                                             int32_t destoffset_1,
                                             int32_t srcoffset_3,
                                             int32_t num_arrays_4,
                                             int32_t x_elems_5,
                                             int32_t y_elems_6, int32_t mulx_7,
                                             int32_t muly_8, __global
                                             unsigned char *destmem_0, __global
                                             unsigned char *srcmem_2)
{
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = destoffset_1 + our_array_offset_30;
    int32_t idata_offset_34 = srcoffset_3 + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_8);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_8) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local int8_t *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                     get_local_id_0_38)] = ((__global
                                                             int8_t *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                 index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_8) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_8);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global int8_t *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                        index_out_36)] = ((__local
                                                           int8_t *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                              17 +
                                                                              get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_i8_small(uint block_9_backing_offset_0,
                                         int32_t destoffset_1,
                                         int32_t srcoffset_3,
                                         int32_t num_arrays_4,
                                         int32_t x_elems_5, int32_t y_elems_6,
                                         int32_t mulx_7, int32_t muly_8,
                                         __global unsigned char *destmem_0,
                                         __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *block_9_backing_0 = &shared_mem[block_9_backing_offset_0];
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = destoffset_1 + our_array_offset_30;
    int32_t idata_offset_34 = srcoffset_3 + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {
        ((__global int8_t *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                        index_out_36)] = ((__global
                                                           int8_t *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    
  error_0:
    return;
}
__kernel void mkImgPatcheszisegmap_38762(__global int *global_failure,
                                         int failure_is_an_option, __global
                                         int *global_failure_args,
                                         int32_t h_32683, int32_t w_32684,
                                         int32_t c_32685, int32_t p_32686,
                                         int32_t n_cols_32689,
                                         int32_t n_rows_32691, __global
                                         unsigned char *img_mem_39418, __global
                                         unsigned char *mem_39429)
{
    #define segmap_group_sizze_39262 (mkImgPatcheszisegmap_group_sizze_38773)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40298;
    int32_t local_tid_40299;
    int32_t group_sizze_40302;
    int32_t wave_sizze_40301;
    int32_t group_tid_40300;
    
    global_tid_40298 = get_global_id(0);
    local_tid_40299 = get_local_id(0);
    group_sizze_40302 = get_local_size(0);
    wave_sizze_40301 = LOCKSTEP_WIDTH;
    group_tid_40300 = get_group_id(0);
    
    int32_t phys_tid_38762;
    
    phys_tid_38762 = global_tid_40298;
    
    int32_t gtid_38757;
    
    gtid_38757 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40300) *
                                      sext_i32_i64(segmap_group_sizze_39262) +
                                      sext_i32_i64(local_tid_40299),
                                      sext_i32_i64(n_cols_32689) *
                                      sext_i32_i64(p_32686) *
                                      sext_i32_i64(p_32686) *
                                      sext_i32_i64(c_32685)));
    
    int32_t gtid_38758;
    
    gtid_38758 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40300) *
                                      sext_i32_i64(segmap_group_sizze_39262) +
                                      sext_i32_i64(local_tid_40299) -
                                      squot64(sext_i32_i64(group_tid_40300) *
                                              sext_i32_i64(segmap_group_sizze_39262) +
                                              sext_i32_i64(local_tid_40299),
                                              sext_i32_i64(n_cols_32689) *
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(c_32685)) *
                                      (sext_i32_i64(n_cols_32689) *
                                       sext_i32_i64(p_32686) *
                                       sext_i32_i64(p_32686) *
                                       sext_i32_i64(c_32685)),
                                      sext_i32_i64(p_32686) *
                                      sext_i32_i64(p_32686) *
                                      sext_i32_i64(c_32685)));
    
    int32_t gtid_38759;
    
    gtid_38759 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40300) *
                                      sext_i32_i64(segmap_group_sizze_39262) +
                                      sext_i32_i64(local_tid_40299) -
                                      squot64(sext_i32_i64(group_tid_40300) *
                                              sext_i32_i64(segmap_group_sizze_39262) +
                                              sext_i32_i64(local_tid_40299),
                                              sext_i32_i64(n_cols_32689) *
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(c_32685)) *
                                      (sext_i32_i64(n_cols_32689) *
                                       sext_i32_i64(p_32686) *
                                       sext_i32_i64(p_32686) *
                                       sext_i32_i64(c_32685)) -
                                      squot64(sext_i32_i64(group_tid_40300) *
                                              sext_i32_i64(segmap_group_sizze_39262) +
                                              sext_i32_i64(local_tid_40299) -
                                              squot64(sext_i32_i64(group_tid_40300) *
                                                      sext_i32_i64(segmap_group_sizze_39262) +
                                                      sext_i32_i64(local_tid_40299),
                                                      sext_i32_i64(n_cols_32689) *
                                                      sext_i32_i64(p_32686) *
                                                      sext_i32_i64(p_32686) *
                                                      sext_i32_i64(c_32685)) *
                                              (sext_i32_i64(n_cols_32689) *
                                               sext_i32_i64(p_32686) *
                                               sext_i32_i64(p_32686) *
                                               sext_i32_i64(c_32685)),
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(c_32685)) *
                                      (sext_i32_i64(p_32686) *
                                       sext_i32_i64(p_32686) *
                                       sext_i32_i64(c_32685)),
                                      sext_i32_i64(p_32686) *
                                      sext_i32_i64(c_32685)));
    
    int32_t gtid_38760;
    
    gtid_38760 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40300) *
                                      sext_i32_i64(segmap_group_sizze_39262) +
                                      sext_i32_i64(local_tid_40299) -
                                      squot64(sext_i32_i64(group_tid_40300) *
                                              sext_i32_i64(segmap_group_sizze_39262) +
                                              sext_i32_i64(local_tid_40299),
                                              sext_i32_i64(n_cols_32689) *
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(c_32685)) *
                                      (sext_i32_i64(n_cols_32689) *
                                       sext_i32_i64(p_32686) *
                                       sext_i32_i64(p_32686) *
                                       sext_i32_i64(c_32685)) -
                                      squot64(sext_i32_i64(group_tid_40300) *
                                              sext_i32_i64(segmap_group_sizze_39262) +
                                              sext_i32_i64(local_tid_40299) -
                                              squot64(sext_i32_i64(group_tid_40300) *
                                                      sext_i32_i64(segmap_group_sizze_39262) +
                                                      sext_i32_i64(local_tid_40299),
                                                      sext_i32_i64(n_cols_32689) *
                                                      sext_i32_i64(p_32686) *
                                                      sext_i32_i64(p_32686) *
                                                      sext_i32_i64(c_32685)) *
                                              (sext_i32_i64(n_cols_32689) *
                                               sext_i32_i64(p_32686) *
                                               sext_i32_i64(p_32686) *
                                               sext_i32_i64(c_32685)),
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(c_32685)) *
                                      (sext_i32_i64(p_32686) *
                                       sext_i32_i64(p_32686) *
                                       sext_i32_i64(c_32685)) -
                                      squot64(sext_i32_i64(group_tid_40300) *
                                              sext_i32_i64(segmap_group_sizze_39262) +
                                              sext_i32_i64(local_tid_40299) -
                                              squot64(sext_i32_i64(group_tid_40300) *
                                                      sext_i32_i64(segmap_group_sizze_39262) +
                                                      sext_i32_i64(local_tid_40299),
                                                      sext_i32_i64(n_cols_32689) *
                                                      sext_i32_i64(p_32686) *
                                                      sext_i32_i64(p_32686) *
                                                      sext_i32_i64(c_32685)) *
                                              (sext_i32_i64(n_cols_32689) *
                                               sext_i32_i64(p_32686) *
                                               sext_i32_i64(p_32686) *
                                               sext_i32_i64(c_32685)) -
                                              squot64(sext_i32_i64(group_tid_40300) *
                                                      sext_i32_i64(segmap_group_sizze_39262) +
                                                      sext_i32_i64(local_tid_40299) -
                                                      squot64(sext_i32_i64(group_tid_40300) *
                                                              sext_i32_i64(segmap_group_sizze_39262) +
                                                              sext_i32_i64(local_tid_40299),
                                                              sext_i32_i64(n_cols_32689) *
                                                              sext_i32_i64(p_32686) *
                                                              sext_i32_i64(p_32686) *
                                                              sext_i32_i64(c_32685)) *
                                                      (sext_i32_i64(n_cols_32689) *
                                                       sext_i32_i64(p_32686) *
                                                       sext_i32_i64(p_32686) *
                                                       sext_i32_i64(c_32685)),
                                                      sext_i32_i64(p_32686) *
                                                      sext_i32_i64(p_32686) *
                                                      sext_i32_i64(c_32685)) *
                                              (sext_i32_i64(p_32686) *
                                               sext_i32_i64(p_32686) *
                                               sext_i32_i64(c_32685)),
                                              sext_i32_i64(p_32686) *
                                              sext_i32_i64(c_32685)) *
                                      (sext_i32_i64(p_32686) *
                                       sext_i32_i64(c_32685)),
                                      sext_i32_i64(c_32685)));
    
    int32_t gtid_38761;
    
    gtid_38761 = sext_i64_i32(sext_i32_i64(group_tid_40300) *
        sext_i32_i64(segmap_group_sizze_39262) + sext_i32_i64(local_tid_40299) -
        squot64(sext_i32_i64(group_tid_40300) *
                sext_i32_i64(segmap_group_sizze_39262) +
                sext_i32_i64(local_tid_40299), sext_i32_i64(n_cols_32689) *
                sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                sext_i32_i64(c_32685)) * (sext_i32_i64(n_cols_32689) *
                                          sext_i32_i64(p_32686) *
                                          sext_i32_i64(p_32686) *
                                          sext_i32_i64(c_32685)) -
        squot64(sext_i32_i64(group_tid_40300) *
                sext_i32_i64(segmap_group_sizze_39262) +
                sext_i32_i64(local_tid_40299) -
                squot64(sext_i32_i64(group_tid_40300) *
                        sext_i32_i64(segmap_group_sizze_39262) +
                        sext_i32_i64(local_tid_40299),
                        sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                        sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) *
                (sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                 sext_i32_i64(p_32686) * sext_i32_i64(c_32685)),
                sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                sext_i32_i64(c_32685)) * (sext_i32_i64(p_32686) *
                                          sext_i32_i64(p_32686) *
                                          sext_i32_i64(c_32685)) -
        squot64(sext_i32_i64(group_tid_40300) *
                sext_i32_i64(segmap_group_sizze_39262) +
                sext_i32_i64(local_tid_40299) -
                squot64(sext_i32_i64(group_tid_40300) *
                        sext_i32_i64(segmap_group_sizze_39262) +
                        sext_i32_i64(local_tid_40299),
                        sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                        sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) *
                (sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                 sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) -
                squot64(sext_i32_i64(group_tid_40300) *
                        sext_i32_i64(segmap_group_sizze_39262) +
                        sext_i32_i64(local_tid_40299) -
                        squot64(sext_i32_i64(group_tid_40300) *
                                sext_i32_i64(segmap_group_sizze_39262) +
                                sext_i32_i64(local_tid_40299),
                                sext_i32_i64(n_cols_32689) *
                                sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                                sext_i32_i64(c_32685)) *
                        (sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                         sext_i32_i64(p_32686) * sext_i32_i64(c_32685)),
                        sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                        sext_i32_i64(c_32685)) * (sext_i32_i64(p_32686) *
                                                  sext_i32_i64(p_32686) *
                                                  sext_i32_i64(c_32685)),
                sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) *
        (sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) -
        squot64(sext_i32_i64(group_tid_40300) *
                sext_i32_i64(segmap_group_sizze_39262) +
                sext_i32_i64(local_tid_40299) -
                squot64(sext_i32_i64(group_tid_40300) *
                        sext_i32_i64(segmap_group_sizze_39262) +
                        sext_i32_i64(local_tid_40299),
                        sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                        sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) *
                (sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                 sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) -
                squot64(sext_i32_i64(group_tid_40300) *
                        sext_i32_i64(segmap_group_sizze_39262) +
                        sext_i32_i64(local_tid_40299) -
                        squot64(sext_i32_i64(group_tid_40300) *
                                sext_i32_i64(segmap_group_sizze_39262) +
                                sext_i32_i64(local_tid_40299),
                                sext_i32_i64(n_cols_32689) *
                                sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                                sext_i32_i64(c_32685)) *
                        (sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                         sext_i32_i64(p_32686) * sext_i32_i64(c_32685)),
                        sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                        sext_i32_i64(c_32685)) * (sext_i32_i64(p_32686) *
                                                  sext_i32_i64(p_32686) *
                                                  sext_i32_i64(c_32685)) -
                squot64(sext_i32_i64(group_tid_40300) *
                        sext_i32_i64(segmap_group_sizze_39262) +
                        sext_i32_i64(local_tid_40299) -
                        squot64(sext_i32_i64(group_tid_40300) *
                                sext_i32_i64(segmap_group_sizze_39262) +
                                sext_i32_i64(local_tid_40299),
                                sext_i32_i64(n_cols_32689) *
                                sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                                sext_i32_i64(c_32685)) *
                        (sext_i32_i64(n_cols_32689) * sext_i32_i64(p_32686) *
                         sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) -
                        squot64(sext_i32_i64(group_tid_40300) *
                                sext_i32_i64(segmap_group_sizze_39262) +
                                sext_i32_i64(local_tid_40299) -
                                squot64(sext_i32_i64(group_tid_40300) *
                                        sext_i32_i64(segmap_group_sizze_39262) +
                                        sext_i32_i64(local_tid_40299),
                                        sext_i32_i64(n_cols_32689) *
                                        sext_i32_i64(p_32686) *
                                        sext_i32_i64(p_32686) *
                                        sext_i32_i64(c_32685)) *
                                (sext_i32_i64(n_cols_32689) *
                                 sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                                 sext_i32_i64(c_32685)), sext_i32_i64(p_32686) *
                                sext_i32_i64(p_32686) * sext_i32_i64(c_32685)) *
                        (sext_i32_i64(p_32686) * sext_i32_i64(p_32686) *
                         sext_i32_i64(c_32685)), sext_i32_i64(p_32686) *
                        sext_i32_i64(c_32685)) * (sext_i32_i64(p_32686) *
                                                  sext_i32_i64(c_32685)),
                sext_i32_i64(c_32685)) * sext_i32_i64(c_32685));
    if ((((slt32(gtid_38757, n_rows_32691) && slt32(gtid_38758,
                                                    n_cols_32689)) &&
          slt32(gtid_38759, p_32686)) && slt32(gtid_38760, p_32686)) &&
        slt32(gtid_38761, c_32685)) {
        int32_t index_primexp_39303 = add32(gtid_38757, gtid_38759);
        bool binop_x_39310 = sle32(0, index_primexp_39303);
        bool binop_y_39312 = slt32(index_primexp_39303, h_32683);
        bool index_primexp_39313 = binop_x_39310 && binop_y_39312;
        int32_t index_primexp_39298 = add32(gtid_38758, gtid_38760);
        bool binop_x_39305 = sle32(0, index_primexp_39298);
        bool binop_y_39307 = slt32(index_primexp_39298, w_32684);
        bool index_primexp_39308 = binop_x_39305 && binop_y_39307;
        bool index_ok_39276 = index_primexp_39308 && index_primexp_39313;
        bool index_certs_39277;
        
        if (!index_ok_39276) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 53) == -1) {
                    global_failure_args[0] = index_primexp_39303;
                    global_failure_args[1] = index_primexp_39298;
                    global_failure_args[2] = gtid_38761;
                    global_failure_args[3] = h_32683;
                    global_failure_args[4] = w_32684;
                    global_failure_args[5] = c_32685;
                    ;
                }
                return;
            }
        }
        
        int32_t i32_arg_39278 = ((__global
                                  int32_t *) img_mem_39418)[sext_i32_i64(index_primexp_39303) *
                                                            sext_i32_i64(c_32685 *
                                                            w_32684) +
                                                            sext_i32_i64(index_primexp_39298) *
                                                            sext_i32_i64(c_32685) +
                                                            sext_i32_i64(gtid_38761)];
        int8_t unsign_arg_39279 = zext_i32_i8(i32_arg_39278);
        
        ((__global int8_t *) mem_39429)[sext_i32_i64(gtid_38757) *
                                        sext_i32_i64(c_32685 * p_32686 *
                                        p_32686 * n_cols_32689) +
                                        sext_i32_i64(gtid_38758) *
                                        sext_i32_i64(c_32685 * p_32686 *
                                        p_32686) + sext_i32_i64(gtid_38759) *
                                        sext_i32_i64(c_32685 * p_32686) +
                                        sext_i32_i64(gtid_38760) *
                                        sext_i32_i64(c_32685) +
                                        sext_i32_i64(gtid_38761)] =
            unsign_arg_39279;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_39262
}
__kernel void propagateFixKzicopy_40323(int32_t nc_31615, int32_t i_31649,
                                        __global unsigned char *mem_39428,
                                        int32_t ctx_val_39482, __global
                                        unsigned char *mem_39519)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40323;
    int32_t copy_ltid_40324;
    int32_t copy_gid_40325;
    
    copy_gtid_40323 = get_global_id(0);
    copy_ltid_40324 = get_local_id(0);
    copy_gid_40325 = get_group_id(0);
    if (slt32(copy_gtid_40323, sext_i64_i32(sext_i32_i64(nc_31615) * 8))) {
        ((__global int32_t *) mem_39519)[sext_i32_i64(copy_gtid_40323 -
                                         squot32(copy_gtid_40323, 8) * 8) *
                                         sext_i32_i64(nc_31615) +
                                         sext_i32_i64(squot32(copy_gtid_40323,
                                                              8))] = ((__global
                                                                       int32_t *) mem_39428)[sext_i32_i64(i_31649 *
                                                                                             ctx_val_39482) +
                                                                                             (sext_i32_i64(squot32(copy_gtid_40323,
                                                                                                                   8)) *
                                                                                              8 +
                                                                                              sext_i32_i64(copy_gtid_40323 -
                                                                                              squot32(copy_gtid_40323,
                                                                                                      8) *
                                                                                              8))];
    }
    
  error_0:
    return;
}
__kernel void propagateFixKzicopy_40328(int32_t nc_31615, int32_t i_31649,
                                        __global unsigned char *mem_39432,
                                        int32_t ctx_val_39482, __global
                                        unsigned char *mem_39523)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40328;
    int32_t copy_ltid_40329;
    int32_t copy_gid_40330;
    
    copy_gtid_40328 = get_global_id(0);
    copy_ltid_40329 = get_local_id(0);
    copy_gid_40330 = get_group_id(0);
    if (slt32(copy_gtid_40328, sext_i64_i32(sext_i32_i64(nc_31615) * 8))) {
        ((__global float *) mem_39523)[sext_i32_i64(copy_gtid_40328 -
                                       squot32(copy_gtid_40328, 8) * 8) *
                                       sext_i32_i64(nc_31615) +
                                       sext_i32_i64(squot32(copy_gtid_40328,
                                                            8))] = ((__global
                                                                     float *) mem_39432)[sext_i32_i64(i_31649 *
                                                                                         ctx_val_39482) +
                                                                                         (sext_i32_i64(squot32(copy_gtid_40328,
                                                                                                               8)) *
                                                                                          8 +
                                                                                          sext_i32_i64(copy_gtid_40328 -
                                                                                          squot32(copy_gtid_40328,
                                                                                                  8) *
                                                                                          8))];
    }
    
  error_0:
    return;
}
__kernel void propagateFixKzicopy_40366(int32_t nc_31615, int32_t i_31649,
                                        __global unsigned char *mem_39428,
                                        int32_t ctx_val_39482, __global
                                        unsigned char *mem_39679)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40366;
    int32_t copy_ltid_40367;
    int32_t copy_gid_40368;
    
    copy_gtid_40366 = get_global_id(0);
    copy_ltid_40367 = get_local_id(0);
    copy_gid_40368 = get_group_id(0);
    if (slt32(copy_gtid_40366, sext_i64_i32(sext_i32_i64(nc_31615) * 8))) {
        ((__global int32_t *) mem_39679)[sext_i32_i64(copy_gtid_40366 -
                                         squot32(copy_gtid_40366, 8) * 8) *
                                         sext_i32_i64(nc_31615) +
                                         sext_i32_i64(squot32(copy_gtid_40366,
                                                              8))] = ((__global
                                                                       int32_t *) mem_39428)[sext_i32_i64(i_31649 *
                                                                                             ctx_val_39482) +
                                                                                             (sext_i32_i64(squot32(copy_gtid_40366,
                                                                                                                   8)) *
                                                                                              8 +
                                                                                              sext_i32_i64(copy_gtid_40366 -
                                                                                              squot32(copy_gtid_40366,
                                                                                                      8) *
                                                                                              8))];
    }
    
  error_0:
    return;
}
__kernel void propagateFixKzicopy_40371(int32_t nc_31615, int32_t i_31649,
                                        __global unsigned char *mem_39432,
                                        int32_t ctx_val_39482, __global
                                        unsigned char *mem_39683)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40371;
    int32_t copy_ltid_40372;
    int32_t copy_gid_40373;
    
    copy_gtid_40371 = get_global_id(0);
    copy_ltid_40372 = get_local_id(0);
    copy_gid_40373 = get_group_id(0);
    if (slt32(copy_gtid_40371, sext_i64_i32(sext_i32_i64(nc_31615) * 8))) {
        ((__global float *) mem_39683)[sext_i32_i64(copy_gtid_40371 -
                                       squot32(copy_gtid_40371, 8) * 8) *
                                       sext_i32_i64(nc_31615) +
                                       sext_i32_i64(squot32(copy_gtid_40371,
                                                            8))] = ((__global
                                                                     float *) mem_39432)[sext_i32_i64(i_31649 *
                                                                                         ctx_val_39482) +
                                                                                         (sext_i32_i64(squot32(copy_gtid_40371,
                                                                                                               8)) *
                                                                                          8 +
                                                                                          sext_i32_i64(copy_gtid_40371 -
                                                                                          squot32(copy_gtid_40371,
                                                                                                  8) *
                                                                                          8))];
    }
    
  error_0:
    return;
}
__kernel void propagateFixKzicopy_40410(int32_t nc_31615, int32_t i_31649,
                                        __global unsigned char *mem_39428,
                                        int32_t ctx_val_39482, __global
                                        unsigned char *mem_39831)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40410;
    int32_t copy_ltid_40411;
    int32_t copy_gid_40412;
    
    copy_gtid_40410 = get_global_id(0);
    copy_ltid_40411 = get_local_id(0);
    copy_gid_40412 = get_group_id(0);
    if (slt32(copy_gtid_40410, sext_i64_i32(sext_i32_i64(nc_31615) * 8))) {
        ((__global int32_t *) mem_39831)[sext_i32_i64(copy_gtid_40410 -
                                         squot32(copy_gtid_40410, 8) * 8) *
                                         sext_i32_i64(nc_31615) +
                                         sext_i32_i64(squot32(copy_gtid_40410,
                                                              8))] = ((__global
                                                                       int32_t *) mem_39428)[sext_i32_i64(i_31649 *
                                                                                             ctx_val_39482) +
                                                                                             (sext_i32_i64(squot32(copy_gtid_40410,
                                                                                                                   8)) *
                                                                                              8 +
                                                                                              sext_i32_i64(copy_gtid_40410 -
                                                                                              squot32(copy_gtid_40410,
                                                                                                      8) *
                                                                                              8))];
    }
    
  error_0:
    return;
}
__kernel void propagateFixKzicopy_40415(int32_t nc_31615, int32_t i_31649,
                                        __global unsigned char *mem_39432,
                                        int32_t ctx_val_39482, __global
                                        unsigned char *mem_39835)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40415;
    int32_t copy_ltid_40416;
    int32_t copy_gid_40417;
    
    copy_gtid_40415 = get_global_id(0);
    copy_ltid_40416 = get_local_id(0);
    copy_gid_40417 = get_group_id(0);
    if (slt32(copy_gtid_40415, sext_i64_i32(sext_i32_i64(nc_31615) * 8))) {
        ((__global float *) mem_39835)[sext_i32_i64(copy_gtid_40415 -
                                       squot32(copy_gtid_40415, 8) * 8) *
                                       sext_i32_i64(nc_31615) +
                                       sext_i32_i64(squot32(copy_gtid_40415,
                                                            8))] = ((__global
                                                                     float *) mem_39432)[sext_i32_i64(i_31649 *
                                                                                         ctx_val_39482) +
                                                                                         (sext_i32_i64(squot32(copy_gtid_40415,
                                                                                                               8)) *
                                                                                          8 +
                                                                                          sext_i32_i64(copy_gtid_40415 -
                                                                                          squot32(copy_gtid_40415,
                                                                                                  8) *
                                                                                          8))];
    }
    
  error_0:
    return;
}
__kernel void propagateFixKzisegmap_34700(__global int *global_failure,
                                          int failure_is_an_option, __global
                                          int *global_failure_args,
                                          int32_t m_31574, int32_t nc_31615,
                                          int32_t upper_bound_31636,
                                          int32_t im1_31646,
                                          int32_t num_groups_34831,
                                          int32_t binop_x_39322,
                                          int32_t num_threads_39324,
                                          int32_t per_chunk_39330, __global
                                          unsigned char *indir_mem_39419,
                                          __global
                                          unsigned char *orig2leaf_mem_39420,
                                          __global
                                          unsigned char *nat_leaves0_mem_39422,
                                          __global unsigned char *mem_39428,
                                          __global unsigned char *mem_39443,
                                          int32_t ctx_val_39482, __global
                                          unsigned char *mem_39511, __global
                                          unsigned char *mem_39515)
{
    #define segmap_group_sizze_34830 (propagateFixKzisegmap_group_sizze_34703)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40307;
    int32_t local_tid_40308;
    int32_t group_sizze_40311;
    int32_t wave_sizze_40310;
    int32_t group_tid_40309;
    
    global_tid_40307 = get_global_id(0);
    local_tid_40308 = get_local_id(0);
    group_sizze_40311 = get_local_size(0);
    wave_sizze_40310 = LOCKSTEP_WIDTH;
    group_tid_40309 = get_group_id(0);
    
    int32_t phys_tid_34700;
    
    phys_tid_34700 = global_tid_40307;
    
    int32_t phys_group_id_40312;
    
    phys_group_id_40312 = get_group_id(0);
    for (int32_t i_40313 = 0; i_40313 < sdiv_up32(sdiv_up32(nc_31615,
                                                            segmap_group_sizze_34830) -
                                                  phys_group_id_40312,
                                                  num_groups_34831);
         i_40313++) {
        int32_t virt_group_id_40314 = phys_group_id_40312 + i_40313 *
                num_groups_34831;
        int32_t gtid_34699 = sext_i64_i32(sext_i32_i64(virt_group_id_40314) *
                sext_i32_i64(segmap_group_sizze_34830) +
                sext_i32_i64(local_tid_40308));
        
        if (slt32(gtid_34699, nc_31615)) {
            int32_t new_index_39323 = gtid_34699 + binop_x_39322;
            int32_t x_34835 = ((__global
                                int32_t *) nat_leaves0_mem_39422)[sext_i32_i64(new_index_39323)];
            int32_t mem_39494[8];
            
            for (int32_t i_40315 = 0; i_40315 < 8; i_40315++) {
                mem_39494[sext_i32_i64(i_40315)] = -1;
            }
            
            int32_t res_34838;
            int32_t n_inds_34841 = 0;
            
            for (int32_t q_34840 = 0; q_34840 < 8; q_34840++) {
                int32_t estimateIndex_arg_34843 = ((__global
                                                    int32_t *) mem_39428)[sext_i32_i64(im1_31646) *
                                                                          sext_i32_i64(ctx_val_39482) +
                                                                          sext_i32_i64(gtid_34699) *
                                                                          8 +
                                                                          sext_i32_i64(q_34840)];
                bool x_34844 = sle32(0, estimateIndex_arg_34843);
                bool y_34845 = slt32(estimateIndex_arg_34843, m_31574);
                bool bounds_check_34846 = x_34844 && y_34845;
                bool index_certs_34847;
                
                if (!bounds_check_34846) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 54) ==
                            -1) {
                            global_failure_args[0] = estimateIndex_arg_34843;
                            global_failure_args[1] = m_31574;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                int32_t orig_par_ind_34848 = ((__global
                                               int32_t *) indir_mem_39419)[sext_i32_i64(estimateIndex_arg_34843)];
                int32_t orig_par_y_34849 = sdiv32(orig_par_ind_34848, nc_31615);
                int32_t y_34850 = mul32(nc_31615, orig_par_y_34849);
                int32_t orig_par_x_34851 = sub32(orig_par_ind_34848, y_34850);
                bool cond_34852 = slt32(orig_par_y_34849, upper_bound_31636);
                int32_t cand_y_34853;
                
                if (cond_34852) {
                    int32_t res_34854 = add32(1, orig_par_y_34849);
                    
                    cand_y_34853 = res_34854;
                } else {
                    cand_y_34853 = upper_bound_31636;
                }
                
                int32_t x_34855 = mul32(nc_31615, cand_y_34853);
                int32_t orig_cand_ind_34856 = add32(orig_par_x_34851, x_34855);
                bool x_34857 = sle32(0, orig_cand_ind_34856);
                bool y_34858 = slt32(orig_cand_ind_34856, m_31574);
                bool bounds_check_34859 = x_34857 && y_34858;
                bool index_certs_34860;
                
                if (!bounds_check_34859) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 55) ==
                            -1) {
                            global_failure_args[0] = orig_cand_ind_34856;
                            global_failure_args[1] = m_31574;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                int32_t cand_leaf_ind_34861 = ((__global
                                                int32_t *) orig2leaf_mem_39420)[sext_i32_i64(orig_cand_ind_34856)];
                bool not_found_34862 = cand_leaf_ind_34861 == x_34835;
                bool not_found_34863 = !not_found_34862;
                bool res_34864 = slt32(0, n_inds_34841);
                bool x_34865 = not_found_34863 && res_34864;
                bool not_found_34866;
                int32_t not_found_34867;
                bool not_found_34868;
                bool loop_while_34869;
                int32_t j_34870;
                bool not_found_34871;
                
                loop_while_34869 = x_34865;
                j_34870 = 0;
                not_found_34871 = not_found_34863;
                while (loop_while_34869) {
                    bool x_34872 = sle32(0, j_34870);
                    bool y_34873 = slt32(j_34870, 8);
                    bool bounds_check_34874 = x_34872 && y_34873;
                    bool index_certs_34875;
                    
                    if (!bounds_check_34874) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          56) == -1) {
                                global_failure_args[0] = j_34870;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    int32_t x_34876 = mem_39494[sext_i32_i64(j_34870)];
                    bool cond_34877 = x_34876 == cand_leaf_ind_34861;
                    bool x_34878 = !cond_34877;
                    int32_t loopres_34879;
                    
                    if (cond_34877) {
                        loopres_34879 = j_34870;
                    } else {
                        int32_t res_34880 = add32(1, j_34870);
                        
                        loopres_34879 = res_34880;
                    }
                    
                    bool res_34881 = slt32(loopres_34879, n_inds_34841);
                    bool x_34882 = x_34878 && res_34881;
                    bool loop_while_tmp_40318 = x_34882;
                    int32_t j_tmp_40319 = loopres_34879;
                    bool not_found_tmp_40320 = x_34878;
                    
                    loop_while_34869 = loop_while_tmp_40318;
                    j_34870 = j_tmp_40319;
                    not_found_34871 = not_found_tmp_40320;
                }
                not_found_34866 = loop_while_34869;
                not_found_34867 = j_34870;
                not_found_34868 = not_found_34871;
                
                int32_t loopres_34883;
                
                if (not_found_34868) {
                    bool x_34885 = sle32(0, n_inds_34841);
                    bool y_34886 = slt32(n_inds_34841, 8);
                    bool bounds_check_34887 = x_34885 && y_34886;
                    bool index_certs_34888;
                    
                    if (!bounds_check_34887) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          57) == -1) {
                                global_failure_args[0] = n_inds_34841;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    for (int32_t i_40321 = 0; i_40321 < 1; i_40321++) {
                        mem_39494[sext_i32_i64(n_inds_34841 + i_40321)] =
                            ((__global
                              int32_t *) mem_39443)[(sext_i32_i64(orig_cand_ind_34856) +
                                                     sext_i32_i64(i_40321) -
                                                     squot64(sext_i32_i64(orig_cand_ind_34856) +
                                                             sext_i32_i64(i_40321),
                                                             sext_i32_i64(per_chunk_39330)) *
                                                     sext_i32_i64(per_chunk_39330)) *
                                                    sext_i32_i64(num_threads_39324) +
                                                    squot64(sext_i32_i64(orig_cand_ind_34856) +
                                                            sext_i32_i64(i_40321),
                                                            sext_i32_i64(per_chunk_39330))];
                    }
                    
                    int32_t res_34891 = add32(1, n_inds_34841);
                    
                    loopres_34883 = res_34891;
                } else {
                    loopres_34883 = n_inds_34841;
                }
                
                int32_t n_inds_tmp_40316 = loopres_34883;
                
                n_inds_34841 = n_inds_tmp_40316;
            }
            res_34838 = n_inds_34841;
            ((__global int32_t *) mem_39511)[sext_i32_i64(gtid_34699)] =
                res_34838;
            for (int32_t i_40322 = 0; i_40322 < 8; i_40322++) {
                ((__global int32_t *) mem_39515)[sext_i32_i64(i_40322) *
                                                 sext_i32_i64(nc_31615) +
                                                 sext_i32_i64(gtid_34699)] =
                    mem_39494[sext_i32_i64(i_40322)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_34830
}
__kernel void propagateFixKzisegmap_34912(__global int *global_failure,
                                          int failure_is_an_option, __global
                                          int *global_failure_args,
                                          int32_t d_31575, int32_t nr_31586,
                                          int32_t nc_31615,
                                          int32_t num_leaves_31627,
                                          int32_t ppl_31631, int32_t i_31649,
                                          int32_t num_groups_35039, __global
                                          unsigned char *mem_39450, __global
                                          unsigned char *mem_39457, __global
                                          unsigned char *mem_39511, __global
                                          unsigned char *mem_39515, __global
                                          unsigned char *mem_39519, __global
                                          unsigned char *mem_39523, __global
                                          unsigned char *mem_39561, __global
                                          unsigned char *mem_39671, __global
                                          unsigned char *mem_39675, __global
                                          unsigned char *double_buffer_mem_40098,
                                          __global
                                          unsigned char *double_buffer_mem_40099)
{
    #define segmap_group_sizze_35038 (propagateFixKzisegmap_group_sizze_34915)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40333;
    int32_t local_tid_40334;
    int32_t group_sizze_40337;
    int32_t wave_sizze_40336;
    int32_t group_tid_40335;
    
    global_tid_40333 = get_global_id(0);
    local_tid_40334 = get_local_id(0);
    group_sizze_40337 = get_local_size(0);
    wave_sizze_40336 = LOCKSTEP_WIDTH;
    group_tid_40335 = get_group_id(0);
    
    int32_t phys_tid_34912;
    
    phys_tid_34912 = global_tid_40333;
    
    int32_t phys_group_id_40338;
    
    phys_group_id_40338 = get_group_id(0);
    for (int32_t i_40339 = 0; i_40339 < sdiv_up32(sdiv_up32(nc_31615,
                                                            segmap_group_sizze_35038) -
                                                  phys_group_id_40338,
                                                  num_groups_35039);
         i_40339++) {
        int32_t virt_group_id_40340 = phys_group_id_40338 + i_40339 *
                num_groups_35039;
        int32_t gtid_34911 = sext_i64_i32(sext_i32_i64(virt_group_id_40340) *
                sext_i32_i64(segmap_group_sizze_35038) +
                sext_i32_i64(local_tid_40334));
        
        if (slt32(gtid_34911, nc_31615)) {
            int32_t x_35043 = ((__global
                                int32_t *) mem_39511)[sext_i32_i64(gtid_34911)];
            int32_t mem_39556[8];
            float mem_39558[8];
            int32_t mem_39613[8];
            float mem_39615[8];
            
            for (int32_t i_40341 = 0; i_40341 < 8; i_40341++) {
                ((__global
                  int32_t *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_34912) +
                                                      sext_i32_i64(i_40341) *
                                                      sext_i32_i64(num_groups_35039 *
                                                      segmap_group_sizze_35038)] =
                    ((__global int32_t *) mem_39519)[sext_i32_i64(gtid_34911) +
                                                     sext_i32_i64(i_40341) *
                                                     sext_i32_i64(nc_31615)];
            }
            for (int32_t i_40342 = 0; i_40342 < 8; i_40342++) {
                ((__global
                  float *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_34912) +
                                                    sext_i32_i64(i_40342) *
                                                    sext_i32_i64(num_groups_35039 *
                                                    segmap_group_sizze_35038)] =
                    ((__global float *) mem_39523)[sext_i32_i64(gtid_34911) +
                                                   sext_i32_i64(i_40342) *
                                                   sext_i32_i64(nc_31615)];
            }
            for (int32_t q_35052 = 0; q_35052 < x_35043; q_35052++) {
                int32_t i_35055 = q_35052;
                bool x_35056 = sle32(0, i_35055);
                bool y_35057 = slt32(i_35055, 8);
                bool bounds_check_35058 = x_35056 && y_35057;
                bool index_certs_35059;
                
                if (!bounds_check_35058) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 58) ==
                            -1) {
                            global_failure_args[0] = i_35055;
                            global_failure_args[1] = 8;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                int32_t leaf_ind_35060 = ((__global
                                           int32_t *) mem_39515)[sext_i32_i64(i_35055) *
                                                                 sext_i32_i64(nc_31615) +
                                                                 sext_i32_i64(gtid_34911)];
                int32_t bruteForcePar_arg_35061 = mul32(ppl_31631,
                                                        leaf_ind_35060);
                bool x_35062 = sle32(0, leaf_ind_35060);
                bool y_35063 = slt32(leaf_ind_35060, num_leaves_31627);
                bool bounds_check_35064 = x_35062 && y_35063;
                bool index_certs_35065;
                
                if (!bounds_check_35064) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 59) ==
                            -1) {
                            global_failure_args[0] = leaf_ind_35060;
                            global_failure_args[1] = num_leaves_31627;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                for (int32_t i_40345 = 0; i_40345 < 8; i_40345++) {
                    mem_39556[sext_i32_i64(i_40345)] = ((__global
                                                         int32_t *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_34912) +
                                                                                             sext_i32_i64(i_40345) *
                                                                                             sext_i32_i64(num_groups_35039 *
                                                                                             segmap_group_sizze_35038)];
                }
                for (int32_t i_40346 = 0; i_40346 < 8; i_40346++) {
                    mem_39558[sext_i32_i64(i_40346)] = ((__global
                                                         float *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_34912) +
                                                                                           sext_i32_i64(i_40346) *
                                                                                           sext_i32_i64(num_groups_35039 *
                                                                                           segmap_group_sizze_35038)];
                }
                for (int32_t i_39355 = 0; i_39355 < ppl_31631; i_39355++) {
                    float res_35071;
                    float res_35073 = 0.0F;
                    float x_35074;
                    float y_35075;
                    
                    for (int32_t i_35072 = 0; i_35072 < d_31575; i_35072++) {
                        x_35074 = ((__global
                                    float *) mem_39450)[sext_i32_i64(i_31649 *
                                                        nc_31615 + gtid_34911) +
                                                        sext_i32_i64(i_35072) *
                                                        sext_i32_i64(nc_31615 *
                                                        nr_31586)];
                        y_35075 = ((__global
                                    float *) mem_39457)[sext_i32_i64(i_39355 *
                                                        (num_leaves_31627 *
                                                         d_31575) +
                                                        leaf_ind_35060) +
                                                        sext_i32_i64(i_35072) *
                                                        sext_i32_i64(num_leaves_31627)];
                        
                        float zz_35076;
                        
                        zz_35076 = x_35074 - y_35075;
                        
                        float y_35077 = zz_35076 * zz_35076;
                        float loopres_35078 = res_35073 + y_35077;
                        float res_tmp_40348 = loopres_35078;
                        
                        res_35073 = res_tmp_40348;
                    }
                    res_35071 = res_35073;
                    ((__global
                      float *) mem_39561)[sext_i32_i64(phys_tid_34912) +
                                          sext_i32_i64(i_39355) *
                                          sext_i32_i64(num_groups_35039 *
                                          segmap_group_sizze_35038)] =
                        res_35071;
                }
                
                bool knn_35079;
                int32_t knn_35083;
                bool loop_while_35084;
                int32_t j_35088;
                
                loop_while_35084 = 1;
                j_35088 = 0;
                while (loop_while_35084) {
                    int32_t res_35089;
                    float res_35090;
                    int32_t redout_39357;
                    float redout_39358;
                    
                    redout_39357 = ppl_31631;
                    redout_39358 = INFINITY;
                    for (int32_t i_39359 = 0; i_39359 < ppl_31631; i_39359++) {
                        float x_35104 = ((__global
                                          float *) mem_39561)[sext_i32_i64(phys_tid_34912) +
                                                              sext_i32_i64(i_39359) *
                                                              sext_i32_i64(num_groups_35039 *
                                                              segmap_group_sizze_35038)];
                        bool cond_35095 = redout_39358 < x_35104;
                        int32_t res_35096;
                        float res_35097;
                        
                        if (cond_35095) {
                            res_35096 = redout_39357;
                            res_35097 = redout_39358;
                        } else {
                            bool cond_35098 = x_35104 < redout_39358;
                            float res_35099;
                            
                            if (cond_35098) {
                                res_35099 = x_35104;
                            } else {
                                res_35099 = redout_39358;
                            }
                            
                            int32_t res_35100;
                            
                            if (cond_35098) {
                                res_35100 = i_39359;
                            } else {
                                bool cond_35101 = sle32(redout_39357, i_39359);
                                int32_t res_35102;
                                
                                if (cond_35101) {
                                    res_35102 = redout_39357;
                                } else {
                                    res_35102 = i_39359;
                                }
                                res_35100 = res_35102;
                            }
                            res_35096 = res_35100;
                            res_35097 = res_35099;
                        }
                        
                        int32_t redout_tmp_40354 = res_35096;
                        float redout_tmp_40355 = res_35097;
                        
                        redout_39357 = redout_tmp_40354;
                        redout_39358 = redout_tmp_40355;
                    }
                    res_35089 = redout_39357;
                    res_35090 = redout_39358;
                    
                    int32_t i_35105 = sub32(7, j_35088);
                    bool x_35106 = sle32(0, i_35105);
                    bool y_35107 = slt32(i_35105, 8);
                    bool bounds_check_35108 = x_35106 && y_35107;
                    bool index_certs_35109;
                    
                    if (!bounds_check_35108) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          60) == -1) {
                                global_failure_args[0] = i_35105;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_35110 = mem_39558[sext_i32_i64(i_35105)];
                    bool cond_35111 = res_35090 < y_35110;
                    int32_t loopres_35115;
                    
                    if (cond_35111) {
                        bool x_35116 = sle32(0, res_35089);
                        bool y_35117 = slt32(res_35089, ppl_31631);
                        bool bounds_check_35118 = x_35116 && y_35117;
                        bool index_certs_35119;
                        
                        if (!bounds_check_35118) {
                            {
                                if (atomic_cmpxchg_i32_global(global_failure,
                                                              -1, 61) == -1) {
                                    global_failure_args[0] = res_35089;
                                    global_failure_args[1] = ppl_31631;
                                    ;
                                }
                                local_failure = true;
                                goto error_0;
                            }
                        }
                        ((__global
                          float *) mem_39561)[sext_i32_i64(phys_tid_34912) +
                                              sext_i32_i64(res_35089) *
                                              sext_i32_i64(num_groups_35039 *
                                              segmap_group_sizze_35038)] =
                            INFINITY;
                        
                        int32_t lw_val_35121 = add32(bruteForcePar_arg_35061,
                                                     res_35089);
                        
                        mem_39556[sext_i32_i64(i_35105)] = lw_val_35121;
                        mem_39558[sext_i32_i64(i_35105)] = res_35090;
                        
                        int32_t res_35124 = add32(1, j_35088);
                        
                        loopres_35115 = res_35124;
                    } else {
                        loopres_35115 = j_35088;
                    }
                    
                    bool res_35125 = slt32(loopres_35115, 8);
                    bool x_35126 = cond_35111 && res_35125;
                    bool loop_while_tmp_40349 = x_35126;
                    int32_t j_tmp_40353 = loopres_35115;
                    
                    loop_while_35084 = loop_while_tmp_40349;
                    j_35088 = j_tmp_40353;
                }
                knn_35079 = loop_while_35084;
                knn_35083 = j_35088;
                for (int32_t i_40356 = 0; i_40356 < 8; i_40356++) {
                    mem_39613[sext_i32_i64(i_40356)] = -1;
                }
                for (int32_t i_40357 = 0; i_40357 < 8; i_40357++) {
                    mem_39615[sext_i32_i64(i_40357)] = INFINITY;
                }
                
                int32_t res_35131;
                int32_t res_35132;
                int32_t beg_35136;
                int32_t end_35137;
                
                beg_35136 = 0;
                end_35137 = 7;
                for (int32_t i_35133 = 0; i_35133 < 8; i_35133++) {
                    bool x_35138 = sle32(0, beg_35136);
                    bool y_35139 = slt32(beg_35136, 8);
                    bool bounds_check_35140 = x_35138 && y_35139;
                    bool index_certs_35141;
                    
                    if (!bounds_check_35140) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          62) == -1) {
                                global_failure_args[0] = beg_35136;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float x_35142 = mem_39558[sext_i32_i64(beg_35136)];
                    bool x_35143 = sle32(0, end_35137);
                    bool y_35144 = slt32(end_35137, 8);
                    bool bounds_check_35145 = x_35143 && y_35144;
                    bool index_certs_35146;
                    
                    if (!bounds_check_35145) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          63) == -1) {
                                global_failure_args[0] = end_35137;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_35147 = mem_39558[sext_i32_i64(end_35137)];
                    bool cond_35148 = x_35142 < y_35147;
                    float loopres_35149;
                    
                    if (cond_35148) {
                        loopres_35149 = x_35142;
                    } else {
                        loopres_35149 = y_35147;
                    }
                    
                    int32_t loopres_35150;
                    int32_t loopres_35151;
                    int32_t loopres_35152;
                    
                    if (cond_35148) {
                        int32_t res_35153 = mem_39556[sext_i32_i64(beg_35136)];
                        int32_t res_35154 = add32(1, beg_35136);
                        
                        loopres_35150 = res_35153;
                        loopres_35151 = res_35154;
                        loopres_35152 = end_35137;
                    } else {
                        int32_t res_35155 = mem_39556[sext_i32_i64(end_35137)];
                        int32_t res_35156 = sub32(end_35137, 1);
                        
                        loopres_35150 = res_35155;
                        loopres_35151 = beg_35136;
                        loopres_35152 = res_35156;
                    }
                    mem_39613[sext_i32_i64(i_35133)] = loopres_35150;
                    mem_39615[sext_i32_i64(i_35133)] = loopres_35149;
                    
                    int32_t beg_tmp_40360 = loopres_35151;
                    int32_t end_tmp_40361 = loopres_35152;
                    
                    beg_35136 = beg_tmp_40360;
                    end_35137 = end_tmp_40361;
                }
                res_35131 = beg_35136;
                res_35132 = end_35137;
                for (int32_t i_40362 = 0; i_40362 < 8; i_40362++) {
                    ((__global
                      int32_t *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_34912) +
                                                          sext_i32_i64(i_40362) *
                                                          sext_i32_i64(num_groups_35039 *
                                                          segmap_group_sizze_35038)] =
                        mem_39613[sext_i32_i64(i_40362)];
                }
                for (int32_t i_40363 = 0; i_40363 < 8; i_40363++) {
                    ((__global
                      float *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_34912) +
                                                        sext_i32_i64(i_40363) *
                                                        sext_i32_i64(num_groups_35039 *
                                                        segmap_group_sizze_35038)] =
                        mem_39615[sext_i32_i64(i_40363)];
                }
            }
            for (int32_t i_40364 = 0; i_40364 < 8; i_40364++) {
                ((__global int32_t *) mem_39671)[sext_i32_i64(i_40364) *
                                                 sext_i32_i64(nc_31615) +
                                                 sext_i32_i64(gtid_34911)] =
                    ((__global
                      int32_t *) double_buffer_mem_40098)[sext_i32_i64(phys_tid_34912) +
                                                          sext_i32_i64(i_40364) *
                                                          sext_i32_i64(num_groups_35039 *
                                                          segmap_group_sizze_35038)];
            }
            for (int32_t i_40365 = 0; i_40365 < 8; i_40365++) {
                ((__global float *) mem_39675)[sext_i32_i64(i_40365) *
                                               sext_i32_i64(nc_31615) +
                                               sext_i32_i64(gtid_34911)] =
                    ((__global
                      float *) double_buffer_mem_40099)[sext_i32_i64(phys_tid_34912) +
                                                        sext_i32_i64(i_40365) *
                                                        sext_i32_i64(num_groups_35039 *
                                                        segmap_group_sizze_35038)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_35038
}
__kernel void propagateFixKzisegmap_35287(__global int *global_failure,
                                          int failure_is_an_option, __global
                                          int *global_failure_args,
                                          int32_t d_31575, int32_t nr_31586,
                                          int32_t nc_31615,
                                          int32_t num_leaves_31627,
                                          int32_t ppl_31631, int32_t i_31649,
                                          int32_t num_groups_35414, __global
                                          unsigned char *mem_39450, __global
                                          unsigned char *mem_39457, __global
                                          unsigned char *mem_39511, __global
                                          unsigned char *mem_39515, __global
                                          unsigned char *mem_39831, __global
                                          unsigned char *mem_39835, __global
                                          unsigned char *mem_39873, __global
                                          unsigned char *mem_39983, __global
                                          unsigned char *mem_39987, __global
                                          unsigned char *double_buffer_mem_40110,
                                          __global
                                          unsigned char *double_buffer_mem_40111)
{
    #define segmap_group_sizze_35413 (propagateFixKzisegmap_group_sizze_35290)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40420;
    int32_t local_tid_40421;
    int32_t group_sizze_40424;
    int32_t wave_sizze_40423;
    int32_t group_tid_40422;
    
    global_tid_40420 = get_global_id(0);
    local_tid_40421 = get_local_id(0);
    group_sizze_40424 = get_local_size(0);
    wave_sizze_40423 = LOCKSTEP_WIDTH;
    group_tid_40422 = get_group_id(0);
    
    int32_t phys_tid_35287;
    
    phys_tid_35287 = global_tid_40420;
    
    int32_t phys_group_id_40425;
    
    phys_group_id_40425 = get_group_id(0);
    for (int32_t i_40426 = 0; i_40426 < sdiv_up32(sdiv_up32(nc_31615,
                                                            segmap_group_sizze_35413) -
                                                  phys_group_id_40425,
                                                  num_groups_35414);
         i_40426++) {
        int32_t virt_group_id_40427 = phys_group_id_40425 + i_40426 *
                num_groups_35414;
        int32_t gtid_35286 = sext_i64_i32(sext_i32_i64(virt_group_id_40427) *
                sext_i32_i64(segmap_group_sizze_35413) +
                sext_i32_i64(local_tid_40421));
        
        if (slt32(gtid_35286, nc_31615)) {
            int32_t x_35418 = ((__global
                                int32_t *) mem_39511)[sext_i32_i64(gtid_35286)];
            int32_t mem_39868[8];
            float mem_39870[8];
            int32_t mem_39925[8];
            float mem_39927[8];
            
            for (int32_t i_40428 = 0; i_40428 < 8; i_40428++) {
                ((__global
                  int32_t *) double_buffer_mem_40110)[sext_i32_i64(phys_tid_35287) +
                                                      sext_i32_i64(i_40428) *
                                                      sext_i32_i64(num_groups_35414 *
                                                      segmap_group_sizze_35413)] =
                    ((__global int32_t *) mem_39831)[sext_i32_i64(gtid_35286) +
                                                     sext_i32_i64(i_40428) *
                                                     sext_i32_i64(nc_31615)];
            }
            for (int32_t i_40429 = 0; i_40429 < 8; i_40429++) {
                ((__global
                  float *) double_buffer_mem_40111)[sext_i32_i64(phys_tid_35287) +
                                                    sext_i32_i64(i_40429) *
                                                    sext_i32_i64(num_groups_35414 *
                                                    segmap_group_sizze_35413)] =
                    ((__global float *) mem_39835)[sext_i32_i64(gtid_35286) +
                                                   sext_i32_i64(i_40429) *
                                                   sext_i32_i64(nc_31615)];
            }
            for (int32_t q_35425 = 0; q_35425 < x_35418; q_35425++) {
                int32_t i_35428 = q_35425;
                bool x_35429 = sle32(0, i_35428);
                bool y_35430 = slt32(i_35428, 8);
                bool bounds_check_35431 = x_35429 && y_35430;
                bool index_certs_35432;
                
                if (!bounds_check_35431) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 70) ==
                            -1) {
                            global_failure_args[0] = i_35428;
                            global_failure_args[1] = 8;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                int32_t leaf_ind_35433 = ((__global
                                           int32_t *) mem_39515)[sext_i32_i64(i_35428) *
                                                                 sext_i32_i64(nc_31615) +
                                                                 sext_i32_i64(gtid_35286)];
                int32_t bruteForcePar_arg_35434 = mul32(ppl_31631,
                                                        leaf_ind_35433);
                bool x_35435 = sle32(0, leaf_ind_35433);
                bool y_35436 = slt32(leaf_ind_35433, num_leaves_31627);
                bool bounds_check_35437 = x_35435 && y_35436;
                bool index_certs_35438;
                
                if (!bounds_check_35437) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 71) ==
                            -1) {
                            global_failure_args[0] = leaf_ind_35433;
                            global_failure_args[1] = num_leaves_31627;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                for (int32_t i_40432 = 0; i_40432 < 8; i_40432++) {
                    mem_39868[sext_i32_i64(i_40432)] = ((__global
                                                         int32_t *) double_buffer_mem_40110)[sext_i32_i64(phys_tid_35287) +
                                                                                             sext_i32_i64(i_40432) *
                                                                                             sext_i32_i64(num_groups_35414 *
                                                                                             segmap_group_sizze_35413)];
                }
                for (int32_t i_40433 = 0; i_40433 < 8; i_40433++) {
                    mem_39870[sext_i32_i64(i_40433)] = ((__global
                                                         float *) double_buffer_mem_40111)[sext_i32_i64(phys_tid_35287) +
                                                                                           sext_i32_i64(i_40433) *
                                                                                           sext_i32_i64(num_groups_35414 *
                                                                                           segmap_group_sizze_35413)];
                }
                for (int32_t i_39362 = 0; i_39362 < ppl_31631; i_39362++) {
                    float res_35444;
                    float res_35446 = 0.0F;
                    float x_35447;
                    float y_35448;
                    
                    for (int32_t i_35445 = 0; i_35445 < d_31575; i_35445++) {
                        x_35447 = ((__global
                                    float *) mem_39450)[sext_i32_i64(i_31649 *
                                                        nc_31615 + gtid_35286) +
                                                        sext_i32_i64(i_35445) *
                                                        sext_i32_i64(nc_31615 *
                                                        nr_31586)];
                        y_35448 = ((__global
                                    float *) mem_39457)[sext_i32_i64(i_39362 *
                                                        (num_leaves_31627 *
                                                         d_31575) +
                                                        leaf_ind_35433) +
                                                        sext_i32_i64(i_35445) *
                                                        sext_i32_i64(num_leaves_31627)];
                        
                        float zz_35449;
                        
                        zz_35449 = x_35447 - y_35448;
                        
                        float y_35450 = zz_35449 * zz_35449;
                        float loopres_35451 = res_35446 + y_35450;
                        float res_tmp_40435 = loopres_35451;
                        
                        res_35446 = res_tmp_40435;
                    }
                    res_35444 = res_35446;
                    ((__global
                      float *) mem_39873)[sext_i32_i64(phys_tid_35287) +
                                          sext_i32_i64(i_39362) *
                                          sext_i32_i64(num_groups_35414 *
                                          segmap_group_sizze_35413)] =
                        res_35444;
                }
                
                bool knn_35452;
                int32_t knn_35456;
                bool loop_while_35457;
                int32_t j_35461;
                
                loop_while_35457 = 1;
                j_35461 = 0;
                while (loop_while_35457) {
                    int32_t res_35462;
                    float res_35463;
                    int32_t redout_39364;
                    float redout_39365;
                    
                    redout_39364 = ppl_31631;
                    redout_39365 = INFINITY;
                    for (int32_t i_39366 = 0; i_39366 < ppl_31631; i_39366++) {
                        float x_35477 = ((__global
                                          float *) mem_39873)[sext_i32_i64(phys_tid_35287) +
                                                              sext_i32_i64(i_39366) *
                                                              sext_i32_i64(num_groups_35414 *
                                                              segmap_group_sizze_35413)];
                        bool cond_35468 = redout_39365 < x_35477;
                        int32_t res_35469;
                        float res_35470;
                        
                        if (cond_35468) {
                            res_35469 = redout_39364;
                            res_35470 = redout_39365;
                        } else {
                            bool cond_35471 = x_35477 < redout_39365;
                            float res_35472;
                            
                            if (cond_35471) {
                                res_35472 = x_35477;
                            } else {
                                res_35472 = redout_39365;
                            }
                            
                            int32_t res_35473;
                            
                            if (cond_35471) {
                                res_35473 = i_39366;
                            } else {
                                bool cond_35474 = sle32(redout_39364, i_39366);
                                int32_t res_35475;
                                
                                if (cond_35474) {
                                    res_35475 = redout_39364;
                                } else {
                                    res_35475 = i_39366;
                                }
                                res_35473 = res_35475;
                            }
                            res_35469 = res_35473;
                            res_35470 = res_35472;
                        }
                        
                        int32_t redout_tmp_40441 = res_35469;
                        float redout_tmp_40442 = res_35470;
                        
                        redout_39364 = redout_tmp_40441;
                        redout_39365 = redout_tmp_40442;
                    }
                    res_35462 = redout_39364;
                    res_35463 = redout_39365;
                    
                    int32_t i_35478 = sub32(7, j_35461);
                    bool x_35479 = sle32(0, i_35478);
                    bool y_35480 = slt32(i_35478, 8);
                    bool bounds_check_35481 = x_35479 && y_35480;
                    bool index_certs_35482;
                    
                    if (!bounds_check_35481) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          72) == -1) {
                                global_failure_args[0] = i_35478;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_35483 = mem_39870[sext_i32_i64(i_35478)];
                    bool cond_35484 = res_35463 < y_35483;
                    int32_t loopres_35488;
                    
                    if (cond_35484) {
                        bool x_35489 = sle32(0, res_35462);
                        bool y_35490 = slt32(res_35462, ppl_31631);
                        bool bounds_check_35491 = x_35489 && y_35490;
                        bool index_certs_35492;
                        
                        if (!bounds_check_35491) {
                            {
                                if (atomic_cmpxchg_i32_global(global_failure,
                                                              -1, 73) == -1) {
                                    global_failure_args[0] = res_35462;
                                    global_failure_args[1] = ppl_31631;
                                    ;
                                }
                                local_failure = true;
                                goto error_0;
                            }
                        }
                        ((__global
                          float *) mem_39873)[sext_i32_i64(phys_tid_35287) +
                                              sext_i32_i64(res_35462) *
                                              sext_i32_i64(num_groups_35414 *
                                              segmap_group_sizze_35413)] =
                            INFINITY;
                        
                        int32_t lw_val_35494 = add32(bruteForcePar_arg_35434,
                                                     res_35462);
                        
                        mem_39868[sext_i32_i64(i_35478)] = lw_val_35494;
                        mem_39870[sext_i32_i64(i_35478)] = res_35463;
                        
                        int32_t res_35497 = add32(1, j_35461);
                        
                        loopres_35488 = res_35497;
                    } else {
                        loopres_35488 = j_35461;
                    }
                    
                    bool res_35498 = slt32(loopres_35488, 8);
                    bool x_35499 = cond_35484 && res_35498;
                    bool loop_while_tmp_40436 = x_35499;
                    int32_t j_tmp_40440 = loopres_35488;
                    
                    loop_while_35457 = loop_while_tmp_40436;
                    j_35461 = j_tmp_40440;
                }
                knn_35452 = loop_while_35457;
                knn_35456 = j_35461;
                for (int32_t i_40443 = 0; i_40443 < 8; i_40443++) {
                    mem_39925[sext_i32_i64(i_40443)] = -1;
                }
                for (int32_t i_40444 = 0; i_40444 < 8; i_40444++) {
                    mem_39927[sext_i32_i64(i_40444)] = INFINITY;
                }
                
                int32_t res_35504;
                int32_t res_35505;
                int32_t beg_35509;
                int32_t end_35510;
                
                beg_35509 = 0;
                end_35510 = 7;
                for (int32_t i_35506 = 0; i_35506 < 8; i_35506++) {
                    bool x_35511 = sle32(0, beg_35509);
                    bool y_35512 = slt32(beg_35509, 8);
                    bool bounds_check_35513 = x_35511 && y_35512;
                    bool index_certs_35514;
                    
                    if (!bounds_check_35513) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          74) == -1) {
                                global_failure_args[0] = beg_35509;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float x_35515 = mem_39870[sext_i32_i64(beg_35509)];
                    bool x_35516 = sle32(0, end_35510);
                    bool y_35517 = slt32(end_35510, 8);
                    bool bounds_check_35518 = x_35516 && y_35517;
                    bool index_certs_35519;
                    
                    if (!bounds_check_35518) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          75) == -1) {
                                global_failure_args[0] = end_35510;
                                global_failure_args[1] = 8;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    float y_35520 = mem_39870[sext_i32_i64(end_35510)];
                    bool cond_35521 = x_35515 < y_35520;
                    float loopres_35522;
                    
                    if (cond_35521) {
                        loopres_35522 = x_35515;
                    } else {
                        loopres_35522 = y_35520;
                    }
                    
                    int32_t loopres_35523;
                    int32_t loopres_35524;
                    int32_t loopres_35525;
                    
                    if (cond_35521) {
                        int32_t res_35526 = mem_39868[sext_i32_i64(beg_35509)];
                        int32_t res_35527 = add32(1, beg_35509);
                        
                        loopres_35523 = res_35526;
                        loopres_35524 = res_35527;
                        loopres_35525 = end_35510;
                    } else {
                        int32_t res_35528 = mem_39868[sext_i32_i64(end_35510)];
                        int32_t res_35529 = sub32(end_35510, 1);
                        
                        loopres_35523 = res_35528;
                        loopres_35524 = beg_35509;
                        loopres_35525 = res_35529;
                    }
                    mem_39925[sext_i32_i64(i_35506)] = loopres_35523;
                    mem_39927[sext_i32_i64(i_35506)] = loopres_35522;
                    
                    int32_t beg_tmp_40447 = loopres_35524;
                    int32_t end_tmp_40448 = loopres_35525;
                    
                    beg_35509 = beg_tmp_40447;
                    end_35510 = end_tmp_40448;
                }
                res_35504 = beg_35509;
                res_35505 = end_35510;
                for (int32_t i_40449 = 0; i_40449 < 8; i_40449++) {
                    ((__global
                      int32_t *) double_buffer_mem_40110)[sext_i32_i64(phys_tid_35287) +
                                                          sext_i32_i64(i_40449) *
                                                          sext_i32_i64(num_groups_35414 *
                                                          segmap_group_sizze_35413)] =
                        mem_39925[sext_i32_i64(i_40449)];
                }
                for (int32_t i_40450 = 0; i_40450 < 8; i_40450++) {
                    ((__global
                      float *) double_buffer_mem_40111)[sext_i32_i64(phys_tid_35287) +
                                                        sext_i32_i64(i_40450) *
                                                        sext_i32_i64(num_groups_35414 *
                                                        segmap_group_sizze_35413)] =
                        mem_39927[sext_i32_i64(i_40450)];
                }
            }
            for (int32_t i_40451 = 0; i_40451 < 8; i_40451++) {
                ((__global int32_t *) mem_39983)[sext_i32_i64(i_40451) *
                                                 sext_i32_i64(nc_31615) +
                                                 sext_i32_i64(gtid_35286)] =
                    ((__global
                      int32_t *) double_buffer_mem_40110)[sext_i32_i64(phys_tid_35287) +
                                                          sext_i32_i64(i_40451) *
                                                          sext_i32_i64(num_groups_35414 *
                                                          segmap_group_sizze_35413)];
            }
            for (int32_t i_40452 = 0; i_40452 < 8; i_40452++) {
                ((__global float *) mem_39987)[sext_i32_i64(i_40452) *
                                               sext_i32_i64(nc_31615) +
                                               sext_i32_i64(gtid_35286)] =
                    ((__global
                      float *) double_buffer_mem_40111)[sext_i32_i64(phys_tid_35287) +
                                                        sext_i32_i64(i_40452) *
                                                        sext_i32_i64(num_groups_35414 *
                                                        segmap_group_sizze_35413)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_35413
}
__kernel void propagateFixKzisegmap_35593(__global int *global_failure,
                                          int failure_is_an_option, __global
                                          int *global_failure_args,
                                          int32_t m_31574, int32_t n_31578,
                                          __global
                                          unsigned char *indir_mem_39419,
                                          __global unsigned char *mem_39428,
                                          int32_t ctx_val_39482, __global
                                          unsigned char *mem_40042)
{
    #define segmap_group_sizze_35639 (propagateFixKzisegmap_group_sizze_35598)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40455;
    int32_t local_tid_40456;
    int32_t group_sizze_40459;
    int32_t wave_sizze_40458;
    int32_t group_tid_40457;
    
    global_tid_40455 = get_global_id(0);
    local_tid_40456 = get_local_id(0);
    group_sizze_40459 = get_local_size(0);
    wave_sizze_40458 = LOCKSTEP_WIDTH;
    group_tid_40457 = get_group_id(0);
    
    int32_t phys_tid_35593;
    
    phys_tid_35593 = global_tid_40455;
    
    int32_t gtid_35591;
    
    gtid_35591 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40457) *
                                      sext_i32_i64(segmap_group_sizze_35639) +
                                      sext_i32_i64(local_tid_40456), 8));
    
    int32_t gtid_35592;
    
    gtid_35592 = sext_i64_i32(sext_i32_i64(group_tid_40457) *
        sext_i32_i64(segmap_group_sizze_35639) + sext_i32_i64(local_tid_40456) -
        squot64(sext_i32_i64(group_tid_40457) *
                sext_i32_i64(segmap_group_sizze_35639) +
                sext_i32_i64(local_tid_40456), 8) * 8);
    if (slt32(gtid_35591, n_31578) && slt32(gtid_35592, 8)) {
        int32_t binop_x_39287 = 8 * gtid_35591;
        int32_t binop_x_39288 = gtid_35592 + binop_x_39287;
        int32_t new_index_39290 = squot32(binop_x_39288, ctx_val_39482);
        int32_t binop_y_39298 = new_index_39290 * ctx_val_39482;
        int32_t binop_x_39299 = binop_x_39288 - binop_y_39298;
        int32_t new_index_39300 = squot32(binop_x_39299, 8);
        int32_t binop_y_39320 = 8 * new_index_39300;
        int32_t new_index_39321 = binop_x_39299 - binop_y_39320;
        int32_t x_35644 = ((__global
                            int32_t *) mem_39428)[sext_i32_i64(new_index_39290) *
                                                  sext_i32_i64(ctx_val_39482) +
                                                  sext_i32_i64(new_index_39300) *
                                                  8 +
                                                  sext_i32_i64(new_index_39321)];
        bool x_35645 = sle32(0, x_35644);
        bool y_35646 = slt32(x_35644, m_31574);
        bool bounds_check_35647 = x_35645 && y_35646;
        bool index_certs_35648;
        
        if (!bounds_check_35647) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 76) == -1) {
                    global_failure_args[0] = x_35644;
                    global_failure_args[1] = m_31574;
                    ;
                }
                return;
            }
        }
        
        int32_t res_35649 = ((__global
                              int32_t *) indir_mem_39419)[sext_i32_i64(x_35644)];
        
        ((__global int32_t *) mem_40042)[sext_i32_i64(gtid_35591) * 8 +
                                         sext_i32_i64(gtid_35592)] = res_35649;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_35639
}
__kernel void propagateFixKzisegmap_intragroup_34910(__global
                                                     int *global_failure,
                                                     int failure_is_an_option,
                                                     __global
                                                     int *global_failure_args,
                                                     uint red_arr_mem_40396_backing_offset_0,
                                                     uint red_arr_mem_40394_backing_offset_1,
                                                     uint mem_39767_backing_offset_2,
                                                     uint mem_39765_backing_offset_3,
                                                     uint mem_39722_backing_offset_4,
                                                     uint mem_39718_backing_offset_5,
                                                     uint mem_39716_backing_offset_6,
                                                     int32_t d_31575,
                                                     int32_t nr_31586,
                                                     int32_t nc_31615,
                                                     int32_t num_leaves_31627,
                                                     int32_t ppl_31631,
                                                     int32_t i_31649, __global
                                                     unsigned char *ref_pts_mem_39418,
                                                     __global
                                                     unsigned char *mem_39450,
                                                     __global
                                                     unsigned char *mem_39511,
                                                     __global
                                                     unsigned char *mem_39515,
                                                     __global
                                                     unsigned char *mem_39679,
                                                     __global
                                                     unsigned char *mem_39683,
                                                     __global
                                                     unsigned char *mem_39823,
                                                     __global
                                                     unsigned char *mem_39827,
                                                     __global
                                                     unsigned char *double_buffer_mem_40104,
                                                     __global
                                                     unsigned char *double_buffer_mem_40105)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40396_backing_6 =
                  &shared_mem[red_arr_mem_40396_backing_offset_0];
    volatile char *red_arr_mem_40394_backing_5 =
                  &shared_mem[red_arr_mem_40394_backing_offset_1];
    volatile char *mem_39767_backing_4 =
                  &shared_mem[mem_39767_backing_offset_2];
    volatile char *mem_39765_backing_3 =
                  &shared_mem[mem_39765_backing_offset_3];
    volatile char *mem_39722_backing_2 =
                  &shared_mem[mem_39722_backing_offset_4];
    volatile char *mem_39718_backing_1 =
                  &shared_mem[mem_39718_backing_offset_5];
    volatile char *mem_39716_backing_0 =
                  &shared_mem[mem_39716_backing_offset_6];
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40376;
    int32_t local_tid_40377;
    int32_t group_sizze_40380;
    int32_t wave_sizze_40379;
    int32_t group_tid_40378;
    
    global_tid_40376 = get_global_id(0);
    local_tid_40377 = get_local_id(0);
    group_sizze_40380 = get_local_size(0);
    wave_sizze_40379 = LOCKSTEP_WIDTH;
    group_tid_40378 = get_group_id(0);
    
    int32_t phys_tid_34910;
    
    phys_tid_34910 = group_tid_40378;
    
    int32_t ltid_pre_40381;
    
    ltid_pre_40381 = local_tid_40377;
    
    int32_t gtid_34894;
    
    gtid_34894 = group_tid_40378;
    
    int32_t x_35166;
    
    x_35166 = ((__global int32_t *) mem_39511)[sext_i32_i64(gtid_34894)];
    
    __local char *mem_39716;
    
    mem_39716 = (__local char *) mem_39716_backing_0;
    
    __local char *mem_39718;
    
    mem_39718 = (__local char *) mem_39718_backing_1;
    
    __local char *mem_39722;
    
    mem_39722 = (__local char *) mem_39722_backing_2;
    
    __local char *mem_39765;
    
    mem_39765 = (__local char *) mem_39765_backing_3;
    
    __local char *mem_39767;
    
    mem_39767 = (__local char *) mem_39767_backing_4;
    for (int32_t i_40382 = 0; i_40382 < sdiv_up32(8 - local_tid_40377,
                                                  ppl_31631); i_40382++) {
        ((__global
          int32_t *) double_buffer_mem_40104)[sext_i32_i64(phys_tid_34910 * 8) +
                                              sext_i32_i64(i_40382 * ppl_31631 +
                                              local_tid_40377)] = ((__global
                                                                    int32_t *) mem_39679)[sext_i32_i64(gtid_34894) +
                                                                                          sext_i32_i64(i_40382 *
                                                                                          ppl_31631 +
                                                                                          local_tid_40377) *
                                                                                          sext_i32_i64(nc_31615)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_40383 = 0; i_40383 < sdiv_up32(8 - local_tid_40377,
                                                  ppl_31631); i_40383++) {
        ((__global
          float *) double_buffer_mem_40105)[sext_i32_i64(phys_tid_34910 * 8) +
                                            sext_i32_i64(i_40383 * ppl_31631 +
                                            local_tid_40377)] = ((__global
                                                                  float *) mem_39683)[sext_i32_i64(gtid_34894) +
                                                                                      sext_i32_i64(i_40383 *
                                                                                      ppl_31631 +
                                                                                      local_tid_40377) *
                                                                                      sext_i32_i64(nc_31615)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t q_35175 = 0; q_35175 < x_35166; q_35175++) {
        int32_t i_35178 = q_35175;
        bool x_35179 = sle32(0, i_35178);
        bool y_35180 = slt32(i_35178, 8);
        bool bounds_check_35181 = x_35179 && y_35180;
        bool index_certs_35182;
        
        if (!bounds_check_35181) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 64) == -1) {
                    global_failure_args[0] = i_35178;
                    global_failure_args[1] = 8;
                    ;
                }
                local_failure = true;
                goto error_0;
            }
        }
        
        int32_t leaf_ind_35183 = ((__global
                                   int32_t *) mem_39515)[sext_i32_i64(i_35178) *
                                                         sext_i32_i64(nc_31615) +
                                                         sext_i32_i64(gtid_34894)];
        int32_t bruteForcePar_arg_35184 = mul32(ppl_31631, leaf_ind_35183);
        bool x_35185 = sle32(0, leaf_ind_35183);
        bool y_35186 = slt32(leaf_ind_35183, num_leaves_31627);
        bool bounds_check_35187 = x_35185 && y_35186;
        bool index_certs_35188;
        
        if (!bounds_check_35187) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 65) == -1) {
                    global_failure_args[0] = leaf_ind_35183;
                    global_failure_args[1] = num_leaves_31627;
                    ;
                }
                local_failure = true;
                goto error_0;
            }
        }
        for (int32_t i_40386 = 0; i_40386 < sdiv_up32(8 - local_tid_40377,
                                                      ppl_31631); i_40386++) {
            ((__local int32_t *) mem_39716)[sext_i32_i64(i_40386 * ppl_31631 +
                                            local_tid_40377)] = ((__global
                                                                  int32_t *) double_buffer_mem_40104)[sext_i32_i64(phys_tid_34910 *
                                                                                                      8) +
                                                                                                      sext_i32_i64(i_40386 *
                                                                                                      ppl_31631 +
                                                                                                      local_tid_40377)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_40387 = 0; i_40387 < sdiv_up32(8 - local_tid_40377,
                                                      ppl_31631); i_40387++) {
            ((__local float *) mem_39718)[sext_i32_i64(i_40387 * ppl_31631 +
                                          local_tid_40377)] = ((__global
                                                                float *) double_buffer_mem_40105)[sext_i32_i64(phys_tid_34910 *
                                                                                                  8) +
                                                                                                  sext_i32_i64(i_40387 *
                                                                                                  ppl_31631 +
                                                                                                  local_tid_40377)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t gtid_34897 = ltid_pre_40381;
        int32_t phys_tid_34898 = local_tid_40377;
        
        if (slt32(gtid_34897, ppl_31631)) {
            float res_35194;
            float res_35196 = 0.0F;
            float x_35197;
            float y_35198;
            
            for (int32_t i_35195 = 0; i_35195 < d_31575; i_35195++) {
                x_35197 = ((__global float *) mem_39450)[sext_i32_i64(i_31649 *
                                                         nc_31615 +
                                                         gtid_34894) +
                                                         sext_i32_i64(i_35195) *
                                                         sext_i32_i64(nc_31615 *
                                                         nr_31586)];
                y_35198 = ((__global
                            float *) ref_pts_mem_39418)[sext_i32_i64(leaf_ind_35183 *
                                                        (d_31575 * ppl_31631) +
                                                        gtid_34897 * d_31575) +
                                                        sext_i32_i64(i_35195)];
                
                float zz_35199;
                
                zz_35199 = x_35197 - y_35198;
                
                float y_35200 = zz_35199 * zz_35199;
                float loopres_35201 = res_35196 + y_35200;
                float res_tmp_40388 = loopres_35201;
                
                res_35196 = res_tmp_40388;
            }
            res_35194 = res_35196;
            ((__local float *) mem_39722)[sext_i32_i64(gtid_34897)] = res_35194;
        }
        
      error_0:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        bool knn_35202;
        int32_t knn_35206;
        bool loop_while_35207;
        int32_t j_35211;
        
        loop_while_35207 = 1;
        j_35211 = 0;
        while (loop_while_35207) {
            int32_t res_35212;
            float res_35213;
            int32_t gtid_34908 = ltid_pre_40381;
            int32_t phys_tid_34909 = local_tid_40377;
            __local char *red_arr_mem_40394;
            
            red_arr_mem_40394 = (__local char *) red_arr_mem_40394_backing_5;
            
            __local char *red_arr_mem_40396;
            
            red_arr_mem_40396 = (__local char *) red_arr_mem_40396_backing_6;
            if (slt32(gtid_34908, ppl_31631)) {
                float x_35227 = ((__local
                                  float *) mem_39722)[sext_i32_i64(gtid_34908)];
                
                ((__local
                  int32_t *) red_arr_mem_40394)[sext_i32_i64(gtid_34908)] =
                    gtid_34908;
                ((__local
                  float *) red_arr_mem_40396)[sext_i32_i64(gtid_34908)] =
                    x_35227;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t offset_40398;
            int32_t skip_waves_40399;
            int32_t x_35214;
            float x_35215;
            int32_t x_35216;
            float x_35217;
            
            offset_40398 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_40377, ppl_31631)) {
                    x_35214 = ((__local
                                int32_t *) red_arr_mem_40394)[sext_i32_i64(local_tid_40377 +
                                                              offset_40398)];
                    x_35215 = ((__local
                                float *) red_arr_mem_40396)[sext_i32_i64(local_tid_40377 +
                                                            offset_40398)];
                }
            }
            offset_40398 = 1;
            while (slt32(offset_40398, wave_sizze_40379)) {
                if (slt32(local_tid_40377 + offset_40398, ppl_31631) &&
                    ((local_tid_40377 - squot32(local_tid_40377,
                                                wave_sizze_40379) *
                      wave_sizze_40379) & (2 * offset_40398 - 1)) == 0) {
                    // read array element
                    {
                        x_35216 = ((volatile __local
                                    int32_t *) red_arr_mem_40394)[sext_i32_i64(local_tid_40377 +
                                                                  offset_40398)];
                        x_35217 = ((volatile __local
                                    float *) red_arr_mem_40396)[sext_i32_i64(local_tid_40377 +
                                                                offset_40398)];
                    }
                    // apply reduction operation
                    {
                        bool cond_35218 = x_35215 < x_35217;
                        int32_t res_35219;
                        float res_35220;
                        
                        if (cond_35218) {
                            res_35219 = x_35214;
                            res_35220 = x_35215;
                        } else {
                            bool cond_35221 = x_35217 < x_35215;
                            float res_35222;
                            
                            if (cond_35221) {
                                res_35222 = x_35217;
                            } else {
                                res_35222 = x_35215;
                            }
                            
                            int32_t res_35223;
                            
                            if (cond_35221) {
                                res_35223 = x_35216;
                            } else {
                                bool cond_35224 = sle32(x_35214, x_35216);
                                int32_t res_35225;
                                
                                if (cond_35224) {
                                    res_35225 = x_35214;
                                } else {
                                    res_35225 = x_35216;
                                }
                                res_35223 = res_35225;
                            }
                            res_35219 = res_35223;
                            res_35220 = res_35222;
                        }
                        x_35214 = res_35219;
                        x_35215 = res_35220;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int32_t *) red_arr_mem_40394)[sext_i32_i64(local_tid_40377)] =
                            x_35214;
                        ((volatile __local
                          float *) red_arr_mem_40396)[sext_i32_i64(local_tid_40377)] =
                            x_35215;
                    }
                }
                offset_40398 *= 2;
            }
            skip_waves_40399 = 1;
            while (slt32(skip_waves_40399, squot32(ppl_31631 +
                                                   wave_sizze_40379 - 1,
                                                   wave_sizze_40379))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_40398 = skip_waves_40399 * wave_sizze_40379;
                if (slt32(local_tid_40377 + offset_40398, ppl_31631) &&
                    ((local_tid_40377 - squot32(local_tid_40377,
                                                wave_sizze_40379) *
                      wave_sizze_40379) == 0 && (squot32(local_tid_40377,
                                                         wave_sizze_40379) &
                                                 (2 * skip_waves_40399 - 1)) ==
                     0)) {
                    // read array element
                    {
                        x_35216 = ((__local
                                    int32_t *) red_arr_mem_40394)[sext_i32_i64(local_tid_40377 +
                                                                  offset_40398)];
                        x_35217 = ((__local
                                    float *) red_arr_mem_40396)[sext_i32_i64(local_tid_40377 +
                                                                offset_40398)];
                    }
                    // apply reduction operation
                    {
                        bool cond_35218 = x_35215 < x_35217;
                        int32_t res_35219;
                        float res_35220;
                        
                        if (cond_35218) {
                            res_35219 = x_35214;
                            res_35220 = x_35215;
                        } else {
                            bool cond_35221 = x_35217 < x_35215;
                            float res_35222;
                            
                            if (cond_35221) {
                                res_35222 = x_35217;
                            } else {
                                res_35222 = x_35215;
                            }
                            
                            int32_t res_35223;
                            
                            if (cond_35221) {
                                res_35223 = x_35216;
                            } else {
                                bool cond_35224 = sle32(x_35214, x_35216);
                                int32_t res_35225;
                                
                                if (cond_35224) {
                                    res_35225 = x_35214;
                                } else {
                                    res_35225 = x_35216;
                                }
                                res_35223 = res_35225;
                            }
                            res_35219 = res_35223;
                            res_35220 = res_35222;
                        }
                        x_35214 = res_35219;
                        x_35215 = res_35220;
                    }
                    // write result of operation
                    {
                        ((__local
                          int32_t *) red_arr_mem_40394)[sext_i32_i64(local_tid_40377)] =
                            x_35214;
                        ((__local
                          float *) red_arr_mem_40396)[sext_i32_i64(local_tid_40377)] =
                            x_35215;
                    }
                }
                skip_waves_40399 *= 2;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            res_35212 = ((__local int32_t *) red_arr_mem_40394)[0];
            res_35213 = ((__local float *) red_arr_mem_40396)[0];
            
            int32_t i_35228 = sub32(7, j_35211);
            bool x_35229 = sle32(0, i_35228);
            bool y_35230 = slt32(i_35228, 8);
            bool bounds_check_35231 = x_35229 && y_35230;
            bool index_certs_35232;
            
            if (!bounds_check_35231) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 66) ==
                        -1) {
                        global_failure_args[0] = i_35228;
                        global_failure_args[1] = 8;
                        ;
                    }
                    local_failure = true;
                    goto error_3;
                }
            }
            
            float y_35233 = ((__local
                              float *) mem_39718)[sext_i32_i64(i_35228)];
            bool cond_35234 = res_35213 < y_35233;
            int32_t loopres_35238;

            barrier(CLK_LOCAL_MEM_FENCE); // COSMIN 1
            barrier(CLK_GLOBAL_MEM_FENCE);
            
            if (cond_35234) {
                bool x_35239 = sle32(0, res_35212);
                bool y_35240 = slt32(res_35212, ppl_31631);
                bool bounds_check_35241 = x_35239 && y_35240;
                bool index_certs_35242;
                
                if (!bounds_check_35241) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 67) ==
                            -1) {
                            global_failure_args[0] = res_35212;
                            global_failure_args[1] = ppl_31631;
                            ;
                        }
                        local_failure = true;
                        goto error_3;
                    }
                }
                //barrier(CLK_LOCAL_MEM_FENCE);
                if (local_tid_40377 == 0) {
                    ((__local float *) mem_39722)[sext_i32_i64(res_35212)] =
                        INFINITY;
                }
                //barrier(CLK_LOCAL_MEM_FENCE);
                
                int32_t lw_val_35244 = add32(bruteForcePar_arg_35184,
                                             res_35212);
                
                barrier(CLK_LOCAL_MEM_FENCE);
                if (local_tid_40377 == 0) {
                    ((__local int32_t *) mem_39716)[sext_i32_i64(i_35228)] =
                        lw_val_35244;
                }
                //barrier(CLK_LOCAL_MEM_FENCE);
                //barrier(CLK_LOCAL_MEM_FENCE);
                if (local_tid_40377 == 0) {
                    ((__local float *) mem_39718)[sext_i32_i64(i_35228)] =
                        res_35213;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                int32_t res_35247 = add32(1, j_35211);
                
                loopres_35238 = res_35247;
            } else {
                loopres_35238 = j_35211;
            }
            
            bool res_35248 = slt32(loopres_35238, 8);
            bool x_35249 = cond_35234 && res_35248;
            bool loop_while_tmp_40389 = x_35249;
            int32_t j_tmp_40393 = loopres_35238;
            
            loop_while_35207 = loop_while_tmp_40389;
            j_35211 = j_tmp_40393;
        }

        knn_35202 = loop_while_35207;
        knn_35206 = j_35211;
        for (int32_t i_40400 = 0; i_40400 < sdiv_up32(8 - local_tid_40377,
                                                      ppl_31631); i_40400++) {
            ((__local int32_t *) mem_39765)[sext_i32_i64(i_40400 * ppl_31631 +
                                            local_tid_40377)] = -1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_40401 = 0; i_40401 < sdiv_up32(8 - local_tid_40377,
                                                      ppl_31631); i_40401++) {
            ((__local float *) mem_39767)[sext_i32_i64(i_40401 * ppl_31631 +
                                          local_tid_40377)] = INFINITY;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t res_35254;
        int32_t res_35255;
        int32_t beg_35259;
        int32_t end_35260;
        
        beg_35259 = 0;
        end_35260 = 7;
        for (int32_t i_35256 = 0; i_35256 < 8; i_35256++) {
            bool x_35261 = sle32(0, beg_35259);
            bool y_35262 = slt32(beg_35259, 8);
            bool bounds_check_35263 = x_35261 && y_35262;
            bool index_certs_35264;
            
            if (!bounds_check_35263) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 68) ==
                        -1) {
                        global_failure_args[0] = beg_35259;
                        global_failure_args[1] = 8;
                        ;
                    }
                    local_failure = true;
                    goto error_3;
                }
            }
            
            float x_35265 = ((__local
                              float *) mem_39718)[sext_i32_i64(beg_35259)];
            bool x_35266 = sle32(0, end_35260);
            bool y_35267 = slt32(end_35260, 8);
            bool bounds_check_35268 = x_35266 && y_35267;
            bool index_certs_35269;
            
            if (!bounds_check_35268) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 69) ==
                        -1) {
                        global_failure_args[0] = end_35260;
                        global_failure_args[1] = 8;
                        ;
                    }
                    local_failure = true;
                    goto error_3;
                }
            }
            
            float y_35270 = ((__local
                              float *) mem_39718)[sext_i32_i64(end_35260)];
            bool cond_35271 = x_35265 < y_35270;
            float loopres_35272;
            
            if (cond_35271) {
                loopres_35272 = x_35265;
            } else {
                loopres_35272 = y_35270;
            }
            
            int32_t loopres_35273;
            int32_t loopres_35274;
            int32_t loopres_35275;
            
            if (cond_35271) {
                int32_t res_35276 = ((__local
                                      int32_t *) mem_39716)[sext_i32_i64(beg_35259)];
                int32_t res_35277 = add32(1, beg_35259);
                
                loopres_35273 = res_35276;
                loopres_35274 = res_35277;
                loopres_35275 = end_35260;
            } else {
                int32_t res_35278 = ((__local
                                      int32_t *) mem_39716)[sext_i32_i64(end_35260)];
                int32_t res_35279 = sub32(end_35260, 1);
                
                loopres_35273 = res_35278;
                loopres_35274 = beg_35259;
                loopres_35275 = res_35279;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_tid_40377 == 0) {
                ((__local int32_t *) mem_39765)[sext_i32_i64(i_35256)] =
                    loopres_35273;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_tid_40377 == 0) {
                ((__local float *) mem_39767)[sext_i32_i64(i_35256)] =
                    loopres_35272;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t beg_tmp_40404 = loopres_35274;
            int32_t end_tmp_40405 = loopres_35275;
            
            beg_35259 = beg_tmp_40404;
            end_35260 = end_tmp_40405;
        }
        res_35254 = beg_35259;
        res_35255 = end_35260;
        for (int32_t i_40406 = 0; i_40406 < sdiv_up32(8 - local_tid_40377,
                                                      ppl_31631); i_40406++) {
            ((__global
              int32_t *) double_buffer_mem_40104)[sext_i32_i64(phys_tid_34910 *
                                                  8) + sext_i32_i64(i_40406 *
                                                  ppl_31631 +
                                                  local_tid_40377)] = ((__local
                                                                        int32_t *) mem_39765)[sext_i32_i64(i_40406 *
                                                                                              ppl_31631 +
                                                                                              local_tid_40377)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_40407 = 0; i_40407 < sdiv_up32(8 - local_tid_40377,
                                                      ppl_31631); i_40407++) {
            ((__global
              float *) double_buffer_mem_40105)[sext_i32_i64(phys_tid_34910 *
                                                8) + sext_i32_i64(i_40407 *
                                                ppl_31631 + local_tid_40377)] =
                ((__local float *) mem_39767)[sext_i32_i64(i_40407 * ppl_31631 +
                                              local_tid_40377)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_tid_40377 == 0) {
        for (int32_t i_40408 = 0; i_40408 < 8; i_40408++) {
            ((__global int32_t *) mem_39823)[sext_i32_i64(gtid_34894) * 8 +
                                             sext_i32_i64(i_40408)] = ((__global
                                                                        int32_t *) double_buffer_mem_40104)[sext_i32_i64(phys_tid_34910 *
                                                                                                            8) +
                                                                                                            sext_i32_i64(i_40408)];
        }
    }
    if (local_tid_40377 == 0) {
        for (int32_t i_40409 = 0; i_40409 < 8; i_40409++) {
            ((__global float *) mem_39827)[sext_i32_i64(gtid_34894) * 8 +
                                           sext_i32_i64(i_40409)] = ((__global
                                                                      float *) double_buffer_mem_40105)[sext_i32_i64(phys_tid_34910 *
                                                                                                        8) +
                                                                                                        sext_i32_i64(i_40409)];
        }
    }
    
  error_3:
    return;
}
__kernel void reducePatchDimzisegmap_33764(__global int *global_failure,
                                           int32_t n_30993, int32_t d_30994,
                                           int32_t d_red_30995, int32_t d_30996,
                                           int32_t num_groups_33788, __global
                                           unsigned char *comps_mem_39419,
                                           __global
                                           unsigned char *means_mem_39420,
                                           __global unsigned char *mem_39424,
                                           __global unsigned char *mem_39428,
                                           __global unsigned char *mem_39445)
{
    #define segmap_group_sizze_33787 (reducePatchDimzisegmap_group_sizze_33767)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40300;
    int32_t local_tid_40301;
    int32_t group_sizze_40304;
    int32_t wave_sizze_40303;
    int32_t group_tid_40302;
    
    global_tid_40300 = get_global_id(0);
    local_tid_40301 = get_local_id(0);
    group_sizze_40304 = get_local_size(0);
    wave_sizze_40303 = LOCKSTEP_WIDTH;
    group_tid_40302 = get_group_id(0);
    
    int32_t phys_tid_33764;
    
    phys_tid_33764 = global_tid_40300;
    
    int32_t phys_group_id_40305;
    
    phys_group_id_40305 = get_group_id(0);
    for (int32_t i_40306 = 0; i_40306 < sdiv_up32(sdiv_up32(n_30993,
                                                            segmap_group_sizze_33787) -
                                                  phys_group_id_40305,
                                                  num_groups_33788);
         i_40306++) {
        int32_t virt_group_id_40307 = phys_group_id_40305 + i_40306 *
                num_groups_33788;
        int32_t gtid_33763 = sext_i64_i32(sext_i32_i64(virt_group_id_40307) *
                sext_i32_i64(segmap_group_sizze_33787) +
                sext_i32_i64(local_tid_40301));
        
        if (slt32(gtid_33763, n_30993)) {
            for (int32_t i_39355 = 0; i_39355 < d_red_30995; i_39355++) {
                float res_33794;
                float redout_39357 = 0.0F;
                
                for (int32_t i_39358 = 0; i_39358 < d_30994; i_39358++) {
                    int8_t x_33798 = ((__global
                                       int8_t *) mem_39424)[sext_i32_i64(i_39358) *
                                                            sext_i32_i64(n_30993) +
                                                            sext_i32_i64(gtid_33763)];
                    float x_33799 = ((__global
                                      float *) comps_mem_39419)[sext_i32_i64(i_39355) *
                                                                sext_i32_i64(d_30996) +
                                                                sext_i32_i64(i_39358)];
                    float x_33800 = ((__global
                                      float *) means_mem_39420)[sext_i32_i64(i_39358)];
                    float res_33801 = uitofp_i8_f32(x_33798);
                    float x_33802 = res_33801 - x_33800;
                    float res_33803 = x_33799 * x_33802;
                    float res_33797 = res_33803 + redout_39357;
                    float redout_tmp_40309 = res_33797;
                    
                    redout_39357 = redout_tmp_40309;
                }
                res_33794 = redout_39357;
                ((__global float *) mem_39428)[sext_i32_i64(phys_tid_33764) +
                                               sext_i32_i64(i_39355) *
                                               sext_i32_i64(num_groups_33788 *
                                               segmap_group_sizze_33787)] =
                    res_33794;
            }
            for (int32_t i_40310 = 0; i_40310 < d_red_30995; i_40310++) {
                ((__global float *) mem_39445)[sext_i32_i64(i_40310) *
                                               sext_i32_i64(n_30993) +
                                               sext_i32_i64(gtid_33763)] =
                    ((__global
                      float *) mem_39428)[sext_i32_i64(phys_tid_33764) +
                                          sext_i32_i64(i_40310) *
                                          sext_i32_i64(num_groups_33788 *
                                          segmap_group_sizze_33787)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_33787
}
__kernel void reducePatchDimzisegmap_33806(__global int *global_failure,
                                           int32_t n_30993, int32_t d_30994,
                                           int32_t d_red_30995, __global
                                           unsigned char *img_mem_39418,
                                           __global
                                           unsigned char *means_mem_39420,
                                           __global unsigned char *mem_39450,
                                           __global unsigned char *mem_39456)
{
    #define segmap_group_sizze_33884 (reducePatchDimzisegmap_group_sizze_33811)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40311;
    int32_t local_tid_40312;
    int32_t group_sizze_40315;
    int32_t wave_sizze_40314;
    int32_t group_tid_40313;
    
    global_tid_40311 = get_global_id(0);
    local_tid_40312 = get_local_id(0);
    group_sizze_40315 = get_local_size(0);
    wave_sizze_40314 = LOCKSTEP_WIDTH;
    group_tid_40313 = get_group_id(0);
    
    int32_t phys_tid_33806;
    
    phys_tid_33806 = global_tid_40311;
    
    int32_t gtid_33804;
    
    gtid_33804 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40313) *
                                      sext_i32_i64(segmap_group_sizze_33884) +
                                      sext_i32_i64(local_tid_40312),
                                      sext_i32_i64(d_red_30995)));
    
    int32_t gtid_33805;
    
    gtid_33805 = sext_i64_i32(sext_i32_i64(group_tid_40313) *
        sext_i32_i64(segmap_group_sizze_33884) + sext_i32_i64(local_tid_40312) -
        squot64(sext_i32_i64(group_tid_40313) *
                sext_i32_i64(segmap_group_sizze_33884) +
                sext_i32_i64(local_tid_40312), sext_i32_i64(d_red_30995)) *
        sext_i32_i64(d_red_30995));
    if (slt32(gtid_33804, n_30993) && slt32(gtid_33805, d_red_30995)) {
        float res_33895;
        float redout_39359 = 0.0F;
        
        for (int32_t i_39360 = 0; i_39360 < d_30994; i_39360++) {
            int8_t x_33899 = ((__global
                               int8_t *) img_mem_39418)[sext_i32_i64(gtid_33804) *
                                                        sext_i32_i64(d_30994) +
                                                        sext_i32_i64(i_39360)];
            float x_33900 = ((__global
                              float *) mem_39450)[sext_i32_i64(i_39360) *
                                                  sext_i32_i64(d_red_30995) +
                                                  sext_i32_i64(gtid_33805)];
            float x_33901 = ((__global
                              float *) means_mem_39420)[sext_i32_i64(i_39360)];
            float res_33902 = uitofp_i8_f32(x_33899);
            float x_33903 = res_33902 - x_33901;
            float res_33904 = x_33900 * x_33903;
            float res_33898 = res_33904 + redout_39359;
            float redout_tmp_40316 = res_33898;
            
            redout_39359 = redout_tmp_40316;
        }
        res_33895 = redout_39359;
        ((__global float *) mem_39456)[sext_i32_i64(gtid_33804) *
                                       sext_i32_i64(d_red_30995) +
                                       sext_i32_i64(gtid_33805)] = res_33895;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_33884
}
__kernel void reducePatchDimzisegred_large_33846(__global int *global_failure,
                                                 uint sync_arr_mem_40353_backing_offset_0,
                                                 uint red_arr_mem_40351_backing_offset_1,
                                                 int32_t d_30994,
                                                 int32_t d_red_30995,
                                                 int32_t d_30996,
                                                 int32_t num_groups_33912,
                                                 __global
                                                 unsigned char *img_mem_39418,
                                                 __global
                                                 unsigned char *comps_mem_39419,
                                                 __global
                                                 unsigned char *means_mem_39420,
                                                 __global
                                                 unsigned char *mem_39462,
                                                 int32_t groups_per_segment_40337,
                                                 int32_t elements_per_thread_40338,
                                                 int32_t virt_num_groups_40339,
                                                 int32_t threads_per_segment_40341,
                                                 __global
                                                 unsigned char *group_res_arr_mem_40342,
                                                 __global
                                                 unsigned char *reducePatchDimzicounter_mem_40344)
{
    #define segred_group_sizze_33911 (reducePatchDimzisegred_group_sizze_33840)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *sync_arr_mem_40353_backing_1 =
                  &shared_mem[sync_arr_mem_40353_backing_offset_0];
    volatile char *red_arr_mem_40351_backing_0 =
                  &shared_mem[red_arr_mem_40351_backing_offset_1];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40346;
    int32_t local_tid_40347;
    int32_t group_sizze_40350;
    int32_t wave_sizze_40349;
    int32_t group_tid_40348;
    
    global_tid_40346 = get_global_id(0);
    local_tid_40347 = get_local_id(0);
    group_sizze_40350 = get_local_size(0);
    wave_sizze_40349 = LOCKSTEP_WIDTH;
    group_tid_40348 = get_group_id(0);
    
    int32_t phys_tid_33846;
    
    phys_tid_33846 = global_tid_40346;
    
    __local char *red_arr_mem_40351;
    
    red_arr_mem_40351 = (__local char *) red_arr_mem_40351_backing_0;
    
    __local char *sync_arr_mem_40353;
    
    sync_arr_mem_40353 = (__local char *) sync_arr_mem_40353_backing_1;
    
    int32_t phys_group_id_40355;
    
    phys_group_id_40355 = get_group_id(0);
    for (int32_t i_40356 = 0; i_40356 < sdiv_up32(virt_num_groups_40339 -
                                                  phys_group_id_40355,
                                                  num_groups_33912);
         i_40356++) {
        int32_t virt_group_id_40357 = phys_group_id_40355 + i_40356 *
                num_groups_33912;
        int32_t flat_segment_id_40358 = squot32(virt_group_id_40357,
                                                groups_per_segment_40337);
        int32_t global_tid_40359 = srem32(virt_group_id_40357 *
                                          segred_group_sizze_33911 +
                                          local_tid_40347,
                                          segred_group_sizze_33911 *
                                          groups_per_segment_40337);
        int32_t gtid_33832 = squot32(flat_segment_id_40358, d_red_30995);
        int32_t gtid_33833 = flat_segment_id_40358 -
                squot32(flat_segment_id_40358, d_red_30995) * d_red_30995;
        int32_t gtid_33845;
        float x_acc_40360;
        int32_t chunk_sizze_40361;
        
        chunk_sizze_40361 = smin32(elements_per_thread_40338,
                                   sdiv_up32(d_30994 - global_tid_40359,
                                             threads_per_segment_40341));
        
        float x_33915;
        float x_33916;
        
        // neutral-initialise the accumulators
        {
            x_acc_40360 = 0.0F;
        }
        for (int32_t i_40365 = 0; i_40365 < chunk_sizze_40361; i_40365++) {
            gtid_33845 = global_tid_40359 + threads_per_segment_40341 * i_40365;
            // apply map function
            {
                int8_t x_33920 = ((__global
                                   int8_t *) img_mem_39418)[sext_i32_i64(gtid_33832) *
                                                            sext_i32_i64(d_30994) +
                                                            sext_i32_i64(gtid_33845)];
                float x_33921 = ((__global
                                  float *) comps_mem_39419)[sext_i32_i64(gtid_33833) *
                                                            sext_i32_i64(d_30996) +
                                                            sext_i32_i64(gtid_33845)];
                float x_33922 = ((__global
                                  float *) means_mem_39420)[sext_i32_i64(gtid_33845)];
                float res_33923 = uitofp_i8_f32(x_33920);
                float x_33924 = res_33923 - x_33922;
                float res_33925 = x_33921 * x_33924;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_33915 = x_acc_40360;
                }
                // load new values
                {
                    x_33916 = res_33925;
                }
                // apply reduction operator
                {
                    float res_33917 = x_33915 + x_33916;
                    
                    // store in accumulator
                    {
                        x_acc_40360 = res_33917;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_33915 = x_acc_40360;
            ((__local
              float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347)] =
                x_33915;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_40366;
        int32_t skip_waves_40367;
        float x_40362;
        float x_40363;
        
        offset_40366 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_40347, segred_group_sizze_33911)) {
                x_40362 = ((__local
                            float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347 +
                                                        offset_40366)];
            }
        }
        offset_40366 = 1;
        while (slt32(offset_40366, wave_sizze_40349)) {
            if (slt32(local_tid_40347 + offset_40366,
                      segred_group_sizze_33911) && ((local_tid_40347 -
                                                     squot32(local_tid_40347,
                                                             wave_sizze_40349) *
                                                     wave_sizze_40349) & (2 *
                                                                          offset_40366 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_40363 = ((volatile __local
                                float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347 +
                                                            offset_40366)];
                }
                // apply reduction operation
                {
                    float res_40364 = x_40362 + x_40363;
                    
                    x_40362 = res_40364;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347)] =
                        x_40362;
                }
            }
            offset_40366 *= 2;
        }
        skip_waves_40367 = 1;
        while (slt32(skip_waves_40367, squot32(segred_group_sizze_33911 +
                                               wave_sizze_40349 - 1,
                                               wave_sizze_40349))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_40366 = skip_waves_40367 * wave_sizze_40349;
            if (slt32(local_tid_40347 + offset_40366,
                      segred_group_sizze_33911) && ((local_tid_40347 -
                                                     squot32(local_tid_40347,
                                                             wave_sizze_40349) *
                                                     wave_sizze_40349) == 0 &&
                                                    (squot32(local_tid_40347,
                                                             wave_sizze_40349) &
                                                     (2 * skip_waves_40367 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_40363 = ((__local
                                float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347 +
                                                            offset_40366)];
                }
                // apply reduction operation
                {
                    float res_40364 = x_40362 + x_40363;
                    
                    x_40362 = res_40364;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347)] =
                        x_40362;
                }
            }
            skip_waves_40367 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_40347 == 0) {
                x_acc_40360 = x_40362;
            }
        }
        if (groups_per_segment_40337 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_40347 == 0) {
                    ((__global float *) mem_39462)[sext_i32_i64(gtid_33832) *
                                                   sext_i32_i64(d_red_30995) +
                                                   sext_i32_i64(gtid_33833)] =
                        x_acc_40360;
                }
            }
        } else {
            int32_t old_counter_40368;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_40347 == 0) {
                    ((__global
                      float *) group_res_arr_mem_40342)[sext_i32_i64(virt_group_id_40357) *
                                                        sext_i32_i64(segred_group_sizze_33911)] =
                        x_acc_40360;
                    mem_fence_global();
                    old_counter_40368 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) reducePatchDimzicounter_mem_40344)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_40358,
                                                                                                                            10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_40353)[0] =
                        old_counter_40368 == groups_per_segment_40337 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_40369;
            
            is_last_group_40369 = ((__local bool *) sync_arr_mem_40353)[0];
            if (is_last_group_40369) {
                if (local_tid_40347 == 0) {
                    old_counter_40368 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) reducePatchDimzicounter_mem_40344)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_40358,
                                                                                                                            10240)))],
                                              (int) (0 -
                                                     groups_per_segment_40337));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_40370 =
                            sdiv_up32(groups_per_segment_40337,
                                      segred_group_sizze_33911);
                    
                    x_33915 = 0.0F;
                    for (int32_t i_40371 = 0; i_40371 < read_per_thread_40370;
                         i_40371++) {
                        int32_t group_res_id_40372 = local_tid_40347 *
                                read_per_thread_40370 + i_40371;
                        int32_t index_of_group_res_40373 =
                                flat_segment_id_40358 *
                                groups_per_segment_40337 + group_res_id_40372;
                        
                        if (slt32(group_res_id_40372,
                                  groups_per_segment_40337)) {
                            x_33916 = ((__global
                                        float *) group_res_arr_mem_40342)[sext_i32_i64(index_of_group_res_40373) *
                                                                          sext_i32_i64(segred_group_sizze_33911)];
                            
                            float res_33917;
                            
                            res_33917 = x_33915 + x_33916;
                            x_33915 = res_33917;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347)] =
                    x_33915;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_40374;
                    int32_t skip_waves_40375;
                    float x_40362;
                    float x_40363;
                    
                    offset_40374 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_40347, segred_group_sizze_33911)) {
                            x_40362 = ((__local
                                        float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347 +
                                                                    offset_40374)];
                        }
                    }
                    offset_40374 = 1;
                    while (slt32(offset_40374, wave_sizze_40349)) {
                        if (slt32(local_tid_40347 + offset_40374,
                                  segred_group_sizze_33911) &&
                            ((local_tid_40347 - squot32(local_tid_40347,
                                                        wave_sizze_40349) *
                              wave_sizze_40349) & (2 * offset_40374 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_40363 = ((volatile __local
                                            float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347 +
                                                                        offset_40374)];
                            }
                            // apply reduction operation
                            {
                                float res_40364 = x_40362 + x_40363;
                                
                                x_40362 = res_40364;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347)] =
                                    x_40362;
                            }
                        }
                        offset_40374 *= 2;
                    }
                    skip_waves_40375 = 1;
                    while (slt32(skip_waves_40375,
                                 squot32(segred_group_sizze_33911 +
                                         wave_sizze_40349 - 1,
                                         wave_sizze_40349))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_40374 = skip_waves_40375 * wave_sizze_40349;
                        if (slt32(local_tid_40347 + offset_40374,
                                  segred_group_sizze_33911) &&
                            ((local_tid_40347 - squot32(local_tid_40347,
                                                        wave_sizze_40349) *
                              wave_sizze_40349) == 0 &&
                             (squot32(local_tid_40347, wave_sizze_40349) & (2 *
                                                                            skip_waves_40375 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_40363 = ((__local
                                            float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347 +
                                                                        offset_40374)];
                            }
                            // apply reduction operation
                            {
                                float res_40364 = x_40362 + x_40363;
                                
                                x_40362 = res_40364;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_40351)[sext_i32_i64(local_tid_40347)] =
                                    x_40362;
                            }
                        }
                        skip_waves_40375 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_40347 == 0) {
                            ((__global
                              float *) mem_39462)[sext_i32_i64(gtid_33832) *
                                                  sext_i32_i64(d_red_30995) +
                                                  sext_i32_i64(gtid_33833)] =
                                x_40362;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_33911
}
__kernel void reducePatchDimzisegred_small_33846(__global int *global_failure,
                                                 uint red_arr_mem_40324_backing_offset_0,
                                                 int32_t n_30993,
                                                 int32_t d_30994,
                                                 int32_t d_red_30995,
                                                 int32_t d_30996,
                                                 int32_t num_groups_33912,
                                                 __global
                                                 unsigned char *img_mem_39418,
                                                 __global
                                                 unsigned char *comps_mem_39419,
                                                 __global
                                                 unsigned char *means_mem_39420,
                                                 __global
                                                 unsigned char *mem_39462,
                                                 int32_t segment_sizze_nonzzero_40317)
{
    #define segred_group_sizze_33911 (reducePatchDimzisegred_group_sizze_33840)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40324_backing_0 =
                  &shared_mem[red_arr_mem_40324_backing_offset_0];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40319;
    int32_t local_tid_40320;
    int32_t group_sizze_40323;
    int32_t wave_sizze_40322;
    int32_t group_tid_40321;
    
    global_tid_40319 = get_global_id(0);
    local_tid_40320 = get_local_id(0);
    group_sizze_40323 = get_local_size(0);
    wave_sizze_40322 = LOCKSTEP_WIDTH;
    group_tid_40321 = get_group_id(0);
    
    int32_t phys_tid_33846;
    
    phys_tid_33846 = global_tid_40319;
    
    __local char *red_arr_mem_40324;
    
    red_arr_mem_40324 = (__local char *) red_arr_mem_40324_backing_0;
    
    int32_t phys_group_id_40326;
    
    phys_group_id_40326 = get_group_id(0);
    for (int32_t i_40327 = 0; i_40327 < sdiv_up32(sdiv_up32(n_30993 *
                                                            d_red_30995,
                                                            squot32(segred_group_sizze_33911,
                                                                    segment_sizze_nonzzero_40317)) -
                                                  phys_group_id_40326,
                                                  num_groups_33912);
         i_40327++) {
        int32_t virt_group_id_40328 = phys_group_id_40326 + i_40327 *
                num_groups_33912;
        int32_t gtid_33832 = squot32(squot32(local_tid_40320,
                                             segment_sizze_nonzzero_40317) +
                                     virt_group_id_40328 *
                                     squot32(segred_group_sizze_33911,
                                             segment_sizze_nonzzero_40317),
                                     d_red_30995);
        int32_t gtid_33833 = squot32(local_tid_40320,
                                     segment_sizze_nonzzero_40317) +
                virt_group_id_40328 * squot32(segred_group_sizze_33911,
                                              segment_sizze_nonzzero_40317) -
                squot32(squot32(local_tid_40320, segment_sizze_nonzzero_40317) +
                        virt_group_id_40328 * squot32(segred_group_sizze_33911,
                                                      segment_sizze_nonzzero_40317),
                        d_red_30995) * d_red_30995;
        int32_t gtid_33845 = srem32(local_tid_40320, d_30994);
        
        // apply map function if in bounds
        {
            if (slt32(0, d_30994) && ((slt32(gtid_33832, n_30993) &&
                                       slt32(gtid_33833, d_red_30995)) &&
                                      slt32(local_tid_40320, d_30994 *
                                            squot32(segred_group_sizze_33911,
                                                    segment_sizze_nonzzero_40317)))) {
                int8_t x_33920 = ((__global
                                   int8_t *) img_mem_39418)[sext_i32_i64(gtid_33832) *
                                                            sext_i32_i64(d_30994) +
                                                            sext_i32_i64(gtid_33845)];
                float x_33921 = ((__global
                                  float *) comps_mem_39419)[sext_i32_i64(gtid_33833) *
                                                            sext_i32_i64(d_30996) +
                                                            sext_i32_i64(gtid_33845)];
                float x_33922 = ((__global
                                  float *) means_mem_39420)[sext_i32_i64(gtid_33845)];
                float res_33923 = uitofp_i8_f32(x_33920);
                float x_33924 = res_33923 - x_33922;
                float res_33925 = x_33921 * x_33924;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320)] =
                        res_33925;
                }
            } else {
                ((__local
                  float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, d_30994)) {
            // perform segmented scan to imitate reduction
            {
                float x_33915;
                float x_33916;
                float x_40329;
                float x_40330;
                int32_t skip_threads_40332;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_40320, d_30994 *
                              squot32(segred_group_sizze_33911,
                                      segment_sizze_nonzzero_40317))) {
                        x_33916 = ((volatile __local
                                    float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320)];
                        if ((local_tid_40320 - squot32(local_tid_40320, 32) *
                             32) == 0) {
                            x_33915 = x_33916;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_40332 = 1;
                    while (slt32(skip_threads_40332, 32)) {
                        if (sle32(skip_threads_40332, local_tid_40320 -
                                  squot32(local_tid_40320, 32) * 32) &&
                            slt32(local_tid_40320, d_30994 *
                                  squot32(segred_group_sizze_33911,
                                          segment_sizze_nonzzero_40317))) {
                            // read operands
                            {
                                x_33915 = ((volatile __local
                                            float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320 -
                                                                        skip_threads_40332)];
                            }
                            // perform operation
                            {
                                bool inactive_40333 =
                                     slt32(srem32(local_tid_40320, d_30994),
                                           local_tid_40320 - (local_tid_40320 -
                                                              skip_threads_40332));
                                
                                if (inactive_40333) {
                                    x_33915 = x_33916;
                                }
                                if (!inactive_40333) {
                                    float res_33917 = x_33915 + x_33916;
                                    
                                    x_33915 = res_33917;
                                }
                            }
                        }
                        if (sle32(wave_sizze_40322, skip_threads_40332)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_40332, local_tid_40320 -
                                  squot32(local_tid_40320, 32) * 32) &&
                            slt32(local_tid_40320, d_30994 *
                                  squot32(segred_group_sizze_33911,
                                          segment_sizze_nonzzero_40317))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320)] =
                                    x_33915;
                                x_33916 = x_33915;
                            }
                        }
                        if (sle32(wave_sizze_40322, skip_threads_40332)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_40332 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_40320 - squot32(local_tid_40320, 32) * 32) ==
                        31 && slt32(local_tid_40320, d_30994 *
                                    squot32(segred_group_sizze_33911,
                                            segment_sizze_nonzzero_40317))) {
                        ((volatile __local
                          float *) red_arr_mem_40324)[sext_i32_i64(squot32(local_tid_40320,
                                                                           32))] =
                            x_33915;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_40334;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_40320, 32) == 0 &&
                            slt32(local_tid_40320, d_30994 *
                                  squot32(segred_group_sizze_33911,
                                          segment_sizze_nonzzero_40317))) {
                            x_40330 = ((volatile __local
                                        float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320)];
                            if ((local_tid_40320 - squot32(local_tid_40320,
                                                           32) * 32) == 0) {
                                x_40329 = x_40330;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_40334 = 1;
                        while (slt32(skip_threads_40334, 32)) {
                            if (sle32(skip_threads_40334, local_tid_40320 -
                                      squot32(local_tid_40320, 32) * 32) &&
                                (squot32(local_tid_40320, 32) == 0 &&
                                 slt32(local_tid_40320, d_30994 *
                                       squot32(segred_group_sizze_33911,
                                               segment_sizze_nonzzero_40317)))) {
                                // read operands
                                {
                                    x_40329 = ((volatile __local
                                                float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320 -
                                                                            skip_threads_40334)];
                                }
                                // perform operation
                                {
                                    bool inactive_40335 =
                                         slt32(srem32(local_tid_40320 * 32 +
                                                      32 - 1, d_30994),
                                               local_tid_40320 * 32 + 32 - 1 -
                                               ((local_tid_40320 -
                                                 skip_threads_40334) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_40335) {
                                        x_40329 = x_40330;
                                    }
                                    if (!inactive_40335) {
                                        float res_40331 = x_40329 + x_40330;
                                        
                                        x_40329 = res_40331;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_40322, skip_threads_40334)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_40334, local_tid_40320 -
                                      squot32(local_tid_40320, 32) * 32) &&
                                (squot32(local_tid_40320, 32) == 0 &&
                                 slt32(local_tid_40320, d_30994 *
                                       squot32(segred_group_sizze_33911,
                                               segment_sizze_nonzzero_40317)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320)] =
                                        x_40329;
                                    x_40330 = x_40329;
                                }
                            }
                            if (sle32(wave_sizze_40322, skip_threads_40334)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_40334 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_40320, 32) == 0 ||
                          !slt32(local_tid_40320, d_30994 *
                                 squot32(segred_group_sizze_33911,
                                         segment_sizze_nonzzero_40317)))) {
                        // read operands
                        {
                            x_33916 = x_33915;
                            x_33915 = ((__local
                                        float *) red_arr_mem_40324)[sext_i32_i64(squot32(local_tid_40320,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_40336 = slt32(srem32(local_tid_40320,
                                                               d_30994),
                                                        local_tid_40320 -
                                                        (squot32(local_tid_40320,
                                                                 32) * 32 - 1));
                            
                            if (inactive_40336) {
                                x_33915 = x_33916;
                            }
                            if (!inactive_40336) {
                                float res_33917 = x_33915 + x_33916;
                                
                                x_33915 = res_33917;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320)] =
                                x_33915;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_40320, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_40324)[sext_i32_i64(local_tid_40320)] =
                            x_33916;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_40328 * squot32(segred_group_sizze_33911,
                                                    segment_sizze_nonzzero_40317) +
                      local_tid_40320, n_30993 * d_red_30995) &&
                slt32(local_tid_40320, squot32(segred_group_sizze_33911,
                                               segment_sizze_nonzzero_40317))) {
                ((__global
                  float *) mem_39462)[sext_i32_i64(squot32(virt_group_id_40328 *
                                                           squot32(segred_group_sizze_33911,
                                                                   segment_sizze_nonzzero_40317) +
                                                           local_tid_40320,
                                                           d_red_30995)) *
                                      sext_i32_i64(d_red_30995) +
                                      sext_i32_i64(virt_group_id_40328 *
                                      squot32(segred_group_sizze_33911,
                                              segment_sizze_nonzzero_40317) +
                                      local_tid_40320 -
                                      squot32(virt_group_id_40328 *
                                              squot32(segred_group_sizze_33911,
                                                      segment_sizze_nonzzero_40317) +
                                              local_tid_40320, d_red_30995) *
                                      d_red_30995)] = ((__local
                                                        float *) red_arr_mem_40324)[sext_i32_i64((local_tid_40320 +
                                                                                                  1) *
                                                                                    segment_sizze_nonzzero_40317 -
                                                                                    1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_33911
}
__kernel void selectBestNNzisegmap_32876(__global int *global_failure,
                                         int failure_is_an_option, __global
                                         int *global_failure_args,
                                         int32_t n_30870, int32_t h1_30872,
                                         int32_t w1_30873, int32_t c_30874,
                                         int32_t h2_30875, int32_t w2_30876,
                                         int32_t c_30877, int32_t p_30878,
                                         int32_t n_colsA_30887,
                                         int32_t n_colsB_30889,
                                         int32_t patch_len_30891,
                                         int32_t segmap_usable_groups_33466,
                                         __global unsigned char *imgA_mem_39419,
                                         __global unsigned char *imgB_mem_39420,
                                         __global unsigned char *mem_39454,
                                         __global unsigned char *mem_39458,
                                         __global unsigned char *mem_39473,
                                         __global unsigned char *mem_39476,
                                         __global unsigned char *mem_39479)
{
    #define segmap_group_sizze_33463 (selectBestNNzisegmap_group_sizze_32879)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40336;
    int32_t local_tid_40337;
    int32_t group_sizze_40340;
    int32_t wave_sizze_40339;
    int32_t group_tid_40338;
    
    global_tid_40336 = get_global_id(0);
    local_tid_40337 = get_local_id(0);
    group_sizze_40340 = get_local_size(0);
    wave_sizze_40339 = LOCKSTEP_WIDTH;
    group_tid_40338 = get_group_id(0);
    
    int32_t phys_tid_32876;
    
    phys_tid_32876 = global_tid_40336;
    
    int32_t gtid_32875;
    
    gtid_32875 = sext_i64_i32(sext_i32_i64(group_tid_40338) *
        sext_i32_i64(segmap_group_sizze_33463) + sext_i32_i64(local_tid_40337));
    if (slt32(gtid_32875, n_30870)) {
        int32_t y_33472 = sdiv32(gtid_32875, n_colsA_30887);
        int32_t y_33473 = mul32(n_colsA_30887, y_33472);
        int32_t x_33474 = sub32(gtid_32875, y_33473);
        
        for (int32_t i_39359 = 0; i_39359 < patch_len_30891; i_39359++) {
            int32_t ij_33477 = sdiv32(i_39359, c_30874);
            int32_t y_33478 = mul32(c_30874, ij_33477);
            int32_t k_33479 = sub32(i_39359, y_33478);
            int32_t i_33480 = sdiv32(ij_33477, p_30878);
            int32_t y_33481 = mul32(p_30878, i_33480);
            int32_t j_33482 = sub32(ij_33477, y_33481);
            int32_t i_33483 = add32(y_33472, i_33480);
            bool x_33484 = sle32(0, i_33483);
            bool y_33485 = slt32(i_33483, h1_30872);
            bool bounds_check_33486 = x_33484 && y_33485;
            int32_t i_33487 = add32(x_33474, j_33482);
            bool x_33488 = sle32(0, i_33487);
            bool y_33489 = slt32(i_33487, w1_30873);
            bool bounds_check_33490 = x_33488 && y_33489;
            bool x_33491 = sle32(0, k_33479);
            bool y_33492 = slt32(k_33479, c_30874);
            bool bounds_check_33493 = x_33491 && y_33492;
            bool y_33494 = bounds_check_33486 && bounds_check_33493;
            bool index_ok_33495 = bounds_check_33490 && y_33494;
            bool index_certs_33496;
            
            if (!index_ok_33495) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 79) ==
                        -1) {
                        global_failure_args[0] = i_33483;
                        global_failure_args[1] = i_33487;
                        global_failure_args[2] = k_33479;
                        global_failure_args[3] = h1_30872;
                        global_failure_args[4] = w1_30873;
                        global_failure_args[5] = c_30874;
                        ;
                    }
                    return;
                }
            }
            
            int32_t i32_arg_33497 = ((__global
                                      int32_t *) imgA_mem_39419)[sext_i32_i64(i_33483) *
                                                                 sext_i32_i64(c_30874 *
                                                                 w1_30873) +
                                                                 sext_i32_i64(i_33487) *
                                                                 sext_i32_i64(c_30874) +
                                                                 sext_i32_i64(k_33479)];
            float res_33498 = sitofp_i32_f32(i32_arg_33497);
            
            ((__global float *) mem_39458)[sext_i32_i64(phys_tid_32876) +
                                           sext_i32_i64(i_39359) *
                                           sext_i32_i64(segmap_usable_groups_33466 *
                                           segmap_group_sizze_33463)] =
                res_33498;
        }
        
        int32_t res_33499;
        float res_33500;
        int32_t nn_ind_33502;
        float nn_dst_33503;
        
        nn_ind_33502 = -1;
        nn_dst_33503 = INFINITY;
        for (int32_t q_33501 = 0; q_33501 < 8; q_33501++) {
            int32_t indB_33504 = ((__global
                                   int32_t *) mem_39454)[sext_i32_i64(q_33501) *
                                                         sext_i32_i64(n_30870) +
                                                         sext_i32_i64(gtid_32875)];
            int32_t ii_33505 = sdiv32(indB_33504, n_colsB_30889);
            int32_t y_33506 = mul32(n_colsB_30889, ii_33505);
            int32_t jj_33507 = sub32(indB_33504, y_33506);
            float res_33508;
            float redout_39288 = 0.0F;
            
            for (int32_t i_39289 = 0; i_39289 < patch_len_30891; i_39289++) {
                int32_t ij_33513 = sdiv32(i_39289, c_30874);
                int32_t y_33514 = mul32(c_30874, ij_33513);
                int32_t k_33515 = sub32(i_39289, y_33514);
                int32_t i_33516 = sdiv32(ij_33513, p_30878);
                int32_t y_33517 = mul32(p_30878, i_33516);
                int32_t j_33518 = sub32(ij_33513, y_33517);
                int32_t i_33519 = add32(ii_33505, i_33516);
                bool x_33520 = sle32(0, i_33519);
                bool y_33521 = slt32(i_33519, h2_30875);
                bool bounds_check_33522 = x_33520 && y_33521;
                int32_t i_33523 = add32(jj_33507, j_33518);
                bool x_33524 = sle32(0, i_33523);
                bool y_33525 = slt32(i_33523, w2_30876);
                bool bounds_check_33526 = x_33524 && y_33525;
                bool x_33527 = sle32(0, k_33515);
                bool y_33528 = slt32(k_33515, c_30874);
                bool bounds_check_33529 = x_33527 && y_33528;
                bool y_33530 = bounds_check_33522 && bounds_check_33529;
                bool index_ok_33531 = bounds_check_33526 && y_33530;
                bool index_certs_33532;
                
                if (!index_ok_33531) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 80) ==
                            -1) {
                            global_failure_args[0] = i_33519;
                            global_failure_args[1] = i_33523;
                            global_failure_args[2] = k_33515;
                            global_failure_args[3] = h2_30875;
                            global_failure_args[4] = w2_30876;
                            global_failure_args[5] = c_30874;
                            ;
                        }
                        return;
                    }
                }
                
                int32_t i32_arg_33533 = ((__global
                                          int32_t *) imgB_mem_39420)[sext_i32_i64(i_33519) *
                                                                     sext_i32_i64(c_30877 *
                                                                     w2_30876) +
                                                                     sext_i32_i64(i_33523) *
                                                                     sext_i32_i64(c_30877) +
                                                                     sext_i32_i64(k_33515)];
                float res_33534 = sitofp_i32_f32(i32_arg_33533);
                float a_v_33539 = ((__global
                                    float *) mem_39458)[sext_i32_i64(phys_tid_32876) +
                                                        sext_i32_i64(i_39289) *
                                                        sext_i32_i64(segmap_usable_groups_33466 *
                                                        segmap_group_sizze_33463)];
                float d_33540 = res_33534 - a_v_33539;
                float res_33541 = d_33540 * d_33540;
                float res_33511 = res_33541 + redout_39288;
                float redout_tmp_40344 = res_33511;
                
                redout_39288 = redout_tmp_40344;
            }
            res_33508 = redout_39288;
            
            float dst_33542 = res_33508;
            bool cond_33543 = res_33508 < nn_dst_33503;
            int32_t loopres_33544;
            
            if (cond_33543) {
                loopres_33544 = indB_33504;
            } else {
                loopres_33544 = nn_ind_33502;
            }
            
            float loopres_33545;
            
            if (cond_33543) {
                loopres_33545 = dst_33542;
            } else {
                loopres_33545 = nn_dst_33503;
            }
            
            int32_t nn_ind_tmp_40342 = loopres_33544;
            float nn_dst_tmp_40343 = loopres_33545;
            
            nn_ind_33502 = nn_ind_tmp_40342;
            nn_dst_33503 = nn_dst_tmp_40343;
        }
        res_33499 = nn_ind_33502;
        res_33500 = nn_dst_33503;
        ((__global float *) mem_39473)[sext_i32_i64(gtid_32875)] = res_33500;
        ((__global int32_t *) mem_39476)[sext_i32_i64(gtid_32875)] = res_33499;
        ((__global float *) mem_39479)[sext_i32_i64(gtid_32875)] = res_33500;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_33463
}
__kernel void selectBestNNzisegmap_33132(__global int *global_failure,
                                         int failure_is_an_option, __global
                                         int *global_failure_args,
                                         int32_t n_30870, int32_t c_30874,
                                         int32_t h2_30875, int32_t w2_30876,
                                         int32_t c_30877, int32_t p_30878,
                                         int32_t n_colsB_30889,
                                         int32_t patch_len_30891, __global
                                         unsigned char *imgB_mem_39420, __global
                                         unsigned char *mem_39510, __global
                                         unsigned char *mem_39515, __global
                                         unsigned char *mem_39519, __global
                                         unsigned char *mem_39522, __global
                                         unsigned char *mem_39525)
{
    #define segmap_group_sizze_33688 (selectBestNNzisegmap_group_sizze_33135)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40362;
    int32_t local_tid_40363;
    int32_t group_sizze_40366;
    int32_t wave_sizze_40365;
    int32_t group_tid_40364;
    
    global_tid_40362 = get_global_id(0);
    local_tid_40363 = get_local_id(0);
    group_sizze_40366 = get_local_size(0);
    wave_sizze_40365 = LOCKSTEP_WIDTH;
    group_tid_40364 = get_group_id(0);
    
    int32_t phys_tid_33132;
    
    phys_tid_33132 = global_tid_40362;
    
    int32_t gtid_33131;
    
    gtid_33131 = sext_i64_i32(sext_i32_i64(group_tid_40364) *
        sext_i32_i64(segmap_group_sizze_33688) + sext_i32_i64(local_tid_40363));
    if (slt32(gtid_33131, n_30870)) {
        int32_t res_33697;
        float res_33698;
        int32_t nn_ind_33700;
        float nn_dst_33701;
        
        nn_ind_33700 = -1;
        nn_dst_33701 = INFINITY;
        for (int32_t q_33699 = 0; q_33699 < 8; q_33699++) {
            int32_t indB_33702 = ((__global
                                   int32_t *) mem_39510)[sext_i32_i64(q_33699) *
                                                         sext_i32_i64(n_30870) +
                                                         sext_i32_i64(gtid_33131)];
            int32_t ii_33703 = sdiv32(indB_33702, n_colsB_30889);
            int32_t y_33704 = mul32(n_colsB_30889, ii_33703);
            int32_t jj_33705 = sub32(indB_33702, y_33704);
            float res_33706;
            float redout_39312 = 0.0F;
            
            for (int32_t i_39313 = 0; i_39313 < patch_len_30891; i_39313++) {
                int32_t ij_33711 = sdiv32(i_39313, c_30874);
                int32_t y_33712 = mul32(c_30874, ij_33711);
                int32_t k_33713 = sub32(i_39313, y_33712);
                int32_t i_33714 = sdiv32(ij_33711, p_30878);
                int32_t y_33715 = mul32(p_30878, i_33714);
                int32_t j_33716 = sub32(ij_33711, y_33715);
                int32_t i_33717 = add32(ii_33703, i_33714);
                bool x_33718 = sle32(0, i_33717);
                bool y_33719 = slt32(i_33717, h2_30875);
                bool bounds_check_33720 = x_33718 && y_33719;
                int32_t i_33721 = add32(jj_33705, j_33716);
                bool x_33722 = sle32(0, i_33721);
                bool y_33723 = slt32(i_33721, w2_30876);
                bool bounds_check_33724 = x_33722 && y_33723;
                bool x_33725 = sle32(0, k_33713);
                bool y_33726 = slt32(k_33713, c_30874);
                bool bounds_check_33727 = x_33725 && y_33726;
                bool y_33728 = bounds_check_33720 && bounds_check_33727;
                bool index_ok_33729 = bounds_check_33724 && y_33728;
                bool index_certs_33730;
                
                if (!index_ok_33729) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 84) ==
                            -1) {
                            global_failure_args[0] = i_33717;
                            global_failure_args[1] = i_33721;
                            global_failure_args[2] = k_33713;
                            global_failure_args[3] = h2_30875;
                            global_failure_args[4] = w2_30876;
                            global_failure_args[5] = c_30874;
                            ;
                        }
                        return;
                    }
                }
                
                int32_t i32_arg_33731 = ((__global
                                          int32_t *) imgB_mem_39420)[sext_i32_i64(i_33717) *
                                                                     sext_i32_i64(c_30877 *
                                                                     w2_30876) +
                                                                     sext_i32_i64(i_33721) *
                                                                     sext_i32_i64(c_30877) +
                                                                     sext_i32_i64(k_33713)];
                float res_33732 = sitofp_i32_f32(i32_arg_33731);
                float a_v_33737 = ((__global
                                    float *) mem_39515)[sext_i32_i64(i_39313) *
                                                        sext_i32_i64(n_30870) +
                                                        sext_i32_i64(gtid_33131)];
                float d_33738 = res_33732 - a_v_33737;
                float res_33739 = d_33738 * d_33738;
                float res_33709 = res_33739 + redout_39312;
                float redout_tmp_40369 = res_33709;
                
                redout_39312 = redout_tmp_40369;
            }
            res_33706 = redout_39312;
            
            float dst_33740 = res_33706;
            bool cond_33741 = res_33706 < nn_dst_33701;
            int32_t loopres_33742;
            
            if (cond_33741) {
                loopres_33742 = indB_33702;
            } else {
                loopres_33742 = nn_ind_33700;
            }
            
            float loopres_33743;
            
            if (cond_33741) {
                loopres_33743 = dst_33740;
            } else {
                loopres_33743 = nn_dst_33701;
            }
            
            int32_t nn_ind_tmp_40367 = loopres_33742;
            float nn_dst_tmp_40368 = loopres_33743;
            
            nn_ind_33700 = nn_ind_tmp_40367;
            nn_dst_33701 = nn_dst_tmp_40368;
        }
        res_33697 = nn_ind_33700;
        res_33698 = nn_dst_33701;
        ((__global float *) mem_39519)[sext_i32_i64(gtid_33131)] = res_33698;
        ((__global int32_t *) mem_39522)[sext_i32_i64(gtid_33131)] = res_33697;
        ((__global float *) mem_39525)[sext_i32_i64(gtid_33131)] = res_33698;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_33688
}
__kernel void selectBestNNzisegmap_33234(__global int *global_failure,
                                         int failure_is_an_option, __global
                                         int *global_failure_args,
                                         int32_t n_30870, int32_t h1_30872,
                                         int32_t w1_30873, int32_t c_30874,
                                         int32_t p_30878, int32_t n_colsA_30887,
                                         int32_t patch_len_30891, __global
                                         unsigned char *imgA_mem_39419, __global
                                         unsigned char *mem_39505)
{
    #define segmap_group_sizze_33655 (selectBestNNzisegmap_group_sizze_33239)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40357;
    int32_t local_tid_40358;
    int32_t group_sizze_40361;
    int32_t wave_sizze_40360;
    int32_t group_tid_40359;
    
    global_tid_40357 = get_global_id(0);
    local_tid_40358 = get_local_id(0);
    group_sizze_40361 = get_local_size(0);
    wave_sizze_40360 = LOCKSTEP_WIDTH;
    group_tid_40359 = get_group_id(0);
    
    int32_t phys_tid_33234;
    
    phys_tid_33234 = global_tid_40357;
    
    int32_t gtid_33232;
    
    gtid_33232 = sext_i64_i32(squot64(sext_i32_i64(group_tid_40359) *
                                      sext_i32_i64(segmap_group_sizze_33655) +
                                      sext_i32_i64(local_tid_40358),
                                      sext_i32_i64(patch_len_30891)));
    
    int32_t gtid_33233;
    
    gtid_33233 = sext_i64_i32(sext_i32_i64(group_tid_40359) *
        sext_i32_i64(segmap_group_sizze_33655) + sext_i32_i64(local_tid_40358) -
        squot64(sext_i32_i64(group_tid_40359) *
                sext_i32_i64(segmap_group_sizze_33655) +
                sext_i32_i64(local_tid_40358), sext_i32_i64(patch_len_30891)) *
        sext_i32_i64(patch_len_30891));
    if (slt32(gtid_33232, n_30870) && slt32(gtid_33233, patch_len_30891)) {
        int32_t index_primexp_39311 = sdiv32(gtid_33232, n_colsA_30887);
        int32_t binop_y_39307 = mul32(n_colsA_30887, index_primexp_39311);
        int32_t index_primexp_39308 = sub32(gtid_33232, binop_y_39307);
        int32_t ij_33663 = sdiv32(gtid_33233, c_30874);
        int32_t y_33664 = mul32(c_30874, ij_33663);
        int32_t k_33665 = sub32(gtid_33233, y_33664);
        int32_t i_33666 = sdiv32(ij_33663, p_30878);
        int32_t y_33667 = mul32(p_30878, i_33666);
        int32_t j_33668 = sub32(ij_33663, y_33667);
        int32_t i_33669 = add32(i_33666, index_primexp_39311);
        bool x_33670 = sle32(0, i_33669);
        bool y_33671 = slt32(i_33669, h1_30872);
        bool bounds_check_33672 = x_33670 && y_33671;
        int32_t i_33673 = add32(j_33668, index_primexp_39308);
        bool x_33674 = sle32(0, i_33673);
        bool y_33675 = slt32(i_33673, w1_30873);
        bool bounds_check_33676 = x_33674 && y_33675;
        bool x_33677 = sle32(0, k_33665);
        bool y_33678 = slt32(k_33665, c_30874);
        bool bounds_check_33679 = x_33677 && y_33678;
        bool y_33680 = bounds_check_33672 && bounds_check_33679;
        bool index_ok_33681 = bounds_check_33676 && y_33680;
        bool index_certs_33682;
        
        if (!index_ok_33681) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 83) == -1) {
                    global_failure_args[0] = i_33669;
                    global_failure_args[1] = i_33673;
                    global_failure_args[2] = k_33665;
                    global_failure_args[3] = h1_30872;
                    global_failure_args[4] = w1_30873;
                    global_failure_args[5] = c_30874;
                    ;
                }
                return;
            }
        }
        
        int32_t i32_arg_33683 = ((__global
                                  int32_t *) imgA_mem_39419)[sext_i32_i64(i_33669) *
                                                             sext_i32_i64(c_30874 *
                                                             w1_30873) +
                                                             sext_i32_i64(i_33673) *
                                                             sext_i32_i64(c_30874) +
                                                             sext_i32_i64(k_33665)];
        float res_33684 = sitofp_i32_f32(i32_arg_33683);
        
        ((__global float *) mem_39505)[sext_i32_i64(gtid_33232) *
                                       sext_i32_i64(patch_len_30891) +
                                       sext_i32_i64(gtid_33233)] = res_33684;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_33655
}
__kernel void selectBestNNzisegmap_intragroup_32874(__global
                                                    int *global_failure,
                                                    int failure_is_an_option,
                                                    __global
                                                    int *global_failure_args,
                                                    uint red_arr_mem_40353_backing_offset_0,
                                                    uint mem_39489_backing_offset_1,
                                                    int32_t n_30870,
                                                    int32_t h1_30872,
                                                    int32_t w1_30873,
                                                    int32_t c_30874,
                                                    int32_t h2_30875,
                                                    int32_t w2_30876,
                                                    int32_t c_30877,
                                                    int32_t p_30878,
                                                    int32_t n_colsA_30887,
                                                    int32_t n_colsB_30889,
                                                    int32_t patch_len_30891,
                                                    __global
                                                    unsigned char *imgA_mem_39419,
                                                    __global
                                                    unsigned char *imgB_mem_39420,
                                                    __global
                                                    unsigned char *mem_39484,
                                                    __global
                                                    unsigned char *mem_39493,
                                                    __global
                                                    unsigned char *mem_39496,
                                                    __global
                                                    unsigned char *mem_39499)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40353_backing_1 =
                  &shared_mem[red_arr_mem_40353_backing_offset_0];
    volatile char *mem_39489_backing_0 =
                  &shared_mem[mem_39489_backing_offset_1];
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40345;
    int32_t local_tid_40346;
    int32_t group_sizze_40349;
    int32_t wave_sizze_40348;
    int32_t group_tid_40347;
    
    global_tid_40345 = get_global_id(0);
    local_tid_40346 = get_local_id(0);
    group_sizze_40349 = get_local_size(0);
    wave_sizze_40348 = LOCKSTEP_WIDTH;
    group_tid_40347 = get_group_id(0);
    
    int32_t phys_tid_32874;
    
    phys_tid_32874 = group_tid_40347;
    
    int32_t ltid_pre_40350;
    
    ltid_pre_40350 = local_tid_40346;
    
    int32_t gtid_32844;
    
    gtid_32844 = group_tid_40347;
    
    int32_t y_33554;
    
    y_33554 = sdiv32(gtid_32844, n_colsA_30887);
    
    int32_t y_33555 = mul32(n_colsA_30887, y_33554);
    int32_t x_33556 = sub32(gtid_32844, y_33555);
    __local char *mem_39489;
    
    mem_39489 = (__local char *) mem_39489_backing_0;
    
    int32_t gtid_32847 = ltid_pre_40350;
    int32_t phys_tid_32848 = local_tid_40346;
    
    if (slt32(gtid_32847, patch_len_30891)) {
        int32_t ij_33559 = sdiv32(gtid_32847, c_30874);
        int32_t y_33560 = mul32(c_30874, ij_33559);
        int32_t k_33561 = sub32(gtid_32847, y_33560);
        int32_t i_33562 = sdiv32(ij_33559, p_30878);
        int32_t y_33563 = mul32(p_30878, i_33562);
        int32_t j_33564 = sub32(ij_33559, y_33563);
        int32_t i_33565 = add32(y_33554, i_33562);
        bool x_33566 = sle32(0, i_33565);
        bool y_33567 = slt32(i_33565, h1_30872);
        bool bounds_check_33568 = x_33566 && y_33567;
        int32_t i_33569 = add32(x_33556, j_33564);
        bool x_33570 = sle32(0, i_33569);
        bool y_33571 = slt32(i_33569, w1_30873);
        bool bounds_check_33572 = x_33570 && y_33571;
        bool x_33573 = sle32(0, k_33561);
        bool y_33574 = slt32(k_33561, c_30874);
        bool bounds_check_33575 = x_33573 && y_33574;
        bool y_33576 = bounds_check_33568 && bounds_check_33575;
        bool index_ok_33577 = bounds_check_33572 && y_33576;
        bool index_certs_33578;
        
        if (!index_ok_33577) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 81) == -1) {
                    global_failure_args[0] = i_33565;
                    global_failure_args[1] = i_33569;
                    global_failure_args[2] = k_33561;
                    global_failure_args[3] = h1_30872;
                    global_failure_args[4] = w1_30873;
                    global_failure_args[5] = c_30874;
                    ;
                }
                local_failure = true;
                goto error_0;
            }
        }
        
        int32_t i32_arg_33579 = ((__global
                                  int32_t *) imgA_mem_39419)[sext_i32_i64(i_33565) *
                                                             sext_i32_i64(c_30874 *
                                                             w1_30873) +
                                                             sext_i32_i64(i_33569) *
                                                             sext_i32_i64(c_30874) +
                                                             sext_i32_i64(k_33561)];
        float res_33580 = sitofp_i32_f32(i32_arg_33579);
        
        ((__local float *) mem_39489)[sext_i32_i64(gtid_32847)] = res_33580;
    }
    
  error_0:
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_failure)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t res_33581;
    float res_33582;
    int32_t nn_ind_33584;
    float nn_dst_33585;
    
    nn_ind_33584 = -1;
    nn_dst_33585 = INFINITY;
    for (int32_t q_33583 = 0; q_33583 < 8; q_33583++) {
        int32_t indB_33586 = ((__global
                               int32_t *) mem_39484)[sext_i32_i64(q_33583) *
                                                     sext_i32_i64(n_30870) +
                                                     sext_i32_i64(gtid_32844)];
        int32_t ii_33587 = sdiv32(indB_33586, n_colsB_30889);
        int32_t y_33588 = mul32(n_colsB_30889, ii_33587);
        int32_t jj_33589 = sub32(indB_33586, y_33588);
        float res_33590;
        int32_t gtid_32872 = ltid_pre_40350;
        int32_t phys_tid_32873 = local_tid_40346;
        __local char *red_arr_mem_40353;
        
        red_arr_mem_40353 = (__local char *) red_arr_mem_40353_backing_1;
        if (slt32(gtid_32872, patch_len_30891)) {
            int32_t ij_33595 = sdiv32(gtid_32872, c_30874);
            int32_t y_33596 = mul32(c_30874, ij_33595);
            int32_t k_33597 = sub32(gtid_32872, y_33596);
            int32_t i_33598 = sdiv32(ij_33595, p_30878);
            int32_t y_33599 = mul32(p_30878, i_33598);
            int32_t j_33600 = sub32(ij_33595, y_33599);
            int32_t i_33601 = add32(ii_33587, i_33598);
            bool x_33602 = sle32(0, i_33601);
            bool y_33603 = slt32(i_33601, h2_30875);
            bool bounds_check_33604 = x_33602 && y_33603;
            int32_t i_33605 = add32(jj_33589, j_33600);
            bool x_33606 = sle32(0, i_33605);
            bool y_33607 = slt32(i_33605, w2_30876);
            bool bounds_check_33608 = x_33606 && y_33607;
            bool x_33609 = sle32(0, k_33597);
            bool y_33610 = slt32(k_33597, c_30874);
            bool bounds_check_33611 = x_33609 && y_33610;
            bool y_33612 = bounds_check_33604 && bounds_check_33611;
            bool index_ok_33613 = bounds_check_33608 && y_33612;
            bool index_certs_33614;
            
            if (!index_ok_33613) {
                {
                    if (atomic_cmpxchg_i32_global(global_failure, -1, 82) ==
                        -1) {
                        global_failure_args[0] = i_33601;
                        global_failure_args[1] = i_33605;
                        global_failure_args[2] = k_33597;
                        global_failure_args[3] = h2_30875;
                        global_failure_args[4] = w2_30876;
                        global_failure_args[5] = c_30874;
                        ;
                    }
                    local_failure = true;
                    goto error_1;
                }
            }
            
            int32_t i32_arg_33615 = ((__global
                                      int32_t *) imgB_mem_39420)[sext_i32_i64(i_33601) *
                                                                 sext_i32_i64(c_30877 *
                                                                 w2_30876) +
                                                                 sext_i32_i64(i_33605) *
                                                                 sext_i32_i64(c_30877) +
                                                                 sext_i32_i64(k_33597)];
            float res_33616 = sitofp_i32_f32(i32_arg_33615);
            float a_v_33621 = ((__local
                                float *) mem_39489)[sext_i32_i64(gtid_32872)];
            float d_33622 = res_33616 - a_v_33621;
            float res_33623 = d_33622 * d_33622;
            
            ((__local float *) red_arr_mem_40353)[sext_i32_i64(gtid_32872)] =
                res_33623;
        }
        
      error_1:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_40355;
        int32_t skip_waves_40356;
        float x_33591;
        float x_33592;
        
        offset_40355 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_40346, patch_len_30891)) {
                x_33591 = ((__local
                            float *) red_arr_mem_40353)[sext_i32_i64(local_tid_40346 +
                                                        offset_40355)];
            }
        }
        offset_40355 = 1;
        while (slt32(offset_40355, wave_sizze_40348)) {
            if (slt32(local_tid_40346 + offset_40355, patch_len_30891) &&
                ((local_tid_40346 - squot32(local_tid_40346, wave_sizze_40348) *
                  wave_sizze_40348) & (2 * offset_40355 - 1)) == 0) {
                // read array element
                {
                    x_33592 = ((volatile __local
                                float *) red_arr_mem_40353)[sext_i32_i64(local_tid_40346 +
                                                            offset_40355)];
                }
                // apply reduction operation
                {
                    float res_33593 = x_33591 + x_33592;
                    
                    x_33591 = res_33593;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_40353)[sext_i32_i64(local_tid_40346)] =
                        x_33591;
                }
            }
            offset_40355 *= 2;
        }
        skip_waves_40356 = 1;
        while (slt32(skip_waves_40356, squot32(patch_len_30891 +
                                               wave_sizze_40348 - 1,
                                               wave_sizze_40348))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_40355 = skip_waves_40356 * wave_sizze_40348;
            if (slt32(local_tid_40346 + offset_40355, patch_len_30891) &&
                ((local_tid_40346 - squot32(local_tid_40346, wave_sizze_40348) *
                  wave_sizze_40348) == 0 && (squot32(local_tid_40346,
                                                     wave_sizze_40348) & (2 *
                                                                          skip_waves_40356 -
                                                                          1)) ==
                 0)) {
                // read array element
                {
                    x_33592 = ((__local
                                float *) red_arr_mem_40353)[sext_i32_i64(local_tid_40346 +
                                                            offset_40355)];
                }
                // apply reduction operation
                {
                    float res_33593 = x_33591 + x_33592;
                    
                    x_33591 = res_33593;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_40353)[sext_i32_i64(local_tid_40346)] =
                        x_33591;
                }
            }
            skip_waves_40356 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        res_33590 = ((__local float *) red_arr_mem_40353)[0];
        
        float dst_33624 = res_33590;
        bool cond_33625 = res_33590 < nn_dst_33585;
        int32_t loopres_33626;
        
        if (cond_33625) {
            loopres_33626 = indB_33586;
        } else {
            loopres_33626 = nn_ind_33584;
        }
        
        float loopres_33627;
        
        if (cond_33625) {
            loopres_33627 = dst_33624;
        } else {
            loopres_33627 = nn_dst_33585;
        }
        
        int32_t nn_ind_tmp_40351 = loopres_33626;
        float nn_dst_tmp_40352 = loopres_33627;
        
        nn_ind_33584 = nn_ind_tmp_40351;
        nn_dst_33585 = nn_dst_tmp_40352;
    }
    res_33581 = nn_ind_33584;
    res_33582 = nn_dst_33585;
    if (local_tid_40346 == 0) {
        ((__global float *) mem_39493)[sext_i32_i64(gtid_32844)] = res_33582;
    }
    if (local_tid_40346 == 0) {
        ((__global int32_t *) mem_39496)[sext_i32_i64(gtid_32844)] = res_33581;
    }
    if (local_tid_40346 == 0) {
        ((__global float *) mem_39499)[sext_i32_i64(gtid_32844)] = res_33582;
    }
    
  error_3:
    return;
}
__kernel void selectBestNNzisegred_nonseg_32749(__global int *global_failure,
                                                int failure_is_an_option,
                                                __global
                                                int *global_failure_args,
                                                uint red_arr_mem_40314_backing_offset_0,
                                                uint sync_arr_mem_40312_backing_offset_1,
                                                int32_t n_30870,
                                                int32_t h1_30872,
                                                int32_t w1_30873,
                                                int32_t c_30874,
                                                int32_t h2_30875,
                                                int32_t w2_30876,
                                                int32_t c_30877,
                                                int32_t p_30878,
                                                int32_t n_colsA_30887,
                                                int32_t n_colsB_30889,
                                                int32_t patch_len_30891,
                                                int32_t num_groups_32753,
                                                __global
                                                unsigned char *imgA_mem_39419,
                                                __global
                                                unsigned char *imgB_mem_39420,
                                                __global
                                                unsigned char *mem_39425,
                                                __global
                                                unsigned char *mem_39429,
                                                __global
                                                unsigned char *mem_39443,
                                                __global
                                                unsigned char *mem_39446,
                                                __global
                                                unsigned char *mem_39449,
                                                __global
                                                unsigned char *selectBestNNzicounter_mem_40302,
                                                __global
                                                unsigned char *group_res_arr_mem_40304,
                                                int32_t num_threads_40306)
{
    #define segred_group_sizze_32752 (selectBestNNzisegred_group_sizze_32739)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40314_backing_1 =
                  &shared_mem[red_arr_mem_40314_backing_offset_0];
    volatile char *sync_arr_mem_40312_backing_0 =
                  &shared_mem[sync_arr_mem_40312_backing_offset_1];
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_40307;
    int32_t local_tid_40308;
    int32_t group_sizze_40311;
    int32_t wave_sizze_40310;
    int32_t group_tid_40309;
    
    global_tid_40307 = get_global_id(0);
    local_tid_40308 = get_local_id(0);
    group_sizze_40311 = get_local_size(0);
    wave_sizze_40310 = LOCKSTEP_WIDTH;
    group_tid_40309 = get_group_id(0);
    
    int32_t phys_tid_32749;
    
    phys_tid_32749 = global_tid_40307;
    
    __local char *sync_arr_mem_40312;
    
    sync_arr_mem_40312 = (__local char *) sync_arr_mem_40312_backing_0;
    
    __local char *red_arr_mem_40314;
    
    red_arr_mem_40314 = (__local char *) red_arr_mem_40314_backing_1;
    
    int32_t dummy_32747;
    
    dummy_32747 = 0;
    
    int32_t gtid_32748;
    
    gtid_32748 = 0;
    
    float x_acc_40316;
    int32_t chunk_sizze_40317;
    
    chunk_sizze_40317 = smin32(sdiv_up32(n_30870, segred_group_sizze_32752 *
                                         num_groups_32753), sdiv_up32(n_30870 -
                                                                      phys_tid_32749,
                                                                      num_threads_40306));
    
    float x_32758;
    float x_32759;
    
    // neutral-initialise the accumulators
    {
        x_acc_40316 = 0.0F;
    }
    for (int32_t i_40321 = 0; i_40321 < chunk_sizze_40317; i_40321++) {
        gtid_32748 = phys_tid_32749 + num_threads_40306 * i_40321;
        // apply map function
        {
            int32_t y_32763 = sdiv32(gtid_32748, n_colsA_30887);
            int32_t y_32764 = mul32(n_colsA_30887, y_32763);
            int32_t x_32765 = sub32(gtid_32748, y_32764);
            
            for (int32_t i_39355 = 0; i_39355 < patch_len_30891; i_39355++) {
                int32_t ij_32768 = sdiv32(i_39355, c_30874);
                int32_t y_32769 = mul32(c_30874, ij_32768);
                int32_t k_32770 = sub32(i_39355, y_32769);
                int32_t i_32771 = sdiv32(ij_32768, p_30878);
                int32_t y_32772 = mul32(p_30878, i_32771);
                int32_t j_32773 = sub32(ij_32768, y_32772);
                int32_t i_32774 = add32(y_32763, i_32771);
                bool x_32775 = sle32(0, i_32774);
                bool y_32776 = slt32(i_32774, h1_30872);
                bool bounds_check_32777 = x_32775 && y_32776;
                int32_t i_32778 = add32(x_32765, j_32773);
                bool x_32779 = sle32(0, i_32778);
                bool y_32780 = slt32(i_32778, w1_30873);
                bool bounds_check_32781 = x_32779 && y_32780;
                bool x_32782 = sle32(0, k_32770);
                bool y_32783 = slt32(k_32770, c_30874);
                bool bounds_check_32784 = x_32782 && y_32783;
                bool y_32785 = bounds_check_32777 && bounds_check_32784;
                bool index_ok_32786 = bounds_check_32781 && y_32785;
                bool index_certs_32787;
                
                if (!index_ok_32786) {
                    {
                        if (atomic_cmpxchg_i32_global(global_failure, -1, 77) ==
                            -1) {
                            global_failure_args[0] = i_32774;
                            global_failure_args[1] = i_32778;
                            global_failure_args[2] = k_32770;
                            global_failure_args[3] = h1_30872;
                            global_failure_args[4] = w1_30873;
                            global_failure_args[5] = c_30874;
                            ;
                        }
                        local_failure = true;
                        goto error_0;
                    }
                }
                
                int32_t i32_arg_32788 = ((__global
                                          int32_t *) imgA_mem_39419)[sext_i32_i64(i_32774) *
                                                                     sext_i32_i64(c_30874 *
                                                                     w1_30873) +
                                                                     sext_i32_i64(i_32778) *
                                                                     sext_i32_i64(c_30874) +
                                                                     sext_i32_i64(k_32770)];
                float res_32789 = sitofp_i32_f32(i32_arg_32788);
                
                ((__global float *) mem_39429)[sext_i32_i64(phys_tid_32749) +
                                               sext_i32_i64(i_39355) *
                                               sext_i32_i64(num_groups_32753 *
                                               segred_group_sizze_32752)] =
                    res_32789;
            }
            
            int32_t res_32790;
            float res_32791;
            int32_t nn_ind_32793;
            float nn_dst_32794;
            
            nn_ind_32793 = -1;
            nn_dst_32794 = INFINITY;
            for (int32_t q_32792 = 0; q_32792 < 8; q_32792++) {
                int32_t indB_32795 = ((__global
                                       int32_t *) mem_39425)[sext_i32_i64(q_32792) *
                                                             sext_i32_i64(n_30870) +
                                                             sext_i32_i64(gtid_32748)];
                int32_t ii_32796 = sdiv32(indB_32795, n_colsB_30889);
                int32_t y_32797 = mul32(n_colsB_30889, ii_32796);
                int32_t jj_32798 = sub32(indB_32795, y_32797);
                float res_32799;
                float redout_39284 = 0.0F;
                
                for (int32_t i_39285 = 0; i_39285 < patch_len_30891;
                     i_39285++) {
                    int32_t ij_32804 = sdiv32(i_39285, c_30874);
                    int32_t y_32805 = mul32(c_30874, ij_32804);
                    int32_t k_32806 = sub32(i_39285, y_32805);
                    int32_t i_32807 = sdiv32(ij_32804, p_30878);
                    int32_t y_32808 = mul32(p_30878, i_32807);
                    int32_t j_32809 = sub32(ij_32804, y_32808);
                    int32_t i_32810 = add32(ii_32796, i_32807);
                    bool x_32811 = sle32(0, i_32810);
                    bool y_32812 = slt32(i_32810, h2_30875);
                    bool bounds_check_32813 = x_32811 && y_32812;
                    int32_t i_32814 = add32(jj_32798, j_32809);
                    bool x_32815 = sle32(0, i_32814);
                    bool y_32816 = slt32(i_32814, w2_30876);
                    bool bounds_check_32817 = x_32815 && y_32816;
                    bool x_32818 = sle32(0, k_32806);
                    bool y_32819 = slt32(k_32806, c_30874);
                    bool bounds_check_32820 = x_32818 && y_32819;
                    bool y_32821 = bounds_check_32813 && bounds_check_32820;
                    bool index_ok_32822 = bounds_check_32817 && y_32821;
                    bool index_certs_32823;
                    
                    if (!index_ok_32822) {
                        {
                            if (atomic_cmpxchg_i32_global(global_failure, -1,
                                                          78) == -1) {
                                global_failure_args[0] = i_32810;
                                global_failure_args[1] = i_32814;
                                global_failure_args[2] = k_32806;
                                global_failure_args[3] = h2_30875;
                                global_failure_args[4] = w2_30876;
                                global_failure_args[5] = c_30874;
                                ;
                            }
                            local_failure = true;
                            goto error_0;
                        }
                    }
                    
                    int32_t i32_arg_32824 = ((__global
                                              int32_t *) imgB_mem_39420)[sext_i32_i64(i_32810) *
                                                                         sext_i32_i64(c_30877 *
                                                                         w2_30876) +
                                                                         sext_i32_i64(i_32814) *
                                                                         sext_i32_i64(c_30877) +
                                                                         sext_i32_i64(k_32806)];
                    float res_32825 = sitofp_i32_f32(i32_arg_32824);
                    float a_v_32830 = ((__global
                                        float *) mem_39429)[sext_i32_i64(phys_tid_32749) +
                                                            sext_i32_i64(i_39285) *
                                                            sext_i32_i64(num_groups_32753 *
                                                            segred_group_sizze_32752)];
                    float d_32831 = res_32825 - a_v_32830;
                    float res_32832 = d_32831 * d_32831;
                    float res_32802 = res_32832 + redout_39284;
                    float redout_tmp_40325 = res_32802;
                    
                    redout_39284 = redout_tmp_40325;
                }
                res_32799 = redout_39284;
                
                float dst_32833 = res_32799;
                bool cond_32834 = res_32799 < nn_dst_32794;
                int32_t loopres_32835;
                
                if (cond_32834) {
                    loopres_32835 = indB_32795;
                } else {
                    loopres_32835 = nn_ind_32793;
                }
                
                float loopres_32836;
                
                if (cond_32834) {
                    loopres_32836 = dst_32833;
                } else {
                    loopres_32836 = nn_dst_32794;
                }
                
                int32_t nn_ind_tmp_40323 = loopres_32835;
                float nn_dst_tmp_40324 = loopres_32836;
                
                nn_ind_32793 = nn_ind_tmp_40323;
                nn_dst_32794 = nn_dst_tmp_40324;
            }
            res_32790 = nn_ind_32793;
            res_32791 = nn_dst_32794;
            // save map-out results
            {
                ((__global int32_t *) mem_39446)[sext_i32_i64(dummy_32747) *
                                                 sext_i32_i64(n_30870) +
                                                 sext_i32_i64(gtid_32748)] =
                    res_32790;
                ((__global float *) mem_39449)[sext_i32_i64(dummy_32747) *
                                               sext_i32_i64(n_30870) +
                                               sext_i32_i64(gtid_32748)] =
                    res_32791;
            }
            // load accumulator
            {
                x_32758 = x_acc_40316;
            }
            // load new values
            {
                x_32759 = res_32791;
            }
            // apply reduction operator
            {
                float res_32760 = x_32758 + x_32759;
                
                // store in accumulator
                {
                    x_acc_40316 = res_32760;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_32758 = x_acc_40316;
        ((__local float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308)] =
            x_32758;
    }
    
  error_0:
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_failure)
        return;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_40326;
    int32_t skip_waves_40327;
    float x_40318;
    float x_40319;
    
    offset_40326 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_40308, segred_group_sizze_32752)) {
            x_40318 = ((__local
                        float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308 +
                                                    offset_40326)];
        }
    }
    offset_40326 = 1;
    while (slt32(offset_40326, wave_sizze_40310)) {
        if (slt32(local_tid_40308 + offset_40326, segred_group_sizze_32752) &&
            ((local_tid_40308 - squot32(local_tid_40308, wave_sizze_40310) *
              wave_sizze_40310) & (2 * offset_40326 - 1)) == 0) {
            // read array element
            {
                x_40319 = ((volatile __local
                            float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308 +
                                                        offset_40326)];
            }
            // apply reduction operation
            {
                float res_40320 = x_40318 + x_40319;
                
                x_40318 = res_40320;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308)] =
                    x_40318;
            }
        }
        offset_40326 *= 2;
    }
    skip_waves_40327 = 1;
    while (slt32(skip_waves_40327, squot32(segred_group_sizze_32752 +
                                           wave_sizze_40310 - 1,
                                           wave_sizze_40310))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_40326 = skip_waves_40327 * wave_sizze_40310;
        if (slt32(local_tid_40308 + offset_40326, segred_group_sizze_32752) &&
            ((local_tid_40308 - squot32(local_tid_40308, wave_sizze_40310) *
              wave_sizze_40310) == 0 && (squot32(local_tid_40308,
                                                 wave_sizze_40310) & (2 *
                                                                      skip_waves_40327 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_40319 = ((__local
                            float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308 +
                                                        offset_40326)];
            }
            // apply reduction operation
            {
                float res_40320 = x_40318 + x_40319;
                
                x_40318 = res_40320;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308)] =
                    x_40318;
            }
        }
        skip_waves_40327 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (local_tid_40308 == 0) {
            x_acc_40316 = x_40318;
        }
    }
    
    int32_t old_counter_40328;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_40308 == 0) {
            ((__global
              float *) group_res_arr_mem_40304)[sext_i32_i64(group_tid_40309) *
                                                sext_i32_i64(segred_group_sizze_32752)] =
                x_acc_40316;
            mem_fence_global();
            old_counter_40328 = atomic_add_i32_global(&((volatile __global
                                                         int *) selectBestNNzicounter_mem_40302)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_40312)[0] = old_counter_40328 ==
                num_groups_32753 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_40329;
    
    is_last_group_40329 = ((__local bool *) sync_arr_mem_40312)[0];
    if (is_last_group_40329) {
        if (local_tid_40308 == 0) {
            old_counter_40328 = atomic_add_i32_global(&((volatile __global
                                                         int *) selectBestNNzicounter_mem_40302)[0],
                                                      (int) (0 -
                                                             num_groups_32753));
        }
        // read in the per-group-results
        {
            int32_t read_per_thread_40330 = sdiv_up32(num_groups_32753,
                                                      segred_group_sizze_32752);
            
            x_32758 = 0.0F;
            for (int32_t i_40331 = 0; i_40331 < read_per_thread_40330;
                 i_40331++) {
                int32_t group_res_id_40332 = local_tid_40308 *
                        read_per_thread_40330 + i_40331;
                int32_t index_of_group_res_40333 = group_res_id_40332;
                
                if (slt32(group_res_id_40332, num_groups_32753)) {
                    x_32759 = ((__global
                                float *) group_res_arr_mem_40304)[sext_i32_i64(index_of_group_res_40333) *
                                                                  sext_i32_i64(segred_group_sizze_32752)];
                    
                    float res_32760;
                    
                    res_32760 = x_32758 + x_32759;
                    x_32758 = res_32760;
                }
            }
        }
        ((__local float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308)] =
            x_32758;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_40334;
            int32_t skip_waves_40335;
            float x_40318;
            float x_40319;
            
            offset_40334 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_40308, segred_group_sizze_32752)) {
                    x_40318 = ((__local
                                float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308 +
                                                            offset_40334)];
                }
            }
            offset_40334 = 1;
            while (slt32(offset_40334, wave_sizze_40310)) {
                if (slt32(local_tid_40308 + offset_40334,
                          segred_group_sizze_32752) && ((local_tid_40308 -
                                                         squot32(local_tid_40308,
                                                                 wave_sizze_40310) *
                                                         wave_sizze_40310) &
                                                        (2 * offset_40334 -
                                                         1)) == 0) {
                    // read array element
                    {
                        x_40319 = ((volatile __local
                                    float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308 +
                                                                offset_40334)];
                    }
                    // apply reduction operation
                    {
                        float res_40320 = x_40318 + x_40319;
                        
                        x_40318 = res_40320;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308)] =
                            x_40318;
                    }
                }
                offset_40334 *= 2;
            }
            skip_waves_40335 = 1;
            while (slt32(skip_waves_40335, squot32(segred_group_sizze_32752 +
                                                   wave_sizze_40310 - 1,
                                                   wave_sizze_40310))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_40334 = skip_waves_40335 * wave_sizze_40310;
                if (slt32(local_tid_40308 + offset_40334,
                          segred_group_sizze_32752) && ((local_tid_40308 -
                                                         squot32(local_tid_40308,
                                                                 wave_sizze_40310) *
                                                         wave_sizze_40310) ==
                                                        0 &&
                                                        (squot32(local_tid_40308,
                                                                 wave_sizze_40310) &
                                                         (2 * skip_waves_40335 -
                                                          1)) == 0)) {
                    // read array element
                    {
                        x_40319 = ((__local
                                    float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308 +
                                                                offset_40334)];
                    }
                    // apply reduction operation
                    {
                        float res_40320 = x_40318 + x_40319;
                        
                        x_40318 = res_40320;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_40314)[sext_i32_i64(local_tid_40308)] =
                            x_40318;
                    }
                }
                skip_waves_40335 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_40308 == 0) {
                    ((__global float *) mem_39443)[0] = x_40318;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_32752
}
__kernel void selectBestNNzisegred_nonseg_33448(__global int *global_failure,
                                                uint red_arr_mem_40384_backing_offset_0,
                                                uint sync_arr_mem_40382_backing_offset_1,
                                                int32_t n_30870,
                                                int32_t num_groups_33750,
                                                __global
                                                unsigned char *res_map_acc_mem_39529,
                                                __global
                                                unsigned char *mem_39534,
                                                __global
                                                unsigned char *selectBestNNzicounter_mem_40372,
                                                __global
                                                unsigned char *group_res_arr_mem_40374,
                                                int32_t num_threads_40376)
{
    #define segred_group_sizze_33749 (selectBestNNzisegred_group_sizze_33440)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    volatile char *red_arr_mem_40384_backing_1 =
                  &shared_mem[red_arr_mem_40384_backing_offset_0];
    volatile char *sync_arr_mem_40382_backing_0 =
                  &shared_mem[sync_arr_mem_40382_backing_offset_1];
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_40377;
    int32_t local_tid_40378;
    int32_t group_sizze_40381;
    int32_t wave_sizze_40380;
    int32_t group_tid_40379;
    
    global_tid_40377 = get_global_id(0);
    local_tid_40378 = get_local_id(0);
    group_sizze_40381 = get_local_size(0);
    wave_sizze_40380 = LOCKSTEP_WIDTH;
    group_tid_40379 = get_group_id(0);
    
    int32_t phys_tid_33448;
    
    phys_tid_33448 = global_tid_40377;
    
    __local char *sync_arr_mem_40382;
    
    sync_arr_mem_40382 = (__local char *) sync_arr_mem_40382_backing_0;
    
    __local char *red_arr_mem_40384;
    
    red_arr_mem_40384 = (__local char *) red_arr_mem_40384_backing_1;
    
    int32_t dummy_33446;
    
    dummy_33446 = 0;
    
    int32_t gtid_33447;
    
    gtid_33447 = 0;
    
    float x_acc_40386;
    int32_t chunk_sizze_40387;
    
    chunk_sizze_40387 = smin32(sdiv_up32(n_30870, segred_group_sizze_33749 *
                                         num_groups_33750), sdiv_up32(n_30870 -
                                                                      phys_tid_33448,
                                                                      num_threads_40376));
    
    float x_33753;
    float x_33754;
    
    // neutral-initialise the accumulators
    {
        x_acc_40386 = 0.0F;
    }
    for (int32_t i_40391 = 0; i_40391 < chunk_sizze_40387; i_40391++) {
        gtid_33447 = phys_tid_33448 + num_threads_40376 * i_40391;
        // apply map function
        {
            float x_33756 = ((__global
                              float *) res_map_acc_mem_39529)[sext_i32_i64(gtid_33447)];
            
            // save map-out results
            { }
            // load accumulator
            {
                x_33753 = x_acc_40386;
            }
            // load new values
            {
                x_33754 = x_33756;
            }
            // apply reduction operator
            {
                float res_33755 = x_33753 + x_33754;
                
                // store in accumulator
                {
                    x_acc_40386 = res_33755;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_33753 = x_acc_40386;
        ((__local float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378)] =
            x_33753;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_40392;
    int32_t skip_waves_40393;
    float x_40388;
    float x_40389;
    
    offset_40392 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_40378, segred_group_sizze_33749)) {
            x_40388 = ((__local
                        float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378 +
                                                    offset_40392)];
        }
    }
    offset_40392 = 1;
    while (slt32(offset_40392, wave_sizze_40380)) {
        if (slt32(local_tid_40378 + offset_40392, segred_group_sizze_33749) &&
            ((local_tid_40378 - squot32(local_tid_40378, wave_sizze_40380) *
              wave_sizze_40380) & (2 * offset_40392 - 1)) == 0) {
            // read array element
            {
                x_40389 = ((volatile __local
                            float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378 +
                                                        offset_40392)];
            }
            // apply reduction operation
            {
                float res_40390 = x_40388 + x_40389;
                
                x_40388 = res_40390;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378)] =
                    x_40388;
            }
        }
        offset_40392 *= 2;
    }
    skip_waves_40393 = 1;
    while (slt32(skip_waves_40393, squot32(segred_group_sizze_33749 +
                                           wave_sizze_40380 - 1,
                                           wave_sizze_40380))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_40392 = skip_waves_40393 * wave_sizze_40380;
        if (slt32(local_tid_40378 + offset_40392, segred_group_sizze_33749) &&
            ((local_tid_40378 - squot32(local_tid_40378, wave_sizze_40380) *
              wave_sizze_40380) == 0 && (squot32(local_tid_40378,
                                                 wave_sizze_40380) & (2 *
                                                                      skip_waves_40393 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_40389 = ((__local
                            float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378 +
                                                        offset_40392)];
            }
            // apply reduction operation
            {
                float res_40390 = x_40388 + x_40389;
                
                x_40388 = res_40390;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378)] =
                    x_40388;
            }
        }
        skip_waves_40393 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (local_tid_40378 == 0) {
            x_acc_40386 = x_40388;
        }
    }
    
    int32_t old_counter_40394;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_40378 == 0) {
            ((__global
              float *) group_res_arr_mem_40374)[sext_i32_i64(group_tid_40379) *
                                                sext_i32_i64(segred_group_sizze_33749)] =
                x_acc_40386;
            mem_fence_global();
            old_counter_40394 = atomic_add_i32_global(&((volatile __global
                                                         int *) selectBestNNzicounter_mem_40372)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_40382)[0] = old_counter_40394 ==
                num_groups_33750 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_40395;
    
    is_last_group_40395 = ((__local bool *) sync_arr_mem_40382)[0];
    if (is_last_group_40395) {
        if (local_tid_40378 == 0) {
            old_counter_40394 = atomic_add_i32_global(&((volatile __global
                                                         int *) selectBestNNzicounter_mem_40372)[0],
                                                      (int) (0 -
                                                             num_groups_33750));
        }
        // read in the per-group-results
        {
            int32_t read_per_thread_40396 = sdiv_up32(num_groups_33750,
                                                      segred_group_sizze_33749);
            
            x_33753 = 0.0F;
            for (int32_t i_40397 = 0; i_40397 < read_per_thread_40396;
                 i_40397++) {
                int32_t group_res_id_40398 = local_tid_40378 *
                        read_per_thread_40396 + i_40397;
                int32_t index_of_group_res_40399 = group_res_id_40398;
                
                if (slt32(group_res_id_40398, num_groups_33750)) {
                    x_33754 = ((__global
                                float *) group_res_arr_mem_40374)[sext_i32_i64(index_of_group_res_40399) *
                                                                  sext_i32_i64(segred_group_sizze_33749)];
                    
                    float res_33755;
                    
                    res_33755 = x_33753 + x_33754;
                    x_33753 = res_33755;
                }
            }
        }
        ((__local float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378)] =
            x_33753;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_40400;
            int32_t skip_waves_40401;
            float x_40388;
            float x_40389;
            
            offset_40400 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_40378, segred_group_sizze_33749)) {
                    x_40388 = ((__local
                                float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378 +
                                                            offset_40400)];
                }
            }
            offset_40400 = 1;
            while (slt32(offset_40400, wave_sizze_40380)) {
                if (slt32(local_tid_40378 + offset_40400,
                          segred_group_sizze_33749) && ((local_tid_40378 -
                                                         squot32(local_tid_40378,
                                                                 wave_sizze_40380) *
                                                         wave_sizze_40380) &
                                                        (2 * offset_40400 -
                                                         1)) == 0) {
                    // read array element
                    {
                        x_40389 = ((volatile __local
                                    float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378 +
                                                                offset_40400)];
                    }
                    // apply reduction operation
                    {
                        float res_40390 = x_40388 + x_40389;
                        
                        x_40388 = res_40390;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378)] =
                            x_40388;
                    }
                }
                offset_40400 *= 2;
            }
            skip_waves_40401 = 1;
            while (slt32(skip_waves_40401, squot32(segred_group_sizze_33749 +
                                                   wave_sizze_40380 - 1,
                                                   wave_sizze_40380))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_40400 = skip_waves_40401 * wave_sizze_40380;
                if (slt32(local_tid_40378 + offset_40400,
                          segred_group_sizze_33749) && ((local_tid_40378 -
                                                         squot32(local_tid_40378,
                                                                 wave_sizze_40380) *
                                                         wave_sizze_40380) ==
                                                        0 &&
                                                        (squot32(local_tid_40378,
                                                                 wave_sizze_40380) &
                                                         (2 * skip_waves_40401 -
                                                          1)) == 0)) {
                    // read array element
                    {
                        x_40389 = ((__local
                                    float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378 +
                                                                offset_40400)];
                    }
                    // apply reduction operation
                    {
                        float res_40390 = x_40388 + x_40389;
                        
                        x_40388 = res_40390;
                    }
                    // write result of operation
                    {
                        ((__local
                          float *) red_arr_mem_40384)[sext_i32_i64(local_tid_40378)] =
                            x_40388;
                    }
                }
                skip_waves_40401 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_40378 == 0) {
                    ((__global float *) mem_39534)[0] = x_40388;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_33749
}
__kernel void selectBestNN_BADzicopy_40308(int32_t n_30845, int32_t kk_30846,
                                           __global
                                           unsigned char *knn_inds_mem_39418,
                                           __global unsigned char *mem_39426)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_40308;
    int32_t copy_ltid_40309;
    int32_t copy_gid_40310;
    
    copy_gtid_40308 = get_global_id(0);
    copy_ltid_40309 = get_local_id(0);
    copy_gid_40310 = get_group_id(0);
    if (slt32(copy_gtid_40308, sext_i64_i32(sext_i32_i64(n_30845)))) {
        ((__global int32_t *) mem_39426)[sext_i32_i64(copy_gtid_40308)] =
            ((__global
              int32_t *) knn_inds_mem_39418)[sext_i32_i64(copy_gtid_40308) *
                                             sext_i32_i64(kk_30846)];
    }
    
  error_0:
    return;
}
