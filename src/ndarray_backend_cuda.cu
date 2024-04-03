#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t convertIdxWithStride(size_t idx, CudaVec shape, CudaVec strides) {
  size_t res = 0;
  for (size_t i = shape.size; i > 0; i--) {
    assert(shape.data[i - 1] > 0 && "Shape data must be greater than 0");
    size_t mod = idx % shape.data[i - 1];
    res += mod * strides.data[i - 1];
    idx /= shape.data[i - 1];
  }
  return res;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    size_t idx = offset + convertIdxWithStride(gid, shape, strides);
    out[gid] = a[idx];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;

  if (gid < size) {
    // assign a[gid] to out[resolved_idx]
    size_t idx = offset + convertIdxWithStride(gid, shape, strides);
    out[idx] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}


__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;

  if (gid < size) {
    size_t idx = offset + convertIdxWithStride(gid, shape, strides);
    out[idx] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


template<typename BinaryOp>
__global__ void EwiseBinaryKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = BinaryOp::apply(a[gid], b[gid]);
  }
}

template<typename BinaryOp>
void EwiseBinaryOp(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(a.size);
  EwiseBinaryKernel<BinaryOp><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

template<typename BinaryOp>
__global__ void ScalarBinaryKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = BinaryOp::apply(a[gid], val);
  }
}

template<typename BinaryOp>
void ScalarBinaryOp(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(a.size);
  ScalarBinaryKernel<BinaryOp><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

template<typename UnaryOp>
__global__ void EwiseUnaryKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = UnaryOp::apply(a[gid]);
  }
}

template<typename BinaryOp>
void EwiseUnaryOp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseUnaryKernel<BinaryOp><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

// Operators
struct MulOp {
  __device__ static inline scalar_t apply(scalar_t a, scalar_t b) {
    return a * b;
  }
};

struct DivOp {
  __device__ static inline scalar_t apply(scalar_t a, scalar_t b) {
    return a / b;
  }
};

struct PowerOp {
  __device__ static inline scalar_t apply(scalar_t a, scalar_t b) {
    return pow(a, b);
  }
};

struct MaximumOp {
  __device__ static inline scalar_t apply(scalar_t a, scalar_t b) {
    return max(a, b);
  }
};

struct EqOp {
  __device__ static inline scalar_t apply(scalar_t a, scalar_t b) {
    return scalar_t(a == b);
  }
};

struct GeOp {
  __device__ static inline scalar_t apply(scalar_t a, scalar_t b) {
    return scalar_t(a >= b);
  }
};

struct LogOp {
  __device__ static inline scalar_t apply(scalar_t a) {
    return log(a);
  }
};

struct ExpOp {
  __device__ static inline scalar_t apply(scalar_t a) {
    return exp(a);
  }
};

struct TanhOp {
  __device__ static inline scalar_t apply(scalar_t a) {
    return tanh(a);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
const size_t L = 64, S = 64, V = 4; // Tune these parameters
// V = 2 would cause a launchOutOfResources error (too many threads)

__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* out, uint32_t M, uint32_t N, uint32_t P) {
  __shared__ scalar_t sA[S][L];
  __shared__ scalar_t sB[S][L];
  size_t x_block = blockIdx.x;
  size_t y_block = blockIdx.y;
  scalar_t a[V];
  scalar_t b[V];
  scalar_t c[V][V] = {0}; // only for this thread

  for (size_t i = 0; i < N; i += S) {
    __syncthreads();
    // thread cooperative fetching into sA and sB here
    // a[y_block * L : y_block * L + L, i : i + S].T -> sA
    // b[i : i + S, x_block * L : x_block * L + L] -> sB
    size_t nthreads = blockDim.x * blockDim.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (size_t j = 0; j < (S * L + nthreads - 1) / nthreads; j++) {
      size_t y = (j * nthreads + tid) / L;
      size_t x = (j * nthreads + tid) % L;
      // might be too hard to get right
      // block might be out of bounds, need to pad with 0
      if (i + y >= N) { // out of bounds access
        sA[y][x] = 0;
        sB[y][x] = 0;
        continue;
      }
      if (y_block * L + x < M) {
        // How to understand this index?
        // sA[y][x] = A[y_block * L : y_block * L + L, i : i + S].T[y, x]
        // = A[y_block * L : y_block * L + L, i : i + S][x, y]
        // = A[y_block * L + x, i + y]
        // With constraint y_block * L + x < M
        sA[y][x] = A[(y_block * L + x) * N + i + y];
      } else {
        sA[y][x] = 0;
      }
      if (x_block * L + x < P) {
        sB[y][x] = B[(i + y) * P + x_block * L + x];
      } else {
        sB[y][x] = 0;
      }
    }
    __syncthreads();
    for (size_t k = 0; k < S; k++) {
      // get kth row from sA and sB
      // sA[k, y * V : y * V + V] -> a
      // sB[k, x * V : x * V + V] -> b
      for (size_t j = 0; j < V; j++) {
        a[j] = sA[k][threadIdx.y * V + j];
      }
      for (size_t j = 0; j < V; j++) {
        b[j] = sB[k][threadIdx.x * V + j];
      }
      // multiply them together
      for (size_t y = 0; y < V; y++) {
        for (size_t x = 0; x < V; x++) {
          c[y][x] += a[y] * b[x];
        }
      }
    }
  }

  size_t y_base = y_block * L + threadIdx.y * V;
  size_t x_base = x_block * L + threadIdx.x * V;
  for (size_t i = y_base; i < min(y_base + V, size_t(M)); i++) {
    for (size_t j = x_base; j < min(x_base + V, size_t(P)); j++) {
      out[i * P + j] = c[i - y_base][j - x_base];
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  size_t by = max((M + L - 1) / L, (P + L - 1) / L);
  size_t bx = (N + S - 1) / S;
  dim3 numBlocks(by, bx);
  dim3 threadsPerBlock(L / V, L / V);
  MatmulKernel<<<numBlocks, threadsPerBlock>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid < out_size) {
    scalar_t max_val = a[gid * reduce_size];
    for (size_t i = 1; i < reduce_size; i++) {
      max_val = max(max_val, a[gid * reduce_size + i]);
    }
    out[gid] = max_val;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid < out_size) {
    scalar_t sum = 0;
    for (size_t i = 0; i < reduce_size; i++) {
      sum += a[gid * reduce_size + i];
    }
    out[gid] = sum;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
   * can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseBinaryOp<MulOp>);
  m.def("scalar_mul", ScalarBinaryOp<MulOp>);
  m.def("ewise_div", EwiseBinaryOp<DivOp>);
  m.def("scalar_div", ScalarBinaryOp<DivOp>);
  m.def("scalar_power", ScalarBinaryOp<PowerOp>);

  m.def("ewise_maximum", EwiseBinaryOp<MaximumOp>);
  m.def("scalar_maximum", ScalarBinaryOp<MaximumOp>);
  m.def("ewise_eq", EwiseBinaryOp<EqOp>);
  m.def("scalar_eq", ScalarBinaryOp<EqOp>);
  m.def("ewise_ge", EwiseBinaryOp<GeOp>);
  m.def("scalar_ge", ScalarBinaryOp<GeOp>);

  m.def("ewise_log", EwiseUnaryOp<LogOp>);
  m.def("ewise_exp", EwiseUnaryOp<ExpOp>);
  m.def("ewise_tanh", EwiseUnaryOp<TanhOp>);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
