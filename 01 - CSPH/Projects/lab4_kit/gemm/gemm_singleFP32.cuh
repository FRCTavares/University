#define LBLK 32
static inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

/* Find element based on row-major ordering */
#define RM(r, c, width) ((r) * (width) + (c))
/* Find element based on row-major ordering for 3 dimensions */
#define RM3(r, c, d, width) RM(RM(r, c, width), d, width)

/* Find element based on column-major ordering */
#define CM(r, c, height) ((c) * (height) + (r))


const char* cublasComputeTypeToString(cublasComputeType_t computeType);
__global__ void cudaBlockKernel_FP32(int N, float *dmatA, float *dmatB, float *dmatC);