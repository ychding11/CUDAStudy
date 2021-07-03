 
#include <iostream>
#include <chrono>  // for high_resolution_clock
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <cassert>
#include "cutil_math.h" // from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h
#include "helper_string.h"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#define CUDA_CALL_CHECK(x)                             \
do{                                                    \
    if((x) != cudaSuccess)                             \
    {                                                  \
        cudaError_t cudaStatus = x;                    \
        printf("Error at %s:%d\t",__FILE__,__LINE__);  \
        printf("%s %d\t",#x, (cudaStatus));            \
        printf("%s\n",cudaGetErrorString(cudaStatus)); \
        system("pause");                               \
        return EXIT_FAILURE;                           \
    }                                                  \
} while(0)


#define M_PI 3.14159265359f  // pi

// __deviceV__ : executed on the device (GPU) and callable only from the device

// random number generator from https://github.com/gz/rust-raytracer
__device__ static float getrandom(unsigned int *seed0, unsigned int *seed1)
{
 *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
 *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

 unsigned int ires = ((*seed0) << 16) + (*seed1);

 // Convert to float
 union
 {
  float f;
  unsigned int ui;
 } res;
 res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

 return (res.f - 2.f) / 2.f;
}


// __global__ : executed on the device (GPU) and called only from host (CPU) 
int TestSmallPTOnGPU(int width, int height, int samps, float vfov = 45.f)
{
    float3* output_h = new float3[width * height]; // pointer to memory for image on the host (system RAM)
    float3* output_d;    // pointer to memory for image on the device (GPU VRAM)

    CUDA_CALL_CHECK( cudaSetDevice(0) );

    // allocate memory on the CUDA device (GPU VRAM)
    CUDA_CALL_CHECK( cudaMalloc(&output_d, width * height * sizeof(float3)) );
        
    dim3 block(8, 8, 1);   
    dim3 grid(width / block.x, height / block.y, 1);

    printf("\nStart rendering... %d, %d, %d, %f\n", width, height, samps, vfov);
 
    // Record start time                          
    auto start = std::chrono::high_resolution_clock::now();

    //< schedule threads on device and launch CUDA kernel from host
    //< use default stream
    CUDA_CALL_CHECK(cudaDeviceSynchronize());

    // Check for any errors launching the kernel
    CUDA_CALL_CHECK(cudaGetLastError());

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    printf("Render Done! Time=%lf seconds\n", elapsed.count());

    // copy result from device back to host
    CUDA_CALL_CHECK(cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
 
    // free CUDA memory
    CUDA_CALL_CHECK( cudaFree(output_d) );  

    //SaveToPPM(output_h, width, height);

    printf("Saved image to 'smallptcuda.ppm'\n");
    delete[] output_h;
    return 0;
}

static void help(const char *name)
{
    printf("Usage:  %s [OPTION]...\n", name);
    printf("A simple path tracer by cuda. \n");
    printf("\n");

    printf("Options:\n");
    printf("--help\t\t Display this help menu, exit\n\n");
    printf("-width=value\t\t set width, int\n");
    printf("-height=value\t\t set height, int\n");
    printf("-samples=value\t\t set samples per pixel, int \n");
    printf("-vfov=value\t\t set vertical fov, float\n");

    exit(0);
}

//int main(int argc, char *argv[])
//{
//    int width = 800, height = 800, samps = 800;
//    float vfov = 45.f;
//    bool showHelp = false;
//    
//    if (argc > 1)
//    {
//        if (checkCmdLineFlag(argc, (const char**)argv, "help"))
//        {
//            help(argv[0]);
//        }
//        if (checkCmdLineFlag(argc, (const char **)argv, "width"))
//            width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
//        if (checkCmdLineFlag(argc, (const char **)argv, "height"))
//            height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
//        if (checkCmdLineFlag(argc, (const char **)argv, "samples"))
//            samps = getCmdLineArgumentInt(argc, (const char **)argv, "samples");
//        if (checkCmdLineFlag(argc, (const char **)argv, "vfov"))
//            vfov = getCmdLineArgumentFloat(argc, (const char **)argv, "vfov");
//    }
//
//    TestSmallPTOnGPU(width, height, samps, vfov);
//
//    system("ffplay smallptcuda.ppm");
//    system("PAUSE");
//}
//


/** a useful function to compute the number of threads */
__host__ __device__
int divup(int x, int y)
{ return x / y + (x % y ? 1 : 0); }

/** gets the color, given the dwell */
void dwell_color(int *r, int *g, int *b, int dwell);

/** a simple complex type */
struct complex
{
    __host__ __device__
    complex(float re, float im = 0)
    {
        this->re = re;
        this->im = im;
    }
    float re, im;
};

// operator overloads for complex numbers
inline __host__ __device__
complex operator+ (const complex &a, const complex &b)
{
    return complex(a.re + b.re, a.im + b.im);
}

inline __host__ __device__
complex operator- (const complex &a)
{
    return complex(-a.re, -a.im);
}

inline __host__ __device__
complex operator- (const complex &a, const complex &b)
{
    return complex(a.re - b.re, a.im - b.im);
}

inline __host__ __device__
complex operator* (const complex &a, const complex &b)
{
    return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}

inline __host__ __device__
float abs2(const complex &a)
{
    return a.re * a.re + a.im * a.im;
}

inline __host__ __device__
complex operator/ (const complex &a, const complex &b)
{
    float invabs2 = 1 / abs2(b);
    return complex((a.re * b.re + a.im * b.im) * invabs2,
        (a.im * b.re - b.im * a.re) * invabs2);
}

#define MAX_DWELL 512
/** block size along */
#define BSX 64
#define BSY 4
/** maximum recursion depth */
#define MAX_DEPTH 4
/** region below which do per-pixel */
#define MIN_SIZE 32
/** subdivision factor along each axis */
#define SUBDIV 4
/** subdivision when launched from host */
#define INIT_SUBDIV 32

/** find the dwell for the pixel */
__device__
int pixel_dwell(int w, int h, complex cmin, complex cmax, int x, int y)
{
    complex dc = cmax - cmin;
    float fx = (float)x / w, fy = (float)y / h;
    complex c = cmin + complex(fx * dc.re, fy * dc.im);
    int dwell = 0;
    complex z = c;
    while (dwell < MAX_DWELL && abs2(z) < 2 * 2)
    {
        z = z * z + c;
        dwell++;
    }
    return dwell;
}

/** checking for an error */
__device__
void check_error(int x0, int y0, int d)
{
    int err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error @ region (%d..%d, %d..%d)\n", x0, x0 + d, y0, y0 + d);
        assert(0);
    }
}

/**
 * kernel : fill the image region with a specific value
 *
**/
__global__
void dwell_fill_k (int *dwells, int w, int x0, int y0, int d, int dwell)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < d && y < d)
    {
        x += x0, y += y0;
        dwells[y * w + x] = dwell;
    }
}

/**
 * kernel : fill the image region with per-pixel calculation 
 * a leaf node
 *
**/
__global__
void mandelbrot_pixel_k (int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < d && y < d)
    {
        x += x0, y += y0;
        dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y);
    }
}

/** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral element, -1 = dwells are different */
#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)

__device__
int same_dwell(int d1, int d2)
{
    if (d1 == d2)
        return d1;
    else if (d1 == NEUT_DWELL || d2 == NEUT_DWELL)
        return min(d1, d2);
    else
        return DIFF_DWELL;
}  // same_dwell

/** evaluates the common border dwell, if it exists */
__device__
int border_dwell (int w, int h, complex cmin, complex cmax, int x0, int y0, int d)
{
    // check whether all boundary pixels have the same dwell
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int bs = blockDim.x * blockDim.y;
    int comm_dwell = NEUT_DWELL;
    // for all boundary pixels, distributed across threads
    for (int r = tid; r < d; r += bs)
    {
        // for each boundary: b = 0 is east, then counter-clockwise
        for (int b = 0; b < 4; b++)
        {
            int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
            int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
            int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
            comm_dwell = same_dwell(comm_dwell, dwell);
        }
    }
    // for all boundary pixels
    // reduce across threads in the block
    __shared__ int ldwells[BSX * BSY];
    int nt = min(d, BSX * BSY);
    if (tid < nt)
        ldwells[tid] = comm_dwell;

    __syncthreads();

    for (; nt > 1; nt /= 2)
    {
        if (tid < nt / 2)
            ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2]);
        __syncthreads();
    }
    return ldwells[0];
}  // border_dwell

/** computes the dwells for Mandelbrot image using dynamic parallelism; one block is launched per pixel
        @param dwells the output array
        @param w the width of the output image
        @param h the height of the output image
        @param cmin the complex value associated with the left-bottom corner of the
        image
        @param cmax the complex value associated with the right-top corner of the
        image
        @param x0 the starting x coordinate of the portion to compute
        @param y0 the starting y coordinate of the portion to compute
        @param d the size of the portion to compute (the portion is always a square)
        @param depth kernel invocation depth
        @remarks the algorithm reverts to per-pixel Mandelbrot evaluation once either maximum depth or minimum size is reached
 */
__global__
void mandelbrot_block_k (int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int depth)
{
    x0 += d * blockIdx.x,
    y0 += d * blockIdx.y;
    int comm_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        if (comm_dwell != DIFF_DWELL)
        {
            // uniform dwell, just fill
            dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
            dwell_fill_k <<<grid, bs >>> (dwells, w, x0, y0, d, comm_dwell);
        }
        else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE)
        {
            // subdivide recursively
            dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
            mandelbrot_block_k <<<grid, bs >>> (dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1);
        }
        else
        {
            // leaf, per-pixel kernel
            dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
            mandelbrot_pixel_k <<<grid, bs >>> (dwells, w, h, cmin, cmax, x0, y0, d);
        }

        //cucheck_dev(cudaGetLastError());
        check_error(x0, y0, d);
    }
}

/**
  * converter : dwell --> color 
  *
 **/
#define CUT_DWELL (MAX_DWELL / 4)
inline void dwell_color(int *r, int *g, int *b, int dwell)
{
    // black for the Mandelbrot set
    if (dwell >= MAX_DWELL)
    {
        *r = *g = *b = 0;
    }
    else
    {
        if (dwell < 0) dwell = 0;
        if (dwell <= CUT_DWELL)
        {
            // from black to blue the first half
            *r = *g = 0;
            *b = 128 + dwell * 127 / (CUT_DWELL);
        }
        else
        {
            // from blue to white for the second half
            *b = 255;
            *r = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
        }
    }
}

void Save_PPM(int* data, int w, int h)
{
    // Write image to PPM file, a very simple image file format
    FILE *f = fopen("smallptcuda.ppm", "w");          
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);

    for (int i = 0; i < w * h; i++)  // loop over pixels, write RGB values
    {
        int r, g, b;
        dwell_color(&r, &g, &b, data[i]);
        fprintf(f, "%d %d %d ", r, g, b);
    }
    fclose(f);
}

/** data size */
#define H (1 * 1024)
#define W (1 * 1024)

int main(int argc, char **argv)
{
    int w = W, h = H;
    size_t dwell_sz = w * h * sizeof(int);
    int *h_dwells = nullptr,
        *d_dwells = nullptr;
    CUDA_CALL_CHECK(cudaMalloc((void**)&d_dwells, dwell_sz));
    h_dwells = (int*)malloc(dwell_sz);

    // Record start time                          
    auto start = std::chrono::high_resolution_clock::now();

    //< schedule threads on device and launch CUDA kernel from host
    //< use default stream
    dim3 bs(BSX, BSY), grid(INIT_SUBDIV, INIT_SUBDIV);
    mandelbrot_block_k <<< grid, bs >>> (d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1), 0, 0, W / INIT_SUBDIV, 1);
    CUDA_CALL_CHECK(cudaThreadSynchronize());

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();

    CUDA_CALL_CHECK(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));

    std::chrono::duration<double> elapsed = finish - start;
    printf("Mandelbrot set computed in %.3lf second  @(%.3lf Mpix/s)\n", elapsed.count(), h * w * 1e-6 / elapsed.count());

    Save_PPM(h_dwells, w, h);
    system("ffplay smallptcuda.ppm");
    system("PAUSE");

    // free data
    cudaFree(d_dwells);
    free(h_dwells);
    return 0;
}
