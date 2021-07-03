 
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

// __device__ : executed on the device (GPU) and callable only from the device
struct Ray
{ 
 float3 orig; // ray origin
 float3 dir;  // ray direction 
 __device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {} 
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance(), only DIFF used here

//< use on GPU
 __constant__ float epsilon = 1e-6f;  // epsilon required to prevent floating point precision artifacts

struct Sphere
{
 float rad;            // radius 
 float3 pos, emi, col; // position, emission, colour 
 Refl_t refl;          // reflection type (e.g. diffuse)

__device__ float intersect_sphere(const Ray &r) const 
{ 
 // ray/sphere intersection
 // returns distance t to intersection point, 0 if no hit  
 // ray equation: p(x,y,z) = ray.orig + t*ray.dir
 // general sphere equation: x^2 + y^2 + z^2 = rad^2 
 // classic quadratic equation of form ax^2 + bx + c = 0 
 // solution x = (-b +- sqrt(b*b - 4ac)) / 2a
 // solve t^2*ray.dir*ray.dir + 2*t*(orig-p)*ray.dir + (orig-p)*(orig-p) - rad*rad = 0 
 // more details in "Realistic Ray Tracing" book by P. Shirley or Scratchapixel.com

  float3 op = pos - r.orig;    // distance from ray.orig to center sphere 
  float t;
  float b = dot(op, r.dir);    // b in quadratic equation
  float disc = b*b - dot(op, op) + rad*rad;  // discriminant quadratic equation
  if (disc<0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
  else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant
  return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0); // pick closest point in front of ray origin
}

};

// SCENE
// 9 spheres forming a Cornell box
// small enough to be in constant GPU memory
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] =
{
 { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 
 { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght 
 { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
 { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt 
 { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm 
 { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
 { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 1
 { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 2
 { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

// param t is distance to closest intersection, initialise t to a huge number outside scene
// param i is the intersected sphere id.
__device__ inline bool intersect_scene(const Ray &r, float &t, int &id)
{
 float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;  
 for (int i = int(n); i--;)  // test all scene objects for intersection
  if ((d = spheres[i].intersect_sphere(r)) && d<t) // if newly computed intersection distance d is smaller than current closest intersection distance
  {  
    t = d;  // keep track of distance along ray to closest intersection point 
    id = i; // and closest intersected object
  }
 return t < inf; // returns true if an intersection with the scene occurred, false when no hit
}

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

// radiance function, the mean of path tracing 
// solves the rendering equation: 
// outgoing radiance (at a point) = emitted radiance + reflected radiance
// reflected radiance is sum (integral) of incoming radiance from all directions in hemisphere above point, 
// multiplied by reflectance function of material (BRDF) and cosine incident angle 
// returns radiance by ray 
__device__ float3 radiance(Ray &r, unsigned int *s1, unsigned int *s2)
{ 
 float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
 float3 mask = make_float3(1.0f, 1.0f, 1.0f); 

 // ray bounce loop (no Russian Roulette used) 
 // iteration up to 4 bounces (replaces recursion in CPU code)
 for (int bounces = 0; bounces < 4; bounces++)
 {  
  float t;           // distance to closest intersection 
  int id = 0;        // index of closest intersected sphere 

// test ray for intersection with scene
  if (!intersect_scene(r, t, id))
   return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

  const Sphere &obj = spheres[id];  // hitobject
  float3 x = r.orig + r.dir*t;          // hitpoint 
  float3 n = normalize(x - obj.pos);    // normal
  float3 nl = dot(n, r.dir) < 0 ? n : n * -1; // front facing normal

  // add emission of current sphere to accumulated colour(first term in rendering equation sum) 
  accucolor += mask * obj.emi;

  // all spheres in the scene are diffuse
  // generate new diffuse ray:
  // origin = hitpoint of previous ray in path
  // random direction in hemisphere above hitpoint (see "Realistic Ray Tracing", P. Shirley)

  // create 2 random numbers
  float r1 = 2 * M_PI * getrandom(s1, s2); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
  float r2 = getrandom(s1, s2);  // pick random number for elevation
  float r2s = sqrtf(r2); 

  // compute local orthonormal basis uvw at hitpoint for calculation random ray direction 
  // first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
  float3 w = nl; 
  float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));  
  float3 v = cross(w,u);

  // cosine weighted importance sampling (favours ray directions closer to normal direction)
  float3 d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));
  r.orig = x + nl * 0.05f; // offset ray origin slightly to prevent self intersection
  r.dir = d;

  mask *= obj.col;    // multiply with colour of object       
  mask *= dot(d,nl);  // weigh light contribution using cosine of angle between incident light and normal
  mask *= 2;          // fudge factor
 }

 return accucolor;
}

// __global__ : executed on the device (GPU) and called only from host (CPU) 
__global__ void render_kernel(float3 *output, int width, int height, int samps, float vfov = 45.f)
{
    // assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    //< index of current pixel in one Dimension 
    //< reverse y axis
    unsigned int i = (height - y - 1) * width + x;

    unsigned int s1 = x;  // seeds for random number generator
    unsigned int s2 = y;

    float3 r = make_float3(0.0f);

    // hardcoded camera ray(origin, direction) 
    Ray cam(make_float3(50, 52, 255.f), normalize(make_float3(0, -0.042612f, -1))); 

    float3 up = make_float3(0, 1, 0);
    //float vfov = 45.f;
    auto ratio = float(height) / float(width);
    auto dir = normalize(cam.dir);
    auto image_u = normalize(cross(dir, up));
    auto image_v = normalize(cross(image_u, dir));
    auto image_w = std::tan(vfov * float (M_PI * (1.f / 180.f) * 0.5f));
    image_u = image_u * image_w;
    image_v = image_v * image_w * ratio;

    //< AA : samples per pixel
    //< It is NOT progressive mode
    float u, v;
    float3 d;
    float invSamps = (1. / samps);
    for (int s = 0; s < (samps >> 2); s++)
	{  
        //< Performance goes down by applying AA. 1.1s --> 1.3s
        {
		    //< ray sampling from camera & collect radiance 
            u = (2 * (x + 0.25f)) / float(width)  - 1.0f ;
            v = (2 * (y + 0.25f)) / float(height) - 1.0f ;
            d = image_u * u + image_v * v + dir;
            r = r + radiance(Ray(cam.orig, normalize(d)), &s1, &s2) * invSamps;

            u = (2 * (x + 0.75f)) / float(width)  - 1.0f ;
            v = (2 * (y + 0.25f)) / float(height) - 1.0f ;
            d = image_u * u + image_v * v + dir;
            r = r + radiance(Ray(cam.orig, normalize(d)), &s1, &s2) * invSamps;

            u = (2 * (x + 0.25f)) / float(width)  - 1.0f ;
            v = (2 * (y + 0.75f)) / float(height) - 1.0f ;
            d = image_u * u + image_v * v + dir;
            r = r + radiance(Ray(cam.orig, normalize(d)), &s1, &s2) * invSamps;

            u = (2 * (x + 0.75f)) / float(width)  - 1.0f ;
            v = (2 * (y + 0.75f)) / float(height) - 1.0f ;
            d = image_u * u + image_v * v + dir;
            r = r + radiance(Ray(cam.orig, normalize(d)), &s1, &s2) * invSamps;
        }
    }

    //< write raw rgb value to buffer on the GPU 
    //< clamp & gamma is done on CPU
    output[i] = make_float3(r.x, r.y, r.z);
    //output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

//< clamp x into [0, 1]
inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; } 

// convert RGB float  [0,1] ==> int [0, 255] and perform gamma correction
inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); } 

void SaveToPPM(float3* output, int w, int h);

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
    render_kernel <<< grid, block >>>(output_d, width, height, samps, vfov);  
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

    SaveToPPM(output_h, width, height);

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
void SaveToPPM(float3* output, int w, int h)
{
    // Write image to PPM file, a very simple image file format
    FILE *f = fopen("smallptcuda.ppm", "w");          
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)  // loop over pixels, write RGB values
        fprintf(f, "%d %d %d ", toInt(output[i].x), toInt(output[i].y), toInt(output[i].z));
    fclose(f);
}


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
    /** real and imaginary part */
    float re, im;
}; // struct complex

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
#define H (16 * 1024)
#define W (16 * 1024)
#define IMAGE_PATH "./mandelbrot.png"

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


    //< save the image to disk 
    //SaveToPPM(h_dwells, width, height);

    Save_PPM(h_dwells, w, h);

    // print performance

    // free data
    cudaFree(d_dwells);
    free(h_dwells);
    return 0;
}






