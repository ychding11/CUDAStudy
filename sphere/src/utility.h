 
#pragma once

#include <iostream>
#include <string>
#include <chrono>  // for high_resolution_clock

//#include <cuda_runtime.h>
//#include <curand.h>
//#include <curand_kernel.h>

/////////////////////////////////////////////////////////////////
// 
/////////////////////////////////////////////////////////////////

class PerformanceProbe
{
private: 
    std::string m_name;
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::steady_clock::time_point m_end_time;
public:
    PerformanceProbe() = delete;
    PerformanceProbe(const std::string name)
        : m_name(name)
    {
         m_start_time = std::chrono::high_resolution_clock::now();
    }

    ~PerformanceProbe()
    {
         m_end_time = std::chrono::high_resolution_clock::now();
         auto elapsed = m_end_time - m_start_time;
         std::cout << "[Perfomance Probe] "
             << m_name << " time = " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " µs. "
             << std::endl;
    }
};

//< no space in X
#define PERFORMANCE_PROBE(X) PerformanceProbe X{#X}

inline float clamp(float x)
{
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
} 

inline int toInt(float x)
{
    return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
}  // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction


inline bool file_exists(const std::string name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

inline void SaveToPPM(float* output, int w, int h)
{
    {
        PERFORMANCE_PROBE(save_ppm);

        // Write image to PPM file, a very simple image file format
        FILE *f = fopen("smallptcuda.ppm", "w");          
        fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
        for (int i = 0; i < w * h * 3; i += 3)  // loop over pixels, write RGB values
        fprintf(f, "%d %d %d ", toInt(output[i]), toInt(output[i + 1]), toInt(output[i + 2]));
        fclose(f);
    }

    system("ffplay smallptcuda.ppm");
}
