#ifndef RAYH_UTILS_H
#define RAYH_UTILS_H

#include <cstdlib>
#include <limits>
#include <cassert>
#include "vec3.h"

#define M_PI     3.1415926
#define MAXFLOAT (std::numeric_limits<float>::max())

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// SRAND48 and DRAND48 don't exist on windows, but these are the equivalent functions
	inline void srand48(long seed)
	{
		srand((unsigned int)seed);
	}
	inline double drand48()
	{
		double ret = double(rand())/( RAND_MAX + 1);
#if 0
        if (ret < 1.0 && ret >= 0.0)
        {

        }
        else
        {
            printf("- random error: %lf\n", ret);
            system("pause");
            exit(1);
        }
#endif
        return ret;
	}
#endif
#endif 


   
