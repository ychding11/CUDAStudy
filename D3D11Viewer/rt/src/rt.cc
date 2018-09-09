#include <iostream>
#include <ctime>
#include <chrono>
#include "sphere.h"
#include "moving_sphere.h"
#include "hitable_list.h"
#include "float.h"
#include "camera.h"
#include "material.h"
#include "bvh.h"
#include "box.h"
#include "surface_texture.h"
#include "aarect.h"
#include "texture.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "pdf.h"
#include "utils.h"

inline vec3 de_nan(const vec3& c)
{
    vec3 temp = c;
    if (!(temp[0] == temp[0])) temp[0] = 0;
    if (!(temp[1] == temp[1])) temp[1] = 0;
    if (!(temp[2] == temp[2])) temp[2] = 0;
    return temp;
}

vec3 color(const ray& r, hitable *world, hitable *light_shape, int depth)
{
    hit_record hrec;
    if (world->hit(r, 0.001, MAXFLOAT, hrec))
    { 
        scatter_record srec;
        vec3 emitted = hrec.mat_ptr->emitted(r, hrec, hrec.u, hrec.v, hrec.p);
        if (depth < 50 && hrec.mat_ptr->scatter(r, hrec, srec))
        {
            if (srec.is_specular)
            {
                return srec.attenuation * color(srec.specular_ray, world, light_shape, depth+1);
            }
            else
            {
                hitable_pdf plight(light_shape, hrec.p);
                mixture_pdf p(&plight, srec.pdf_ptr);
                ray scattered = ray(hrec.p, p.generate(), r.time());
                float pdf_val = p.value(scattered.direction());
                delete srec.pdf_ptr;
                return emitted + srec.attenuation*hrec.mat_ptr->scattering_pdf(r, hrec, scattered)*color(scattered, world, light_shape, depth+1) / pdf_val;
            }
        }
        else 
            return emitted;
    }
    else 
        return vec3(0,0,0);
}

void cornell_box(hitable **scene, camera **cam, float aspect) {
    int i = 0;
    hitable **list = new hitable*[8];
    material *red = new lambertian( new constant_texture(vec3(0.65, 0.05, 0.05)) );
    material *white = new lambertian( new constant_texture(vec3(0.73, 0.73, 0.73)) );
    material *green = new lambertian( new constant_texture(vec3(0.12, 0.45, 0.15)) );
    material *light = new diffuse_light( new constant_texture(vec3(10, 10, 10)) );
    list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
    list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
    list[i++] = new flip_normals(new xz_rect(213, 343, 227, 332, 554, light));
    list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
    list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
    list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
    material *glass = new dielectric(1.5);
    list[i++] = new sphere(vec3(190, 90, 190),90 , glass);
    list[i++] = new translate(new rotate_y(
                    new box(vec3(0, 0, 0), vec3(165, 330, 165), white),  15), vec3(265,0,295));
    *scene = new hitable_list(list,i);
    vec3 lookfrom(278, 278, -800);
    vec3 lookat(278,278,0);
    float dist_to_focus = 10.0;
    float aperture = 0.0;
    float vfov = 40.0;
    *cam = new camera(lookfrom, lookat, vec3(0,1,0),
                      vfov, aspect, aperture, dist_to_focus, 0.0, 1.0);
}

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

inline int  to_int(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

void save_to_ppm(vec3* output, int w, int h, int s, int t)
{
    char filename[128];
    sprintf_s(filename, 128, "image-%d-%d-%d-%d.ppm", w, h, s, t);
    FILE *f = nullptr;
    fopen_s(&f, filename, "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)
        fprintf(f, "%d %d %d ", to_int(output[i].x()), to_int(output[i].y()), to_int(output[i].z()));
    fclose(f);

    fprintf(stdout, "- Save as %s\n\n", filename);
#if 0
    char display_image[128];
    sprintf_s(display_image, 128, "ffplay.exe %s", filename);
    system(display_image);
#endif
}

int update(void* data, int nx = 256, int ny = 256, int ns = 10)
{
    vec3* pic = (vec3*)data; 
    assert(pic);

    hitable *world;
    camera *cam;

    float aspect = float(ny) / float(nx);

    cornell_box(&world, &cam, aspect);

    hitable *light_shape = new xz_rect(213, 343, 227, 332, 554, 0);
    hitable *glass_sphere = new sphere(vec3(190, 90, 190), 90, 0);
    hitable *a[2];
    a[0] = light_shape;
    a[1] = glass_sphere;
    hitable_list hlist(a,2);

    //fprintf(stdout, "- Start Rendering... %dx%d\n", nx, ny);
    // Record start time                          
    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0, j = ny-1; j >= 0; j--)
    {
        //fprintf(stdout, "\rRendering (%d spp) %5.2f%% ", ns, 100.*k / (nx * ny));
        for (int i = 0; i < nx; i++)
        {
            vec3 col(0.f, 0.f, 0.f);
            float invNs = 4.f / float(ns);
            for (int s=0; s < 4; s++)
            {
                float u = float(i+drand48())/ float(nx);
                float v = float(j+drand48())/ float(ny);
                ray r = cam->get_ray(u, v);
                vec3 p = r.point_at_parameter(2.0);
                col += de_nan(color(r, world, &hlist, 0));
            }
            col *= invNs;
            pic[k++] += col;
        }
    }
    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    //fprintf(stdout, "\n- Render Done! Time=%lf seconds\n", elapsed.count());
    //save_to_ppm(pic, nx, ny, ns, int(elapsed.count()));
    return 0;
}

