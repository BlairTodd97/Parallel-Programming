/*
Fractal code for CS 4380 / CS 5351

Copyright (c) 2020 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source or binary form, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

//Blair Todd
//CS4380

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <sys/time.h>
#include "BMP43805351.h"

unsigned char* GPU_Init(const int gpu_frames, const int width);
void GPU_Exec(const int start_frame, const int stop_frame, const int width, unsigned char* d_pic);
void GPU_Fini(const int gpu_frames, const int width, unsigned char* pic, unsigned char* d_pic);

static void fractal(const int start_frame, const int stop_frame, const int width, unsigned char* const pic)
{
  // todo: use OpenMP to parallelize the for-row loop with default(none) and do not specify a schedule

   const double Delta = 0.00304;
   const double xMid = -0.055846456;
   const double yMid = -0.668311119;
   
   #pragma omp parallel default(none)
   
   for (int frame = start_frame; frame < stop_frame; frame++){
      const double delta = Delta * pow(0.975, frame);
      const double xMin = xMid - delta;
      const double yMin = yMid -delta;
      const double dw = 2.0 * delta / width;
   
   #pragma omp for
   for(int row = 0; row < width; row++){
      const double cy = yMin + row * dw;
      for(int col = 0; col < width; col++){
         const double cx = xMin + col * dw;
         double x = cx;
         double y = cy;
         double x2, y2;
         int count = 256;
         do{
            x2 = x * x;
            y2 = y * y;
            y = 2.0 * x * y + cy;
            x = x2 -y2 + cx;
            count--;
         } while((count > 0) && ((x2 + y2) <= 5.0));
         pic[frame * width * width + row * width + col] = (unsigned char)count;
         }
      }
   }
}

int main(int argc, char *argv[])
{
  // set up MPI
  int comm_sz, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) printf("Fractal v2.1\n");

  // check command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s frame_width number_of_frames cpu_percentage\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 8) {fprintf(stderr, "ERROR: frame_width must be at least 8\n"); exit(-1);}
  const int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: number of number_of_frames must be at least 1\n"); exit(-1);}
  if ((frames % comm_sz) != 0) {fprintf(stderr, "ERROR: number_of_frames must be a multiple of the number of processes\n"); exit(-1);}
  const int percentage = atoi(argv[3]);
  if ((percentage < 0) || (percentage > 100)) {fprintf(stderr, "ERROR: cpu_percentage must be between 0 and 100\n"); exit(-1);}

  const int cpu_start = my_rank * frames / comm_sz;
  const int gpu_stop = (my_rank + 1) * frames / comm_sz;
  const int my_range = gpu_stop - cpu_start;
  const int cpu_stop = cpu_start + my_range * percentage / 100;
  const int gpu_start = cpu_stop;

  if (my_rank == 0) {
    printf("frames: %d\n", frames);
    printf("width: %d\n", width);
    printf("CPU percentage: %d\n", percentage);
    printf("MPI tasks: %d\n", comm_sz);
  }

  // allocate picture arrays
  unsigned char* pic = new unsigned char [my_range * width * width];
  unsigned char* d_pic = GPU_Init(gpu_stop - gpu_start, width);
  unsigned char* full_pic = NULL;
  if (my_rank == 0) full_pic = new unsigned char [frames * width * width];

  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);  // for better timing
  gettimeofday(&start, NULL);

  // asynchronously compute the requested frames on the GPU
  GPU_Exec(gpu_start, gpu_stop, width, d_pic);

  // compute the remaining frames on the CPU
  fractal(cpu_start, cpu_stop, width, pic);

  // copy the GPU's result into the appropriate location of the CPU's pic array
  GPU_Fini(gpu_stop - gpu_start, width, &pic[(cpu_stop - cpu_start) * width * width], d_pic);

  // todo: gather the results into full_pic on compute node 0

   int size = my_range * width * width;
   MPI_Gather(full_pic, size, MPI_UNSIGNED_CHAR, pic, size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    gettimeofday(&end, NULL);
    const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("compute time: %.6f s\n", runtime);

    // write result to BMP files
    if ((width <= 256) && (frames <= 64)) {
      for (int frame = 0; frame < frames; frame++) {
        BMP24 bmp(0, 0, width, width);
        for (int y = 0; y < width; y++) {
          for (int x = 0; x < width; x++) {
            bmp.dot(x, y, full_pic[frame * width * width + y * width + x] * 0x010101);
          }
        }
        char name[32];
        sprintf(name, "fractal%d.bmp", frame + 1000);
        bmp.save(name);
      }
    }

    delete [] full_pic;
  }

  MPI_Finalize();
  delete [] pic;
  return 0;
}
