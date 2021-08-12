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

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include "BMP43805351.h"
#include <pthread.h>


//shared mem variables
static const double Delta = 0.00304;
static const double xMid = -0.055846456;
static const double yMid = -0.668311119;

static int threads;
static int frames;
static int width;
static unsigned char* pic;



static void* fractal(void *arg) //changed args to take thread rank
{
 

   const long my_rank = (long)arg; //get personal rank
   long my_start = my_rank *(frames/threads);
   long my_end = (my_rank +1) * (frames/threads);

  // compute pixels of each frame
//  double delta = Delta; taken out due to loop carry
  for (int frame = my_start; frame < my_end; frame++) {  // frames
    
 const double delta = Delta * pow(0.975, frame);

    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    for (int row = 0; row < width; row++) {  // rows
      const double cy = yMin + row * dw;
      for (int col = 0; col < width; col++) {  // columns
        const double cx = xMin + col * dw;
        double x = cx;
        double y = cy;
        double x2, y2;
        int count = 256;
        do {
          x2 = x * x;
          y2 = y * y;
          y = 2.0 * x * y + cy;
          x = x2 - y2 + cx;
          count--;
        } while ((count > 0) && ((x2 + y2) <= 5.0));
        pic[frame * width * width + row * width + col] = (unsigned char)count;
      }
    }
  //  delta *= 0.975;
  }
 return NULL;
}

int main(int argc, char *argv[])
{
  printf("Fractal v2.1\n");

  // check command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s frame_width number_of_frames\n", argv[0]); exit(-1);} //update argument count
  width = atoi(argv[1]);
  if (width < 8) {fprintf(stderr, "ERROR: frame_width must be at least 8\n"); exit(-1);}
  frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: number_of_frames must be at least 1\n"); exit(-1);}
  printf("frames: %d\n", frames);
  printf("width: %d\n", width);

  threads = atoi(argv[3]); // take threads and check for correct number
  if(threads < 1) {fprintf(stderr, "error: threads must be at least 1\n"); exit(-1);}
  printf("threads: %ld\n", threads);


  // allocate picture array

   pic = new unsigned char [frames * width * width];
 
   pthread_t* const handle = new pthread_t[threads-1]; //created handler for threads
   int thread; // counter for when threads are made and joined
  // start time
  timeval start, end;
  gettimeofday(&start, NULL);


  for( thread = 0; thread < threads-1 ; thread++){ //create threads 
     pthread_create(&handle[thread], NULL, fractal, (void*)thread+1);
  }
 
  fractal((void*)0); //start fractal in master thread as well


   for(long thread = 0; thread < threads-1; thread++){ //join threads
      pthread_join(handle[thread], NULL);
   }



  // execute timed code

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // write result to BMP files
  if ((width <= 256) && (frames <= 64)) {
    for (int frame = 0; frame < frames; frame++) {
      BMP24 bmp(0, 0, width, width);
      for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
          bmp.dot(x, y, pic[frame * width * width + y * width + x] * 0x010101);
        }
      }
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      bmp.save(name);
    }
  }

  // clean up
  
  delete [] pic;
  delete [] handle;  
return 0;
}
