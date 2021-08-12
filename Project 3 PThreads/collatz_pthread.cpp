/*
Collatz code for CS 4380 / CS 5351

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
#include <algorithm>
#include <sys/time.h>

#include <pthread.h> // add pthread header


//shared variables "shared memory"
static long threads;
static long bound;
static int maxlen;
static pthread_mutex_t mutex;


static void* collatz(void* arg) //changed arguments for pthreads
{
   //private variables for each thread
   const long my_rank = (long)arg;
   const long beg = my_rank * bound / threads;
   const long end = (my_rank + 1) * bound / threads;


  // compute sequence lengths
  int ml = 0; //private sequence length

  for (long i = beg + 1; i <= end; i++) { // update loop for parallel
    long val = i;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val /= 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
   // maxlen = std::max(maxlen, len); 
   if(ml < len)
      ml = len;
  }

   if(maxlen < ml){ //reduce
      pthread_mutex_lock(&mutex);
      if(maxlen < ml){
         maxlen = ml;
      }
      pthread_mutex_unlock(&mutex);
      }

      return NULL;
}
   

int main(int argc, char *argv[])
{
  printf("Collatz v1.4\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s upper_bound\n", argv[0]); exit(-1);} // update to take additional argument
  bound = atol(argv[1]);
  if (bound < 1) {fprintf(stderr, "ERROR: upper_bound must be at least 1\n"); exit(-1);}
  printf("upper bound: %ld\n", bound);
  
  threads = atoi(argv[2]); // take arg and put to threads
  if(threads < 1){fprintf(stderr, "ERROR: num_threads must be at least 1\n"); exit(-1);} //check for at least 1 thread
  printf("threads: %ld\n", threads);

  pthread_mutex_init(&mutex, NULL);
  pthread_t* const handler = new pthread_t[threads - 1];
  



  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
 // const int maxlen = collatz(bound);

   maxlen = 0;

   for(long thread = 0; thread < threads - 1; thread++){ //create threads
      pthread_create(&handler[thread], NULL, collatz, (void*)thread);
   }

   collatz((void*)0); //work for master thread (rank 0)
 
   for(long thread = 0; thread < threads - 1; thread++){ //join threads
      pthread_join(handler[thread],NULL);
   }


  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);

  // print result
  printf("longest sequence length: %d elements\n", maxlen);

  pthread_mutex_destroy(&mutex); // clean up mutex
  delete [] handler; //clean up memory 
  return 0;
}
