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
#include <mpi.h> //inlcude MPI


static int collatz(const long bound, const int my_Rank, const int comm_sz)
{
	//calculate cycle
   const long my_start = 1;
   const long my_end = (my_Rank + 1) * bound/comm_sz; 

   // compute sequence lengths
   int maxlen = 0;
   for (long i = my_start + my_Rank; i <= bound; i+=comm_sz) {
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
      if(maxlen < len)  
         maxlen = len;         
   }

   return maxlen;
 }

int main(int argc, char *argv[]) //change as little as possible
{ 
  int my_Rank;  // Initialize MPI 
  int comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_Rank);

 if(my_Rank == 0)
    printf("Collatz v1.4\n");
 
  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s upper_bound\n", argv[0]); exit(-1);}
  const long bound = atol(argv[1]);
  if (bound < 1) {fprintf(stderr, "ERROR: upper_bound must be at least 1\n"); exit(-1);}
  if(my_Rank == 0) printf("upper bound: %ld\n", bound); //only rank 0 prints

  // start time
  timeval start, end;

  MPI_Barrier(MPI_COMM_WORLD); //barrier before timed sections

  gettimeofday(&start, NULL);

  // execute timed code
  const int my_maxLen = collatz(bound, my_Rank, comm_sz); //add args
  
  int maxlen;
  MPI_Reduce(&my_maxLen, &maxlen, 1, MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD); // reduce results 

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  
   if(my_Rank == 0) //only one process to output
      printf("compute time: %.6f s\n", runtime);

  // print result
   if(my_Rank == 0) // only one process to output
      printf("longest sequence length: %d elements\n", maxlen);

   MPI_Finalize(); //finalize use of MPI
   return 0;
}
