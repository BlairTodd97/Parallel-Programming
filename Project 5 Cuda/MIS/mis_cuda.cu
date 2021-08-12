/*
Maximal independent set code for CS 4380 / CS 5351

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

#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include "ECLgraph.h"
#include <cuda.h>

static const int ThreadsPerBlock = 1024;

static const unsigned char in = 2;
static const unsigned char out = 1;
static const unsigned char undecided = 0;

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static __device__ int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static __global__ void init(const ECLgraph g, unsigned char* status,unsigned int* rndval)
{ 

 int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < g.nodes){
   status[index] = undecided;
   rndval[index] = hash(index + 712453987);
  }
 }




static __global__ void mis(const ECLgraph g, volatile unsigned char* const status, const unsigned int* const rndval, volatile bool* const goagain)
{
  // initialize arrays
//  for (int v = 0; v < g.nodes; v++) status[v] = undecided;
 // for (int v = 0; v < g.nodes; v++) rndval[v] = hash(v + 712453987);
  // repeat until all nodes' status has been decided
const int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;

 // do {
//    goagain = false;
    // go over all the nodes
  //  int v = 0;
      if (status[idx] == undecided) {
        int i = g.nindex[idx];
        // try to find a neighbor whose random number is lower
        while ((i < g.nindex[idx + 1]) && ((status[g.nlist[i]] == out) || (rndval[idx] < rndval[g.nlist[i]]) || ((rndval[idx] == rndval[g.nlist[i]]) && (idx < g.nlist[i])))) {
          i++;
        }
        if (i < g.nindex[idx + 1]) {
          // found such a neighbor -> status still unknown
          goagain = true;
        
    
        } else {
          // no such neighbor -> all neighbors are "out" and my status is "in"
          status[idx] = in;
          for (int i = g.nindex[idx]; i < g.nindex[idx + 1]; i++) {
            status[g.nlist[i]] = out;
          }
        }
      }
    }
 // } while (goagain);


static void CheckCuda()
{
 cudaError_t e;
 cudaDeviceSynchronize();
 if (cudaSuccess != (e = cudaGetLastError())){
	fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
	exit(-1);
 }

}

int main(int argc, char* argv[])
{
  printf("Maximal Independent Set v1.5\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s input_file\n", argv[0]); exit(-1);}

  ECLgraph g = readECLgraph(argv[1]);

 ECLgraph d_g = g;
 cudaMalloc((void **)&d_g.nindex, sizeof(int) * (g.nodes + 1));
 cudaMalloc((void **)&d_g.nlist, sizeof(int) * g.edges);
 if (cudaSuccess != cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice)){fprintf(stderr, "copying to device failed\n"); exit(-1);}
if (cudaSuccess !=  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice)){fprintf(stderr, "copying to device failed\n"); exit(-1);}


   unsigned char* const status = new unsigned char[g.nodes];
   unsigned int* const rndval = new unsigned int[g.nodes];



//allocate arrays on GPU
 const int num =  d_g.nodes;
 unsigned char  * status_d = new unsigned char[g.nodes];
 unsigned int  * rndval_d = new unsigned int[g.nodes];
 const int size = num * sizeof(unsigned char);
 const int size_0 = num * sizeof(unsigned int);
 cudaMalloc((void **)&status_d, size);
 cudaMalloc((void **)&rndval_d, size_0);

  // read input
//  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // Kernel launch, not sure yet
// bool *goagain = new bool;
// bool *goagain_d = new bool(false);
 int size_1 = num * sizeof(bool);

 init<<<(d_g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock ,ThreadsPerBlock>>>(d_g , status_d, rndval_d);

 if(cudaSuccess != cudaMemcpy(status_d,status_d,size, cudaMemcpyHostToHost)){fprintf(stderr, "ERROR: status init  copying to device failed\n"); exit(-1);}
 if(cudaSuccess != cudaMemcpy(rndval_d, rndval_d, size_0, cudaMemcpyHostToHost)){fprintf(stderr, "ERROR: rndval init copying to device failed\n"); exit(-1);}
  cudaDeviceSynchronize(); 


do{

 bool *goagain = new bool(false);
 bool *goagain_d = new bool(false);
 cudaMalloc((void **)&goagain_d, size_1);
if (cudaSuccess != cudaMemcpy(goagain, goagain_d, size_1, cudaMemcpyHostToDevice)){fprintf(stderr, " ERROR: goagain 1 copying to device failed\n"); exit(-1);}
  mis<<<(g.nodes + ThreadsPerBlock -1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, status_d, rndval_d, goagain_d);
  if (cudaSuccess != cudaMemcpy(goagain, goagain_d, size_1, cudaMemcpyDeviceToHost)){fprintf(stderr, "ERROR: goagian 2 copying from device failed\n"); exit(-1);}
} while(*goagain);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);
  CheckCuda();

//unsigned char* status = new unsigned char[num];

if (cudaSuccess != cudaMemcpy(status_d, status, size, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  // determine and print set size
  int count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == in) {
      count++;
    }
  }
  printf("elements in set: %d (%.1f%%)\n", count, 100.0 * count / g.nodes);

  // verify result
  for (int v = 0; v < g.nodes; v++) {
    if ((status[v] != in) && (status[v] != out)) {fprintf(stderr, "ERROR: found unprocessed node\n"); exit(-1);}
    if (status[v] == in) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n"); exit(-1);}
      }
    } else {
      bool flag = true;
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {
          flag = false;
          break;
        }
      }
      if (flag) {fprintf(stderr, "ERROR: set is not maximal\n"); exit(-1);}
    }
  }
  printf("verification passed\n");

  // clean up
//  freeECLgraph(g);
  cudaFree(status_d);
  cudaFree(rndval_d);
  cudaFree(goagain_d);
  delete [] status;
  delete [] rndval;
  return 0;
}
