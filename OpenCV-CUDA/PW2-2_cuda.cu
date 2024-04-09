#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

#define PI 3.141592653
#define E 2.7182818
#define MaxKernel 21
#define BlockSize 32

__device__ float gaussian(float sigma, int x, int y){
              return pow(E,-((x*x)+(y*y))/(2.*sigma*sigma))/(2. * PI * sigma*sigma);
            }

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernel_size, float sigma)
{
  __shared__ uchar3 shared_pixel[BlockSize + MaxKernel -1][BlockSize + MaxKernel-1];
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  float gauss = 0.0;
  float gauss_sum = 0.0;
  float temp[3] = {0.0f, 0.0f, 0.0f};
  uchar3 val;


  int shared_x = threadIdx.x + MaxKernel/2;
  int shared_y = threadIdx.y + MaxKernel/2;
  shared_pixel[shared_y][shared_x] = src(dst_y, dst_x);
  if((threadIdx.y< MaxKernel/2)&&(threadIdx.x< MaxKernel/2)){
    shared_pixel[shared_y-MaxKernel/2][shared_x-MaxKernel/2] = src(dst_y-MaxKernel/2, dst_x-MaxKernel/2);
    shared_pixel[shared_y+BlockSize][shared_x+BlockSize] = src(dst_y+BlockSize, dst_x+BlockSize);
  }
  if(threadIdx.y< MaxKernel/2){
    shared_pixel[shared_y-MaxKernel/2][shared_x] = src(dst_y-MaxKernel/2, dst_x);
    shared_pixel[shared_y+BlockSize][shared_x] = src(dst_y+BlockSize, dst_x);
  }
  if(threadIdx.x< MaxKernel/2){
    shared_pixel[shared_y][shared_x-MaxKernel/2] = src(dst_y, dst_x-MaxKernel/2);
    shared_pixel[shared_y][shared_x+BlockSize] = src(dst_y, dst_x+BlockSize);
  }
  
  __syncthreads();
  if (dst_x < cols && dst_y < rows){      
      for(int i=-int(kernel_size)/2; i < int(kernel_size)/2; i++){
        for(int j=-(int(kernel_size)/2); j < int(kernel_size/2); j++){
            gauss = gaussian(sigma, i, j);
            if(dst_x<cols/2){
              if((dst_x+i<0) || (dst_y+j<0) || (dst_y+j>rows) || (dst_x+i>cols/2)){
                val = shared_pixel[shared_y][shared_x];
              }else{
                val = shared_pixel[shared_y+i][shared_x+j];
              }
            }else{
              if((dst_x+i<cols/2) || (dst_y+j<0) || (dst_y+j>rows) || (dst_x+i>cols)){
                val = shared_pixel[shared_y][shared_x];
              }else{
                val = shared_pixel[shared_y+i][shared_x+j];
              }
            }
            temp[0] += val.x * gauss;
            temp[1] += val.y * gauss;
            temp[2] += val.z * gauss;
            gauss_sum += gauss;
        }
      }
      dst(dst_y, dst_x).x = temp[0]/gauss_sum;
      dst(dst_y, dst_x).y = temp[1]/gauss_sum;
      dst(dst_y, dst_x).z = temp[2]/gauss_sum;
    }
}
int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, float sigma )
{
  const dim3 block(BlockSize, BlockSize);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, kernel_size, sigma);

}

