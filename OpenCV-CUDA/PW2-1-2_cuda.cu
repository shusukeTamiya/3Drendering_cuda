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

__device__ float gaussian(float sigma, int x, int y){
              return pow(E,-((x*x)+(y*y))/(2.*sigma*sigma))/(2. * PI * sigma*sigma);
            }

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernel_size, float sigma)
{
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  float gauss = 0.0;
  float gauss_sum = 0.0;
  float temp[3] = {0.0f, 0.0f, 0.0f};
  uchar3 val;

  if (dst_x < cols && dst_y < rows){      
      for(int i=-int(kernel_size)/2; i < int(kernel_size)/2; i++){
        for(int j=-(int(kernel_size)/2); j < int(kernel_size/2); j++){
            gauss = gaussian(sigma, i, j);
            if(dst_x<cols/2){
              if((dst_x+i<0) || (dst_y+j<0) || (dst_y+j>rows) || (dst_x+i>cols/2)){
                val = src(dst_y, dst_x);
              }else{
                val = src(dst_y+j, dst_x+i);
              }
            }else{
                if((dst_x+i<cols/2) || (dst_y+j<0) || (dst_y+j>rows) || (dst_x+i>cols)){
                  val = src(dst_y, dst_x);
                }else{
                  val = src(dst_y+j, dst_x+i);
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
      gauss_sum = 0.0;
    }
  
}
int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, float sigma )
{
  const dim3 block(32, 32);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, kernel_size, sigma);

}

