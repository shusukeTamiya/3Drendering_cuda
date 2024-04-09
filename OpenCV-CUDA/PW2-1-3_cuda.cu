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

__device__ void sum_function(uchar3 color, float* sum) {
    sum[0] += static_cast<float>(color.x);
    sum[1] += static_cast<float>(color.y);
    sum[2] += static_cast<float>(color.z);
}


__device__ float* convert_to_float(uchar3 bgr){
  float result[3]={0.0f, 0.0f, 0.0f};
  result[0] = static_cast<float>(bgr.x);
  result[1] = static_cast<float>(bgr.y);
  result[2] = static_cast<float>(bgr.z);
  return result;
}

__device__ float difference_function(float target, float ave){
  float result = 0.0f;
  result = target-ave;
  return result;
}

__device__ void covariance_function(float covariance[3][3], float difference[100][100][3], float neighbor){
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      for(int k=0;k<neighbor;k++){
        for(int l=0;l<neighbor;l++){
          covariance[i][j] += difference[k][l][i]*difference[k][l][j];
        }
      }
      covariance[i][j]/=neighbor*neighbor;
    }
  }
}

__device__  float determinant_function(float matrix[3][3] ){
  float result = 0.0f;
  result = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  return result;
}

__device__ int kernel_size_function(float determinant, float ratio){
  return 2+static_cast<int>(ratio/(0.01*(determinant*determinant)));
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int neighbor, float ratio)
{
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  float gauss = 0.0;
  float gauss_sum = 0.0;
  float temp[3] = {0.0f, 0.0f, 0.0f};
  uchar3 val;
  float sum[3] = {0.0f, 0.0f, 0.0f};
  float ave[3] = {0.0f, 0.0f, 0.0f};
  float difference[100][100][3] = {0.0f};
  float covariance[3][3] = {0.0f};
  float determinant = 0.0f;
  int kernel_size;
  float sigma = 3.0;

// calculate sum
  for(int i = -neighbor/2; i < neighbor/2; ++i){
    for(int j = -neighbor/2; j < neighbor/2; ++j){
      if(dst_x<cols){
        if((dst_x+i<0) || (dst_y+j<0) || (dst_y+j>rows) || (dst_x+i>cols)){
          val = src(dst_y, dst_x);
          sum_function(src(dst_y, dst_x),sum);
        }else{
          val = src(dst_y+j, dst_x+i);
          sum_function(src(dst_y, dst_x),sum);
        }
      }
    }
  }
  //calculate ave
  ave[0] = sum[0]/static_cast<float>((neighbor*neighbor));
  ave[1] = sum[1]/static_cast<float>((neighbor*neighbor));
  ave[2] = sum[2]/static_cast<float>((neighbor*neighbor));


  // calculate difference between target and average
  for(int i = -neighbor/2; i < neighbor/2; ++i){
    for(int j = -neighbor/2; j < neighbor/2; ++j){
      if(dst_x<cols){
        if((dst_x+i<0) || (dst_y+j<0) || (dst_y+j>rows) || (dst_x+i>cols)){
          for(int c=0;c<3;c++){
            difference[i][j][c] = difference_function(convert_to_float(src(dst_y, dst_x))[c],ave[c]);
          }
        }else{
          for(int c=0;c<3;c++){
            difference[i][j][c] = difference_function(convert_to_float(src(dst_y+j, dst_x+i))[c],ave[c]);
          }
        }
      }
    }
  }

  // calculate the covariance
  covariance_function(covariance, difference,static_cast<float>(neighbor));

  // calculate determinant
  determinant = determinant_function(covariance);

  // calculate kernel_size
  kernel_size = kernel_size_function(determinant, ratio);
  if(kernel_size>31 || kernel_size<0){
    kernel_size = 31;
  }



// gaussian filter
  if (dst_x < cols && dst_y < rows){      
      for(int i=-int(kernel_size)/2; i < int(kernel_size)/2; i++){
        for(int j=-(int(kernel_size)/2); j < int(kernel_size/2); j++){
            gauss = gaussian(sigma, i, j);
            if(dst_x<cols){
              if((dst_x+i<0) || (dst_y+j<0) || (dst_y+j>rows) || (dst_x+i>cols)){
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

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int neighbor, float ratio )
{
  const dim3 block(32, 32);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, neighbor, ratio);

}

