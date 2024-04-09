#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__device__ float* dot(float A[3][3], float B[3]){
              float result[3] = {0.0f, 0.0f, 0.0f};
              for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                  result[i] += A[i][j] * B[j];
                }
              }
              return result;
            }; 

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int method_num )
{
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  float left_color[3] = {0., 0., 0.};
  float right_color[3] = {0., 0., 0.};

  float tAnaglyph_left[3][3] = {
        {0., 0., 0.},
        {0., 0., 0.},
        {0.299, 0.587, 0.114}
      };

  float tAnaglyph_right[3][3] = {
        {0.299, 0.587, 0.114},
        {0., 0., 0.},
        {0., 0., 0.}
      };

  float gAnaglyph_left[3][3] = {
        {0., 0., 0.},
        {0., 0., 0.},
        {0.299, 0.587, 0.114}
      };
  
  float gAnaglyph_right[3][3] = {
        {0.299, 0.587, 0.114},
        {0.299, 0.587, 0.114},
        {0., 0., 0.}
      };
  
  
  float cAnaglyph_left[3][3] = {
        {0., 0., 0.},
        {0., 0., 0.},
        {1., 0., 0.}
      };

  float cAnaglyph_right[3][3] = {
        {0., 0., 1.},
        {0., 1., 0.},
        {0., 0., 0.}
      };
  
  float hcAnaglyph_left[3][3] = {
        {0., 0., 0.},
        {0., 0., 0.},
        {0.299, 0.587, 0.114}
      };

  float hcAnaglyph_right[3][3] = {
        {0., 0., 1.},
        {0., 1., 0.},
        {0., 0., 0.}
      };
  
  float oAnaglyph_left[3][3] = {
        {0., 0., 0.},
        {0., 0., 0.},
        {0., 0.7, 0.3}
      };

  float oAnaglyph_right[3][3] = {
        {0., 0., 1.},
        {0., 1., 0.},
        {0., 0., 0.}
      };
  

  if (dst_x < cols && dst_y < rows) {
    uchar3 val = src(dst_y, dst_x);
    left_color[0] = val.x;
    left_color[1] = val.y;
    left_color[2] = val.z;
    
    val = src(dst_y, dst_x+cols);
    right_color[0] = val.x;
    right_color[1] = val.y;
    right_color[2] = val.z;
  }
  
  if (dst_x < cols && dst_y < rows){
      // dst(dst_y, dst_x).x = dot(tAnaglyph_left,left_color)[0]+dot(tAnaglyph_right,right_color)[0];
      // dst(dst_y, dst_x).y = dot(tAnaglyph_left,left_color)[1]+dot(tAnaglyph_right,right_color)[1];
      // dst(dst_y, dst_x).z = dot(tAnaglyph_left,left_color)[2]+dot(tAnaglyph_right,right_color)[2];
      if (method_num == 1){
        dst(dst_y, dst_x).x = dot(tAnaglyph_left,left_color)[0]+dot(tAnaglyph_right,right_color)[0];
        dst(dst_y, dst_x).y = dot(tAnaglyph_left,left_color)[1]+dot(tAnaglyph_right,right_color)[1];
        dst(dst_y, dst_x).z = dot(tAnaglyph_left,left_color)[2]+dot(tAnaglyph_right,right_color)[2];
      }else if (method_num == 2){
        dst(dst_y, dst_x).x = dot(gAnaglyph_left,left_color)[0]+dot(gAnaglyph_right,right_color)[0];
        dst(dst_y, dst_x).y = dot(gAnaglyph_left,left_color)[1]+dot(gAnaglyph_right,right_color)[1];
        dst(dst_y, dst_x).z = dot(gAnaglyph_left,left_color)[2]+dot(gAnaglyph_right,right_color)[2];
      }else if (method_num == 3){
        dst(dst_y, dst_x).x = dot(cAnaglyph_left,left_color)[0]+dot(cAnaglyph_right,right_color)[0];
        dst(dst_y, dst_x).y = dot(cAnaglyph_left,left_color)[1]+dot(cAnaglyph_right,right_color)[1];
        dst(dst_y, dst_x).z = dot(cAnaglyph_left,left_color)[2]+dot(cAnaglyph_right,right_color)[2];
      }else if (method_num == 4){
        dst(dst_y, dst_x).x = dot(hcAnaglyph_left,left_color)[0]+dot(hcAnaglyph_right,right_color)[0];
        dst(dst_y, dst_x).y = dot(hcAnaglyph_left,left_color)[1]+dot(hcAnaglyph_right,right_color)[1];
        dst(dst_y, dst_x).z = dot(hcAnaglyph_left,left_color)[2]+dot(hcAnaglyph_right,right_color)[2];
      }else if (method_num == 5){
        dst(dst_y, dst_x).x = dot(oAnaglyph_left,left_color)[0]+dot(oAnaglyph_right,right_color)[0];
        dst(dst_y, dst_x).y = dot(oAnaglyph_left,left_color)[1]+dot(oAnaglyph_right,right_color)[1];
        dst(dst_y, dst_x).z = dot(oAnaglyph_left,left_color)[2]+dot(oAnaglyph_right,right_color)[2];
      }
      
    }
  
}
int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int method_num )
{
  const dim3 block(32, 32);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, method_num);

}

