#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>  // for high_resolution_clock

using namespace std;

std::vector<float> dot(const float A[3][3], const std::vector<float>& B) {
    std::vector<float> result(3, 0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += A[i][j] * static_cast<float>(B[j]);
        }
    }

    return result;
} 

int main( int argc, char** argv )
{

  cv::Mat_<cv::Vec3b> source = cv::imread ( argv[1], cv::IMREAD_COLOR);
  cv::Mat_<cv::Vec3b> destination_left ( source.rows, source.cols/2 );
  cv::Mat_<cv::Vec3b> destination_right ( source.rows, source.cols/2 );
  cv::Mat_<cv::Vec3b> destination ( source.rows, source.cols/2 );
  cv::Mat_<cv::Vec3b> destination_temp ( source.rows, source.cols/2 );
  cv::imshow("Source Image", source );
  std::string processing_method = argv[2];

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;


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
  

#pragma omp parallel for
  for (int it=0;it<iter;it++)
    {
      for (int i=0;i<source.rows;i++)
	      for (int j=0;j<source.cols;j++)
          for (int c=0;c<3;c++)
            {
              if(j<source.cols/2){
                destination_left(i,j)[c] = source(i,j)[c];
              }else{
                destination_right(i,j-source.cols/2)[c] = source(i,j)[c];
              }
            }
  
  #pragma omp parallel for
      for (int i=0;i<source.rows;i++)
	      for (int j=0;j<source.cols/2;j++)
        {
        
          if(processing_method=="True_Anaglyph"){
            std::vector<float> pixel_left = {
            static_cast<float>(destination_left(i,j)[0]),
            static_cast<float>(destination_left(i,j)[1]), 
            static_cast<float>(destination_left(i,j)[2])};
            std::vector<float> pixel_right = {
            static_cast<float>(destination_right(i,j)[0]),
            static_cast<float>(destination_right(i,j)[1]), 
            static_cast<float>(destination_right(i,j)[2])};
            destination(i,j)[0] = static_cast<unsigned char>(dot(tAnaglyph_left,pixel_left)[0]+dot(tAnaglyph_right,pixel_right)[0]);
            destination(i,j)[1] = static_cast<unsigned char>(dot(tAnaglyph_left,pixel_left)[1]+dot(tAnaglyph_right,pixel_right)[1]);
            destination(i,j)[2] = static_cast<unsigned char>(dot(tAnaglyph_left,pixel_left)[2]+dot(tAnaglyph_right,pixel_right)[2]);
          }else if(processing_method == "Gray_Anaglyph"){
            std::vector<float> pixel_left = {
            static_cast<float>(destination_left(i,j)[0]),
            static_cast<float>(destination_left(i,j)[1]), 
            static_cast<float>(destination_left(i,j)[2])};
            std::vector<float> pixel_right = {
            static_cast<float>(destination_right(i,j)[0]),
            static_cast<float>(destination_right(i,j)[1]), 
            static_cast<float>(destination_right(i,j)[2])};
            destination(i,j)[0] = static_cast<unsigned char>(dot(gAnaglyph_left,pixel_left)[0]+dot(gAnaglyph_right,pixel_right)[0]);
            destination(i,j)[1] = static_cast<unsigned char>(dot(gAnaglyph_left,pixel_left)[1]+dot(gAnaglyph_right,pixel_right)[1]);
            destination(i,j)[2] = static_cast<unsigned char>(dot(gAnaglyph_left,pixel_left)[2]+dot(gAnaglyph_right,pixel_right)[2]);
          }else if(processing_method == "Color_Anaglyph"){
            std::vector<float> pixel_left = {
            static_cast<float>(destination_left(i,j)[0]),
            static_cast<float>(destination_left(i,j)[1]), 
            static_cast<float>(destination_left(i,j)[2])};
            std::vector<float> pixel_right = {
            static_cast<float>(destination_right(i,j)[0]),
            static_cast<float>(destination_right(i,j)[1]), 
            static_cast<float>(destination_right(i,j)[2])};
            destination(i,j)[0] = static_cast<unsigned char>(dot(cAnaglyph_left,pixel_left)[0]+dot(cAnaglyph_right,pixel_right)[0]);
            destination(i,j)[1] = static_cast<unsigned char>(dot(cAnaglyph_left,pixel_left)[1]+dot(cAnaglyph_right,pixel_right)[1]);
            destination(i,j)[2] = static_cast<unsigned char>(dot(cAnaglyph_left,pixel_left)[2]+dot(cAnaglyph_right,pixel_right)[2]);
          }else if(processing_method == "Half_Color_Anaglyph"){
            std::vector<float> pixel_left = {
            static_cast<float>(destination_left(i,j)[0]),
            static_cast<float>(destination_left(i,j)[1]), 
            static_cast<float>(destination_left(i,j)[2])};
            std::vector<float> pixel_right = {
            static_cast<float>(destination_right(i,j)[0]),
            static_cast<float>(destination_right(i,j)[1]), 
            static_cast<float>(destination_right(i,j)[2])};
            destination(i,j)[0] = static_cast<unsigned char>(dot(hcAnaglyph_left,pixel_left)[0]+dot(hcAnaglyph_right,pixel_right)[0]);
            destination(i,j)[1] = static_cast<unsigned char>(dot(hcAnaglyph_left,pixel_left)[1]+dot(hcAnaglyph_right,pixel_right)[1]);
            destination(i,j)[2] = static_cast<unsigned char>(dot(hcAnaglyph_left,pixel_left)[2]+dot(hcAnaglyph_right,pixel_right)[2]);
          }else if(processing_method == "Optimized_Anaglyph"){
            std::vector<float> pixel_left = {
            static_cast<float>(destination_left(i,j)[0]),
            static_cast<float>(destination_left(i,j)[1]), 
            static_cast<float>(destination_left(i,j)[2])};
            std::vector<float> pixel_right = {
            static_cast<float>(destination_right(i,j)[0]),
            static_cast<float>(destination_right(i,j)[1]), 
            static_cast<float>(destination_right(i,j)[2])};
            destination(i,j)[0] = static_cast<unsigned char>(dot(oAnaglyph_left,pixel_left)[0]+dot(oAnaglyph_right,pixel_right)[0]);
            destination(i,j)[1] = static_cast<unsigned char>(dot(oAnaglyph_left,pixel_left)[1]+dot(oAnaglyph_right,pixel_right)[1]);
            destination(i,j)[2] = static_cast<unsigned char>(dot(oAnaglyph_left,pixel_left)[2]+dot(oAnaglyph_right,pixel_right)[2]);
          }
  
          
        }

    }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Processed Image", destination );


  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;
  
  cv::waitKey();
  return 0;
}

