#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>  // for high_resolution_clock

using namespace std;
#define PI 3.141952f
#define E 2.71828f


std::vector<float> dot(const float A[3][3], const std::vector<float>& B) {
    std::vector<float> result(3, 0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += A[i][j] * static_cast<float>(B[j]);
        }
    }

    return result;
}

float gaussian_equation(float sigma, float x, float y){
        return pow(E,-((x*x)+(y*y))/(2.*sigma*sigma))/(2. * PI * sigma*sigma);
      }

cv::Mat_<cv::Vec3b> gaussian(cv::Mat_<cv::Vec3b> matrix, int kernel_size, float sigma) {
    cv::Mat_<cv::Vec3b> result(matrix.rows, matrix.cols);
    cv::Mat_<cv::Vec3b> temp(matrix.rows, matrix.cols);
    float weight = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < matrix.rows; ++i) {
      for (int j = 0; j < matrix.cols; ++j){
        
        for(int k = -kernel_size/2; k < kernel_size/2; ++k){
          for(int l = -kernel_size/2; l < kernel_size/2; ++l){
            weight = gaussian_equation(sigma, k,l);
            if(j<matrix.cols/2){
              if(((i+k<0)&&(j+l<0))||((i+k<0)&&((l+j)>matrix.cols/2))||
              (((i+k)>(matrix.rows/2))&&((l+j)<0)) || ((i+k)>matrix.rows/2)&&((l+j)>matrix.cols/2)){
                temp(i,j) += matrix(i,j)*weight;
              }else if(((i+k)<0) || ((i+k)>matrix.rows/2)){
                temp(i,j) += matrix(i,j+l)*weight;
              }else if(((j+l)<0) || ((j+l)>matrix.cols/2)){
                temp(i,j) += matrix(i+k,j)*weight;
              }else{
                temp(i,j) += matrix(i+k,j+l)*weight;
              }
            }else{
              if(((i+k<matrix.rows/2)&&(j+l<matrix.rows/2))||((i+k<matrix.rows/2)&&((l+j)>matrix.cols))||
              (((i+k)>(matrix.rows))&&((l+j)<matrix.rows/2)) || ((i+k)>matrix.rows)&&((l+j)>matrix.cols)){
                temp(i,j) += matrix(i,j)*weight;
              }else if(((i+k)<matrix.rows/2) || ((i+k)>matrix.rows)){
                temp(i,j) += matrix(i,j+l)*weight;
              }else if(((j+l)<matrix.rows/2) || ((j+l)>matrix.cols)){
                temp(i,j) += matrix(i+k,j)*weight;
              }else{
                temp(i,j) += matrix(i+k,j+l)*weight;
              }
            }
            
            weight_sum += weight;
          }
        }
        result(i,j) = temp(i,j)/weight_sum;
        weight_sum=0.0;

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
  //std::string processing_method = argv[2];
  int kernel_size = std::stoi(argv[2]);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;
  
#pragma omp parallel for
  for (int it=0;it<iter;it++){
    destination = gaussian(source, kernel_size, 1.0); 
    cout << "IPS: " << it << endl;
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

