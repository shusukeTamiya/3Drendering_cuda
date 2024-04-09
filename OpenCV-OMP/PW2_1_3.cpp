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



std::vector<float> convert_to_float(const cv::Vec<unsigned char, 3>& charVec){
  std::vector<float> result(3, 0.0f);
  result[0] = static_cast<float>(charVec[0]);
  result[1] = static_cast<float>(charVec[1]);
  result[2] = static_cast<float>(charVec[2]);
  return result;
}

std::vector<float> sum_function(std::vector<float> bgr, std::vector<float> sum){
  std::vector<float> result(3, 0.0f);
  result[0] = bgr[0]+sum[0];
  result[1] = bgr[1]+sum[1];
  result[2] = bgr[2]+sum[2];
  return result;
}

std::vector<float> difference_function(std::vector<float> bgr, std::vector<float> ave){
  std::vector<float> result(3, 0.0f);
  result[0] = bgr[0]-ave[0];
  result[1] = bgr[1]-ave[1];
  result[2] = bgr[2]-ave[2];
  return result;
}

std::vector<std::vector<float>> covariance_function(const std::vector<std::vector<float>>& difference, int neighbor){
  std::vector<std::vector<float>> result(3, std::vector<float>(3, 0.0f));
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      for(int k=0;k<neighbor*neighbor;k++){
        result[i][j] += difference[k][i]*difference[k][j];
      }
      result[i][j]/=neighbor*neighbor;
    }
  }
  return result;
}

float determinant_function(const std::vector<std::vector<float>>& matrix){
  float result = 0.0f;
  result = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  return result;
}

int calculate_kernel(float determinant, float ratio){
  return 2+static_cast<int>(ratio/(0.01*(determinant*determinant)));
}


cv::Mat_<cv::Vec3b> gaussian(cv::Mat_<cv::Vec3b> matrix, int neighbor, float r) {
    cv::Mat_<cv::Vec3b> result(matrix.rows, matrix.cols);
    cv::Mat_<cv::Vec3b> temp(matrix.rows, matrix.cols);
    cv::Mat_<cv::Vec3b> temp2(neighbor, neighbor);
    std::vector<float> sum(3, 0.0f);
    std::vector<float> ave(3, 0.0f);
    std::vector<std::vector<float>> difference(neighbor*neighbor, std::vector<float>(3, 0.0f));
    std::vector<std::vector<float>> covariance(3, std::vector<float>(3, 0.0f));
    float weight = 0.0;
    float weight_sum = 0.0;
    float sigma = 3.0;
    int kernel_size = 1;
    float determinant = 0.0f;
    
    for (int i = 0; i < matrix.rows; ++i) {
      for (int j = 0; j < matrix.cols; ++j){
        // calculate average
        for(int k = -neighbor/2; k < neighbor/2; ++k){
          for(int l = -neighbor/2; l < neighbor/2; ++l){
              if(((i+k<0)&&(j+l<0))||((i+k<0)&&((l+j)>matrix.cols/2))||
              (((i+k)>(matrix.rows))&&((l+j)<0)) || ((i+k)>matrix.rows)&&((l+j)>matrix.cols/2)){
                sum = sum_function(convert_to_float(matrix(i,j)),sum);
              }else if(((i+k)<0) || ((i+k)>matrix.rows)){
                sum = sum_function(convert_to_float(matrix(i,j+l)),sum);
              }else if(((j+l)<0) || ((j+l)>matrix.cols/2)){
                sum = sum_function(convert_to_float(matrix(i+k,j)),sum);
              }else{
                sum = sum_function(convert_to_float(matrix(i+k,j+k)),sum);
              }
            }
            ave[0] = sum[0]/static_cast<float>((neighbor*neighbor));
            ave[1] = sum[1]/static_cast<float>((neighbor*neighbor));
            ave[2] = sum[2]/static_cast<float>((neighbor*neighbor));
            sum[0] = 0.0f;
            sum[1] = 0.0f;
            sum[2] = 0.0f;
          }

        // determinate difference between target and average
        for(int k = -neighbor/2; k < neighbor/2; ++k){
          for(int l = -neighbor/2; l < neighbor/2; ++l){
            if(j<matrix.cols/2){
              if(((i+k<0)&&(j+l<0))||((i+k<0)&&((l+j)>matrix.cols/2))||
              (((i+k)>(matrix.rows/2))&&((l+j)<0)) || ((i+k)>matrix.rows/2)&&((l+j)>matrix.cols/2)){
                difference[k+l+neighbor] = difference_function(convert_to_float(matrix(i,j)),ave);
              }else if(((i+k)<0) || ((i+k)>matrix.rows/2)){
                difference[k+l+neighbor] = difference_function(convert_to_float(matrix(i,j+l)),ave);
              }else if(((j+l)<0) || ((j+l)>matrix.cols/2)){
                difference[k+l+neighbor] = difference_function(convert_to_float(matrix(i+k,j)),ave);
              }else{
                difference[k+l+neighbor] = difference_function(convert_to_float(matrix(i+k,j+l)),ave);
              }
            }else{
              if(((i+k<matrix.rows/2)&&(j+l<matrix.rows/2))||((i+k<matrix.rows/2)&&((l+j)>matrix.cols))||
              (((i+k)>(matrix.rows))&&((l+j)<matrix.rows/2)) || ((i+k)>matrix.rows)&&((l+j)>matrix.cols)){
                difference[k+l+neighbor] = difference_function(convert_to_float(matrix(i,j)),ave);
              }else if(((i+k)<matrix.rows/2) || ((i+k)>matrix.rows)){
                difference[k+l+neighbor] = difference_function(convert_to_float(matrix(i,j+l)),ave);
              }else if(((j+l)<matrix.rows/2) || ((j+l)>matrix.cols)){
                difference[k+l+neighbor] = difference_function(convert_to_float(matrix(i+k,j)),ave);
              }else{
                difference[k+l+neighbor] = difference_function(convert_to_float(matrix(i+k,j+l)),ave);
              }
            }
          }
        }
        // calculate the covariance
        covariance = covariance_function(difference,neighbor);

        // calculate the determinant
        determinant = abs(determinant_function(covariance));

        // caluculate the kernel_size
        kernel_size = calculate_kernel(determinant,r);
        if(kernel_size>31 || kernel_size<0){
          kernel_size = 31;
        }

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
  int neighbor = std::stoi(argv[2]);
  float ratio = std::stof(argv[3]);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;
  
#pragma omp parallel for
  for (int it=0;it<iter;it++){
    destination = gaussian(source, neighbor, ratio); 
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

