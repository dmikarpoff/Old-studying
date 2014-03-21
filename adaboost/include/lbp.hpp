#ifndef lbp_hpp
#define lbp_hpp
#include <opencv2/core/core.hpp>
using namespace cv;

bool isUniform(uint x);

uint calcNumOfUniform(uint numOfPatterns);

int stupidNumOfTr(uint x);

void drawHist(const Mat& hist, int histSize, string name);

int getUniformMask(const Mat& img, Mat &res);

template <typename _Tp> 
void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors); 

void elbp(InputArray src, OutputArray dst, int radius, int neighbors);

Mat histc_(const Mat& src, int minVal, int maxVal, bool normed);

Mat histc(InputArray _src, int minVal, int maxVal, bool normed);

Mat spatial_histogram(InputArray _src, int numPatterns, int grid_x, int grid_y, bool /*normed*/);

Mat elbp(InputArray src, int radius, int neighbors);
#endif
