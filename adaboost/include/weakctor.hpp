#ifndef weaktor_hpp
#define weaktor_hpp
#include <vector>
#include <opencv2/core/core.hpp>

using std::vector;
using namespace cv;

struct WeakParams
{
  int x;
  int y;
  int width;
  int height;
  int radius;
  int neighbors;
  WeakParams(int x_, int y_, int w, int h, int r, int n)
    : x(x_),
      y(y_),
      width(w),
      height(h),
      radius(r),
      neighbors(n) {};
  WeakParams() {};
};

class WeakCtor 
{
  vector <Mat> samples;
  vector <int> labels;
  Mat mean;
  int size;
  vector<double> lutpos;
  vector<double> lutneg;
  double maxdist;
  WeakParams params;
  public:
  WeakCtor()
  {
  };
  WeakCtor(const vector<int> &labels)
    :
      labels(labels),
      size(labels.size())
  {};

  WeakParams getParams() const
  { return params; };
  void loadFeature(const vector<Mat> &samples, const WeakParams& p);
  void saveHist();
  double classify(const Mat& mat) const;
  void drawLut() const;
  int getSize() const
  { return size; }
  vector<double> getPos() const {return lutpos; };
  vector<double> getNeg() const {return lutneg; };
  Mat getMean() const {return mean; };
  double getMaxDist() const {return maxdist; };
  void setMean(const Mat & m) {mean = m; };
  void setPosNeg(const vector<double> pos, const vector<double> neg) {lutpos = pos; lutneg = neg; };
  void setMaxDist(double md) { maxdist = md; };
//  vector<double> weights();
  private:
  double fdr(const vector<double> &chisq);
  vector<double> calcLUThist(int label);
  void findOptimal();
  void meanHist();
  double chisquare(const Mat& sample) const;
  double calcMean(int label, const vector<double> &chisq);
  double calcDisp(int label, const vector<double> &chisq, double meanv);
  double meanDeriv(int label, int bin);
  double dispDeriv(int label, int bin, const vector<double> &chisq, double meanv, double meanderiv);
  double fdrDeriv(int bin, const vector<double> &chisq);
  vector<double> chisquares (const vector<Mat> & samples);
};
#endif
