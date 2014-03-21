#ifndef lbph_hpp
#define lbph_hpp
#include <vector>
#include <opencv2/core/core.hpp>
#include "weakctor.hpp"
#include "adaboost.hpp"
using namespace cv;

class LBPHm
{
private:
    std::vector<cv::Mat> _histograms;
    cv::Mat _labels;
    vector <WeakCtor> weak;
    AdaBoost ada;
    AdaBoost evoAda;

public:

    LBPHm() {};
    ~LBPHm() { }

    void boost(const vector<Triple> &src, double samples, bool multiscale); 

    void evoboost(const vector<Triple> src, double samples); 

    vector<int> weakpredictor(const Mat & img) const;
    
    int fastpredictor(const Mat & img) const;

    vector<int> evopredictor(const Mat & img) const;

    void save(std::string filename);

    void saveevo(std::string filename);

    void load(const std::string & filename);
};

#endif
