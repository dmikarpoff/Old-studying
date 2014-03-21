#ifndef adaboost_hpp
#define adaboost_hpp
#include "utils.hpp"
#include <vector>
#include "weakctor.hpp"

class AdaBoost
{
  vector<Triple> triples;
  vector<WeakCtor> weak;
  vector<WeakCtor> prefclass;
  vector<double> betha;
  vector<double> alpha;
  vector<WeakParams> params;
  int size;
  int triplessize;
  double alphas;
  public:
  AdaBoost(const vector<Triple>& src, const vector<WeakCtor>& weak_, int tsize)
    : triples(src),
      triplessize(tsize),
      weak(weak_)
      {
      for (vector<WeakCtor>::const_iterator it = weak_.begin(); it < weak_.end(); it++)
        params.push_back(it->getParams());
      };
  AdaBoost() {};
  AdaBoost(const vector<WeakParams> &par, const vector<WeakCtor> &pref, const vector<double> &al, int sz)
  : params(par),
    prefclass(pref),
    alpha(al),
    size(sz)
  { 
    alphas = 0;
    for (int i = 0; i < al.size(); i++)
    alphas += al[i];
  } ;
  void train(const vector< vector< Mat > > & features);
  vector<int> predict(const vector<Mat>& cells) const;
  int singlepredict(const vector<Mat>& cells) const;
  vector<WeakParams> getParams() const
  { return params; };
  vector<double> getAlpha() {return alpha; };
  vector<WeakCtor> getPref() { return prefclass; };
  int getSize() { return size; };
};
#endif
