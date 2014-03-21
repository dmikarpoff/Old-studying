#ifndef evolution_hpp
#define evolution_hpp
#include <vector>
#include "weakctor.hpp"
using std::vector;
class WeakEvo
{
private:
  vector<WeakParams> firstgen;

public:
  WeakEvo(const vector<WeakParams> & pref)
    : firstgen(pref) {};

  vector<WeakParams> generate(int size, int imrows, int imcols);
};

#endif
