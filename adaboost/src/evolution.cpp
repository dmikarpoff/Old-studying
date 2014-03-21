#include "../include/evolution.hpp"
#include <ctime>

vector<WeakParams> WeakEvo::generate(int size, int imrows, int imcols)
{
  srand(time(NULL));
  vector<WeakParams> res(size);
  int firsgensize = firstgen.size();
  for (int i = 0; i < size; i++)  
  {
    int choice = rand() % firsgensize;
    int randparam = rand() % 4;
    int x = firstgen[choice].x;
    int y = firstgen[choice].y;
    int h = firstgen[choice].height;
    int w = firstgen[choice].width;
    int randaddition = rand() % 7 - 3;
    switch (randparam)
    {
      case 0:
        if (x + w + randaddition < imcols && x + w + randaddition >0 && x + randaddition >= 0)
          x += randaddition;
        break;
      case 1:
        if (y + h + randaddition < imrows && y + h + randaddition >0 && y + randaddition >=0)
          y += randaddition;
        break;
      case 2:
        if (x + w + randaddition < imcols && x + w + randaddition >0 && w + randaddition >= 2)
          w += randaddition;
        break;
      case 3:
        if (y + h + randaddition < imrows && y + h + randaddition >0 && h + randaddition >= 2)
          h += randaddition;
        break;
    }
    res[i] = WeakParams(x, y, w, h, 0, 8);
  }
  for (int i = 0; i < firsgensize; i++)
    res.push_back(firstgen[i]);
  return res;
}
