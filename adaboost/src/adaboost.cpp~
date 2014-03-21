#include "../include/adaboost.hpp"
#include <iostream>

void AdaBoost::train(const vector<vector<Mat> > &features)
{
  vector <vector<Mat> > cells = features;

  int neg = 0;
  for (int i = 0; i < triplessize; i++)
    if (triples[i].label == 0)
      neg++;
//  vector<double> weights(labels.size(), 1/(double)(labels.size()));
  vector<double> weights(triplessize, 1/2.0/(triplessize-neg));
  for (int i = 0; i < triplessize; i++)
    if (triples[i].label == 0)
      weights[i] = 1/2.0/neg;

  int preferred = 0;
  double preverror = 0;
  size = 50;
  vector<int> used;

  for (int i = 0; i < size; i++)
  {
    double norm = 0;
    for (int t = 0; t < weights.size(); t++)
      norm += weights[t];

    for (int t = 0; t < weights.size(); t++)
      weights[t] = weights[t] / norm;

    double error = DBL_MAX;
    vector<int> correctness(triplessize);

    bool flag = false;
#pragma omp parallel for
    for (int k = 0; k < weak.size(); k++)
    {
      double currerror = 0;
      for (int j = 0; j < triplessize; j++)
      {
        double corr = weak[k].classify(cells[j][k]);
        currerror += weights[j] * std::fabs(corr - triples[j].label);
      }
      if (currerror < error)
      {
        error = currerror;
        preferred = k;
      }
    }
    for (int j = 0; j < triplessize; j++)
    {
      double corr = weak[preferred].classify(cells[j][preferred]);
      if (corr > 1/2.0 && triples[j].label == 1)
        correctness[j] = 1;
      else if (corr < 1/2.0 && triples[j].label == 0)
        correctness[j] = 1;
      else correctness[j] = 0;
    }
for (int g = 0; g < used.size(); g++)
      if (used[g] == preferred)
        flag = true;
    
    if (!flag) std::cout << "round " << i << " error: " << error << " preferred: " << preferred << " " << std::endl;
//    if (flag) std::cout << "not used" << std::endl;

    if (!flag)
    {
      prefclass.push_back(weak[preferred]);
      betha.push_back(error / (1 - error));
      used.push_back(preferred);
    }
    for (int j = 0; j < weights.size(); j++)
      if (correctness[j])
        weights[j] = weights[j] * (error / (1 - error));
  }
  alphas = 0;
  for (int i = 0; i < betha.size(); i++)
  {
    alpha.push_back(-std::log(betha[i]));
    alphas += alpha[i];
  }
  params.erase(params.begin(), params.end());
  for (int i = 0; i < prefclass.size(); i++)
  {
    params.push_back(prefclass[i].getParams());
//    if (i < 10)
//      prefclass[i].drawLut();
  }
  size = prefclass.size();
}


vector<int> AdaBoost::predict(const vector<Mat>& sample) const
{
  vector<int> predictions(0);
  double res = 0;  
  double al = 0;
   for (int i = 0; i < size; i++)
   {
     res += alpha[i] * prefclass[i].classify(sample[i]);
     al += alpha[i];
//     if (i % 3 == 0)
       if (res >= 1/2.0*al)
       		predictions.push_back(1);
       else 
			predictions.push_back(0);
   }
   return predictions;
//   if (res >= 1/2.0*alphas)
//     return 1;
//   else return 0;
}

int AdaBoost::singlepredict(const vector<Mat> & sample) const
{
  double res = 0;  
#pragma omp parallel for
  for (int i = 0; i < size; i++)
  {
    res += alpha[i] * prefclass[i].classify(sample[i]);
  }
	std::cout << res / alphas << std::endl;
    if (res >= 1/2.0*alphas)
      return 1;
    else return 0;
}
