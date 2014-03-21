#ifndef UTILS
#define UTILS
#include <opencv2/core/core.hpp>

using namespace cv;
struct Triple
{
  Mat img;
  int num;
  int label;
  Triple(const Mat &img, int num, int label)
    : img(img),
      num(num),
      label(label)
  {};
};

struct Face
{
  static int ids;
  Rect r;
  bool available;
  bool del;
  int timer;
  int id;
  int duration;
  int male;
  int female;
  Face(Rect r_)
    : r(r_),
      available(true),
      del(false),
      timer(50),
      male(0),
      female(0),
      duration(0)
  {
      id = ids++;
  };
  Face() {};
  Face(const Face& face)
    : r(face.r),
      available(face.available),
      del(face.del),
      timer(face.timer),
      id(face.id),
      duration(face.duration),
      male(face.male),
      female(face.female)
  {};
};

double dist(Rect r1, Rect r2);

Mat align(const Mat& img_, Rect face, bool test);

void read_csv(vector<Triple>& triples, char separator = ';');

void help();

void detectAndDisplay(Mat&& frame);
#endif
