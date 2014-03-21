#include "../include/lbph.hpp"
#include <ctime>
#include "../include/lbp.hpp"
#include <iostream>
#include <fstream>
#include "../include/evolution.hpp"

void LBPHm::boost(const vector<Triple> &src, double samples, bool multiscale) 
{
    const int numofweak = 12000;
    int samplessize = src.size()*samples;
    vector<vector<Mat> > features(src.size()*samples, vector<Mat>(numofweak));
    vector<WeakParams> params(numofweak, WeakParams());
    vector<int>_lbls(samplessize, 0);
#pragma omp parallel for
    for (int i = 0; i < samplessize; i++)
        _lbls[i] = src[i].label;
    weak = vector<WeakCtor>(numofweak, WeakCtor(_lbls));
    srand(std::time(NULL));
#pragma omp parallel for
    for (int i = 0; i < numofweak; i++)
    {
        int y = rand();// % (src[0].rows -4 - 5);
        int x = rand();// % (src[0].cols -4 - 5);
        int neighbors = rand() % 5 + 8;
        int radius;
        if (!multiscale) radius = rand() % 3 + 2;
        if (multiscale) radius = 0;
        int width = 0;
        int height = 0; 

        if (multiscale)
        {
            y = y % (src[0].img.rows -(5*2)- 5);
            x = x % (src[0].img.rows -(5*2) - 5);
            width = (rand() % (src[0].img.cols - (5*2)- x - 5) + 5);
            height = (rand() % (src[0].img.rows -(5*2) - y - 5) + 5);
            neighbors = 8;
        } else {
            y = y % (src[0].img.rows -(radius*2)- 5);
            x = x % (src[0].img.rows -(radius*2) - 5);
            width = (rand() % (src[0].img.cols - (radius*2)- x - 5) + 5);
            height = (rand() % (src[0].img.rows -(radius*2) - y - 5) + 5);
        }
        params[i] = WeakParams(x, y, width, height, radius, neighbors);
    }

#pragma omp parallel for
    for(size_t sampleIdx = 0; sampleIdx < samplessize; sampleIdx++)
    {
        vector<Mat> lbp_image(15);  

        vector<Mat> lbp_multiscale(5);
        for (int i = 0; i < 5; i++)
            lbp_multiscale[i] = elbp(src[sampleIdx].img, 1+i, 8);

        for (int i = 0; i < numofweak; i++)
        {
            int y = params[i].y;
            int x = params[i].x;
            int width = params[i].width;
            int height = params[i].height;
            int neighbors = params[i].neighbors;
            int radius = params[i].radius;
            double uniformnum = calcNumOfUniform(static_cast<int>(std::pow(2.0, static_cast<double>(neighbors)))); /* number of possible patterns */

            Mat cell;
            if (multiscale)
            {
                vector<Mat> src_cell(5);
                vector<Mat> cell_hist(5);
                for (int k = 0; k < 5; k++)
                {
                    src_cell[k] = Mat(lbp_multiscale[k], Rect(x, y, width, height));
                    cell_hist[k] = histc(src_cell[k], 0, (uniformnum-1), true);
                    if (k > 0 && k == 1)
                        hconcat(cell_hist[0], cell_hist[1], cell);
                    else if (k > 1)
                        hconcat(cell, cell_hist[k], cell);

                }
            } else {
                Mat src_cell = Mat(lbp_image[(radius - 2) * 5 + (neighbors - 8)], Rect(x, y, width, height));
                cell = histc(src_cell, 0, (uniformnum-1), true);
            }
            features[sampleIdx][i] = cell;
        }
	for (auto it = lbp_multiscale.begin(); it < lbp_multiscale.end(); it++)
		it->release();
    }

    vector<vector<Mat> > transpose(features[0].size(), vector<Mat>(samplessize));
    for (int i = 0; i < features.size(); i++)
        for (int j = 0; j < features[i].size(); j++)
            transpose[j][i] = features[i][j];

    std::cout << "weak finding" << std::endl;
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < transpose.size(); i++)
        weak[i].loadFeature(transpose[i], params[i]);

    std::cout << "adaboost start" << std::endl;
    ada = AdaBoost(src, weak, samplessize);
    ada.train(features);    
}

void LBPHm::evoboost(const vector<Triple> src, double samples) 
{
    const int numofweak = 4000;
    int samplessize = src.size()*samples;
    vector<vector<Mat> > features(samplessize, vector<Mat>(numofweak));

    vector<int>_lbls(src.size()*samples, 0);
    for (int i = 0; i < samplessize; i++)
        _lbls[i] = src[i].label;

    weak = vector<WeakCtor>(numofweak, WeakCtor(_lbls));

    vector<WeakParams> params = ada.getParams();
    WeakEvo evo(params);
    params = evo.generate(numofweak, src[0].img.rows - 14, src[0].img.cols - 14);

    std::cout << "evolution started" << std::endl;
#pragma omp parallel for
    for(size_t sampleIdx = 0; sampleIdx < samplessize; sampleIdx++)
    {
        vector<Mat> lbp_image(15);  
        vector<Mat> lbp_multiscale(5);

        for (int i = 0; i < 5; i++)
            lbp_multiscale[i] = elbp(src[sampleIdx].img, 1+i, 8);

        for (int i = 0; i < numofweak; i++)
        {
            int y = params[i].y;
            int x = params[i].x;
            int width = params[i].width;
            int height = params[i].height;
            int neighbors = params[i].neighbors;
            int radius = params[i].radius;
            double uniformnum = calcNumOfUniform(static_cast<int>(std::pow(2.0, static_cast<double>(neighbors)))); /* number of possible patterns */

            Mat cell;
            if (radius == 0)
            {
                vector<Mat> src_cell(5);
                vector<Mat> cell_hist(5);
                for (int k = 0; k < 5; k++)
                {
                    src_cell[k] = Mat(lbp_multiscale[k], Rect(x, y, width, height));
                    cell_hist[k] = histc(src_cell[k], 0, (uniformnum-1), true);
                    if (k > 0 && k == 1)
                        hconcat(cell_hist[0], cell_hist[1], cell);
                    else if (k > 1)
                        hconcat(cell, cell_hist[k], cell);

                }
            } else {
                Mat src_cell = Mat(lbp_image[(radius - 2) * 5 + (neighbors - 8)], Rect(x, y, width, height));
                cell = histc(src_cell, 0, (uniformnum-1), true);
            }
            features[sampleIdx][i] = cell;
        }
    }

    vector<vector<Mat> > transpose(features[0].size(), vector<Mat>(samplessize));
    for (int i = 0; i < features.size(); i++)
        for (int j = 0; j < features[i].size(); j++)
            transpose[j][i] = features[i][j];
    std::cout << "weak finding" << std::endl;
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < transpose.size(); i++)
        weak[i].loadFeature(transpose[i], params[i]);

    std::cout << "adaboost start" << std::endl;
    evoAda = AdaBoost(src, weak, samplessize);
    evoAda.train(features);    
}


vector<int> LBPHm::weakpredictor(const Mat & img) const
{
    Mat _img;
    img.copyTo(_img);
    vector <double> predictions(0);
    vector <WeakParams> params = ada.getParams();
    vector <Mat> cells(0);
    for (vector<WeakParams>::iterator it = params.begin(); it < params.end(); it++)
    {
        if (it->radius)
        {
            Mat lbp_image = elbp(_img, it->radius, it->neighbors);
            Mat src_cell = Mat(lbp_image, Rect(it->x, it->y, it->width, it->height));
            double uniformnum = calcNumOfUniform(static_cast<int>(std::pow(2.0, static_cast<double>(it->neighbors)))); /* number of possible patterns */
            Mat cell_hist = histc(src_cell, 0, (uniformnum-1), true);
            cells.push_back(cell_hist);
            //std::cout << it->x << " " << it->y << std::endl;
        } else {
            Mat cell;
            vector<Mat> lbp_multiscale(5);
            for (int i = 0; i < 5; i++)
                lbp_multiscale[i] = elbp(_img, 1+i, 8);

            vector<Mat> src_cell(5);
            vector<Mat> cell_hist(5);
            double uniformnum = calcNumOfUniform(static_cast<int>(std::pow(2.0, static_cast<double>(it->neighbors)))); /* number of possible patterns */
            for (int k = 0; k < 5; k++)
            {
                src_cell[k] = Mat(lbp_multiscale[k], Rect(it->x, it->y, it->width, it->height));
                cell_hist[k] = histc(src_cell[k], 0, (uniformnum-1), true);
                if (k > 0 && k == 1)
                    hconcat(cell_hist[0], cell_hist[1], cell);
                else if (k > 1)
                    hconcat(cell, cell_hist[k], cell);
            }
            cells.push_back(cell);
        }
    }
    return ada.predict(cells); 
}

int LBPHm::fastpredictor(const Mat & img) const
{
    Mat _img;
    img.copyTo(_img);
    vector <double> predictions(0);
    vector <WeakParams> params = ada.getParams();
    vector <Mat> cells(0);
    for (vector<WeakParams>::iterator it = params.begin(); it < params.end(); it++)
    {
        if (it->radius)
        {
            Mat lbp_image = elbp(_img, it->radius, it->neighbors);
            Mat src_cell = Mat(lbp_image, Rect(it->x, it->y, it->width, it->height));
            double uniformnum = calcNumOfUniform(static_cast<int>(std::pow(2.0, static_cast<double>(it->neighbors)))); /* number of possible patterns */
            Mat cell_hist = histc(src_cell, 0, (uniformnum-1), true);
            cells.push_back(cell_hist);
            //std::cout << it->x << " " << it->y << std::endl;
        } else {
            Mat cell;
            vector<Mat> lbp_multiscale(5);
#pragma omp parallel for
            for (int i = 0; i < 5; i++)
                lbp_multiscale[i] = elbp(_img, 1+i, 8);

            vector<Mat> src_cell(5);
            vector<Mat> cell_hist(5);
            double uniformnum = calcNumOfUniform(static_cast<int>(std::pow(2.0, static_cast<double>(it->neighbors)))); /* number of possible patterns */
            for (int k = 0; k < 5; k++)
            {
                src_cell[k] = Mat(lbp_multiscale[k], Rect(it->x, it->y, it->width, it->height));
                cell_hist[k] = histc(src_cell[k], 0, (uniformnum-1), true);
                if (k > 0 && k == 1)
                    hconcat(cell_hist[0], cell_hist[1], cell);
                else if (k > 1)
                    hconcat(cell, cell_hist[k], cell);
            }
            cells.push_back(cell);
        }
    }
    return ada.singlepredict(cells); 
}



vector<int> LBPHm::evopredictor(const Mat & img) const
{
    Mat _img;
    img.copyTo(_img);
    vector <double> predictions(0);
    vector <WeakParams> params = evoAda.getParams();
    vector <Mat> cells(0);
    for (vector<WeakParams>::iterator it = params.begin(); it < params.end(); it++)
    {
        if (it->radius)
        {
            Mat lbp_image = elbp(_img, it->radius, it->neighbors);
            Mat src_cell = Mat(lbp_image, Rect(it->x, it->y, it->width, it->height));
            double uniformnum = calcNumOfUniform(static_cast<int>(std::pow(2.0, static_cast<double>(it->neighbors)))); /* number of possible patterns */
            Mat cell_hist = histc(src_cell, 0, (uniformnum-1), true);
            cells.push_back(cell_hist);
            //std::cout << it->x << " " << it->y << std::endl;
        } else {
            Mat cell;
            vector<Mat> lbp_multiscale(5);
            for (int i = 0; i < 5; i++)
                lbp_multiscale[i] = elbp(_img, 1+i, 8);

            vector<Mat> src_cell(5);
            vector<Mat> cell_hist(5);
            double uniformnum = calcNumOfUniform(static_cast<int>(std::pow(2.0, static_cast<double>(it->neighbors)))); /* number of possible patterns */
            for (int k = 0; k < 5; k++)
            {
                src_cell[k] = Mat(lbp_multiscale[k], Rect(it->x, it->y, it->width, it->height));
                cell_hist[k] = histc(src_cell[k], 0, (uniformnum-1), true);
                if (k > 0 && k == 1)
                    hconcat(cell_hist[0], cell_hist[1], cell);
                else if (k > 1)
                    hconcat(cell, cell_hist[k], cell);
            }
            cells.push_back(cell);
        }
    }
    return evoAda.predict(cells); 
}


void LBPHm::save(std::string filename)
{
    vector<WeakParams> params = ada.getParams();
    std::ofstream file(filename.c_str(), std::ofstream::out );

    file << ada.getSize() << std::endl;
    file << params.size() << std::endl;
    for (vector<WeakParams>::const_iterator it = params.begin(); it < params.end(); it++)
        file << it->x << " " << it->y << " " << it-> width << " " << it->height << " " << it->radius << " " << it->neighbors << std::endl;
    vector<double> al = ada.getAlpha();
    file << al.size() << std::endl;
    for (vector<double>::const_iterator it = al.begin(); it < al.end(); it++)
        file << *it << " ";
    file << std::endl;
    vector<WeakCtor> pref = ada.getPref();
    file << pref.size() << std::endl;
    for (vector<WeakCtor>::const_iterator it = pref.begin(); it < pref.end(); it++)
    {
        vector<double> pos = it->getPos();
        vector<double> neg = it->getNeg();
        Mat m = it->getMean();
        double md = it->getMaxDist();
        file << md << std::endl;
        file << m.cols << std::endl;
        for (int i = 0; i < m.cols; i++)
            file << m.at<float>(0, i) << " ";
        file << std::endl;
        for (int i = 0; i < 32; i++)
            file << pos[i] << " ";
        file << std::endl;
        for (int i = 0; i < 32; i++)
            file << neg[i] << " ";
        file << std::endl;
    }
}

void LBPHm::saveevo(std::string filename)
{
    vector<WeakParams> params = evoAda.getParams();
    std::ofstream file(filename.c_str(), std::ofstream::out );

    file << evoAda.getSize() << std::endl;
    file << params.size() << std::endl;
    for (vector<WeakParams>::const_iterator it = params.begin(); it < params.end(); it++)
        file << it->x << " " << it->y << " " << it-> width << " " << it->height << " " << it->radius << " " << it->neighbors << std::endl;
    vector<double> al = evoAda.getAlpha();
    file << al.size() << std::endl;
    for (vector<double>::const_iterator it = al.begin(); it < al.end(); it++)
        file << *it << " ";
    file << std::endl;
    vector<WeakCtor> pref = evoAda.getPref();
    file << pref.size() << std::endl;
    for (vector<WeakCtor>::const_iterator it = pref.begin(); it < pref.end(); it++)
    {
        vector<double> pos = it->getPos();
        vector<double> neg = it->getNeg();
        Mat m = it->getMean();
        double md = it->getMaxDist();
        file << md << std::endl;
        file << m.cols << std::endl;
        for (int i = 0; i < m.cols; i++)
            file << m.at<float>(0, i) << " ";
        file << std::endl;
        for (int i = 0; i < 32; i++)
            file << pos[i] << " ";
        file << std::endl;
        for (int i = 0; i < 32; i++)
            file << neg[i] << " ";
        file << std::endl;
    }
}


void LBPHm::load(const std::string &filename)
{
    int paramssize;
    int size;
    std::ifstream file(filename.c_str(), std::ifstream::in);
    file >> size;
    file >> paramssize;
    vector<WeakParams> params(paramssize);
    for (int i = 0; i < paramssize; i++)
    {
        int x, y, width, height, radius, neighbors;
        file >> x;
        file >> y;
        file >> width;
        file >> height;
        file >> radius;
        file >> neighbors;
        params[i] = WeakParams(x, y, width, height, radius, neighbors);
    }
    int alsize;
    file >> alsize;
    vector<double> al(alsize);
    for (int i = 0; i < alsize; i++)
        file >> al[i];
    int prefsize;
    file >> prefsize;
    vector<WeakCtor> pref(prefsize);
    //done
    for (int i = 0; i < prefsize; i++)
    {
        vector<double> pos(32);
        vector<double> neg(32);
        int mcols;
        double md;
        file >> md;
        file >> mcols;
        Mat m;
        vector<float> mf(mcols);
        for (int j = 0; j < mcols; j++)
            file >> mf[j];
        m = Mat(mf);
        m = m.t();
        for (int j = 0; j < 32; j++)
            file >> pos[j];
        for (int j = 0; j < 32; j++)
            file >> neg[j];
        pref[i].setMean(m);
        pref[i].setPosNeg(pos, neg);
        pref[i].setMaxDist(md);
    }
    ada = AdaBoost(params, pref, al, size);
    evoAda = AdaBoost(params, pref, al, size);
}
