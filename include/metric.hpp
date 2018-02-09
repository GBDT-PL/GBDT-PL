//
//  metrics.hpp
//  LinearGBM
//


//

#ifndef metrics_hpp
#define metrics_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include "booster_config.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm> 

using std::vector;
using std::string; 

struct ScoreIndexPair {
    ScoreIndexPair(double _score, int _index): score(_score), index(_index) {}
    
    double score;
    int index;
};

struct ScoreIndexRel {
    ScoreIndexRel(double _score, int _index, double _rel): score(_score), index(_index), rel(_rel) {}
    
    double score;
    int index;
    double rel;
};

class Metric {
protected:
    BoosterConfig booster_config;
    vector<double> &labels;
    vector<double> &scores;
public:
    string name;    
    Metric(const BoosterConfig &booster_configg, vector<double> &labelss, vector<double> &scoress); 
    virtual double Eval() = 0;
};

class AUC: public Metric {
private:
public:
    AUC(const BoosterConfig &booster_config, vector<double> &labelss, vector<double> &scoress);
    double Eval();
};

class RMSE: public Metric {
private:
public:
    RMSE(const BoosterConfig &booster_config, vector<double> &labelss, vector<double> &scoress);
    double Eval();
};

class NDCG: public Metric {
private:
    vector<int> &queries;
    vector<int> query_boundaries;
    double NDCGAt(const double *rels, const double *scores, int k, int n);
    double DCG(const double* rels, const double* scores, int k, int n); 
public:
    NDCG(const BoosterConfig &booster_config, vector<double> &labelss,
         vector<double> &scoress, vector<int> &queries);
    double Eval();
};

#endif /* metrics_hpp */
