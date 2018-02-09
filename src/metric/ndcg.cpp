//
//  ndcg.cpp
//  LinearGBMVector
//


//

#include <stdio.h>
#include "metric.hpp"
#include <cassert>

NDCG::NDCG(const BoosterConfig &booster_config, vector<double> &labelss,
           vector<double> &scoress, vector<int> &queriess):
Metric(booster_config, labelss, scoress), queries(queriess) {
    name = "ndcg";
    query_boundaries.push_back(0);
    int prevq = queries[0];
    for(int i = 0; i < queries.size(); ++i) {
        if(queries[i] != prevq) {
            query_boundaries.push_back(i);
        }
        prevq = queries[i]; 
    }
}

double NDCG::Eval() {
    int num_data = static_cast<int>(labels.size());
    vector<double> ndcg_probs(num_data, 0.0);
    vector<double> ndcg_scores(num_data, 0.0);
    vector<double> ndcg_max_scores(num_data, 0.0);
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int k = 0; k < num_data; ++k) {
        double max_score = scores[k];
        for(int s = 1; s < booster_config.num_classes; ++s) {
            if(scores[s * num_data + k] > max_score) {
                max_score = scores[s * num_data + k];       
            }
        }
        ndcg_max_scores[k] = max_score;
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int k = 0; k < num_data; ++k) {
        ndcg_probs[k] = 0.0;
        for(int s = 0; s < booster_config.num_classes; ++s) {
            ndcg_probs[k] += exp(scores[s * num_data + k] - ndcg_max_scores[k]);    
        }
    }
    
#pragma omp parallel for schedule(static) num_threads(booster_config.num_threads)
    for(int k = 0; k < num_data; ++k) {
        ndcg_scores[k] = 0.0;
        for(int s = 0; s < booster_config.num_classes; ++s) {
            ndcg_scores[k] += (s + 1) * exp(scores[s * num_data + k] -
                                             ndcg_max_scores[k]) / ndcg_probs[k];   
        }
    }
    
    vector<double> ndcg(4, 0.0);
    
    for(int k = 0; k < query_boundaries.size(); ++k) {
        int n;
        if(k == query_boundaries.size() - 1) {
            n = num_data - query_boundaries[k];
        }
        else {
            n = query_boundaries[k + 1] - query_boundaries[k];
        }
        
        double score = NDCGAt(labels.data() + query_boundaries[k],
                              ndcg_scores.data() + query_boundaries[k],
                              1, n);
        ndcg[0] += score;
        
        score = NDCGAt(labels.data() + query_boundaries[k],
                              ndcg_scores.data() + query_boundaries[k],
                              3, n);
        ndcg[1] += score;
        
        score = NDCGAt(labels.data() + query_boundaries[k],
                              ndcg_scores.data() + query_boundaries[k],
                              5, n);
        ndcg[2] += score;
        
        score = NDCGAt(labels.data() + query_boundaries[k],
                              ndcg_scores.data() + query_boundaries[k],
                              10, n);
        ndcg[3] += score;
    }
    ndcg[0] /= query_boundaries.size();
    ndcg[1] /= query_boundaries.size();
    ndcg[2] /= query_boundaries.size();
    ndcg[3] /= query_boundaries.size();
    
    cout << "ndcg@" << 1 << " " << ndcg[0] << " ";
    cout << "ndcg@" << 3 << " " << ndcg[1] << " ";
    cout << "ndcg@" << 5 << " " << ndcg[2] << " ";
    cout << "ndcg@" << 10 << " " << ndcg[3] << endl;
    
    return ndcg[0]; 
}

static bool compare(ScoreIndexPair *a, ScoreIndexPair *b) {
    return (a->score > b->score);
}

double NDCG::DCG(const double *rels, const double *scores, int k, int n) {
    std::vector<ScoreIndexPair*> pairs(n, nullptr);
    for(int i = 0; i < n; ++i) {
        pairs[i] = new ScoreIndexPair(scores[i], i);
    }
    
    std::sort(pairs.begin(), pairs.end(), compare);
    
    double dcg = 0.0;

    for(int i = 0; i < pairs.size() && i < k; ++i) {
        dcg += (pow(2, rels[pairs[i]->index]) - 1) / log2(2 + i);                   
    }
    
    for(int i = 0; i < n; ++i) {
        delete pairs[i];
    }
    
    return dcg; 
}

double NDCG::NDCGAt(const double *rels, const double *scores, int k, int n) {
    double dcg = DCG(rels, scores, k, n);
    double max_dcg = DCG(rels, rels, k, n);
    if(max_dcg == 0.0) {
        assert(dcg == 0.0);
        return 1.0;
    }
    assert(dcg <= max_dcg);
    return dcg / max_dcg;
}

