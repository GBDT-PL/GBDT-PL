//
//  main.cpp
//  LinearGBM
//


//

#include <iostream> 
#include "tree.hpp"
#include "datamat.hpp"
#include <vector>
#include <string>
#include <cmath>
#include "booster.hpp"
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <cstdlib>
#include <x86intrin.h>
#include <mkl.h>
#include <omp.h>
#include <functional>
#include <fstream>

using std::cout;
using std::endl;
using std::vector;
using std::function;
using std::ofstream;

int main(int argc, const char * argv[]) {			
    std::vector<std::vector<double>> train_data, test_data; 
    std::vector<double> train_label, test_label;
    std::vector<int> train_queries, test_queries;   
    if(argc < 13) {
    	cout << "usage: ./main <loss> <metric> <num bins> <max var> <node type> <verbose> <boosting> <label index> <group id index> <train file> <test file> <dataset name> <predict fname>" << endl;
	return -1;
    }
    BoosterConfig booster_config(500, 254, 0.0, 0.01, argv[1], 0.1,
                                argv[2], 1, 24, std::atoi(argv[3]),
                                 100.0, std::atoi(argv[4]), "leaf", argv[5], 5, std::atoi(argv[6]), 0.0, argv[7], 0.2, 0.1); 
    
    DataMat data_mat(booster_config, "train", std::atoi(argv[8]), std::atoi(argv[9]),
                     argv[10]); 
    DataMat test_mat(booster_config, "test", std::atoi(argv[8]), std::atoi(argv[9]),
                     argv[11],
                     &data_mat);
    
    /*BoosterConfig booster_config(500, 254, 0.0, 0.01, "logistic", 0.1,        
                                 "auc", 1, 32, 63,
                                 100.0, 5, "grow_by_leaf", 5, 1);
    
    DataMat data_mat(booster_config, "train", 0, -1,
                     "../LinearGBM2/data/higgs_train.csv",
                     train_data, train_label, train_queries);
    DataMat test_mat(booster_config, "test", 0, -1,
                     "../LinearGBM2/data/higgs_test.csv",
                     test_data, test_label, test_queries, &data_mat);*/                                 
    
    Booster booster(booster_config, data_mat, test_mat);
    
    booster.Train();
    
    if(std::atoi(argv[6]) == 2) {
    std::vector<std::vector<double>> predict_matrix;
    std::vector<double> predict_label;
    std::vector<int> predict_queries;
    DataMat predict_data(booster_config, "predict", std::atoi(argv[8]), std::atoi(argv[9]),
                     argv[11],  
                     &data_mat);
    
    vector<double> result;
    booster.Predict(predict_data, result);

    ofstream fout(std::string(argv[12]) + "/" + std::string(argv[12]) + "_" + std::string(argv[3]) + "_" + std::string(argv[7]) + ".result_2");
    for(int i = 0; i < result.size(); ++i) {
    	fout << result[i] << endl;
    }
    fout.close();
    }

    return 0;
}
