import gbdtpl

if __name__ == "__main__":
    params = {"num_trees":100, "num_leaves":255, "min_sum_hessians":100.0, "lambda":0.01, "objective":"logistic",
              "learning_rate":0.1, "eval_metric":"auc", "num_classes":1, "num_threads":24, "num_bins":63, "min_gain":0.0,
              "max_var":5, "grow_by":"leaf", "leaf_type":"additive_linear", "max_depth":5, "verbose":1, "sparse_ratio":0.0,
              "boosting_type":"gbdt", "goss_alpha":0.0, "goss_beta":0.0}

    train_data = gbdtpl.DataMat("train", params, 0, -1, "../../../LinearGBM2/data/small_higgs_train.csv")
    test_data = gbdtpl.DataMat("test", params, 0, -1, "../../../LinearGBM2/data/small_higgs_test.csv", train_data)
    booster = gbdtpl.Booster(params, train_data, test_data) 
    booster.Train()
    results = booster.Predict(test_data)
    print results
