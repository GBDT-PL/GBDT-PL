import gbdtpl

if __name__ == "__main__":
    params = {"num_trees":100, "num_leaves":255, "min_sum_hessians":100.0, "l2_reg":0.01, "l1_reg":0.01, "objective":"l2",
            "learning_rate":0.1, "eval_metric":"rmse", "normalization":"no", "num_classes":1, "num_threads":24, "num_bins":63, "min_gain":0.0,
              "max_var":5, "grow_by":"leaf", "leaf_type":"half_additive", "max_depth":5, "verbose":2, "sparse_ratio":0.0,
              "boosting_type":"gbdt", "goss_alpha":0.0, "goss_beta":0.0}

    train_data = gbdtpl.DataMat("train", params, 0, -1, "../../../LinearGBM2/data/CASP/casp_train.csv")
    test_data = gbdtpl.DataMat("test", params, 0, -1, "../../../LinearGBM2/data/CASP/casp_test.csv", train_data)
    booster = gbdtpl.Booster(params, train_data, test_data) 
    booster.Train()
    predict_data = gbdtpl.DataMat("predict", params, -1, -1, "../../../LinearGBM2/data/CASP/casp_test.csv", train_data)
    results = booster.Predict(predict_data)
    print(results)
    print("best_iteration", booster.get_best_iteration())
    print("scores of test data", booster.get_scores_per_iteration("test"))
