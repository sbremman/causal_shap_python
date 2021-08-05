from sample_causal import sample_causal
import numpy as np
import pandas as pd
from feature_combinations import powerset, feature_exact, feature_not_exact, feature_combinations, weight_matrix
import math
import xgboost
from explainer import Explainer
import matplotlib.pyplot as plt
import shap
import time

if __name__ == '__main__':
    """
    m = 10
    n_samples = 50
    mu = np.ones(10)
    #np.random.seed(seed=1)

    cov_mat = np.cov(np.random.rand(m,n_samples))
    x_test = np.random.multivariate_normal(mu,cov_mat,1)
    cnms = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
    x_test = pd.DataFrame(x_test, columns=cnms)
    index_given = np.array([3, 6])
    ordering = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    confounding = [True, False, True]
    print(sample_causal(index_given, n_samples, mu, cov_mat, m, x_test, ordering, confounding))"""

    bike = pd.read_csv("day.csv")
    # Difference in days, which takes DST into account
    bike["trend"] = bike["instant"]-1
    bike["cosyear"] = np.cos(math.pi*bike["trend"]/365*2)
    bike["sinyear"] = np.sin(math.pi*bike["trend"]/365*2)
    # Unnormalize variables (see data set information in link above)
    bike["temp"] = bike["temp"] * (39 - (-8)) + (-8)
    bike["atemp"] = bike["atemp"] * (50 - (-16)) + (-16)
    bike["windspeed"] = 67 * bike["windspeed"]
    bike["hum"] = 100 * bike["hum"]

    x_var = ["trend", "cosyear", "sinyear", "temp", "atemp", "windspeed", "hum"]
    y_var = ["cnt"]

    X_data = bike[x_var]
    Y_data = bike[y_var]

    train_index = pd.read_csv('train_index.csv')["Resample1"].to_numpy()-1
    test_index = bike.index.difference(train_index)

    X_train = X_data.iloc[train_index]
    Y_train_nc = Y_data.iloc[train_index] # not centered
    Y_train = Y_train_nc.subtract(Y_train_nc.mean())

    X_test = X_data.iloc[test_index]
    Y_test_nc = Y_data.iloc[test_index] # not centered
    Y_test = Y_test_nc.subtract(Y_train_nc.mean())

    
    

    #index_647 = X_test.index[X_test["trend"] == 647].tolist()
    #X_test.loc[index_647][["cosyear", "temp"]].plot(kind='bar')
    #plt.show()
    #exit()
    model = xgboost.XGBRegressor()
    model.load_model("R-bike-model.json")
    """model = xgboost.XGBRegressor(
        gamma=0,
        reg_alpha=0,
        learning_rate=0.3,
        max_depth=6,
        n_estimators=100,
        subsample=1,
        random_state=34,
        min_child_weight=1,
        max_delta_step=0,
        sampling_method="uniform",
        tree_method="exact",
        scale_pos_weight=1
    )"""
    """model = xgboost.XGBRegressor(
        gamma=0,
        learning_rate=0.3,
        max_depth=6,
        n_estimators=10,
        subsample=1,
        random_state=34
    )"""

    #print(model.predict(X_test))
    #exit()

    """explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    #shap.plots.beeswarm(shap_values)
    shap.plots.waterfall(shap_values[139])"""

    print("Start: Explainer_symmetric")
    explainer_symmetric = Explainer(X_train, model)
    print("Done: Explainer_symmetric")
    p = Y_train.mean()

    partial_order = [[0], [1, 2], [3, 4, 5, 6]]

    confounding = [False, True, False]
    #partial_order = [[0, 1, 2, 3, 4, 5, 6]]

    #confounding = [False]

    start_time_explain_causal = time.perf_counter()
    explanation_causal = explainer_symmetric.explain_causal(X_test, p, ordering=partial_order, confounding=confounding, seed=2020)
    stop_time_explain_causal = time.perf_counter()
    print(f"Sample causal took {stop_time_explain_causal - start_time_explain_causal:0.4f} seconds")

    #exit()
    print("expla_start")
    for i in range(144):
        print(i)
        print(explanation_causal[0][i])
        print("\n\n")
    #print(explanation_causal[0][126])
    print("expla_stop")
    """test = explanation_causal[0]


    test2 = explanation_causal_old
    test.op_history = test2.op_history
    test.feature_names = test2.feature_names
    #index_647 = explanation_causal["x_test"].index[explanation_causal["x_test"]["trend"] == 647].tolist()
    #index_702 = explanation_causal["x_test"].index[explanation_causal["x_test"]["trend"] == 702].tolist()

    #explanation_causal["dt"].loc[index_647+index_702][["cosyear", "temp"]].plot(kind='bar')
    #explanation_causal["dt"].loc[index_702][["cosyear", "temp"]].plot(kind='bar')
    #plt.show()
    test_1 = test[139]
    test2_1 = test2[139]"""

    #shap.plots.waterfall(explanation_causal[0][126][0:2])
    shap.plots.waterfall(explanation_causal[0][126][[1, 3]])
    shap.plots.waterfall(explanation_causal[0][139][[1, 3]])
    #display(shap.plots.force(explanation_causal[0].base_values[0], explanation_causal[0].values, feature_names = explanation_causal[0].feature_names))

    #shap.plots.waterfall(explanation_causal[140])
    #shap.plots.waterfall(explanation_causal[141])
    #shap.plots.beeswarm(explanation_causal)
    #shap.plots.force(explanation_causal.base_values[0], explanation_causal.values)





