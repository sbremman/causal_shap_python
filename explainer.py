import sys
import time

import pandas as pd
import numpy as np
from feature_combinations import feature_combinations, feature_matrix, weight_matrix
from prepare_data_causal import prepare_data_causal
from shap import Explanation
import sys
import torch
import math
import random


class Explainer:

    def __init__(self, x_train, model, n_combinations=None, feature_labels=None, asymmetric=None, ordering=None,
                 deep=False):
        # checks input argument
        if not isinstance(x_train, pd.DataFrame):
            sys.exit("x should be a pandas dataframe")

        self.exact = n_combinations is None

        # TODO: Later maybe create a separate module for deep
        if deep:
            if type(model) is tuple:
                a, b = model
                try:
                    a.named_parameters()
                    self.model_type = 'pytorch'
                except:
                    self.model_type = 'tensorflow'

            else:
                try:
                    model.named_parameters()
                    self.model_type = 'pytorch'
                except:
                    self.model_type = 'tensorflow'

        else:
            self.model_type = model._get_type()


        # Checks input argument
        # TODO feature_labels = features(model, x_train.columns.tolist(), feature_labels)
        feature_labels = x_train.columns.tolist()
        self.n_features = len(feature_labels)

        # TODO could do this if going to support other formats than DataFrame as input
        # Converts to data.table, otherwise copy to x_train  --------------
        # x_train <- data.table::as.data.table(x)

        # TODO assumes that all features are used by model.
        # Removes variables that are not included in the model
        # cnms_remove <- setdiff(colnames(x), feature_labels)
        # if (length(cnms_remove) > 0) x_train[, (cnms_remove) := NULL]
        # data.table::setcolorder(x_train, feature_labels)

        # Checks model and features
        self.p = self.predict_model(model, x_train.head())
        if self.p.ndim != 1:
            self.num_outputs = self.p.shape[1]
        else:
            self.num_outputs = 1

        dt_combinations = feature_combinations(m=self.n_features, exact=self.exact, n_combinations=n_combinations)
        # Get weighted matrix
        weighted_mat = weight_matrix(X=dt_combinations, normalize_W_weights=True)

        # Get feature matrix
        feature_mat = feature_matrix(feature_sample=dt_combinations["features"], m=self.n_features)

        self.model = model
        self.S = feature_mat
        self.W = weighted_mat
        self.X = dt_combinations
        self.x_train = x_train
        self.feature_labels = feature_labels
        self.return_val = None
        self.x = None
        self.p = None

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def predict_model(self, model, data):
        if self.model_type == "pytorch":
            batch_size = 2000
            data_temp = np.array_split(data, max(1, math.floor(len(data.values)/batch_size)))

            prediction = model(torch.from_numpy(data_temp[0].values)).detach().numpy()

            prediction = np.empty([len(data.values), prediction.shape[1]])

            data_temp_length = len(data_temp)

            for i in range(data_temp_length):
                print("Predict_model: "+str(i)+" of: "+str(data_temp_length), end='\r')
                prediction[i*batch_size:(i+1)*batch_size, :] = model(torch.from_numpy(data_temp[i].values)).detach().numpy()

        elif self.model_type == "tensorflow":
            sys.exit("Tensorflow not yet supported")

        else:
            prediction = model.predict(data)

        return prediction


    def explain_causal(self, x, prediction_zero, mu=None, cov_mat=None, ordering=None, confounding=False, asymmetric=False, seed=None, shap_format=True):
        #Set seed if given
        if seed is not None:
            random.seed(a=seed)
            np.random.seed(seed)

        # Add arguments to explainer object
        self.x_test = x.reset_index(drop=True)

        # If mu is not provided directly, use mean of training data
        if mu is None:
            self.mu = self.x_train.mean().values
        else:
            self.mu = mu

        # If cov_mat is not provided directly, use sample covariance of training data
        if cov_mat is None:
            cov_mat = self.x_train.cov()

        # Make sure that covariance matrix is positive-definite
        eigen_values = np.linalg.eigvals(cov_mat.values)
        if np.any(eigen_values < 1e-06):
            exit("Covariance matrix is not positive-definite, not yet implemented")
        else:
            self.cov_mat = cov_mat.to_numpy()

        self.ordering = ordering
        self.confounding = confounding

        # Generate data
        dt = prepare_data_causal(self, asymmetric=asymmetric, ordering=ordering)

        if self.return_val is not None:
            print("Return_val is not none")
            return dt

        # Predict
        r = self.prediction(dt, prediction_zero)

        if shap_format:
            return self.r_to_shap_format(r)


        return r

    def r_to_shap_format(self, r):

        base_values = r['dt'].pop('none').values
        values = r['dt'].values
        data = r['x_test'].values


        base_values = base_values.reshape([self.num_outputs, data.shape[0]])
        values = values.reshape([self.num_outputs, data.shape[0], data.shape[1]])

        data = np.array([data]*self.num_outputs)

        display_data = None
        instance_names = None
        feature_names = r['x_test'].columns
        output_names = None
        output_indexes = None
        lower_bounds = None
        upper_bounds = None
        main_effects = None
        hierarchical_values = None
        clustering = None

        explanation_list = []
        if self.num_outputs > 0:
            for i in range(self.num_outputs):
                temp_values = values[i]
                temp_base_values = base_values[i]
                temp_data = data[i]
                out = Explanation(temp_values, temp_base_values, temp_data, display_data, instance_names, feature_names, output_names, output_indexes, lower_bounds, upper_bounds, main_effects, hierarchical_values, clustering)
                explanation_list.append(out)

        if len(explanation_list) == 1:
            explanation_list = explanation_list[0]

        return explanation_list

    def prediction(self, dt, prediction_zero):
        # TODO Check that data is in pandas dataframe
        # TODO check that dataframe has "id", "id_combination" and "w" columns

        # Setup
        cnms = list(self.x_test.columns.values)

        # Check that the number of test observations equals max(id) + 1
        if len(self.x_test.index) != dt['id'].max()+1:
            print("\n\n\n\n!!!!Number of test observations ("+str(len(self.x_test.index))+") did not equal max(id) + 1 ("+str(dt['id'].max()+1)+")!!!!\n\n\n\n")
            exit("Number of test observations did not equal max(id)")

        # Predictions
        p_hat = self.predict_model(self.model, dt[cnms])
        if self.num_outputs > 1:
            for i in range(self.num_outputs):
                dt['p_hat_'+str(i)] = p_hat[:, i]
                dt.loc[dt.id_combination == 0, 'p_hat_'+str(i)] = float(prediction_zero[i])
        else:
            dt['p_hat_'+str(0)] = p_hat[:]
            dt.loc[dt.id_combination == 0, 'p_hat_'+str(0)] = float(prediction_zero[0])


        p_all = self.predict_model(self.model, self.x_test)

        # TODO this should be implemented, however, on the bike-dataset, it at least does not matter:
        #dt.loc[dt.id_combination == dt.id_combination.max(), 'p_hat'] = p_all["id"]
        #print(dt)

        # Calculate contributions
        dt_list = []
        for i in range(self.num_outputs):
            dt_temp = dt[['id', 'id_combination', 'w']]
            dt_temp['p_hat*w'] = dt['p_hat_'+str(i)]*dt['w']
            dt_res = dt_temp.groupby(['id', 'id_combination']).sum()
            dt_res['k'] = dt_res["p_hat*w"]/dt_res['w']
            del dt_res["p_hat*w"]
            del dt_res['w']
            dt_mat = dt_res.unstack().values[:, :]
            kshap = (self.W.dot(dt_mat.T)).T
            cnms_temp = ['none']+cnms
            dt_kshap = pd.DataFrame(data=kshap, columns=cnms_temp)
            dt_list.append(dt_kshap)

        r = {"dt": pd.concat(dt_list),
             "model": self.model,
             "p": p_all,
             "x_test": self.x_test}

        return r


