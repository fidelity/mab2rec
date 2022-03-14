# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import pandas as pd

from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from mab2rec.pipeline import train, score, benchmark
from mab2rec.utils import Constants, default_metrics, load_item_features, load_data_frame, load_list

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep

# Data files
train_data = os.path.join(ROOT_DIR, "data", "data_train.csv")
test_data = os.path.join(ROOT_DIR, "data", "data_test.csv")
user_features = os.path.join(ROOT_DIR, "data", "features_user.csv")
item_features = os.path.join(ROOT_DIR, "data", "features_item.csv")

# Evaluation metrics
metrics = default_metrics([3, 5, 10])


class InvalidTest(unittest.TestCase):

    # =====================
    # BanditRecommender
    # =====================

    def test_invalid_learning_policy(self):
        with self.assertRaises(TypeError):
            BanditRecommender(NeighborhoodPolicy.Radius(radius=12))

    def test_invalid_neighborhood_policy(self):
        with self.assertRaises(TypeError):
            BanditRecommender(LearningPolicy.EpsilonGreedy(), LearningPolicy.Softmax())

    def test_invalid_init_arms_int(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init(1)

    def test_invalid_init_arms_tuple(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init(1, 2)

    def test_invalid_add_arm_value(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init([1, 2])
            rec.add_arm(1)

    def test_invalid_remove_arm_value(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init([1, 2])
            rec.remove_arm(3)

    def test_invalid_remove_arm_no_init(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec.remove_arm(3)

    def test_invalid_set_arms(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec.set_arms(None)

    def test_invalid_partial_fit(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec.partial_fit([1, 1, 2, 2], [0, 1, 1, 1])

    def test_invalid_partial_fit_with_init(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init([1, 2])
            rec.partial_fit([1, 1, 2, 2], [0, 1, 1, 1])

    def test_invalid_predict_not_fit(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init([1, 2])
            _ = rec.predict()

    def test_invalid_predict_expectations_not_fit(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init([1, 2])
            _ = rec.predict_expectations()

    def test_invalid_recommend_not_fit(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy(), top_k=2)
            rec._init([1, 2])
            _ = rec.recommend()

    def test_invalid_recommend_no_contexts(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB(), top_k=2)
            rec.fit([1, 1, 2, 2], [0, 1, 1, 1], [[0, 1, 2], [3, 1, 2], [0, 3, 1], [2, 1, 1]])
            _ = rec.recommend()

    def test_invalid_recommend_excluded_arms(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB(), top_k=2)
            rec.fit([1, 1, 2, 2], [0, 1, 1, 1], [[0, 1, 2], [3, 1, 2], [0, 3, 1], [2, 1, 1]])
            _ = rec.recommend(excluded_arms=[[1], [1]])

    def test_invalid_recommend_excluded_arms_dim(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB(), top_k=2)
            rec.fit([1, 1, 2, 2], [0, 1, 1, 1], [[0, 1, 2], [3, 1, 2], [0, 3, 1], [2, 1, 1]])
            _ = rec.recommend(contexts=[[0, 1, 2]], excluded_arms=[[1], [1]])

    def test_invalid_warm_start_not_fit(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            rec._init([1, 2, 3])
            rec.warm_start(arm_to_features={1: [0.5, 0.5], 2: [1, 0.5], 3: [1, 0]}, distance_quantile=0.5)

    def test_invalid_warm_start_missing_arm(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init([1, 2, 3])
            rec.fit([1, 1, 2, 2, 3], [0, 1, 1, 1, 0])
            rec.warm_start(arm_to_features={1: [0.5, 0.5], 2: [1, 0.5]}, distance_quantile=0.5)

    def test_invalid_warm_start_unknown_arm(self):
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init([1, 2, 3])
            rec.fit([1, 1, 2, 2, 3], [0, 1, 1, 1, 0])
            rec.warm_start(arm_to_features={1: [0.5, 0.5], 2: [1, 0.5], 3: [1, 0], 4: [0, 1]}, distance_quantile=0.5)

    def test_invalid_warm_start_distance(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            rec._init([1, 2, 3])
            rec.fit([1, 1, 2, 2, 3], [0, 1, 1, 1, 0])
            rec.warm_start(arm_to_features={1: [0.5, 0.5], 2: [1, 0.5], 3: [1, 0]}, distance_quantile=50)

    # =====================
    # Train
    # =====================
    
    def test_train_invalid_recommender(self):
        with self.assertRaises(TypeError):
            rec = LearningPolicy.LinUCB()
            train(rec, train_data, user_features)
    
    def test_train_invalid_data(self):
        data = pd.read_csv(train_data)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            train(rec, data.values)
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            train(rec, data.rename(columns={Constants.user_id: "user"}))
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            train(rec, data.rename(columns={Constants.item_id: "item"}))
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            train(rec, data.rename(columns={Constants.item_id: "click"}))
   
    def test_train_invalid_user_features(self):
        user_features_df = pd.read_csv(user_features)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features_df.values)
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features_df.rename(columns={Constants.user_id: "user"}))

    def test_train_invalid_user_features_list(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, user_features_list=pd.Series(["u1", "u2"]))
    
    def test_train_invalid_user_features_dtypes(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, user_features_dtypes=["int8", "int8"])
    
    def test_train_invalid_item_features(self):
        item_features_df = pd.read_csv(item_features)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, item_features=item_features_df.values)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, item_features=list(item_features_df))
    
    def test_train_invalid_item_list(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, item_list=pd.Series(["235", "313", "433"]))
    
    def test_train_invalid_item_eligibility(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, item_eligibility=[[234], [456]])
    
    def test_train_invalid_warm_start(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, warm_start=1)
    
    def test_train_invalid_warm_start_distance(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, warm_start=True, warm_start_distance=50)
    
    def test_train_invalid_user_id_col(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, user_id_col=0)
    
    def test_train_invalid_item_id_col(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, item_id_col=1)
    
    def test_train_invalid_response_col(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, item_id_col=2)
    
    def test_train_invalid_batch_size(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, batch_size="1000")
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, batch_size=-50)
    
    def test_train_invalid_save_file(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features, save_file=1)

    # =====================
    # Score
    # =====================

    def test_score_invalid_recommender(self):
        with self.assertRaises(TypeError):
            rec = LearningPolicy.LinUCB()
            score(rec, train_data, user_features)
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            score(rec, train_data, user_features)

    def test_score_invalid_data(self):
        data = pd.read_csv(train_data)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            train(rec, data)
            score(rec, data.values)
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            train(rec, data)
            score(rec, data.rename(columns={Constants.user_id: "user"}))
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            train(rec, data)
            score(rec, data.rename(columns={Constants.item_id: "item"}))
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
            train(rec, data)
            score(rec, data.rename(columns={Constants.item_id: "click"}))

    def test_score_invalid_user_features(self):
        user_features_df = pd.read_csv(user_features)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features_df)
            score(rec, train_data, user_features_df.values)
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features_df)
            score(rec, train_data, user_features_df.rename(columns={Constants.user_id: "user"}))

    def test_score_invalid_user_features_list(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, user_features_list=pd.Series(["u1", "u2"]))

    def test_score_invalid_user_features_dtypes(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, user_features_dtypes=["int8", "int8"])

    def test_score_invalid_item_features(self):
        item_features_df = pd.read_csv(item_features)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, item_features=item_features_df.values)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, item_features=list(item_features_df))

    def test_score_invalid_item_list(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, item_list=pd.Series(["235", "313", "433"]))

    def test_score_invalid_item_eligibility(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, item_eligibility=[[234], [456]])

    def test_score_invalid_warm_start(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, warm_start=1)

    def test_score_invalid_warm_start_distance(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, warm_start=True, warm_start_distance=50)

    def test_score_invalid_user_id_col(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, user_id_col=0)

    def test_score_invalid_item_id_col(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, item_id_col=1)

    def test_score_invalid_response_col(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, item_id_col=2)

    def test_score_invalid_batch_size(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, batch_size="1000")
        with self.assertRaises(ValueError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, batch_size=-50)

    def test_score_invalid_save_file(self):
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data, user_features)
            score(rec, train_data, user_features, save_file=1)

    # =====================
    # Benchmark
    # =====================

    def test_benchmark_invalid_recommender(self):
        with self.assertRaises(TypeError):
            rec = LearningPolicy.LinUCB()
            benchmark(rec, metrics, train_data, test_data, user_features=user_features)
        with self.assertRaises(TypeError):
            rec = BanditRecommender(LearningPolicy.LinUCB())
            benchmark(rec, metrics, train_data, test_data, user_features=user_features)
        with self.assertRaises(TypeError):
            rec = {"LinUCB": LearningPolicy.LinUCB()}
            benchmark(rec, metrics, train_data, test_data, user_features=user_features)

    def test_benchmark_invalid_data(self):
        train_data_df = pd.read_csv(train_data)
        test_data_df = pd.read_csv(test_data)
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data_df.values, test_data, user_features=user_features)
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data_df.values, user_features=user_features)
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data, test_data, cv=5, user_features=user_features)
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data_df.rename(columns={Constants.user_id: "user"}),
                      test_data, cv=5, user_features=user_features)
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data_df.rename(columns={Constants.item_id: "item"}),
                      test_data, cv=5, user_features=user_features)
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data_df.rename(columns={Constants.response: "click"}),
                      test_data, cv=5, user_features=user_features)

    def test_benchmark_invalid_user_features(self):
        user_features_df = pd.read_csv(user_features)
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features_df.values)
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data, test_data,
                      user_features=user_features_df.rename(columns={Constants.user_id: "user"}))

    def test_benchmark_invalid_user_features_list(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features,
                      user_features_list=pd.Series(["u1", "u2"]))
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data, test_data, user_features_list=pd.Series(["u1", "u2"]))

    def test_benchmark_invalid_user_features_dtypes(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features,
                      user_features_dtypes=["int8", "int8"])
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data, test_data, user_features_dtypes=["int8", "int8"])

    def test_benchmark_invalid_item_features(self):
        item_features_df = pd.read_csv(item_features)
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features,
                      item_features=item_features_df.values)
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features,
                      item_features=list(item_features_df))

    def test_benchmark_invalid_item_list(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features,
                      item_list=pd.Series(["235", "313", "433"]))

    def test_benchmark_invalid_item_eligibility(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features,
                      item_eligibility=[[234], [456]])

    def test_benchmark_invalid_warm_start(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features, warm_start=1)

    def test_benchmark_invalid_warm_start_distance(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features, warm_start_distance=50)

    def test_benchmark_invalid_warm_start_distance_value(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features,
                      warm_start=True, warm_start_distance=50)

    def test_benchmark_invalid_user_id_col(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features, user_id_col=0)

    def test_benchmark_invalid_item_id_col(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features, item_id_col=1)

    def test_benchmark_invalid_response_col(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features, response_col=2)

    def test_benchmark_invalid_batch_size(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(ValueError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features, batch_size=-2)

    def test_benchmark_invalid_save_file(self):
        rec = {"LinUCB": BanditRecommender(LearningPolicy.LinUCB()),
               "Random": BanditRecommender(LearningPolicy.Random())}
        with self.assertRaises(TypeError):
            benchmark(rec, metrics, train_data, test_data, user_features=user_features, output_dir=True)

    # =====================
    # Utils
    # =====================

    def test_utils_invalid_load_item_features(self):
        item_features_df = pd.read_csv(item_features)
        item_list = item_features_df[Constants.item_id].tolist()
        item_list.append("new_id")
        with self.assertRaises(ValueError):
            load_item_features(item_features_df, item_list)

    def test_utils_invalid_load_data_frame(self):
        item_features_df = pd.read_csv(item_features)
        with self.assertRaises(TypeError):
            load_data_frame(item_features_df.values)

    def test_utils_invalid_load_list(self):
        with self.assertRaises(TypeError):
            load_list((1, 2, 3))
