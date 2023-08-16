# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import json
import pickle
import os
import tempfile
import unittest

import pandas as pd
from mabwiser.linear import _Linear
from jurity.recommenders import DiversityRecoMetrics

from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from mab2rec.pipeline import train, score, benchmark
from mab2rec.utils import Constants, default_metrics, concat_recommendations_list
from tests.test_base import BaseTest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = TEST_DIR + os.sep + ".." + os.sep

# Data files
train_data = os.path.join(ROOT_DIR, "data", "data_train.csv")
test_data = os.path.join(ROOT_DIR, "data", "data_test.csv")
user_features = os.path.join(ROOT_DIR, "data", "features_user.csv")
item_features = os.path.join(ROOT_DIR, "data", "features_item.csv")

item_eligibility = os.path.join(ROOT_DIR, "data", "extended", "data_eligibility.csv")
user_features_dtypes = os.path.join(ROOT_DIR, "data", "extended", "features_user_dtypes.json")

# Import
train_data_df = pd.read_csv(train_data)
test_data_df = pd.read_csv(test_data)
user_features_df = pd.read_csv(user_features)
item_features_df = pd.read_csv(item_features)


class TrainTest(BaseTest):

    def test_learning_policies_train(self):
        for lp in self.lps + self.para_lps:
            rec = BanditRecommender(lp)
            train(rec, train_data, user_features)
            train(rec, train_data_df, user_features_df)

    def test_neighborhood_policies_train(self):
        for cp in self.nps:
            rec = BanditRecommender(self.lps[0], cp)
            train(rec, train_data_df, user_features_df)

    def test_lingreedy_train(self):
        rec = BanditRecommender(LearningPolicy.LinGreedy())
        train(rec, train_data_df, user_features_df)
        self.assertTrue(isinstance(rec.mab._imp, _Linear))
        self.assertEqual(len(rec.mab._imp.arm_to_model), 201)
        self.assertAlmostEqual(rec.mab._imp.arm_to_model[427].beta[0], 0.19919595422109415)
        self.assertAlmostEqual(rec.mab._imp.arm_to_model[173].beta[0], -0.23409821140400933)

    def test_train_twice(self):
        # First train
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df)

        # Second train
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df)

        self.assertEqual(rec.mab._is_initial_fit, rec.mab._is_initial_fit)
        self.assertEqual(rec.mab.learning_policy, rec.mab.learning_policy)

    def test_learning_policies_warm_start(self):
        item_list = train_data_df[Constants.item_id].unique().tolist()
        train_item_list = [118, 125, 132, 173, 250, 275, 423, 427, 591, 751]  # Assume only 10 items in train data
        train_df = train_data_df[train_data_df[Constants.item_id].isin(train_item_list)]
        for lp in self.lps + self.para_lps:
            rec = BanditRecommender(lp)
            train(rec, train_df, user_features_df, item_list=item_list,
                  item_features=item_features_df, warm_start=True, warm_start_distance=0.75)
            self.assertEqual(rec.mab.arms, item_list)

    def test_neighborhood_policies_warm_start(self):
        item_list = train_data_df[Constants.item_id].unique().tolist()
        train_item_list = [118, 125, 132, 173, 250, 275, 423, 427, 591, 751]  # Assume only 10 items in train data
        train_df = train_data_df[train_data_df[Constants.item_id].isin(train_item_list)]
        for cp in self.nps:
            rec = BanditRecommender(self.lps[0], cp)
            train(rec, train_df, user_features_df, item_list=item_list,
                  item_features=item_features_df, warm_start=True, warm_start_distance=0.75)
            self.assertEqual(rec.mab.arms, item_list)

    def test_warm_start_twice(self):
        item_list = train_data_df[Constants.item_id].unique().tolist()
        train_item_list = [118, 125, 132, 173, 250, 275, 423, 427, 591, 751]  # Assume only 10 items in train data
        train_df = train_data_df[train_data_df[Constants.item_id].isin(train_item_list)]

        # First warm start
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_df, user_features_df, item_list=item_list,
              item_features=item_features_df, warm_start=True, warm_start_distance=0.75)
        self.assertEqual(rec.mab.arms, item_list)

        # Second warm start
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_df, user_features_df, item_list=item_list,
              item_features=item_features_df, warm_start=True, warm_start_distance=0.75)
        self.assertEqual(rec.mab.arms, item_list)

    def test_warm_start_input_change(self):
        item_list = train_data_df[Constants.item_id].unique().tolist()
        train_item_list = [118, 125, 132, 173, 250, 275, 423, 427, 591, 751]  # Assume only 10 items in train data
        train_df = train_data_df[train_data_df[Constants.item_id].isin(train_item_list)]

        # Copy inputs
        train_df_copy = train_df.copy()
        user_features_df_copy = user_features_df.copy()
        item_list_copy = item_list.copy()
        item_features_df_copy = item_features_df.copy()

        # Train and warm start
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_df, user_features_df, item_list=item_list,
              item_features=item_features_df, warm_start=True, warm_start_distance=0.75)

        # Compare after
        self.assertTrue(train_df.equals(train_df_copy))
        self.assertTrue(user_features_df.equals(user_features_df_copy))
        self.assertTrue(item_features_df.equals(item_features_df_copy))
        self.assertEqual(item_list, item_list_copy)

    def test_user_features_list(self):
        user_features_list = ["u1", "u2", "u3", "u4", "u5"]  # Only use these user features
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df, user_features_list=user_features_list)
        self.assertEqual(len(rec.mab._imp.arm_to_model[118].beta), len(user_features_list))

    def test_user_features_list_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            user_features_list = ["u1", "u2", "u3", "u4", "u5"]  # Only use these user features
            user_features_list_csv = os.path.join(tmp_dir, "user_features_list.csv")
            pd.DataFrame(user_features_list).to_csv(user_features_list_csv, index=False)
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data_df, user_features_df, user_features_list=user_features_list_csv)
            self.assertEqual(len(rec.mab._imp.arm_to_model[118].beta), len(user_features_list))

    def test_user_features_data_types(self):
        with open(user_features_dtypes, 'r') as f:
            data_types = json.load(f)
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df, user_features_dtypes=data_types)
        self.assertTrue(rec.mab._is_initial_fit)

    def test_user_features_data_types_json(self):
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df, user_features_dtypes=user_features_dtypes)
        self.assertTrue(rec.mab._is_initial_fit)

    def test_user_id_col_change(self):
        train_data_df_update = train_data_df.rename(columns={"user_id": "uid"})
        user_features_df_update = user_features_df.rename(columns={"user_id": "uid"})
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df_update, user_features_df_update, user_id_col="uid")
        self.assertTrue(rec.mab._is_initial_fit)

    def test_item_id_col_change(self):
        train_data_df_update = train_data_df.rename(columns={"item_id": "iid"})
        item_features_df_update = item_features_df.rename(columns={"item_id": "iid"})
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df_update, user_features_df,
              item_features=item_features_df_update, item_id_col="iid")
        self.assertTrue(rec.mab._is_initial_fit)

    def test_response_col_change(self):
        train_data_df_update = train_data_df.rename(columns={"response": "ind"})
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df_update, user_features_df, response_col="ind")
        self.assertTrue(rec.mab._is_initial_fit)

    def test_batch_size_small(self):
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df, batch_size=1000)
        self.assertTrue(rec.mab._is_initial_fit)
        rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
        train(rec, train_data_df, batch_size=1000)
        self.assertTrue(rec.mab._is_initial_fit)

    def test_batch_size_large(self):
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df, batch_size=10000000)
        self.assertTrue(rec.mab._is_initial_fit)

    def test_save_file(self):
        # Specified file name
        with tempfile.TemporaryDirectory() as tmp_dir:
            rec_pickle = os.path.join(tmp_dir, "rec.pkl")
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data_df, user_features_df, save_file=rec_pickle)
            self.assertTrue(os.path.exists(rec_pickle))
            with open(rec_pickle, 'rb') as f:
                rec = pickle.load(f)
            self.assertTrue(isinstance(rec, BanditRecommender))
            self.assertTrue(rec.mab._is_initial_fit)

    def test_save_file_true(self):
        default_name = "recommender.pkl"
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df, save_file=True)
        self.assertTrue(os.path.exists(default_name))
        with open(default_name, 'rb') as f:
            rec = pickle.load(f)
        self.assertTrue(isinstance(rec, BanditRecommender))
        self.assertTrue(rec.mab._is_initial_fit)
        os.remove(default_name)


class ScoreTest(BaseTest):

    def test_learning_policies_score(self):
        for lp in self.lps + self.para_lps:

            # Train
            rec = BanditRecommender(lp)
            train(rec, train_data_df, user_features=user_features_df)

            # Score w Warm Start
            df = score(rec, test_data_df.sample(100), user_features=user_features_df,
                       item_list=test_data_df[Constants.item_id].tolist(), item_features=item_features_df,
                       warm_start=True, warm_start_distance=0.75)
            self.assertEqual(df.shape[1], 3)
            self.assertEqual(df.ndim, 2)

    def test_neighborhood_policies_score(self):
        cp_prev = None
        for cp in self.nps:
            if cp_prev is not None and type(cp) == type(cp_prev):
                cp_prev = deepcopy(cp)
                continue
            # Train
            rec = BanditRecommender(self.lps[0], cp)
            train(rec, train_data_df, user_features_df)

            # Score w Warm Start
            df = score(rec, test_data_df.sample(10), user_features_df,
                       item_list=test_data_df[Constants.item_id].tolist(), item_features=item_features_df,
                       warm_start=True, warm_start_distance=0.75)
            self.assertEqual(df.shape[1], 3)
            self.assertEqual(df.ndim, 2)
            cp_prev = deepcopy(cp)

    def test_lingreedy_score(self):
        rec = BanditRecommender(LearningPolicy.LinGreedy(l2_lambda=1000))
        train(rec, train_data_df, user_features_df)
        df = score(rec, test_data_df, user_features_df)
        score_dict = df.head().to_dict()
        self.assertDictEqual(score_dict['user_id'], {0: 259, 1: 259, 2: 259, 3: 259, 4: 259})
        self.assertDictEqual(score_dict['item_id'], {0: 50, 1: 127, 2: 313, 3: 56, 4: 174},)
        self.assertDictEqual(score_dict['score'], {0: 0.5481800084403381, 1: 0.533300457736932, 2: 0.5318001465799647,
                                                   3: 0.5313858071529368, 4: 0.531052100344854},)

    def test_learning_policy_no_features(self):
        rec = BanditRecommender(LearningPolicy.EpsilonGreedy())
        train(rec, train_data_df)
        df = score(rec, test_data_df)
        self.assertEqual(df.shape[1], 3)
        self.assertEqual(df.ndim, 2)

    def test_load_from_pickle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rec_pickle = os.path.join(tmp_dir, "rec.pkl")
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data_df, user_features_df, save_file=rec_pickle)
            df = score(rec_pickle, test_data_df, user_features_df)
            self.assertEqual(df.shape[1], 3)
            self.assertEqual(df.ndim, 2)

    def test_eligible_items(self):
        item_eligibility_df = pd.read_csv(item_eligibility)

        # Train
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df)

        # Score only eligible items
        df = score(rec, test_data_df, user_features_df, item_eligibility=item_eligibility_df)
        self.assertEqual(df.shape[1], 3)
        self.assertEqual(df.ndim, 2)

    def test_eligible_items_twice(self):
        item_eligibility_df = pd.read_csv(item_eligibility)

        # Train
        rec = BanditRecommender(LearningPolicy.Random())
        train(rec, data=train_data_df)

        # Make recommendations that satisfy eligibility criteria
        df_first = score(rec, data=test_data_df, item_eligibility=item_eligibility_df)

        # Train again
        rec = BanditRecommender(LearningPolicy.Random())
        train(rec, data=train_data_df)

        # Make recommendations that satisfy eligibility criteria
        df_second = score(rec, data=test_data_df, item_eligibility=item_eligibility_df)

        self.assertTrue(df_first.equals(df_second))

    def test_batch_size_small(self):
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df)
        df = score(rec, test_data_df, user_features_df, batch_size=1000)
        self.assertEqual(df.shape[1], 3)
        self.assertEqual(df.ndim, 2)

    def test_batch_size_large(self):
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df)
        df = score(rec, test_data_df, user_features_df, batch_size=10000000)
        self.assertEqual(df.shape[1], 3)
        self.assertEqual(df.ndim, 2)

    def test_save_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_csv = os.path.join(tmp_dir, "results.csv")
            rec = BanditRecommender(LearningPolicy.LinUCB())
            train(rec, train_data_df, user_features_df)
            score(rec, train_data_df, user_features_df, save_file=results_csv)
            df = pd.read_csv(results_csv)
            self.assertTrue(os.path.exists(results_csv))
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertEqual(df.shape[1], 3)
            self.assertEqual(df.ndim, 2)

    def test_save_file_true(self):
        default_name = "results.csv"
        rec = BanditRecommender(LearningPolicy.LinUCB())
        train(rec, train_data_df, user_features_df)
        score(rec, train_data_df, user_features_df, save_file=True)
        self.assertTrue(os.path.exists(default_name))
        os.remove(default_name)


class BenchmarkTest(BaseTest):

    recommenders = {
        "Random": BanditRecommender(LearningPolicy.Random()),
        "LinUCB": BanditRecommender(LearningPolicy.LinUCB(alpha=1.5)),
        "ClustersTS": BanditRecommender(LearningPolicy.ThompsonSampling(), NeighborhoodPolicy.Clusters(n_clusters=10))
    }
    metrics = default_metrics(top_k_values=[3, 5, 10])

    def test_benchmark(self):
        recommenders = deepcopy(self.recommenders)
        recommendations, rec_metrics = benchmark(recommenders, self.metrics, train_data, test_data,
                                                 user_features=user_features_df)
        self.assertEqual(recommendations.keys(), self.recommenders.keys())
        self.assertEqual(rec_metrics.keys(), self.recommenders.keys())
        self.assertTrue(isinstance(recommendations["Random"], pd.DataFrame))
        self.assertEqual(recommendations["Random"].shape[1], 3)
        self.assertEqual(recommendations["Random"].ndim, 2)
        self.assertAlmostEqual(rec_metrics["Random"]["AUC(score)@3"], 0.5154761904761904)
        self.assertAlmostEqual(rec_metrics["Random"]["CTR(score)@3"], 0.2112676056338028)
        self.assertAlmostEqual(rec_metrics["Random"]["Precision@3"], 0.04950495049504949)
        self.assertAlmostEqual(rec_metrics["Random"]["Recall@3"], 0.00774496333427291)
        self.assertAlmostEqual(rec_metrics["Random"]["NDCG@3"], 0.014186541542204544)
        self.assertAlmostEqual(rec_metrics["Random"]["MAP@3"], 0.0286028602860286)
        for rec in recommenders.values():
            self.assertTrue(rec.mab is None)

    @unittest.skip("operating system differences")
    def test_benchmark_cv(self):
        recommenders = deepcopy(self.recommenders)
        recommendations, rec_metrics = benchmark(recommenders, self.metrics, train_data, cv=3,
                                                 user_features=user_features_df)
        self.assertEqual(len(rec_metrics), 3)
        self.assertAlmostEqual(rec_metrics[0]["Random"]["AUC(score)@3"], 0.5679790026246719)
        self.assertAlmostEqual(rec_metrics[0]["Random"]["CTR(score)@3"], 0.2616279069767442)
        self.assertAlmostEqual(rec_metrics[0]["Random"]["Precision@3"], 0.05208333333333333)
        self.assertAlmostEqual(rec_metrics[0]["Random"]["Recall@3"], 0.016161228957922595)
        self.assertAlmostEqual(rec_metrics[0]["Random"]["NDCG@3"], 0.023100096565199995)
        self.assertAlmostEqual(rec_metrics[0]["Random"]["MAP@3"], 0.0398341049382716)

        self.assertEqual(len(recommendations), 3)
        self.assertTrue(isinstance(recommendations[0]["Random"], pd.DataFrame))
        self.assertEqual(recommendations[0]["Random"].shape[1], 3)
        self.assertEqual(recommendations[0]["Random"].ndim, 2)

        recommendations = concat_recommendations_list(recommendations)
        self.assertTrue(isinstance(recommendations["Random"], pd.DataFrame))
        self.assertEqual(recommendations["Random"][Constants.user_id].nunique(),
                         train_data_df[Constants.user_id].nunique())
        self.assertEqual(recommendations["Random"].shape[1], 3)

        for rec in recommenders.values():
            self.assertTrue(rec.mab is None)

    def test_benchmark_diversity_metrics(self):
        recommenders = deepcopy(self.recommenders)
        metrics = []
        metric_params = {'click_column': Constants.score,
                         'user_id_column': Constants.user_id,
                         'item_id_column': Constants.item_id}
        for k in [3, 5]:
            metrics.append(DiversityRecoMetrics.InterListDiversity(**metric_params, k=k,
                                                                   user_sample_size=100))
            metrics.append(DiversityRecoMetrics.IntraListDiversity(**metric_params, k=k,
                                                                   user_sample_size=100,
                                                                   item_features=item_features_df))

        recommendations, rec_metrics = benchmark(recommenders, metrics, train_data, test_data,
                                                 user_features=user_features_df)
        self.assertEqual(recommendations.keys(), self.recommenders.keys())
        self.assertEqual(rec_metrics.keys(), self.recommenders.keys())
        self.assertAlmostEqual(rec_metrics["Random"]["Inter-List Diversity@3"], 0.9856228956228957)
        self.assertAlmostEqual(rec_metrics["Random"]["Inter-List Diversity@5"], 0.9749818181818182)
        self.assertAlmostEqual(rec_metrics["Random"]["Intra-List Diversity@3"], 0.7602157694547105)
        self.assertAlmostEqual(rec_metrics["Random"]["Intra-List Diversity@5"], 0.7547351779782561)