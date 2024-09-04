# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from mab2rec.pipeline import benchmark
from mab2rec.visualization import (plot_inter_diversity_at_k, plot_intra_diversity_at_k, plot_metrics_at_k,
                                   plot_num_items_per_recommendation, plot_recommended_counts,
                                   plot_recommended_counts_by_item, plot_personalization_heatmap)
from mab2rec.utils import default_metrics, print_interaction_stats
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


class VisualizationTest(BaseTest):

    recommenders = {
        "Random": BanditRecommender(LearningPolicy.Random()),
        "LinUCB": BanditRecommender(LearningPolicy.LinUCB(alpha=1.5)),
        "ClustersTS": BanditRecommender(LearningPolicy.ThompsonSampling(), NeighborhoodPolicy.Clusters(n_clusters=10))
    }
    metrics = default_metrics(top_k_values=[3, 5, 10])
    recommendations, rec_metrics = benchmark(recommenders, metrics, train_data, test_data,
                                             user_features=user_features_df)
    recommendations_cv, rec_metrics_cv = benchmark(recommenders, metrics, train_data, cv=3,
                                                   user_features=user_features_df)

    @patch("mab2rec.visualization.plt.show")
    def test_plot_metrics_at_k(self, mock_show):
        plot_metrics_at_k(self.rec_metrics)
        plot_metrics_at_k(self.rec_metrics_cv)
        plt.close()

    @patch("mab2rec.visualization.plt.show")
    def test_plot_inter_diversity_at_k(self, mock_show):
        plot_inter_diversity_at_k(self.recommendations, k_list=[3, 5, 10])
        plot_inter_diversity_at_k(self.recommendations_cv, k_list=[3, 5, 10])
        plt.close()

    @patch("mab2rec.visualization.plt.show")
    def test_plot_intra_diversity_at_k(self, mock_show):
        plot_intra_diversity_at_k(self.recommendations, item_features_df, k_list=[3, 5, 10])
        plot_intra_diversity_at_k(self.recommendations_cv, item_features_df, k_list=[3, 5, 10])
        plt.close()

    @patch("mab2rec.visualization.plt.show")
    def test_plot_recommended_counts(self, mock_show):
        plot_recommended_counts(self.recommendations, test_data_df, k=3, alpha=0.7, average_response=False)
        plot_recommended_counts(self.recommendations, test_data_df, k=3, alpha=0.7, average_response=True)
        plot_recommended_counts(self.recommendations_cv, test_data_df, k=3, alpha=0.7, average_response=False)
        plt.close()

    @patch("mab2rec.visualization.plt.show")
    def test_plot_recommended_counts_by_item(self, mock_show):
        plot_recommended_counts_by_item(self.recommendations, k=3, top_n_items=15, normalize=False)
        plot_recommended_counts_by_item(self.recommendations, k=3, top_n_items=15, normalize=True)
        plot_recommended_counts_by_item(self.recommendations_cv, k=3, top_n_items=15, normalize=False)
        plt.close()

    @patch("mab2rec.visualization.plt.show")
    def test_plot_num_items_per_recommendation(self, mock_show):
        plot_num_items_per_recommendation(self.recommendations, test_data_df, normalize=False)
        plot_num_items_per_recommendation(self.recommendations, test_data_df, normalize=True)
        plot_num_items_per_recommendation(self.recommendations_cv, test_data_df, normalize=False)
        plt.close()

    @patch("mab2rec.visualization.plt.show")
    def test_plot_personalization_heatmap(self, mock_show):
        # Create clusters based on user features
        X = user_features_df.iloc[:, 1:]
        kmeans = KMeans(n_clusters=5, n_init=10, random_state=1652)
        kmeans.fit(X)
        user_clusters = dict(zip(user_features_df['user_id'], kmeans.labels_))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        plot_personalization_heatmap(self.recommendations, user_clusters, k=3, cmap=cmap, vmax=0.2, figsize=(5, 5))
        plot_personalization_heatmap(self.recommendations_cv, user_clusters, k=3, cmap=cmap, vmax=0.2, figsize=(5, 5))
        plt.close()

    @patch("mab2rec.utils.print")
    def test_print_interaction_stats(self, mock_show):
        print_interaction_stats(train_data_df)
