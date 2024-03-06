# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Dict, List, Union, Optional

import pandas as pd
import numpy as np
from mabwiser.utils import Arm, Num

from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy


class BaseTest(unittest.TestCase):

    # A list of valid learning policies
    lps = [LearningPolicy.EpsilonGreedy(),
           LearningPolicy.EpsilonGreedy(epsilon=0),
           LearningPolicy.EpsilonGreedy(epsilon=0.0),
           LearningPolicy.EpsilonGreedy(epsilon=0.5),
           LearningPolicy.EpsilonGreedy(epsilon=1),
           LearningPolicy.EpsilonGreedy(epsilon=1.0),
           LearningPolicy.Popularity(),
           LearningPolicy.Random(),
           LearningPolicy.Softmax(),
           LearningPolicy.Softmax(tau=0.1),
           LearningPolicy.Softmax(tau=0.5),
           LearningPolicy.Softmax(tau=1),
           LearningPolicy.Softmax(tau=1.0),
           LearningPolicy.Softmax(tau=5.0),
           LearningPolicy.ThompsonSampling(),
           LearningPolicy.UCB1(),
           LearningPolicy.UCB1(alpha=0),
           LearningPolicy.UCB1(alpha=0.0),
           LearningPolicy.UCB1(alpha=0.5),
           LearningPolicy.UCB1(alpha=1),
           LearningPolicy.UCB1(alpha=1.0),
           LearningPolicy.UCB1(alpha=5)]

    para_lps = [LearningPolicy.LinGreedy(epsilon=0, l2_lambda=1),
                LearningPolicy.LinGreedy(epsilon=0.5, l2_lambda=1),
                LearningPolicy.LinGreedy(epsilon=1, l2_lambda=1),
                LearningPolicy.LinGreedy(epsilon=0, l2_lambda=0.5),
                LearningPolicy.LinGreedy(epsilon=0.5, l2_lambda=0.5),
                LearningPolicy.LinGreedy(epsilon=1, l2_lambda=0.5),
                LearningPolicy.LinTS(alpha=0.00001, l2_lambda=1),
                LearningPolicy.LinTS(alpha=0.5, l2_lambda=1),
                LearningPolicy.LinTS(alpha=1, l2_lambda=1),
                LearningPolicy.LinTS(alpha=0.00001, l2_lambda=0.5),
                LearningPolicy.LinTS(alpha=0.5, l2_lambda=0.5),
                LearningPolicy.LinTS(alpha=1, l2_lambda=0.5),
                LearningPolicy.LinUCB(alpha=0, l2_lambda=1),
                LearningPolicy.LinUCB(alpha=0.5, l2_lambda=1),
                LearningPolicy.LinUCB(alpha=1, l2_lambda=1),
                LearningPolicy.LinUCB(alpha=0, l2_lambda=0.5),
                LearningPolicy.LinUCB(alpha=0.5, l2_lambda=0.5),
                LearningPolicy.LinUCB(alpha=1, l2_lambda=0.5)]

    # A list of valid context policies
    nps = [NeighborhoodPolicy.LSHNearest(),
           NeighborhoodPolicy.LSHNearest(n_dimensions=1),
           NeighborhoodPolicy.KNearest(),
           NeighborhoodPolicy.KNearest(k=3),
           NeighborhoodPolicy.Radius(),
           NeighborhoodPolicy.TreeBandit(),
           NeighborhoodPolicy.Clusters(),
           NeighborhoodPolicy.Clusters(n_clusters=3),
           NeighborhoodPolicy.Clusters(is_minibatch=True),
           NeighborhoodPolicy.Clusters(n_clusters=3, is_minibatch=True)]

    @staticmethod
    def predict(arms: List[Arm],
                decisions: Union[List, np.ndarray, pd.Series],
                rewards: Union[List, np.ndarray, pd.Series],
                learning_policy: Union[LearningPolicy.EpsilonGreedy, LearningPolicy.Popularity, LearningPolicy.Random,
                                       LearningPolicy.Softmax, LearningPolicy.ThompsonSampling, LearningPolicy.UCB1,
                                       LearningPolicy.LinGreedy, LearningPolicy.LinTS, LearningPolicy.LinUCB],
                neighborhood_policy: Union[None, NeighborhoodPolicy.Clusters, NeighborhoodPolicy.KNearest,
                                           NeighborhoodPolicy.LSHNearest, NeighborhoodPolicy.Radius,
                                           NeighborhoodPolicy.TreeBandit] = None,
                context_history: Union[None, List[Num], List[List[Num]], np.ndarray, pd.DataFrame, pd.Series] = None,
                contexts: Union[None, List[Num], List[List[Num]], np.ndarray, pd.DataFrame, pd.Series] = None,
                apply_sigmoid: bool = True,
                excluded_arms: List[List[Arm]] = None,
                warm_start: bool = False,
                arm_to_features: Dict[Arm, List[Num]] = None,
                top_k: Optional[int] = 5,
                seed: Optional[int] = 123456,
                n_jobs: Optional[int] = 1,
                backend: Optional[str] = None):
        """Sets up a Bandit Recommender and runs the given configuration.
        """

        # Model
        rec = BanditRecommender(learning_policy, neighborhood_policy, top_k, seed, n_jobs, backend)

        # Initialize and train
        rec._init(arms)
        rec.fit(decisions, rewards, context_history)

        # Warm-start
        if warm_start:
            rec.warm_start(arm_to_features, distance_quantile=0.5)

        # Run
        recommendations = rec.recommend(contexts, excluded_arms, return_scores=True, apply_sigmoid=apply_sigmoid)

        return recommendations, rec

    def assertListAlmostEqual(self, list1, list2):
        """
        Asserts that floating values in the given lists (almost) equals to each other
        """
        if not isinstance(list1, list):
            list1 = list(list1)

        if not isinstance(list2, list):
            list2 = list(list2)

        self.assertEqual(len(list1), len(list2))

        for index, val in enumerate(list1):
            self.assertAlmostEqual(val, list2[index])

    @staticmethod
    def is_compatible(lp, np):

        # Special case for TreeBandit lp/np compatibility
        if isinstance(np, NeighborhoodPolicy.TreeBandit):
            return np._is_compatible(lp)

        return True
