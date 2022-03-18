# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, NoReturn, Tuple, Union

import numpy as np
import pandas as pd
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from mabwiser.utils import Arm, Num, check_true
from scipy.special import expit

from mab2rec.utils import Constants
from mab2rec._version import __author__, __email__, __version__, __copyright__

__author__ = __author__
__email__ = __email__
__version__ = __version__
__copyright__ = __copyright__


class BanditRecommender:
    """**Mab2Rec: Multi-Armed Bandit Recommender**

    Mab2Rec is a library to support prototyping and building of bandit-based recommendation algorithms. It is powered
    by MABWiser which supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models.

    Attributes
    ----------

    learning_policy :  MABWiser LearningPolicy
        The learning policy.
    neighborhood_policy : MABWiser NeighborhoodPolicy
        The neighborhood policy.
    top_k : int, default=10
        The number of items to recommend.
    seed : int, Constants.default_seed
        The random seed to initialize the internal random number generator.
    n_jobs : int
        This is used to specify how many concurrent processes/threads should be used for parallelized routines.
        Default value is set to 1.
        If set to -1, all CPUs are used.
        If set to -2, all CPUs but one are used, and so on.
    backend : str, optional
        Specify a parallelization backend implementation supported in the joblib library. Supported options are:
        - “loky” used by default, can induce some communication and memory overhead when exchanging input and
          output data with the worker Python processes.
        - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
        - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
          called function relies a lot on Python objects.
        Default value is None. In this case the default backend selected by joblib will be used.
    mab : MAB
        The multi-armed bandit.

    Examples
    --------
        >>> from mab2rec import BanditRecommender, LearningPolicy
        >>> decisions = ['Arm1', 'Arm1', 'Arm3', 'Arm1', 'Arm2', 'Arm3']
        >>> rewards = [0, 1, 1, 0, 1, 0]
        >>> rec = BanditRecommender(LearningPolicy.EpsilonGreedy(epsilon=0.25), top_k=2)
        >>> rec.fit(decisions, rewards)
        >>> rec.recommend()
        ['Arm2', 'Arm1']
        >>> rec.add_arm('Arm4')
        >>> rec.partial_fit(['Arm4'], [1])
        >>> rec.recommend()[0]
        ['Arm2', 'Arm4']

        >>> from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
        >>> decisions = ['Arm1', 'Arm1', 'Arm3', 'Arm1', 'Arm2', 'Arm3']
        >>> rewards = [0, 1, 1, 0, 1, 0]
        >>> contexts = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0]]
        >>> rec = BanditRecommender(LearningPolicy.EpsilonGreedy(), NeighborhoodPolicy.KNearest(k=3), top_k=2)
        >>> rec.fit(decisions, rewards, contexts)
        >>> rec.recommend([[1, 1, 0], [1, 1, 1], [0, 1, 0]])
        [['Arm2', 'Arm3'], ['Arm3', 'Arm2'], ['Arm3', 'Arm2']]

        >>> from mab2rec import BanditRecommender, LearningPolicy
        >>> decisions = ['Arm1', 'Arm1', 'Arm3', 'Arm1', 'Arm2', 'Arm3']
        >>> rewards = [0, 1, 1, 0, 1, 0]
        >>> contexts = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0]]
        >>> rec = BanditRecommender(LearningPolicy.LinGreedy(epsilon=0.1), top_k=2)
        >>> rec.fit(decisions, rewards, contexts)
        >>> rec.recommend([[1, 1, 0], [1, 1, 1], [0, 1, 0]])
        [['Arm2', 'Arm1'], ['Arm2', 'Arm1'], ['Arm2', 'Arm3']]
        >>> arm_to_features = {'Arm1': [0, 1], 'Arm2': [0, 0], 'Arm3': [0, 0], 'Arm4': [0, 1]}
        >>> rec.add_arm('Arm4')
        >>> rec.warm_start(arm_to_features, distance_quantile=0.75)
        >>> rec.recommend([[1, 1, 0], [1, 1, 1], [0, 1, 0]])
        [['Arm2', 'Arm4'], ['Arm2', 'Arm4'], ['Arm2', 'Arm3']]
    """

    def __init__(self, learning_policy: Union[LearningPolicy.EpsilonGreedy,
                                              LearningPolicy.Popularity,
                                              LearningPolicy.Random,
                                              LearningPolicy.Softmax,
                                              LearningPolicy.ThompsonSampling,
                                              LearningPolicy.UCB1,
                                              LearningPolicy.LinGreedy,
                                              LearningPolicy.LinTS,
                                              LearningPolicy.LinUCB],
                 neighborhood_policy: Union[None,
                                            NeighborhoodPolicy.LSHNearest,
                                            NeighborhoodPolicy.Clusters,
                                            NeighborhoodPolicy.KNearest,
                                            NeighborhoodPolicy.Radius,
                                            NeighborhoodPolicy.TreeBandit] = None,
                 top_k: int = 10,
                 seed: int = Constants.default_seed,
                 n_jobs: int = 1,
                 backend: str = None):
        """Initializes bandit recommender with the given arguments.

        Validates the arguments and raises exception in case there are violations.

        Parameters
        ----------
        learning_policy : LearningPolicy
            The learning policy.
        neighborhood_policy : NeighborhoodPolicy, default=None
            The context policy.
        top_k : int, default=10
            The number of items to recommend.
        seed : numbers.Rational, default=Constants.default_seed
            The random seed to initialize the random number generator.
            Default value is set to Constants.default_seed.value
        top_k : int, default=10
            The number of items to recommend.
        n_jobs : int, default=1
            This is used to specify how many concurrent processes/threads should be used for parallelized routines.
            If set to -1, all CPUs are used.
            If set to -2, all CPUs but one are used, and so on.
        backend : str, default=None
            Specify a parallelization backend implementation supported in the joblib library. Supported options are:
            - “loky” used by default, can induce some communication and memory overhead when exchanging input and
              output data with the worker Python processes.
            - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
            - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
              called function relies a lot on Python objects.
            Default value is None. In this case the default backend selected by joblib will be used.
        """

        # Set given arguments
        self.learning_policy = learning_policy
        self.neighborhood_policy = neighborhood_policy
        self.top_k = top_k
        self.seed = seed
        self.n_jobs = n_jobs
        self.backend = backend

        # Validate that MAB can be instantiated with given arguments
        self.mab = None
        self._validate_mab_args()

    def _init(self, arms: List[Union[Arm]]) -> NoReturn:
        """Initializes recommender with given list of arms.

        Parameters
        ----------
        arms : List[Union[Arm]]
            The list of all of the arms available for decisions.
            Arms can be integers, strings, etc.

        Returns
        -------
        No return.
        """
        self.mab = MAB(arms, self.learning_policy, self.neighborhood_policy, self.seed, self.n_jobs, self.backend)

    def add_arm(self, arm: Arm, binarizer=None) -> NoReturn:
        """Adds an _arm_ to the list of arms.

        Incorporates the arm into the learning and neighborhood policies with no training data.

        Parameters
        ----------
        arm : Arm
            The new arm to be added.
        binarizer : Callable, default=None
            The new binarizer function for Thompson Sampling.

        Returns
        -------
        No return.
        """
        if self.mab is None:
            self._init([arm])
        else:
            self.mab.add_arm(arm, binarizer)

    def fit(self, decisions: Union[List[Arm], np.ndarray, pd.Series],
            rewards: Union[List[Num], np.ndarray, pd.Series],
            contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None) -> NoReturn:
        """Fits the recommender the given *decisions*, their corresponding *rewards* and *contexts*, if any.
        If the recommender arms has not been initialized using the `set_arms`, the recommender arms will be set
        to the list of arms in *decisions*.

        Validates arguments and raises exceptions in case there are violations.

        This function makes the following assumptions:
            - each decision corresponds to an arm of the bandit.
            - there are no ``None``, ``Nan``, or ``Infinity`` values in the contexts.

        Parameters
        ----------
         decisions : Union[List[Arm], np.ndarray, pd.Series]
            The decisions that are made.
         rewards : Union[List[Num], np.ndarray, pd.Series]
            The rewards that are received corresponding to the decisions.
         contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame], default=None
            The context under which each decision is made.

        Returns
        -------
        No return.
        """
        if self.mab is None:
            self._init(np.unique(decisions).tolist())
        self.mab.fit(decisions, rewards, contexts)

    def partial_fit(self, decisions: Union[List[Arm], np.ndarray, pd.Series],
                    rewards: Union[List[Num], np.ndarray, pd.Series],
                    contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None) -> NoReturn:
        """Updates the recommender with the given *decisions*, their corresponding *rewards* and *contexts*, if any.

        Validates arguments and raises exceptions in case there are violations.

        This function makes the following assumptions:
            - each decision corresponds to an arm of the bandit.
            - there are no ``None``, ``Nan``, or ``Infinity`` values in the contexts.

        Parameters
        ----------
         decisions : Union[List[Arm], np.ndarray, pd.Series]
            The decisions that are made.
         rewards : Union[List[Num], np.ndarray, pd.Series]
            The rewards that are received corresponding to the decisions.
         contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame], default=None
            The context under which each decision is made.

        Returns
        -------
        No return.
        """
        self._validate_mab(is_fit=True)
        self.mab.partial_fit(decisions, rewards, contexts)

    def predict(self, contexts: Union[None, List[List[Num]],
                                      np.ndarray, pd.Series, pd.DataFrame] = None) -> Union[Arm, List[Arm]]:
        """Returns the "best" arm (or arms list if multiple contexts are given) based on the expected reward.

        The definition of the *best* depends on the specified learning policy.
        Contextual learning policies and neighborhood policies require contexts data in training.
        In testing, they return the best arm given new context(s).

        Parameters
        ----------
        contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame], default=None
            The context under which each decision is made.
            If contexts is not ``None`` for context-free bandits, the predictions returned will be a
            list of the same length as contexts.

        Returns
        -------
        The recommended arm or recommended arms list.
        """
        self._validate_mab(is_fit=True)
        return self.mab.predict(contexts)

    def predict_expectations(self, contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None) \
            -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        """Returns a dictionary of arms (key) to their expected rewards (value).

        Contextual learning policies and neighborhood policies require contexts data for expected rewards.

        Parameters
        ----------
        contexts : Union[None, List[Num], List[List[Num]], np.ndarray, pd.Series, pd.DataFrame], default=None
            The context for the expected rewards.
            If contexts is not ``None`` for context-free bandits, the predicted expectations returned will be a
            list of the same length as contexts.

        Returns
        -------
        The dictionary of arms (key) to their expected rewards (value), or a list of such dictionaries.
        """
        self._validate_mab(is_fit=True)
        return self.mab.predict_expectations(contexts)

    def recommend(self, contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None,
                  excluded_arms: List[List[Arm]] = None, return_scores: bool = False) \
            -> Union[Union[List[Arm], Tuple[List[Arm], List[Num]],
                     Union[List[List[Arm]], Tuple[List[List[Arm]], List[List[Num]]]]]]:
        """Generate _top-k_ recommendations based on the expected reward.

        Recommend up to k arms with the highest predicted expectations.
        For contextual bandits, only items not included in the excluded arms can be recommended.

        Parameters
        ----------
        contexts : np.ndarray, default=None
            The context under which each decision is made.
            If contexts is not ``None`` for context-free bandits, the recommendations returned will be a
            list of the same length as contexts.
        excluded_arms : List[List[Arm]], default=None
            List of list of arms to exclude from recommended arms.
        return_scores : bool, default=False
            Return score for each recommended item.

        Returns
        -------
        List of tuples of the form ([arm_1, arm_2, ..., arm_k], [score_1, score_2, ..., score_k])
        """
        self._validate_mab(is_fit=True)
        self._validate_get_rec(contexts, excluded_arms)

        # Get predicted expectations
        if contexts is None:
            num_contexts = 1
            expectations = [self.mab.predict_expectations(contexts)]
        else:
            num_contexts = len(contexts)
            expectations = self.mab.predict_expectations(contexts)

        # Take sigmoid of expectations so that values are between 0 and 1
        expectations = expit(pd.DataFrame(expectations)[self.mab.arms].values)

        # Create an exclusion mask, where exclusion_mask[context_ind][arm_ind] denotes if the arm with the
        # index arm_ind was excluded for context with the index context_ind.
        # The value will be True if it is excluded and those arms will not be returned as part of the results.
        arm_to_index = {arm: arm_ind for arm_ind, arm in enumerate(self.mab.arms)}
        exclude_mask = np.zeros((num_contexts, len(self.mab.arms)), dtype=bool)
        if excluded_arms is not None:
            for context_ind, excluded in enumerate(excluded_arms):
                exclude_mask[context_ind][[arm_to_index[arm] for arm in excluded if arm in arm_to_index]] = True

        # Get best `top_k` results by sorting the expectations
        arm_inds = np.flip(np.argsort(expectations)[:, -self.top_k:], axis=1)

        # Get the list of top_k recommended items and corresponding expectations for each context
        recommendations = [[]] * num_contexts
        scores = [[]] * num_contexts
        for context_ind in range(num_contexts):
            recommendations[context_ind] = [self.mab.arms[arm_ind] for arm_ind in arm_inds[context_ind]
                                            if not exclude_mask[context_ind, arm_ind]]
            if return_scores:
                scores[context_ind] = [expectations[context_ind, arm_ind] for arm_ind in arm_inds[context_ind]
                                       if not exclude_mask[context_ind, arm_ind]]

        # Return recommendations and scores
        if return_scores:
            if num_contexts > 1:
                return recommendations, scores
            else:
                return recommendations[0], scores[0]
        else:
            if num_contexts > 1:
                return recommendations
            else:
                return recommendations[0]

    def remove_arm(self, arm: Arm) -> NoReturn:
        """Removes an _arm_ from the list of arms.

        Parameters
        ----------
        arm : Arm
            The existing arm to be removed.

        Returns
        -------
        No return.
        """
        self._validate_mab()
        self.mab.remove_arm(arm)

    def set_arms(self, arms: List[Arm], binarizer=None) -> NoReturn:
        """Initializes the recommender and sets the recommender with given list of arms.
        Existing arms not in the given list of arms are removed and new arms are incorporated into the learning and
        neighborhood policies with no training data.
        If the recommender has already been initialized it will not be re-initialized.

        Parameters
        ----------
        arms : List[Arm]
            The new arm to be added.
        binarizer : Callable, default=None
            The new binarizer function for Thompson Sampling.

        Returns
        -------
        No return.
        """

        # Initialize mab
        if self.mab is None:
            self._init(arms)

        # Remove arms
        arms_to_remove = []
        for existing_arm in self.mab.arms:
            if existing_arm not in arms:
                arms_to_remove.append(existing_arm)
        for arm in arms_to_remove:
            self.remove_arm(arm)

        # Add arms
        for new_arm in arms:
            if new_arm not in self.mab.arms:
                self.add_arm(new_arm, binarizer)

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float = None) -> NoReturn:
        """Warm-start untrained (cold) arms of the multi-armed bandit.

        Validates arguments and raises exceptions in case there are violations.

        Parameters
        ----------
        arm_to_features : Dict[Arm, List[Num]]
            Numeric representation for each arm.
        distance_quantile : float, default=None
            Value between 0 and 1 used to determine if an item can be warm started or not using closest item.
            All cold items will be warm started if 1 and none will be warm started if 0.

        Returns
        -------
        No return.
        """
        self._validate_mab(is_fit=True)
        self.mab.warm_start(arm_to_features, distance_quantile)

    def _validate_mab_args(self):
        _ = MAB([1], self.learning_policy, self.neighborhood_policy, self.seed, self.n_jobs, self.backend)
        check_true(isinstance(self.top_k, int), ValueError("Top k should be an integer."))
        check_true(self.top_k > 0, ValueError("Top k should be positive."))

    def _validate_mab(self, is_fit=False):
        check_true(self.mab is not None, ValueError("Recommender has not been initialized."))
        if is_fit:
            check_true(self.mab._is_initial_fit, ValueError("Recommender has not been fit."))

    @staticmethod
    def _validate_get_rec(contexts, excluded_arms):
        if excluded_arms is not None:
            check_true(contexts is not None,
                       ValueError("Excluded arms should either be None, or a list of exclusion lists per context."))
            check_true(len(excluded_arms) == len(contexts),
                       ValueError("Excluded arms should either be None, or a list of exclusion lists per context."))
