# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict, Optional

from spock import SpockBuilder
from spock.config import spock

from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from mab2rec.utils import Constants


@spock
class EpsilonGreedy:
    """Epsilon Greedy Learning Policy.

    Attributes
    ----------
    epsilon: Num
        The probability of selecting a random arm for exploration.
        Integer or float. Must be between 0 and 1.
        Default value is 0.1.
    """
    epsilon: float = 0.1


@spock
class LinGreedy:
    """LinGreedy Learning Policy

    Attributes
    ----------
    epsilon: Num
        The probability of selecting a random arm for exploration.
        Integer or float. Must be between 0 and 1.
        Default value is 0.1.
    l2_lambda: Num
        The regularization strength.
        Integer or float. Must be greater than zero.
        Default value is 1.0.
    scale: bool
        Whether to scale features to have zero mean and unit variance.
        Uses StandardScaler in sklearn.preprocessing.
        Default value is False.
    """
    epsilon: float = 0.1
    l2_lambda: float = 1.0
    scale: bool = False


@spock
class LinTS:
    """LinTS Learning Policy

    Attributes
    ----------
    alpha: Num
        The multiplier to determine the degree of exploration.
        Integer or float. Must be greater than zero.
        Default value is 1.0.
    l2_lambda: Num
        The regularization strength.
        Integer or float. Must be greater than zero.
        Default value is 1.0.
    scale: bool
        Whether to scale features to have zero mean and unit variance.
        Uses StandardScaler in sklearn.preprocessing.
        Default value is False.
    """
    alpha: float = 1.0
    l2_lambda: float = 1.0
    scale: bool = False


@spock
class LinUCB:
    """LinUCB Learning Policy.

    Attributes
    ----------
    alpha: Num
        The parameter to control the exploration.
        Integer or float. Cannot be negative.
        Default value is 1.0.
    l2_lambda: Num
        The regularization strength.
        Integer or float. Cannot be negative.
        Default value is 1.0.
    scale: bool
        Whether to scale features to have zero mean and unit variance.
        Uses StandardScaler in sklearn.preprocessing.
        Default value is False.
    """
    alpha: float = 1.0
    l2_lambda: float = 1.0
    scale: bool = False


@spock
class Popularity:
    """Randomized Popularity Learning Policy."""
    none: Optional[str] = None


@spock
class Random:
    """Random Learning Policy."""
    none: Optional[str] = None


@spock
class Softmax:
    """Softmax Learning Policy.

      Attributes
      ----------
      tau: Num
           The temperature to control the exploration.
           Integer or float. Must be greater than zero.
           Default value is 1.
    """
    tau: float = 1.0


@spock
class ThompsonSampling:
    """Thompson Sampling Learning Policy."""
    none: Optional[str] = None


@spock
class UCB1:
    """Upper Confidence Bound1 Learning Policy.

      Attributes
      ----------
      alpha: Num
          The parameter to control the exploration.
          Integer of float. Cannot be negative.
          Default value is 1.
    """
    alpha: float = 1.0


@spock
class Clusters:
    """Clusters Neighborhood Policy.

    Attributes
    ----------
    n_clusters: Num
        The number of clusters. Integer. Must be at least 2. Default value is 2.
    is_minibatch: bool
        Boolean flag to use ``MiniBatchKMeans`` or not. Default value is False.
    """
    n_clusters: int = 2
    is_minibatch: bool = False


@spock
class KNearest:
    """KNearest Neighborhood Policy.

    Attributes
    ----------
    k: int
        The number of neighbors to select.
        Integer value. Must be greater than zero.
        Default value is 1.
    metric: str
        The metric used to calculate distance.
        Accepts any of the metrics supported by ``scipy.spatial.distance.cdist``.
        Default value is Euclidean distance.
    """
    k: int = 1
    metric: str = "euclidean"


@spock
class LSHNearest:
    """Locality-Sensitive Hashing Approximate Nearest Neighbors Policy.

    Attributes
    ----------
    n_dimensions: int
        The number of dimensions to use for the hyperplane.
        Integer value. Must be greater than zero.
        Default value is 5.
    n_tables: int
        The number of hash tables.
        Integer value. Must be greater than zero.
        Default value is 3.
    """
    n_dimensions: int = 5
    n_tables: int = 3


@spock
class Radius:
    """Radius Neighborhood Policy.

    Radius is a nearest neighborhood approach that selects the observations
    within a given *radius* to be used with a learning policy.

    Attributes
    ----------
    radius: Num
        The maximum distance within which to select observations.
        Integer or Float. Must be greater than zero.
        Default value is 1.
    metric: str
        The metric used to calculate distance.
        Accepts any of the metrics supported by scipy.spatial.distance.cdist.
        Default value is Euclidean distance.
    """
    radius: float = 0.05
    metric: str = "euclidean"


@spock
class TreeBandit:
    """TreeBandit Neighborhood Policy.

    Attributes
    ----------
    tree_parameters: Dict, **kwarg
        Parameters of the decision tree.
        The keys must match the parameters of sklearn.tree.DecisionTreeClassifier.
        When a parameter is not given, the default parameters from
        sklearn.tree.DecisionTreeClassifier will be chosen.
        Default value is an empty dictionary.
    """
    tree_parameters: Dict = {}


class LearningPolicyOptions(Enum):
    """Learning Policies
    """
    epsilon_greedy = EpsilonGreedy
    lin_greedy = LinGreedy
    lin_ts = LinTS
    lin_ucb = LinUCB
    popularity = Popularity
    random = Random
    softmax = Softmax
    ts = ThompsonSampling
    ucb1 = UCB1


class NeighborhoodPolicyOptions(Enum):
    """Neighborhood Policies"""
    clusters = Clusters
    knearest = KNearest
    lshnearest = LSHNearest
    radius = Radius
    treebandit = TreeBandit


@spock
class Recommender:
    learning_policy: LearningPolicyOptions
    neighborhood_policy: Optional[NeighborhoodPolicyOptions] = None
    top_k: Optional[int] = 10
    seed: Optional[int] = 123456
    n_jobs: Optional[int] = 1
    backend: Optional[str] = None


@spock
class RecommenderScore:
    recommender_pickle: str


@spock
class DataConfig:
    data: str
    user_features: Optional[str]
    user_features_list: Optional[str]
    user_features_dtypes: Optional[str]
    item_features: Optional[str]
    item_list: Optional[str]
    item_eligibility: Optional[str]
    warm_start: Optional[bool] = False
    warm_start_distance: Optional[float] = None
    user_id_col: Optional[str] = Constants.user_id
    item_id_col: Optional[str] = Constants.item_id
    response_col: Optional[str] = Constants.response
    batch_size: Optional[int] = 100000
    save_file: Optional[str] = None


def init_recommender(config):

    # Learning policy
    lp_params = config.learning_policy
    if isinstance(lp_params, EpsilonGreedy):
        lp = LearningPolicy.EpsilonGreedy(epsilon=lp_params.epsilon)
    elif isinstance(lp_params, LinGreedy):
        lp = LearningPolicy.LinGreedy(epsilon=lp_params.epsilon, l2_lambda=lp_params.l2_lambda, scale=lp_params.scale)
    elif isinstance(lp_params, LinTS):
        lp = LearningPolicy.LinTS(alpha=lp_params.alpha, l2_lambda=lp_params.l2_lambda, scale=lp_params.scale)
    elif isinstance(lp_params, LinUCB):
        lp = LearningPolicy.LinUCB(alpha=lp_params.alpha, l2_lambda=lp_params.l2_lambda, scale=lp_params.scale)
    elif isinstance(lp_params, Popularity):
        lp = LearningPolicy.Popularity()
    elif isinstance(lp_params, Random):
        lp = LearningPolicy.Random()
    elif isinstance(lp_params, Softmax):
        lp = LearningPolicy.Softmax(tau=lp_params.tau)
    elif isinstance(lp_params, ThompsonSampling):
        lp = LearningPolicy.ThompsonSampling()
    elif isinstance(lp_params, UCB1):
        lp = LearningPolicy.UCB1(alpha=lp_params.alpha)
    else:
        raise NotImplemented(f'{lp_params} not implemented')

    # Neighborhood policy
    cp_params = config.neighborhood_policy
    if isinstance(lp_params, Clusters):
        cp = NeighborhoodPolicy.Clusters(n_clusters=cp_params.n_clusters, is_minibatch=cp_params.is_minibatch)
    elif isinstance(lp_params, KNearest):
        cp = NeighborhoodPolicy.KNearest(k=cp_params.k, metric=cp_params.metric)
    elif isinstance(lp_params, LSHNearest):
        cp = NeighborhoodPolicy.LSHNearest(n_dimensions=cp_params.n_dimensions, n_tables=cp_params.n_tables)
    elif isinstance(lp_params, Radius):
        cp = NeighborhoodPolicy.Radius(radius=cp_params.radius, metric=cp_params.metric)
    elif isinstance(lp_params, TreeBandit):
        cp = NeighborhoodPolicy.TreeBandit(tree_parameters=cp_params.tree_parameters)
    elif cp_params is None:
        cp = None
    else:
        raise NotImplemented(f'{cp_params} not implemented')

    return BanditRecommender(lp, cp, config.top_k, config.seed, config.n_jobs, config.backend)


def train_config():

    config = SpockBuilder(DataConfig, Recommender,
                          EpsilonGreedy, LinGreedy, LinTS, LinUCB, Popularity, Random, Softmax, ThompsonSampling, UCB1,
                          Clusters, KNearest, LSHNearest, Radius, TreeBandit,
                          desc='Train Recommender').generate()

    data_config = config.DataConfig
    rec = init_recommender(config.Recommender)
    args = {
        'recommender': rec,
        'data': data_config.data,
        'user_features': data_config.user_features,
        'user_features_list': data_config.user_features_list,
        'user_features_dtypes': data_config.user_features_dtypes,
        'item_features': data_config.item_features,
        'item_list': data_config.item_list,
        'item_eligibility': data_config.item_eligibility,
        'warm_start': data_config.warm_start,
        'warm_start_distance': data_config.warm_start_distance,
        'user_id_col': data_config.user_id_col,
        'item_id_col': data_config.item_id_col,
        'response_col': data_config.response_col,
        'batch_size': data_config.batch_size,
        'save_file': data_config.save_file,
    }
    return args


def score_config():

    config = SpockBuilder(DataConfig, RecommenderScore, desc='Score Recommender').generate()

    data_config = config.DataConfig
    args = {
        'recommender': config.RecommenderScore.recommender_pickle,
        'data': data_config.data,
        'user_features': data_config.user_features,
        'user_features_list': data_config.user_features_list,
        'user_features_dtypes': data_config.user_features_dtypes,
        'item_features': data_config.item_features,
        'item_list': data_config.item_list,
        'item_eligibility': data_config.item_eligibility,
        'warm_start': data_config.warm_start,
        'warm_start_distance': data_config.warm_start_distance,
        'user_id_col': data_config.user_id_col,
        'item_id_col': data_config.item_id_col,
        'response_col': data_config.response_col,
        'batch_size': data_config.batch_size,
        'save_file': data_config.save_file,
    }
    return args
