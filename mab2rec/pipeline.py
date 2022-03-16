# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from typing import Dict, List, NoReturn, Tuple, Union

import numpy as np
import pandas as pd
from jurity.recommenders import CombinedMetrics, BinaryRecoMetrics, RankingRecoMetrics
from sklearn.model_selection import GroupKFold
from mabwiser.utils import check_true, Arm

from mab2rec.rec import BanditRecommender
from mab2rec.utils import Constants
from mab2rec.utils import explode_recommendations, load_data, load_response_data, merge_user_features
from mab2rec.utils import load_pickle, save_pickle


def train(recommender: BanditRecommender,
          data: Union[str, pd.DataFrame],
          user_features: Union[str, pd.DataFrame] = None,
          user_features_list: Union[str, List[str]] = None,
          user_features_dtypes: Union[str, Dict] = None,
          item_features: Union[str, pd.DataFrame] = None,
          item_list: Union[str, List[Arm]] = None,
          item_eligibility: Union[str, pd.DataFrame] = None,
          warm_start: bool = False,
          warm_start_distance: float = None,
          user_id_col: str = Constants.user_id,
          item_id_col: str = Constants.item_id,
          response_col: str = Constants.response,
          batch_size: int = 100000,
          save_file: Union[str, bool] = None) -> NoReturn:
    """
    Trains Recommender.

    Parameters
    ----------
    recommender : BanditRecommender
        The recommender algorithm to be trained.
        The recommender object is updated in-place.
    data : Union[str, pd.DataFrame]
        Training data.
        Data should have a row for each training sample (user_id, item_id, response).
        Column names should be consistent with user_id_col, item_id_col and response_col arguments.
        CSV format with file header or Data Frame.
    user_features : Union[str, pd.DataFrame], default=None
        User features containing features for each user_id.
        Each row should include user_id and list of features (user_id, u_1, u_2, ..., u_p).
        CSV format with file header or Data Frame.
    user_features_list : Union[str, List[str]], default=None
        List of user features to use.
        Must be a subset of features in (u_1, u_2, ... u_p).
        If None, all the features in user_features are used.
        CSV format with file header or List.
    user_features_dtypes: Union[str, Dict], default=None
        Data type for each user feature.
        Maps each user feature name to valid data type.
        If none, no data type casting is done upon load and data types or inferred by Pandas library.
        JSON format or Dictionary.
    item_features : Union[str, pd.DataFrame], default=None
        Item features file containing features for each item_id.
        Each row should include item_id and list of features (item_id, i_1, i_2, .... i_q).
        CSV format with file header or Data Frame.
    item_list : Union[str, List[Arm]], default=None
        List of items to train.
        If None, all the items in data are used.
        CSV format with file header or List.
    item_eligibility: Union[str, pd.DataFrame], default=None
        Items each user is eligible for.
        Not used during training.
        CSV format with file header or Data Frame.
    warm_start : bool, default=False
        Whether to warm start untrained (cold) arms after training or not.
    warm_start_distance : float, default=None
        Warm start distance quantile.
        Value between 0 and 1 used to determine if an item can be warm started or not using closest item.
        All cold items will be warm started if 1 and none will be warm started if 0.
        Must be specified if warm_start=True.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    response_col : str, default=Constants.response
        Response column name.
    batch_size : str, default=100000
        Batch size used for chunking data.
    save_file : Union[str, bool], default=None
        File name to save recommender pickle.
        If None, recommender is not saved to file.

    Returns
    -------
    No Return.
    """
    _validate_recommender(recommender)
    _validate_common_args(data, user_features, user_features_list, user_features_dtypes, item_features, item_list,
                          item_eligibility, warm_start, warm_start_distance, user_id_col, item_id_col,
                          response_col, batch_size)
    _validate_save(save_file)

    # Import data
    train_data_df, item_list, user_features_df, \
        item_to_features, _ = load_data(data=data,
                                        user_features=user_features,
                                        user_features_list=user_features_list,
                                        user_features_dtypes=user_features_dtypes,
                                        item_features=item_features,
                                        item_list=item_list,
                                        item_eligibility=item_eligibility,
                                        user_id_col=user_id_col,
                                        item_id_col=item_id_col,
                                        response_col=response_col)

    # Initialize and set arms of recommender
    recommender.set_arms(item_list)

    # Loop through the data in batches and fit recommender
    num_batches = max(1, len(train_data_df) // batch_size)
    for df in np.array_split(train_data_df, num_batches):
        if recommender.mab.is_contextual:
            check_true(user_features_df is not None, ValueError("User features are required for contextual bandits."))
            feature_cols = [c for c in user_features_df.columns if c != user_id_col]
            df = merge_user_features(pd.DataFrame(df), user_features_df, user_id_col)
            if recommender.mab._is_initial_fit:
                recommender.partial_fit(df[item_id_col], df[response_col], df[feature_cols])
            else:
                recommender.fit(df[item_id_col], df[response_col], df[feature_cols])
        else:
            if recommender.mab._is_initial_fit:
                recommender.partial_fit(df[item_id_col], df[response_col])
            else:
                recommender.fit(df[item_id_col], df[response_col])

    # Warm start
    if warm_start:
        recommender.warm_start(item_to_features, warm_start_distance)

    # Save file
    if save_file is not None:
        if isinstance(save_file, str):
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            save_pickle(recommender, save_file)
        elif save_file:
            save_pickle(recommender, "recommender.pkl")


def score(recommender: Union[str, BanditRecommender],
          data: Union[str, pd.DataFrame],
          user_features: Union[str, pd.DataFrame] = None,
          user_features_list: Union[str, List[str]] = None,
          user_features_dtypes: Union[str, Dict] = None,
          item_features: Union[str, pd.DataFrame] = None,
          item_list: Union[str, List[Arm]] = None,
          item_eligibility: Union[str, pd.DataFrame] = None,
          warm_start: bool = False,
          warm_start_distance: float = None,
          user_id_col: str = Constants.user_id,
          item_id_col: str = Constants.item_id,
          response_col: str = Constants.response,
          batch_size: int = 100000,
          save_file: Union[str, bool] = None) -> pd.DataFrame:
    """
    Score Recommender.

    Generates top-k recommendations for users in given data.

    Parameters
    ----------
    recommender : Union[str, BanditRecommender]
        The recommender algorithm to be scored.
        Could be an instantiated BanditRecommender or file path of serialized recommender in pickle file.
    data : Union[str, pd.DataFrame]
        Training data.
        Data should have a row for each training sample (user_id, item_id, response).
        Column names should be consistent with user_id_col, item_id_col and response_col arguments.
        CSV format with file header or Data Frame.
    user_features : Union[str, pd.DataFrame], default=None
        User features containing features for each user_id.
        Each row should include user_id and list of features (user_id, u_1, u_2, ..., u_p).
        CSV format with file header or Data Frame.
    user_features_list : Union[str, List[str]], default=None
        List of user features to use.
        Must be a subset of features in (u_1, u_2, ... u_p).
        If None, all the features in user_features are used.
        CSV format with file header or List.
    user_features_dtypes: Union[str, Dict], default=None
        Data type for each user feature.
        Maps each user feature name to valid data type.
        If none, no data type casting is done upon load and data types or inferred by Pandas library.
        JSON format or Dictionary.
    item_features : Union[str, pd.DataFrame], default=None
        Item features file containing features for each item_id.
        Each row should include item_id and list of features (item_id, i_1, i_2, .... i_q).
        CSV format with file header or Data Frame.
    item_list : Union[str, List[Arm]], default=None
        List of items to train.
        If None, all the items in data are used.
        CSV format with file header or List.
    item_eligibility: Union[str, pd.DataFrame], default=None
        Items each user is eligible for.
        Used to generate excluded_arms lists.
        If None, all the items can be evaluated for recommendation for each user.
        CSV format with file header or Data Frame.
    warm_start : bool, default=False
        Whether to warm start untrained (cold) arms after training or not.
    warm_start_distance : float, default=None
        Warm start distance quantile.
        Value between 0 and 1 used to determine if an item can be warm started or not using closest item.
        All cold items will be warm started if 1 and none will be warm started if 0.
        Must be specified if warm_start=True.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    response_col : str, default=Constants.response
        Response column name.
    batch_size : str, default=100000
        Batch size used for chunking data.
    save_file : str, default=None
        File name to save recommender pickle.
        If None, recommender is not saved to file.

    Returns
    -------
    Scored recommendations.
    """
    # Load recommender
    if isinstance(recommender, str):
        recommender = load_pickle(recommender)
    _validate_recommender(recommender, is_fit=True)
    _validate_common_args(data, user_features, user_features_list, user_features_dtypes, item_features,
                          item_list, item_eligibility, warm_start, warm_start_distance, user_id_col, item_id_col,
                          response_col, batch_size)
    _validate_save(save_file)

    # Import data
    test_data_df, item_list_out, user_features_df, \
        item_to_features, excluded_df = load_data(data=data,
                                                  user_features=user_features,
                                                  user_features_list=user_features_list,
                                                  user_features_dtypes=user_features_dtypes,
                                                  item_features=item_features,
                                                  item_list=item_list,
                                                  item_eligibility=item_eligibility,
                                                  user_id_col=user_id_col,
                                                  item_id_col=item_id_col,
                                                  response_col=response_col)
    # Set arms to recommender
    if item_list is not None:
        recommender.set_arms(item_list_out)

    # Warm start
    if warm_start:
        recommender.warm_start(item_to_features, warm_start_distance)

    # Loop through users in batches and get recommendations
    users = test_data_df[user_id_col].unique().tolist()
    recommendations = []
    scores = []
    num_batches = max(1, len(users) // batch_size)
    for users_of_batch in np.array_split(users, num_batches):
        df = pd.DataFrame({user_id_col: users_of_batch})
        if recommender.mab.is_contextual:

            # Merge user features and get feature column names
            df = merge_user_features(df, user_features_df, user_id_col)
            feature_cols = [c for c in user_features_df.columns if c != user_id_col]

            # Merge excluded item list
            if item_eligibility is not None:
                df = df.merge(excluded_df, how='left', on=user_id_col)
                excluded_arms_batch = df[item_id_col].tolist()
            else:
                excluded_arms_batch = None

            # Get recommendations
            recs_of_batch, scores_of_batch = recommender.recommend(df[feature_cols], excluded_arms_batch,
                                                                   return_scores=True)
        else:
            recs_of_batch = [[]] * len(df)
            scores_of_batch = [[]] * len(df)
            for i in range(len(df)):
                recs_of_batch[i], scores_of_batch[i] = recommender.recommend(return_scores=True)

        recommendations += recs_of_batch
        scores += scores_of_batch

    # Convert recommendations to data frame
    df = pd.DataFrame(users, columns=[user_id_col])
    df[item_id_col] = recommendations
    df[Constants.score] = scores
    df = explode_recommendations(df, user_id_col, [item_id_col, Constants.score])

    # Save to csv
    if save_file is not None:
        if isinstance(save_file, str):
            df.to_csv(save_file, index=False)
        elif save_file:
            df.to_csv("results.csv", index=False)

    return df


def benchmark(recommenders: Dict[str, BanditRecommender],
              metrics: List[Union[BinaryRecoMetrics, RankingRecoMetrics]],
              train_data: Union[str, pd.DataFrame],
              test_data: Union[str, pd.DataFrame] = None,
              cv: int = None,
              user_features: Union[str, pd.DataFrame] = None,
              user_features_list: Union[str, List[str]] = None,
              user_features_dtypes: Union[str, Dict] = None,
              item_features: Union[str, pd.DataFrame] = None,
              item_list: Union[str, List[Arm]] = None,
              item_eligibility: Union[str, pd.DataFrame] = None,
              warm_start: bool = False,
              warm_start_distance: float = None,
              user_id_col: str = Constants.user_id,
              item_id_col: str = Constants.item_id,
              response_col: str = Constants.response,
              batch_size: int = 100000) -> Union[Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]],
                                                 Tuple[List[Dict[str, pd.DataFrame]],
                                                       List[Dict[str, Dict[str, float]]]]]:

    """
    Benchmark Recommenders.

    Benchmark a given set of recommender algorithms by training, scoring and evaluating each algorithm
    If using cross-validation (cv) it benchmarks the algorithms on cv-many folds from the train data,
    otherwise it trains on the train data and evaluates on the test data.

    Parameters
    ----------
    recommenders : Dict[str, BanditRecommender]
        The recommender algorithms to be benchmarked.
        Dictionary with names (key) and recommender algorithms (value).
    metrics : List[Union[BinaryRecoMetrics, RankingRecoMetrics]]
        List of metrics used to evaluate recommendations.
    train_data : Union[str, pd.DataFrame]
        Training data used to train recommenders.
        Data should have a row for each training sample (user_id, item_id, response).
        Column names should be consistent with user_id_col, item_id_col and response_col arguments.
        CSV format with file header or Data Frame.
    test_data : Union[str, pd.DataFrame]
        Test data used to generate recommendations.
        Data should have a row for each training sample (user_id, item_id, response).
        Column names should be consistent with user_id_col, item_id_col and response_col arguments.
        CSV format with file header or Data Frame.
    cv : int, default=None
        Number of folds in the train data to use for cross-fold validation.
        A grouped K-fold iterator is used to ensure that the same user is not contained in different folds.
        Test data must be None when using cv.
    user_features : Union[str, pd.DataFrame], default=None
        User features containing features for each user_id.
        Each row should include user_id and list of features (user_id, u_1, u_2, ..., u_p).
        CSV format with file header or Data Frame.
    user_features_list : Union[str, List[str]], default=None
        List of user features to use.
        Must be a subset of features in (u_1, u_2, ... u_p).
        If None, all the features in user_features are used.
        CSV format with file header or List.
    user_features_dtypes: Union[str, Dict], default=None
        Data type for each user feature.
        Maps each user feature name to valid data type.
        If none, no data type casting is done upon load and data types or inferred by Pandas library.
        JSON format or Dictionary.
    item_features : Union[str, pd.DataFrame], default=None
        Item features file containing features for each item_id.
        Each row should include item_id and list of features (item_id, i_1, i_2, .... i_q).
        CSV format with file header or Data Frame.
    item_list : Union[str, List[Arm]], default=None
        List of items to train.
        If None, all the items in data are used.
        CSV format with file header or List.
    item_eligibility: Union[str, pd.DataFrame], default=None
        Items each user is eligible for.
        Used to generate excluded_arms lists.
        If None, all the items can be evaluated for recommendation for each user.
        CSV format with file header or Data Frame.
    warm_start : bool, default=False
        Whether to warm start untrained (cold) arms after training or not.
    warm_start_distance : float, default=None
        Warm start distance quantile.
        Value between 0 and 1 used to determine if an item can be warm started or not using closest item.
        All cold items will be warm started if 1 and none will be warm started if 0.
        Must be specified if warm_start=True.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    response_col : str, default=Constants.response
        Response column name.
    batch_size : str, default=100000
        Batch size used for chunking data.

    Returns
    -------
    Tuple with recommendations and evaluation metrics for each algorithm.
    The tuple values are lists of dictionaries if cross-validation is used, representing the results on each fold,
    and individual dictionaries otherwise.
    """
    _validate_recommender(recommenders)
    _validate_common_args(train_data, user_features, user_features_list, user_features_dtypes, item_features,
                          item_list, item_eligibility, warm_start, warm_start_distance, user_id_col,
                          item_id_col, response_col, batch_size)
    _validate_bench(recommenders, metrics, train_data, test_data, cv)

    # Convert input arguments to dictionary
    args = locals()
    args.pop('cv')

    if cv is None:
        return _bench(**args)
    else:

        # Read data
        if isinstance(train_data, str):
            df = pd.read_csv(train_data)
        else:
            df = pd.DataFrame(train_data)

        # Initialize lists to store recommendation results and metrics for each fold
        recommendations_list = []
        metrics_list = []

        # Split data into cv folds and run benchmark
        group_kfold = GroupKFold(n_splits=cv)
        for train_index, test_index in group_kfold.split(df, groups=df[user_id_col]):

            # Set train/test data frames
            args['train_data'] = df.iloc[train_index, :]
            args['test_data'] = df.iloc[test_index, :]

            # Run benchmark
            recommendations, metrics = _bench(**args)

            # Append
            recommendations_list.append(recommendations)
            metrics_list.append(metrics)

        return recommendations_list, metrics_list


def _bench(recommenders: Dict[str, BanditRecommender],
           metrics: List[Union[BinaryRecoMetrics, RankingRecoMetrics]],
           train_data: Union[str, pd.DataFrame],
           test_data: Union[str, pd.DataFrame],
           user_features: Union[str, pd.DataFrame] = None,
           user_features_list: Union[str, List[str]] = None,
           user_features_dtypes: Union[str, Dict] = None,
           item_features: Union[str, pd.DataFrame] = None,
           item_list: Union[str, List[str]] = None,
           item_eligibility: Union[str, pd.DataFrame] = None,
           warm_start: bool = False,
           warm_start_distance: float = None,
           user_id_col: str = Constants.user_id,
           item_id_col: str = Constants.item_id,
           response_col: str = Constants.response,
           batch_size: int = 100000) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:

    # Import data
    train_data_df, item_list, user_features_df, \
        item_to_features, _ = load_data(data=train_data,
                                        user_features=user_features,
                                        user_features_list=user_features_list,
                                        user_features_dtypes=user_features_dtypes,
                                        item_features=item_features,
                                        item_list=item_list,
                                        item_eligibility=item_eligibility,
                                        user_id_col=user_id_col,
                                        item_id_col=item_id_col,
                                        response_col=response_col)
    test_data_df = load_response_data(test_data)

    recommendations = dict()
    rec_metrics = dict()
    for name, recommender in recommenders.items():

        # Copy recommender
        rec = deepcopy(recommender)

        # Train
        train(
            recommender=rec,
            data=train_data_df,
            user_features=user_features_df,
            item_features=item_features,
            item_list=item_list,
            warm_start=warm_start,
            warm_start_distance=warm_start_distance,
            user_id_col=user_id_col,
            item_id_col=item_id_col,
            response_col=response_col,
            batch_size=batch_size,
            save_file=None)

        # Score
        recommendations[name] = score(
            recommender=rec,
            data=test_data_df,
            user_features=user_features_df,
            item_features=item_features,
            item_list=item_list,
            item_eligibility=item_eligibility,
            warm_start=warm_start,
            warm_start_distance=warm_start_distance,
            user_id_col=user_id_col,
            item_id_col=item_id_col,
            response_col=response_col,
            batch_size=batch_size,
            save_file=None)

        # Evaluate
        cm = CombinedMetrics(*metrics)
        rec_metrics[name] = cm.get_score(test_data_df.rename(columns={response_col: Constants.score}),
                                         recommendations[name])

    return recommendations, rec_metrics


def _validate_common_args(data, user_features, user_features_list, user_features_dtypes, item_features,
                          item_list, item_eligibility, warm_start, warm_start_distance, user_id_col, item_id_col,
                          response_col, batch_size):
    # Train/test data
    check_true(data is not None, ValueError("Data input cannot be none."))
    check_true(isinstance(data, (str, pd.DataFrame)),
               TypeError("Data should be string of filepath or data frame."))

    # User features
    if user_features is not None:
        check_true(isinstance(user_features, (str, pd.DataFrame)),
                   TypeError("User features should be string of filepath or data frame."))
    if user_features_list is not None:
        check_true(user_features is not None, ValueError("User features should be given if user list is specified."))
        check_true(isinstance(user_features_list, (str, list)),
                   TypeError("User features list should be string of filepath or list."))
    if user_features_dtypes is not None:
        check_true(user_features is not None, ValueError("User features should be given if user dtypes is specified."))
        check_true(isinstance(user_features_dtypes, (str, dict)),
                   TypeError("User features dtypes should be string of filepath or dictionary."))

    # Item features
    if item_features is not None:
        check_true(isinstance(item_features, (str, pd.DataFrame)),
                   TypeError("Item features should be string of filepath or data frame."))
    if item_list is not None:
        check_true(isinstance(item_list, (str, list)),
                   TypeError("Item list should be string of filepath or list."))
    if item_eligibility is not None:
        check_true(isinstance(item_eligibility, (str, pd.DataFrame)),
                   TypeError("Item eligibility should be string of filepath or data frame."))

    # Warm start
    check_true(isinstance(warm_start, bool), TypeError("Warm start flag should be boolean."))
    if warm_start:
        check_true(warm_start_distance is not None, ValueError("Warm start distance cannot be none."))
        check_true(isinstance(warm_start_distance, float), TypeError("Warm start distance should be a float."))
        check_true(isinstance(warm_start_distance, float), TypeError("Warm start distance should be a float."))
        check_true(0 <= warm_start_distance <= 1, ValueError("Warm start distance should be between 0 and 1."))
        check_true(item_features is not None, ValueError("Item features are required to warm start arms."))
    else:
        check_true(warm_start_distance is None, ValueError("Warm start distance should be none if warm start false."))

    # IDs
    check_true(isinstance(user_id_col, str), TypeError("User id should be a string."))
    check_true(isinstance(item_id_col, str), TypeError("Item id should be a string."))
    check_true(isinstance(response_col, str), TypeError("Response column should be a string."))

    # Batch size
    check_true(isinstance(batch_size, int), TypeError("Batch size should be an integer."))
    check_true(batch_size > 0, ValueError("Batch size should be positive."))


def _validate_recommender(recommender, is_fit=False):
    if not isinstance(recommender, dict):
        recommender_dict = {"": recommender}
    else:
        recommender_dict = recommender
    for rec in recommender_dict.values():
        check_true(isinstance(rec, BanditRecommender), TypeError("Recommender should be a BanditRecommender instance."))
        if is_fit:
            check_true(rec.mab is not None, ValueError("Recommender has not been initialized."))
            check_true(rec.mab._is_initial_fit, ValueError("Recommender has not been fit."))


def _validate_save(save_file):
    if save_file is not None:
        check_true(isinstance(save_file, (bool, str)), TypeError("Save file should be boolean or a string filepath."))


def _validate_bench(recommenders, metrics, train_data, test_data, cv):

    # Recommenders
    check_true(recommenders is not None, ValueError("Recommenders cannot be none."))
    check_true(isinstance(recommenders, dict), TypeError("Recommenders should be given as a dictionary."))

    # Metrics
    check_true(isinstance(metrics, list), TypeError("Metrics should be given as a list."))
    for v in metrics:
        check_true(isinstance(v, (BinaryRecoMetrics.AUC,
                                  BinaryRecoMetrics.CTR,
                                  RankingRecoMetrics.Precision,
                                  RankingRecoMetrics.Recall,
                                  RankingRecoMetrics.NDCG,
                                  RankingRecoMetrics.MAP)),
                   TypeError("Evaluation metric values must be BinaryRecoMetrics or RankingRecoMetrics instances."))

    # Train/test data
    check_true(train_data is not None, ValueError("Train data cannot be none."))
    check_true(isinstance(train_data, (str, pd.DataFrame)),
               TypeError("Train data should be string of filepath or data frame."))
    if test_data is not None:
        check_true(isinstance(test_data, (str, pd.DataFrame)),
                   TypeError("Test data should be string of filepath or data frame."))

    # CV
    if cv is not None:
        check_true(isinstance(cv, int), TypeError("Cross-validation (cv) must be an integer."))
        check_true(test_data is None, ValueError("Test data must be None when using Cross-validation (cv)."))
