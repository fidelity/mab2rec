# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from typing import Dict, List, NamedTuple, NoReturn, Union

import numpy as np
import pandas as pd
from jurity.recommenders import BinaryRecoMetrics, RankingRecoMetrics
from mabwiser.utils import Arm, Num
from mabwiser.utils import check_true


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """
    default_seed = 12345
    user_id = 'user_id'
    item_id = 'item_id'
    response = 'response'
    score = 'score'


def explode_recommendations(df: pd.DataFrame, unique_col: str, explode_cols: List[str]):
    """Replicates the explode functionality in pandas 0.25 and later.

    Assumes that there are two levels in the dataframe. The unique column is the first level, and it contains
    de-duplicated values. The columns in explode_cols is the second level, where each of these columns contain a list
    of values. Each column in explode_cols is assumed to contain a list of same length.

    The output is the normalized dataframe where the lists are split into individual rows.
    """
    # First, remove anything with 0 length
    lens = df[explode_cols[0]].str.len()
    df = df[lens != 0].reset_index(drop=True)
    # Calculate lengths of the second level list of values
    lens = df[explode_cols[0]].str.len()
    # Repeat the unique column to get the same number of values as the second level
    unique_vals = np.repeat(df[unique_col], lens.values)
    cols = {unique_col: unique_vals.values}
    # Concatenate all second level values to get a flattened list
    cols.update({key: np.concatenate(df[key]) for key in explode_cols})
    return pd.DataFrame(cols)


def concat_recommendations_list(recommendation_results_list: List[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Concatenates recommendation results split across multiple data frames into a single dataframe.

    Parameters
    ----------
    recommendation_results_list: List[Dict[str, pd.DataFrame]]
        List of dictionaries returned by benchmark function.

    Returns
    -------
    Dictionary with recommendations by algorithm.
    """
    recommendation_results_concat = dict()
    for recommendation_results in recommendation_results_list:
        for name, df in recommendation_results.items():
            if name not in recommendation_results_concat:
                recommendation_results_concat[name] = df
            else:
                recommendation_results_concat[name] = pd.concat((recommendation_results_concat[name], df))
    return recommendation_results_concat


def default_metrics(top_k_values=None, user_id_col: str = Constants.user_id, item_id_col: str = Constants.item_id):
    metric_params = {'click_column': Constants.score, 'user_id_column': user_id_col, 'item_id_column': item_id_col}
    metrics = []
    for k in top_k_values:
        metrics.append(BinaryRecoMetrics.AUC(**metric_params, k=k))
        metrics.append(BinaryRecoMetrics.CTR(**metric_params, k=k))
        metrics.append(RankingRecoMetrics.Precision(**metric_params, k=k))
        metrics.append(RankingRecoMetrics.Recall(**metric_params, k=k))
        metrics.append(RankingRecoMetrics.NDCG(**metric_params, k=k))
        metrics.append(RankingRecoMetrics.MAP(**metric_params, k=k))
    return metrics


def load_data(data: Union[str, pd.DataFrame], user_features: Union[str, pd.DataFrame] = None,
              user_features_list: Union[str, List[str]] = None, user_features_dtypes: Union[str, Dict] = None,
              item_features: Union[str, pd.DataFrame] = None, item_list: Union[str, List[str]] = None,
              item_eligibility: Union[str, pd.DataFrame] = None, user_id_col: str = Constants.user_id,
              item_id_col: str = Constants.item_id, response_col: str = Constants.response):
    """
    Import data.

    Parameters
    ----------
    data: Union[str, pd.DataFrame]
        Data should have a row for each sample (user_id, item_id, response).
        Column names should be consistent with user_id_col, item_id_col and response_col arguments.
        CSV format with file header or Data Frame.
    user_features: Union[str, pd.DataFrame]
        User features containing features for each user_id.
        Each row should include user_id and list of features (user_id, u_1, u_2, ..., u_p).
        CSV format with file header or Data Frame.
    user_features_list: Union[str, List[str]]
        List of user features to use.
        Must be a subset of features in (u_1, u_2, ... u_p).
        If None, all the features in user_features are used.
        CSV format with file header or List.
    user_features_dtypes: Union[str, Dict]
        User features data types file with mappings of features to their dtypes upon loading.
        Data should have a key, value pair for user feature, e.g., {"feature_1": "float32"}
        The keys should be consistent with `user_features` file.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_features: Union[str, pd.DataFrame]
        Item features file containing features for each item_id.
        Each row should include item_id and list of features (item_id, i_1, i_2, .... i_q).
        CSV format with file header or Data Frame.
    item_eligibility: Union[str, pd.DataFrame], default=None
        Items each user is eligible for.
        Used to generate excluded_arms lists.
        If None, all the items can be evaluated for recommendation for each user.
        CSV format with file header or Data Frame.
    item_list: List[Arm]
        List of items.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    response_col : str, default=Constants.response
        Response column name.

    Returns
    -------
    Data frame with response data.
    """

    # Response data
    data_df = load_response_data(data, user_id_col, item_id_col, response_col)

    # Item list
    item_list = load_items(data_df, item_list, item_id_col)

    # User features
    if user_features is not None:
        user_features_df = load_user_features(user_features, user_features_list, user_features_dtypes, user_id_col)
    else:
        user_features_df = None

    # Item features
    if item_features is not None:
        item_to_features = load_item_features(item_features, item_list, item_id_col)
    else:
        item_to_features = None

    # Item eligibility
    if item_eligibility is not None:
        excluded_df = load_excluded_items(item_eligibility, item_list, user_id_col, item_id_col)
    else:
        excluded_df = None

    return data_df, item_list, user_features_df, item_to_features, excluded_df


def load_response_data(data: Union[str, pd.DataFrame], user_id_col: str = Constants.user_id,
                       item_id_col: str = Constants.item_id, response_col: str = Constants.response) -> pd.DataFrame:
    """
    Import response data.

    Parameters
    ----------
    data: Union[str, pd.DataFrame]
        Data should have a row for each sample (user_id, item_id, response).
        Column names should be consistent with user_id_col, item_id_col and response_col arguments.
        CSV format with file header or Data Frame.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    response_col : str, default=Constants.response
        Response column name.

    Returns
    -------
    Data frame with response data.
    """
    df = load_data_frame(data)
    check_true(user_id_col in df.columns, ValueError(f"{user_id_col} not in data file."))
    check_true(item_id_col in df.columns, ValueError(f"{item_id_col} not in data file."))
    check_true(response_col in df.columns, ValueError(f"{response_col} not in data file."))
    return df[[user_id_col, item_id_col, response_col]].astype({response_col: int})


def load_items(data_df: pd.DataFrame, item_list: Union[str, List[str]] = None, item_id_col: str = Constants.item_id):
    """
    Import item list.

    Parameters
    ----------
    data_df: pd.DataFrame
        Data frame with response data.
        Data should have a row for each sample (user_id, item_id, response).
    item_list: Union[str, List[str]], default=None
        List of items.
        If None, the list of items in data_df are returned.
        CSV format with file header or List.
    item_id_col : str, default=Constants.item_id
        Item id column name.

    Returns
    -------
    List of items.
    """
    if item_list is None:
        check_true(item_id_col in data_df.columns, ValueError(f"{item_id_col} not in data file."))
        return data_df[item_id_col].unique().tolist()
    else:
        return load_list(item_list)


def load_item_features(item_features: Union[str, pd.DataFrame], item_list: List[Arm],
                       item_id_col: str = Constants.item_id) -> Dict[Arm, List[Num]]:
    """
    Import item features.

    Parameters
    ----------
    item_features: Union[str, pd.DataFrame]
        Item features file containing features for each item_id.
        Each row should include item_id and list of features (item_id, i_1, i_2, .... i_q).
        CSV format with file header or Data Frame.
    item_list: List[Arm]
        List of items.
    item_id_col: str
        Item id column name.

    Returns
    -------
    Dictionary mapping item features to each item.
    """
    df = load_data_frame(item_features)
    check_true(item_id_col in df.columns, ValueError(f"{item_id_col} not in item features."))
    check_true(len(df) == df[item_id_col].nunique(), ValueError(f"Duplicate item ids in item features."))

    # Convert from data frame to dictionary
    item_to_features = df.set_index(item_id_col).T.to_dict("list")

    # Drop features for items not in item list
    item_to_features = {item: features for item, features in item_to_features.items() if item in item_list}

    # Raise error if features are missing for item in item list
    for item in item_list:
        if item not in item_to_features:
            raise ValueError(f"{item} not found in item features.")

    return item_to_features


def load_user_features(user_features: Union[str, pd.DataFrame], user_features_list: Union[str, List[str]] = None,
                       user_features_dtypes: Union[str, Dict] = None,
                       user_id_col: str = Constants.user_id) -> pd.DataFrame:
    """
    Import user features.

    Parameters
    ----------
    user_features: Union[str, pd.DataFrame]
        User features containing features for each user_id.
        Each row should include user_id and list of features (user_id, u_1, u_2, ..., u_p).
        CSV format with file header or Data Frame.
    user_features_list: Union[str, List[str]]
        List of user features to use.
        Must be a subset of features in (u_1, u_2, ... u_p).
        If None, all the features in user_features are used.
        CSV format with file header or List.
    user_features_dtypes: Union[str, Dict]
        User features data types file with mappings of features to their dtypes upon loading.
        Data should have a key, value pair for user feature, e.g., {"feature_1": "float32"}
        The keys should be consistent with `user_features` file.
    user_id_col: str
        User id column name.

    Returns
    -------
    Data frame with user features.
    """
    # User features data types
    data_types = None
    if user_features_dtypes is not None:
        if isinstance(user_features_dtypes, str):
            with open(user_features_dtypes, 'r') as f:
                data_types = json.load(f)
        else:
            data_types = user_features_dtypes

    # Load data
    if isinstance(user_features, str):
        df = pd.read_csv(user_features, dtype=data_types)
    else:
        df = pd.DataFrame(user_features)
    check_true(user_id_col in df.columns, ValueError(f"{user_id_col} not in user features."))
    check_true(len(df) == df[user_id_col].nunique(), ValueError(f"Duplicate user ids in user features."))

    # Subset
    if user_features_list is not None:
        keep = [user_id_col] + load_list(user_features_list)
        return df[keep]
    else:
        return df


def load_data_frame(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load file as data frame.

    Parameters
    ----------
    data: Union[str, pd.DataFrame]
        CSV file with header if string input, otherwise Data Frame.

    Returns
    -------
    Data frame with user features.
    """
    if isinstance(data, str):
        return pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError("Data must be string of filepath or data frame.")


def load_excluded_items(item_eligibility: Union[str, pd.DataFrame], item_list: List[Arm],
                        user_id_col: str = Constants.user_id, item_id_col: str = Constants.item_id) -> pd.DataFrame:
    """Convert eligibility data to excluded_arms list for each user_id.

    Parameters
    ----------
    item_eligibility: Union[str, pd.DataFrame], default=None
        Items each user is eligible for.
        Used to generate excluded_arms lists.
        If None, all the items can be evaluated for recommendation for each user.
        CSV format with file header or Data Frame.
    item_list: List
        The list of all arms.
    user_id_col: str
        User id column name.
    item_id_col: str
        Item id column name.

    Returns
    -------
    DataFrame with user id and list of list of arms to exclude from recommended arms.
    """

    # Load data
    df = load_data_frame(item_eligibility).copy()
    check_true(user_id_col in df.columns, ValueError(user_id_col + ' missing from eligibility_data.'))
    check_true(item_id_col in df.columns, ValueError(item_id_col + ' missing from eligibility_data.'))

    # Create list of excluded items for each user.
    df['excluded_arms'] = df.apply(lambda x: get_exclusion_list(item_list, x[item_id_col]), axis=1)
    df.drop(item_id_col, axis=1, inplace=True)
    df.columns = [user_id_col, item_id_col]
    return df


def load_pickle(pickle_file: str):
    """
    Returns the loaded pickle object.
    """
    with open(pickle_file, 'rb') as infile:
        return pickle.load(infile)


def load_list(data: Union[str, List]) -> List:
    """
    Load file as list.

    Parameters
    ----------
    data: Union[str, pd.DataFrame]
        CSV file with header if string input, otherwise List.

    Returns
    -------
    Data frame with user features.
    """
    if isinstance(data, str):
        return pd.read_csv(data).iloc[:, 0].tolist()
    elif isinstance(data, list):
        return data
    else:
        raise TypeError("Data must be string of filepath or list.")


def get_exclusion_list(arms, eligible_list):
    return list(set(arms).difference(set(eval(eligible_list))))


def print_interaction_stats(df: pd.DataFrame, user_id_col: str = Constants.user_id,
                            item_id_col: str = Constants.item_id, response_col: str = Constants.response) -> NoReturn:
    """
    Print number of rows, number of users, number of items in interaction data.

    Parameters
    ----------
    df: pd.DataFrame
        Interaction data frame with (user_id, item_id, response) in each row.
    user_id_col: str
        User id column name.
    item_id_col: str
        Item id column name.
    response_col: str
        Response column name.

    Returns
    -------
    No return.
    """

    print(f"Number of rows: {len(df):,}")
    print(f"Number of users: {df[user_id_col].nunique():,}")
    print(f"Number of items: {df[item_id_col].nunique():,}")
    print(f"Mean response rate: {df[response_col].mean():.4f}\n")


def merge_user_features(responses_df: pd.DataFrame, user_features_df: pd.DataFrame,
                        user_id_col: str = Constants.user_id) -> pd.DataFrame:
    """
    Merge responses and user features.

    Parameters
    ----------
    responses_df : pd.DataFrame
        Responses.
    user_features_df : pd.DataFrame
        User features.
    user_id_col: str
        User id column name.

    Returns
    -------
    Data frame with merged responses and user features.
    """
    # Subset features to only include users in response data and then merge
    df = user_features_df[user_features_df[user_id_col].isin(responses_df[user_id_col].values)]
    return responses_df.merge(df, on=user_id_col, how="left")


def save_json(obj, json_file) -> NoReturn:
    """
    Save obj as json file.
    """
    with open(json_file, 'w') as f:
        json.dump(obj, f)


def save_pickle(obj, pickle_file) -> NoReturn:
    """
    Save serializable object as pickle file.
    """
    with open(pickle_file, 'wb') as fp:
        pickle.dump(obj, fp)
