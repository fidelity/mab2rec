# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jurity.recommenders import DiversityRecoMetrics

from mab2rec.utils import Constants
from mab2rec.utils import concat_recommendations_list


def plot_metrics_at_k(metric_results: Union[Dict[str, Dict[str, float]], List[Dict[str, Dict[str, float]]]], **kwargs):
    """
    Plots recommendation metric values (y-axis) for different values of k (x-axis)
    for each of the benchmark algorithms.

    Parameters
    ----------
    metric_results : Union[Dict[str, Dict[str, float]], List[Dict[str, Dict[str, float]]]]
        Nested-dictionary or list of dictionaries with evaluation results returned by benchmark function.
    **kwargs
        Other parameters passed to ``sns.catplot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with metric values.
    """

    if not isinstance(metric_results, list):
        metric_results_list = [metric_results]
    else:
        metric_results_list = metric_results

    # Format data for plot
    out = []
    for metric_results in metric_results_list:
        for algo_name, results in metric_results.items():
            for metric_name, value in results.items():
                d = {'algorithm': algo_name,
                     'metric_name': metric_name.split('@')[0],
                     'k': int(metric_name.split('@')[1]),
                     'value': value}
                out.append(d)
    df = pd.DataFrame(out)

    # Plot
    ax = sns.catplot(x='k', y='value', col='metric_name', hue='algorithm', data=df, kind='point',
                     sharey=False, **kwargs)
    return ax


def plot_inter_diversity_at_k(recommendation_results: Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]],
                              k_list: List[int], user_id_col: str = Constants.user_id,
                              item_id_col: str = Constants.item_id, score_col: str = Constants.score,
                              sample_size: float = None,  seed: int = Constants.default_seed,
                              num_runs: int = 10, n_jobs: int = 1, working_memory: int = None, **kwargs):
    """
    Plots recommendation metric values (y-axis) for different values of k (x-axis)
    for each of the benchmark algorithms.

    Parameters
    ----------
    recommendation_results : Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]]
        Dictionary or list of dictionaries with recommendation results returned by benchmark function.
    k_list : List[int]
        List of top-k values to evaluate.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    score_col: str, default=Constants.score
        Recommendation score column name.
    sample_size: float, default=None
        Proportion of users to randomly sample for evaluation.
        If None, no sampling is performed.
    seed : int, default=Constants.default_seed
        The seed used to create random state.
    num_runs: int
        num_runs is used to report the approximation of Inter-List Diversity over multiple runs on smaller
        samples of users, default=10, for a speed-up on evaluations. The sampling size is defined by
        user_sample_size. The final result is averaged over the multiple runs.
    n_jobs: int
        Number of jobs to use for computation in parallel, leveraged by sklearn.metrics.pairwise_distances_chunked.
        -1 means using all processors. Default=1.
    working_memory: Union[int, None]
        Maximum memory for temporary distance matrix chunks, leveraged by sklearn.metrics.pairwise_distances_chunked.
        When None (default), the value of sklearn.get_config()['working_memory'], i.e. 1024M, is used.
    **kwargs
        Other parameters passed to ``sns.catplot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with metric values.
    """

    if not isinstance(recommendation_results, list):
        recommendation_results_list = [recommendation_results]
    else:
        recommendation_results_list = recommendation_results

    # Calculate metrics
    out = []
    for recommendation_results in recommendation_results_list:
        for algo_name, rec_df in recommendation_results.items():
            for k in k_list:
                metric = DiversityRecoMetrics.InterListDiversity(click_column=score_col, k=k,
                                                                 user_id_column=user_id_col,
                                                                 item_id_column=item_id_col,
                                                                 user_sample_size=sample_size,
                                                                 seed=seed,
                                                                 num_runs=num_runs,
                                                                 n_jobs=n_jobs,
                                                                 working_memory=working_memory)
                inter_list_diversity = metric.get_score(rec_df, rec_df)
                d = {'algorithm': algo_name,
                     'metric_name': 'Inter-list Diversity',
                     'k': k,
                     'value': inter_list_diversity}
                out.append(d)
    df = pd.DataFrame(out)

    # Plot
    ax = sns.catplot(x='k', y='value', col='metric_name', hue='algorithm', data=df, kind='point',
                     sharey=False, **kwargs)

    return ax


def plot_intra_diversity_at_k(recommendation_results: Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]],
                              item_features: pd.DataFrame, k_list: List[int], user_id_col: str = Constants.user_id,
                              item_id_col: str = Constants.item_id, score_col: str = Constants.score,
                              sample_size: float = None,  seed: int = Constants.default_seed, n_jobs: int = 1,
                              num_runs: int = 10, **kwargs):
    """
    Plots recommendation metric values (y-axis) for different values of k (x-axis)
    for each of the benchmark algorithms.

    Parameters
    ----------
    recommendation_results : Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]]
        Dictionary or list of dictionaries with recommendation results returned by benchmark function.
    item_features : pd.DataFrame
        Data frame with features for each item_id.
    k_list : List[int]
        List of top-k values to evaluate.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    score_col: str, default=Constants.score
        Recommendation score column name.
    sample_size: float, default=None
        Proportion of users to randomly sample for evaluation.
        If None, no sampling is performed.
    seed : int, default=Constants.default_seed
        The seed used to create random state.
    num_runs: int
        num_runs is used to report the approximation of Intra-List Diversity over multiple runs on smaller
        samples of users, default=10, for a speed-up on evaluations. The sampling size is defined by
        user_sample_size. The final result is averaged over the multiple runs.
    n_jobs: int
        Number of jobs to use for computation in parallel, leveraged by sklearn.metrics.pairwise_distances.
        -1 means using all processors. Default=1.
    **kwargs
        Other parameters passed to ``sns.catplot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with metric values.
    """

    if not isinstance(recommendation_results, list):
        recommendation_results_list = [recommendation_results]
    else:
        recommendation_results_list = recommendation_results

    # Calculate metrics
    out = []
    for recommendation_results in recommendation_results_list:
        for algo_name, rec_df in recommendation_results.items():
            for k in k_list:
                metric = DiversityRecoMetrics.IntraListDiversity(item_features, click_column=score_col, k=k,
                                                                 user_id_column=user_id_col,
                                                                 item_id_column=item_id_col,
                                                                 user_sample_size=sample_size,
                                                                 seed=seed,
                                                                 num_runs=num_runs,
                                                                 n_jobs=n_jobs)
                intra_list_diversity = metric.get_score(rec_df, rec_df)

                d = {'algorithm': algo_name,
                     'metric_name': 'Intra-list Diversity',
                     'k': k,
                     'value': intra_list_diversity}
                out.append(d)
    df = pd.DataFrame(out)

    # Plot
    ax = sns.catplot(x='k', y='value', col='metric_name', hue='algorithm', data=df, kind='point',
                     sharey=False, **kwargs)

    return ax


def plot_recommended_counts(recommendation_results: Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]],
                            actual_results: pd.DataFrame, k: int, average_response: bool = False,
                            user_id_col: str = Constants.user_id, item_id_col: str = Constants.item_id,
                            response_col: str = Constants.response, **kwargs):
    """
    Plots recommendation counts (y-axis) versus actual counts or average responses (x-axis) for each item.

    Parameters
    ----------
    recommendation_results : Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]]
        Dictionary or list of dictionaries with recommendation results returned by benchmark function.
    actual_results : pd.DataFrame
        Test data frame used to generate recommendations.
        Data should have a row for each sample (user_id, item_id, response).
    k : int
        Top-k recommendations to evaluate.
    average_response : bool, default=False
        Whether to plot the average response/reward or not.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    response_col : str, default=Constants.response
        Response column name.
    **kwargs
        Other parameters passed to ``sns.relplot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with recommended counts.
    """

    # Concatenate recommendation results from different cv-folds
    if isinstance(recommendation_results, list):
        recommendation_results = concat_recommendations_list(recommendation_results)

    # Actual
    if average_response:
        actual_counts = actual_results.groupby(item_id_col)[response_col].mean()
    else:
        actual_counts = actual_results.groupby(item_id_col).size()

    # Recommended
    out = []
    for algo_name, rec_df in recommendation_results.items():
        rec_sorted_df = rec_df.sort_values(Constants.score, ascending=False).groupby(user_id_col).head(k)
        rec_counts = rec_sorted_df.groupby(item_id_col).size()

        for item_id in rec_counts.index:

            if item_id in actual_counts.index:
                a_value = actual_counts[item_id]
            else:
                a_value = 0

            out.append({'algorithm': algo_name,
                        'item_id': item_id,
                        'actual': a_value,
                        'recommended': rec_counts[item_id]})
    df = pd.DataFrame(out)

    # Plot
    g = sns.relplot(x='actual', y='recommended', col='algorithm', data=df, **kwargs)
    if not average_response:
        xy_max = max(df['actual'].max(), df['recommended'].max())
        for ax in g.axes.flat:
            ax.plot([0, xy_max], [0, xy_max], color="darkred", linestyle="--", alpha=0.5)
    return g


def plot_recommended_counts_by_item(recommendation_results: Union[Dict[str, pd.DataFrame],
                                                                  List[Dict[str, pd.DataFrame]]],
                                    k: int, top_n_items: int = None, normalize: bool = False,
                                    user_id_col: str = Constants.user_id, item_id_col: str = Constants.item_id,
                                    **kwargs):
    """
    Plots recommendation counts (y-axis) for different items (x-axis) for each of the benchmark algorithms. Only the
    top_n_items with the most recommendations for each algorithm are shown.

    Parameters
    ----------
    recommendation_results : Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]]
        Dictionary or list of dictionaries with recommendation results returned by benchmark function.
    k : int
        Top-k recommendations to evaluate.
    top_n_items : int, default=None
        Top-n number of items based on number of recommendations to plot.
    normalize : bool, default=False
        Whether to normalize the counts per item to be proportions such that they add to 1.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    **kwargs
        Other parameters passed to ``sns.catplot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with recommended counts by item.
    """

    # Concatenate recommendation results from different cv-folds
    if isinstance(recommendation_results, list):
        recommendation_results = concat_recommendations_list(recommendation_results)

    # Calculate metrics
    out = []
    for algo_name, rec_df in recommendation_results.items():

        rec_sorted_df = rec_df.sort_values(Constants.score, ascending=False).groupby(user_id_col).head(k)
        rec_counts = rec_sorted_df[item_id_col].value_counts(normalize=normalize)

        rank = 0
        for item_id, value in rec_counts.items():
            out.append({'algorithm': algo_name,
                        'k': k,
                        'item_id': item_id,
                        'rank': rank,
                        'value': value})
            rank += 1

    df = pd.DataFrame(out)
    if top_n_items is not None:
        df = df[df['rank'] < top_n_items]
    df.drop(columns='rank', inplace=True)

    ax = sns.catplot(x='item_id', y='value', col='algorithm', data=df, kind='bar', color='grey', **kwargs)
    ax.set_xticklabels([])

    return ax


def plot_num_items_per_recommendation(recommendation_results: Union[Dict[str, pd.DataFrame],
                                                                    List[Dict[str, pd.DataFrame]]],
                                      actual_results: pd.DataFrame, normalize: bool = False,
                                      user_id_col: str = Constants.user_id, **kwargs):
    """
    Plots recommendation counts (y-axis) versus actual counts or average responses (x-axis) for each item.

    Parameters
    ----------
    recommendation_results : Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]]
        Dictionary or list of dictionaries with recommendation results returned by benchmark function.
    actual_results : pd.DataFrame
        Test data frame used to generate recommendations.
        Data should have a row for each sample (user_id, item_id, response).
    normalize : bool, default=False
        Whether to normalize the number of items to be proportions such that they add to 1.
    user_id_col: str
        User id column name.
        Default value is set to Constants.user_id
    **kwargs
        Other parameters passed to ``sns.catplot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with counts or proportions for
        different number of items per recommendation.
    """

    # Concatenate recommendation results from different cv-folds
    if isinstance(recommendation_results, list):
        recommendation_results = concat_recommendations_list(recommendation_results)

    # Distinct users in actual results
    users_df = pd.DataFrame(actual_results[user_id_col].unique(), columns=[user_id_col])

    out = []
    for algo_name, rec_df in recommendation_results.items():

        # Merge recommendations for each user
        df = users_df.merge(rec_df, on=user_id_col, how='left')

        # Calculate distribution of number of items per recommendation
        users_per_num_item = pd.value_counts(df.groupby(user_id_col).size(), normalize=normalize)
        for num_items, value in users_per_num_item.items():
            out.append({'algorithm': algo_name,
                        'k': num_items,
                        'value': value})
    df = pd.DataFrame(out)

    # Plot
    ax = sns.catplot(x='k', y='value', col='algorithm', data=df, kind='bar', color='grey', **kwargs)

    return ax


def plot_personalization_heatmap(recommendation_results: Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]],
                                 user_to_cluster: Dict[Union[int, str], int], k: int,
                                 user_id_col: str = Constants.user_id, item_id_col: str = Constants.item_id,
                                 figsize: Tuple[int, int] = None, **kwargs):
    """
    Plot heatmaps to visualize level of personalization, by calculating the distribution of recommendations
    by item within different user clusters.

    Parameters
    ----------
    recommendation_results : Union[Dict[str, pd.DataFrame], List[Dict[str, pd.DataFrame]]]
        Dictionary or list of dictionaries with recommendation results returned by benchmark function.
    user_to_cluster : Dict[Union[int, str], int]
        Mapping from user_id to cluster.
        Clusters could be derived from clustering algorithm such as KMeans or
        defined based on specific user features (e.g. age bands)
    k : int
        Top-k recommendations to evaluate.
    user_id_col : str, default=Constants.user_id
        User id column name.
    item_id_col : str, default=Constants.item_id
        Item id column name.
    figsize: Tuple[int, int], default=None
        Figure size of heatmap set using plt.figure()
    **kwargs
        Other parameters passed to ``sns.catplot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plot with counts or proportions for
        different number of items per recommendation.
    """

    # Concatenate recommendation results from different cv-folds
    if isinstance(recommendation_results, list):
        recommendation_results = concat_recommendations_list(recommendation_results)

    axes = dict()
    for algo_name, rec_df in recommendation_results.items():

        rec_sorted_df = rec_df.sort_values(Constants.score, ascending=False).groupby(user_id_col).head(k)
        rec_sorted_df['cluster'] = rec_sorted_df[user_id_col].map(user_to_cluster)

        # Calculate percentage of recommendations by item within each cluster
        df = rec_sorted_df.groupby(['cluster', item_id_col]).size()
        df = df.groupby(level=0).apply(lambda x: x / float(x.sum())).reset_index()
        df = df.pivot(index=item_id_col, columns='cluster').fillna(0)
        df.columns = df.columns.droplevel()
        df.sort_index(inplace=True)

        # Plot
        if figsize is not None:
            plt.figure(figsize=figsize)
        ax = sns.heatmap(df, **kwargs)
        ax.set_title(algo_name)
        ax.set_yticklabels([])
        plt.show()
        axes[algo_name] = ax

    return axes
