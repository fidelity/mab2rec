.. _quick:

Quick Start
===========

Individual Recommender
----------------------

.. code-block:: python

    # Example of how to train an individual recommender to generate top-4 recommendations

    # Import
    from mab2rec import BanditRecommender, LearningPolicy
    from mab2rec.pipeline import train, score

    # LinGreedy recommender to select top-4 items with 10% random exploration
    rec = BanditRecommender(LearningPolicy.LinGreedy(epsilon=0.1), top_k=4)

    # Train on (user, item, response) interactions in train data using user features
    train(rec, data='data/data_train.csv',
          user_features='data/features_user.csv')

    # Score recommendations for users in test data. The output df holds
    # user_id, item_id, score columns for every test user for top-k items
    df = score(rec, data='data/data_test.csv',
               user_features='data/features_user.csv')

Multiple Recommenders
---------------------

.. code-block:: python

    # Example of how to benchmark multiple bandit algorithms to generate top-4 recommendations

    from mab2rec import BanditRecommender, LearningPolicy
    from mab2rec.pipeline import benchmark
    from jurity.recommenders import BinaryRecoMetrics, RankingRecoMetrics

    # Recommenders (many more available)
    recommenders = {"Random": BanditRecommender(LearningPolicy.Random()),
                    "Popularity": BanditRecommender(LearningPolicy.Popularity()),
                    "LinGreedy": BanditRecommender(LearningPolicy.LinGreedy(epsilon=0.1))}

    # Column names for the response, user, and item id columns
    metric_params = {'click_column': 'score', 'user_id_column': 'user_id', 'item_id_column':'item_id'}

    # Performance metrics for benchmarking (many more available)
    metrics = []
    for top_k in [3, 5, 10]:
        metrics.append(BinaryRecoMetrics.CTR(**metric_params, k=top_k))
        metrics.append(RankingRecoMetrics.NDCG(**metric_params, k=top_k))

    # Benchmarking with a collection of recommenders and metrics
    # This returns two dictionaries;
    # reco_to_results: recommendations for each algorithm on cross-validation data
    # reco_to_metrics: evaluation metrics for each algorithm
    reco_to_results, reco_to_metrics = benchmark(recommenders,
                                                 metrics=metrics,
                                                 train_data="data/data_train.csv",
                                                 cv=5,
                                                 user_features="data/features_user.csv")

