# Mab2Rec: Multi-Armed Bandits Recommender 

Mab2Rec is a Python library for building bandit-based recommendation algorithms. It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models powered by [MABWiser](https://github.com/fidelity/mabwiser) and fairness and recommenders evaluations powered by [Jurity](https://github.com/fidelity/jurity).
It supports [all bandit policies available in MABWiser](https://github.com/fidelity/mabwiser#available-bandit-policies). The library is designed with rapid experimentation in mind, follows the [PEP-8 standards](https://www.python.org/dev/peps/pep-0008/) and is tested heavily.

An introduction to **content- and context-aware** recommender systems and an overview of the building blocks of the library is [presented at All Things Open 2021](https://www.youtube.com/watch?v=54d_YUalvOA). 

Mab2Rec is developed by the Artificial Intelligence Center of Excellence at Fidelity Investments.

## Usage Patterns

Mab2Rec supports **prototyping** of individual bandit algorithms and **benchmarking** of multiple bandit algorithms. 
This can be performed either programmatically or interactively using Jupyter notebooks. 
The intended usage for a new user is to train and test multiple bandits using the [Interactive Notebooks](https://github.com/fidelity/mab2rec#interactive-notebooks) as guidelines.

## Quick Start

```python
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
```

## Quick Start: Benchmarking
```python
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
for top_k in  [3, 5, 10]:
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
```

## Interactive Notebooks

We provide extensive tutorials in Jupyter notebooks under the [notebooks directory](https://github.com/fidelity/mab2rec/tree/master/notebooks/)
for guidelines on building recommenders, performing model selection, and evaluating performance: 

1. [Data Overview:](https://github.com/fidelity/mab2rec/tree/master/notebooks/1_data_overview.ipynb) Overview of data required to train recommender.
2. [Feature Engineering:](https://github.com/fidelity/mab2rec/tree/master/notebooks/2_feature_engineering.ipynb) An overview of methods to create user and item features from structured, unstructured, and sequential data.
3. [Model Selection:](https://github.com/fidelity/mab2rec/tree/master/notebooks/3_model_selection.ipynb) Model selection by benchmarking recommenders using cross-validation.
4. [Evaluation:](https://github.com/fidelity/mab2rec/tree/master/notebooks/4_evaluation.ipynb) Benchmarking of selected recommenders and baselines on test data with detailed evaluation.
5. [Advanced:](https://github.com/fidelity/mab2rec/tree/master/notebooks/5_advanced.ipynb) Demonstration of advanced functionality.


## Open-Source Building Blocks 

Mab2Rec is built on top of several other open-source software from the AI Center at Fidelity, including:

* [MABWiser](https://github.com/fidelity/mabwiser) to create multi-armed bandit recommendation algorithms ([IJAIT'21](https://www.worldscientific.com/doi/abs/10.1142/S0218213021500214), [ICTAI'19](https://ieeexplore.ieee.org/document/8995418)).
* [TextWiser](https://github.com/fidelity/textwiser) to create item representations via text featurization ([AAAI'21](https://ojs.aaai.org/index.php/AAAI/article/view/17814)).
* [Selective](https://github.com/fidelity/selective) to create user representations via feature selection.
* [Seq2Pat](https://github.com/fidelity/seq2pat) to enhance users representations via sequential pattern mining ([AAAI'22](https://aaai.org/Conferences/AAAI-22/)).
* [Jurity](https://github.com/fidelity/jurity) to evaluate recommendations including fairness metrics ([ICMLA'21](https://ieeexplore.ieee.org/abstract/document/9680169)).
* [Spock](https://github.com/fidelity/spock) to define, manage, and use parameter configurations.

## Installation

Mab2Rec can be installed from PyPI using `pip install mab2rec`. It requires Python 3.6+. 
 
### Install from source code

Alternatively, you can build a wheel package on your platform from scratch using the source code:

```bash
pip install setuptools wheel # if wheel is not installed
python setup.py bdist_wheel
pip install dist/mab2rec-X.X.X-py3-none-any.whl
```

### Test Your Setup

To confirm successful cloning and setup, run the tests. All tests should pass. 

```bash
python -W ignore -m unittest discover -v tests
```

## Support

Please submit bug reports and feature requests as [Issues](https://github.com/fidelity/mab2rec/issues).


## License

Mab2Rec is licensed under the [Apache License 2.0](LICENSE).

<br>
