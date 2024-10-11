[![ci](https://github.com/fidelity/mab2rec/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/fidelity/mab2rec/actions/workflows/ci.yml) [![PyPI version fury.io](https://badge.fury.io/py/mab2rec.svg)](https://pypi.python.org/pypi/mab2rec/) [![PyPI license](https://img.shields.io/pypi/l/mab2rec.svg)](https://pypi.python.org/pypi/mab2rec/) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Downloads](https://static.pepy.tech/personalized-badge/mab2rec?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/mab2rec)

# Mab2Rec: Multi-Armed Bandits Recommender 

Mab2Rec ([AAAI'24](https://ojs.aaai.org/index.php/AAAI/article/view/30341)) is a Python library for building bandit-based recommendation algorithms. It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models powered by [MABWiser](https://github.com/fidelity/mabwiser) and fairness and recommenders evaluations powered by [Jurity](https://github.com/fidelity/jurity).
The library is designed with rapid experimentation in mind, follows the [PEP-8 standards](https://www.python.org/dev/peps/pep-0008/), and is tested heavily.

Mab2Rec is built on top of several other open-source software developed at the AI Center at Fidelity:

* [MABWiser](https://github.com/fidelity/mabwiser) to create multi-armed bandit recommendation algorithms ([Bridge@AAAI'24](http://osullivan.ucc.ie/CPML2024/papers/06.pdf), [TMLR'22](https://openreview.net/pdf?id=sX9d3gfwtE), [IJAIT'21](https://www.worldscientific.com/doi/abs/10.1142/S0218213021500214), [ICTAI'19](https://ieeexplore.ieee.org/document/8995418)).
* [TextWiser](https://github.com/fidelity/textwiser) to create item representations via text featurization ([AAAI'21](https://ojs.aaai.org/index.php/AAAI/article/view/17814)).
* [Selective](https://github.com/fidelity/selective) to create user representations via feature selection ([AMAI'24](https://trebuchet.public.springernature.app/get_content/2c9eb6df-5c2b-42bc-89d6-4e3eb8bc8799?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20240405&utm_content=10.1007/s10472-024-09941-x), [CPAIOR'21](https://link.springer.com/chapter/10.1007/978-3-030-78230-6_27), [DSO@IJCAI'21](https://arxiv.org/abs/2112.03105)).
* [Seq2Pat](https://github.com/fidelity/seq2pat) to create user representations via sequential pattern mining ([AI Magazine'23](https://onlinelibrary.wiley.com/doi/epdf/10.1002/aaai.12081), [AAAI'22](https://ojs.aaai.org/index.php/AAAI/article/view/21542), [Bridge@AAAI'23](http://osullivan.ucc.ie/CPML2023/submissions/09.pdf), [Frontiers'22](https://www.frontiersin.org/articles/10.3389/frai.2022.868085/full), [KDF@AAAI'22](https://arxiv.org/abs/2201.09178), [CMU Blog Post](https://www.cmu.edu/tepper/news/stories/2023/may/fidelity-ai.html))
* [Jurity](https://github.com/fidelity/jurity) to evaluate recommendations including fairness metrics ([ACM'24](https://dl.acm.org/doi/10.1145/3700145), [LION'23](https://link.springer.com/chapter/10.1007/978-3-031-44505-7_29), [CIKM'22](https://ceur-ws.org/Vol-3318/short6.pdf), [ICMLA'21](https://ieeexplore.ieee.org/abstract/document/9680169)).

An introduction to **content- and context-aware** recommender systems and an overview of the building blocks of the library is presented at [AAAI 2024](https://underline.io/lecture/91479-building-higher-order-abstractions-from-the-components-of-recommender-systems) and [All Things Open 2021](https://www.youtube.com/watch?v=54d_YUalvOA). There is a corresponding [blogpost](https://2022.allthingsopen.org/introducing-mab2rec-a-multi-armed-bandit-recommender-library/) to serve as a starting point for practioners to build and deploy bandit-based recommenders using Mab2Rec.

Documentation is available at [fidelity.github.io/mab2rec](https://fidelity.github.io/mab2rec).

## Usage Patterns

Mab2Rec supports prototyping with a **single** bandit algorithm or benchmarking with **multiple** bandit algorithms. 
If you are new user, the best place to start is to experiment with multiple bandits using the [tutorial notebooks](notebooks).

## Quick Start

### Single Recommender

```python
# Example of how to train an singler recommender to generate top-4 recommendations

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

### Multiple Recommenders

```python
# Example of how to benchmark multiple recommenders to generate top-4 recommendations

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
```

## Usage Examples

We provide extensive tutorials in the [notebooks](notebooks) folder with guidelines on building recommenders, performing model selection, and evaluating performance.

1. [Data Overview:](https://github.com/fidelity/mab2rec/tree/master/notebooks/1_data_overview.ipynb) Overview of data required to train recommender.
2. [Feature Engineering:](https://github.com/fidelity/mab2rec/tree/master/notebooks/2_feature_engineering.ipynb) Creating user and item features from structured, unstructured, and sequential data.
3. [Model Selection:](https://github.com/fidelity/mab2rec/tree/master/notebooks/3_model_selection.ipynb) Model selection by benchmarking recommenders using cross-validation.
4. [Evaluation:](https://github.com/fidelity/mab2rec/tree/master/notebooks/4_evaluation.ipynb) Benchmarking of selected recommenders and baselines on test data with detailed evaluation.
5. [Advanced:](https://github.com/fidelity/mab2rec/tree/master/notebooks/5_advanced.ipynb) Demonstration of advanced functionality such as persistency, eligibility, item availability, and memory efficiency.

## Installation

Mab2Rec requires **Python 3.8+** and can be installed from PyPI using ``pip install mab2rec`` or by building from source as shown in [installation instructions](https://fidelity.github.io/mab2rec/installation.html).

## Citation

If you use Mab2Rec in a publication, please cite it as:

```bibtex
    @inproceedings{DBLP:conf/aaai/KadiogluK24,
      author       = {Serdar Kadioglu and Bernard Kleynhans},
      title        = {Building Higher-Order Abstractions from the Components of Recommender Systems},
      booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI} 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver, Canada},
      pages        = {22998--23004},
      publisher    = {{AAAI} Press},
      year         = {2024},
      url          = {https://doi.org/10.1609/aaai.v38i21.30341},
      doi          = {10.1609/AAAI.V38I21.30341}
    }
```

## Support

Please submit bug reports and feature requests as [Issues](https://github.com/fidelity/mab2rec/issues).

## License

Mab2Rec is licensed under the [Apache License 2.0](LICENSE).

<br>
