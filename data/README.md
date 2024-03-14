## Data

### MovieLens 

We use the famous [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) as a concrete application.

The training and testing data are stored in `data_train.csv` and `data_test.csv`, respectively. In the train/test data, the response is considered 1 if the user rated an item (movie) as 5 on a 1-5 point scale and 0 otherwise.

Furthermore, we also have features for both users and items in `features_user.csv` and `features_item.csv` respectively.

In the **extended** folder we include additional datasets relevant to more advanced usage. For more details on the datasets above, including advanced usage, see the [Data Overview](https://github.com/fidelity/mab2rec/blob/main/notebooks/1_data_overview.ipynb) notebook

### Article Recommendation 

If you are interested to explore a larger dataset based on article recommendation as used in [ArXiv'24](https://arxiv.org/abs/2402.07936), [KDD'2023](https://arxiv.org/abs/2307.04996), you can [download](https://github.com/fidelity/mab2rec/releases/download/1.2.1/data.zip) it and try out different Mab2Rec algorithms.

