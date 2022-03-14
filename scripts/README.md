# Script Usage


See example Python scripts for **training** and **scoring** an individual bandit recommender. The example train 
configuration in `train.yaml` specifies a `LinGreedy` bandit.

```bash
# Example of how to train an individual recommender using a script
python train.py --config train.yaml
```

As specified in the configuration file the trained recommender is saved to `tmp_dir/rec.pkl`, which can then be used 
for scoring recommendations for the test data.

```bash
# Example of how to score an individual recommender using a script
python score.py --config score.yaml
```

<br>
