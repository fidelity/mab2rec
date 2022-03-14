# Script Usage


See example Python scripts for **training** and **scoring** an individual bandit recommender. The example train 
configuration in `train.yaml` specifies the recommender policy and relevant data files.

```bash
# Example of how to train an individual recommender using a script
python scripts/train.py --config scripts/train.yaml
```

As specified in the configuration file the trained recommender is saved to `tmp_dir/rec.pkl`, which can then be used 
for scoring recommendations on the test data.

```bash
# Example of how to score an individual recommender using a script
python scripts/score.py --config scripts/score.yaml
```

<br>
