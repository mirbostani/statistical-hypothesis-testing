# Hypothesis Testing: Bootstrap Resampling

A Python implementation of the Bootstrap Resampling method, a non-parametric statistical hypothesis testing, for Question Answering (QA) in NLP

## Help

```shell
$ python hypothesis_testing.py -h
usage: hypothesis_testing.py [-h] -d DATASET_FILE -b BASELINE_PREDICTION_FILE
                             -e EXPERIMENTAL_PREDICTION_FILE [-z SAMPLE_SIZE]
                             [-t TEST_REPETITION] [-a SIGNIFICANCE_LEVEL]
                             [-r RESAMPLING_REPETITION] [-n]

Statistical Hypothesis Testing for QA models on SQuAD v1.1 dataset

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET_FILE, --dataset_file DATASET_FILE
                        SQuAD v1.1 dataset file, e.g., dev-v1.1.json
  -b BASELINE_PREDICTION_FILE, --baseline_prediction_file BASELINE_PREDICTION_FILE
                        Baseline model's prediction file on the input dataset
  -e EXPERIMENTAL_PREDICTION_FILE, --experimental_prediction_file EXPERIMENTAL_PREDICTION_FILE
                        Experimental model's prediction file on the input
                        dataset
  -z SAMPLE_SIZE, --sample_size SAMPLE_SIZE
                        If sample size (k) is less than the size of the input
                        dataset, k number of samples will be chosen randomly
                        among dataset examples.
  -t TEST_REPETITION, --test_repetition TEST_REPETITION
                        Hypothesis testing repetition
  -a SIGNIFICANCE_LEVEL, --significance_level SIGNIFICANCE_LEVEL
                        Hypothesis testing significance level (alpha)
  -r RESAMPLING_REPETITION, --resampling_repetition RESAMPLING_REPETITION
                        Bootstrap resampling repetition
  -n, --display_not_found
                        Display question Ids that have no prediction
```

## Example

First, you should feed your dataset file (e.g., `dev-v1.1.json`) to the baseline and experimental models and get their predictions. A sample of the output file format is as follows:

```json
{
    "57284456ff5b5019007da05f": "to emphasize academics over athletics",
    "572681c1dd62a815002e8799": "rocks, algae",
    "570967c4ed30961900e840bb": "Astra 2A",
    "56ddde6b9a695914005b962a": "Denmark, Iceland and Norway",
    "57275e95f1498d1400e8f6f7": "Sports Programs",
    "57286f373acd2414000df9de": "Department of State Affairs",
    "5726a3c6f1498d1400e8e5b0": "1989",
    ...
}
```

Then, use the script to perform a hypothesis test on the predictions.

```shell
$ python hypothesis_testing.py 
  --dataset_file ./dev-v1.1.json 
  --baseline_prediction_file ./baseline_predictions.json 
  --experimental_prediction_file ./experimental_predictions.json 
  --sample_size 100 
  --significance_level 0.05

100%|████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.73it/s]
{
    "resampling_repetition": 10000,
    "significance_level": 0.05,
    "last_baseline_size": 100,
    "last_baseline_score_1": 64,
    "last_baseline_score_0": 36,
    "last_baseline_exact_match": 64.0,
    "last_experimental_size": 100,
    "last_experimental_score_1": 70,
    "last_experimental_score_0": 30,
    "last_experimental_exact_match": 70.0,
    "last_sample_size": 100,
    "last_sample_score_1": 6,
    "last_sample_score_0": 94,
    "last_sample_score_-1": 0,
    "last_means_size": 10000,
    "last_n_score": 21,
    "last_p_value": 0.0021,
    "last_null_hypothesis_rejected": true,
    "average_p_value": 0.0021,
    "null_hypothesis_rejected": true
}
```