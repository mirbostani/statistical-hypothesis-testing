#!/usr/bin/env python

# Statistical Hypothesis Testing: Bootstrap Resampling
# Author: Morteza Mirbostani
# Github: https://github.com/mirbostani

from collections import Counter

from tqdm import tqdm
import string
import re
import argparse
import json
import sys
import random
import math
import time
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt


class StatsHypothesisTest():

    def __init__(self,
                 dataset,
                 baseline_predictions,
                 experimental_predictions,
                 test_repetition: int,
                 sample_size: int,
                 significance_level: float,
                 resampling_repetition: int,
                 display_not_found: bool = False):
        self.dataset = dataset
        self.baseline_predictions = baseline_predictions
        self.experimental_predictions = experimental_predictions
        self.test_repetition = test_repetition
        self.k = sample_size
        self.alpha = significance_level
        self.B = resampling_repetition
        self.display_not_found = display_not_found

        pval = 0
        for i in tqdm(range(self.test_repetition)):
            (self.baseline_scores,
             self.experimental_scores) = self.generate_scores(
                dataset=self.dataset,
                k=self.k,
                baseline_predictions=self.baseline_predictions,
                experimental_predictions=self.experimental_predictions,
                display_not_found=self.display_not_found)
            (self.sample,
             self.means,
             self.p_value,
             self.n_score) = self.bootstrap_resampling(
                baseline_scores=self.baseline_scores,
                experimental_scores=self.experimental_scores,
                B=self.B)
            pval += self.p_value
        self.avg_p_value = pval / self.test_repetition

    def results(self):
        return {
            "resampling_repetition": self.B,
            "significance_level": self.alpha,
            "last_baseline_size": len(self.baseline_scores),
            "last_baseline_score_1": sum(self.baseline_scores),
            "last_baseline_score_0": len(self.baseline_scores) - sum(self.baseline_scores),
            "last_baseline_exact_match": 100 * sum(self.baseline_scores) / len(self.baseline_scores),
            "last_experimental_size": len(self.experimental_scores),
            "last_experimental_score_1": sum(self.experimental_scores),
            "last_experimental_score_0": len(self.experimental_scores) - sum(self.experimental_scores),
            "last_experimental_exact_match": 100 * sum(self.experimental_scores) / len(self.baseline_scores),
            "last_sample_size": len(self.sample),
            "last_sample_score_1": sum([1 for i, v in enumerate(self.sample) if v == 1]),
            "last_sample_score_0": sum([1 for i, v in enumerate(self.sample) if v == 0]),
            "last_sample_score_-1": sum([1 for i, v in enumerate(self.sample) if v == -1]),
            "last_means_size": len(self.means),
            "last_n_score": self.n_score,  # wrong answers of total B questions
            "last_p_value": self.p_value,
            "last_null_hypothesis_rejected": self.p_value < self.alpha,
            "average_p_value": self.avg_p_value,
            "null_hypothesis_rejected": self.avg_p_value < self.alpha,
        }

    def generate_scores(self,
                        dataset,
                        k,  # sample size
                        baseline_predictions,
                        experimental_predictions,
                        display_not_found: bool = False):
        baseline_scores = []
        experimental_scores = []
        question_ids = []

        # Randomly select `sample_size` samples from dataset
        for article in dataset:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    question_ids.append(qa["id"])

        if k in [-1, None]:
            k = len(question_ids)

        if k < len(question_ids):
            random.seed(time.time())
            sample_question_ids = random.sample(question_ids, k=k)
        else:
            sample_question_ids = question_ids

        # Geenrate scores
        for article in dataset:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:

                    # ignore not selected questions
                    if qa["id"] not in sample_question_ids:
                        continue

                    # correct answers
                    ground_truths = list(
                        map(lambda x: x["text"], qa["answers"]))

                    # baseline score
                    if qa["id"] in baseline_predictions:
                        baseline_prediction = baseline_predictions[qa["id"]]
                        exact_match = self.metric_max_over_ground_truths(
                            metric_fn=self.exact_match_score,
                            prediction=baseline_prediction,
                            ground_truths=ground_truths)
                        baseline_scores.append(1 if exact_match else 0)
                    else:
                        if display_not_found:
                            print("Baseline prediction not found for id '{}'".format(
                                qa["id"]), file=sys.stderr)
                        baseline_scores.append(0)

                    # experimental score
                    if qa["id"] in experimental_predictions:
                        experimental_prediction = experimental_predictions[qa["id"]]
                        exact_match = self.metric_max_over_ground_truths(
                            metric_fn=self.exact_match_score,
                            prediction=experimental_prediction,
                            ground_truths=ground_truths)
                        experimental_scores.append(1 if exact_match else 0)
                    else:
                        if display_not_found:
                            print("Experimental prediction not found for id '{}'".format(qa["id"]),
                                  file=sys.stderr)
                        experimental_scores.append(0)

        return (baseline_scores, experimental_scores)

    def bootstrap_resampling(self,
                             baseline_scores,
                             experimental_scores,
                             B):
        baseline_scores_np = np.array(baseline_scores)
        experimental_scores_np = np.array(experimental_scores)

        if baseline_scores_np.size != experimental_scores_np.size:
            print("Sizes are not equal!", file=sys.stderr)
            return (None, None, None)

        # Compute sample based on score difference
        sample = experimental_scores_np - baseline_scores_np

        # Resample `B` times and compute the statistic (i.e., mean)
        means = [np.random.choice(sample, size=sample.size).mean()
                 for _ in range(B)]

        # Compute p-value
        n_score = 0
        for i in range(B):
            if (means[i] <= 0):
                n_score += 1
        p_value = n_score / B

        return (sample.tolist(), means, p_value, n_score)

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # not used
    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self,
                          prediction,
                          ground_truth):
        return (self.normalize_answer(prediction) ==
                self.normalize_answer(ground_truth))

    def metric_max_over_ground_truths(self,
                                      metric_fn,
                                      prediction,
                                      ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


def main():
    expected_version = "1.1"
    parser = argparse.ArgumentParser(
        description="Statistical Hypothesis Testing for QA models on SQuAD \
            v{} dataset".format(expected_version))
    parser.add_argument(
        "-d",
        "--dataset_file",
        type=str,
        required=True,
        help="SQuAD v{} dataset file, e.g., dev-v{}.json".format(
            expected_version, expected_version))
    parser.add_argument(
        "-b",
        "--baseline_prediction_file",
        type=str,
        required=True,
        help="Baseline model's prediction file on the input dataset")
    parser.add_argument(
        "-e",
        "--experimental_prediction_file",
        type=str,
        required=True,
        help="Experimental model's prediction file on the input dataset")
    parser.add_argument(
        "-z",
        "--sample_size",
        type=int,
        default=-1,
        help="If sample size (k) is less than the size of the input dataset, \
            k number of samples will be chosen randomly among dataset examples.")
    parser.add_argument(
        "-t",
        "--test_repetition",
        type=int,
        default=1,
        help="Hypothesis testing repetition")
    parser.add_argument(
        "-a",
        "--significance_level",
        type=float,
        default=0.05,  # 5%
        help="Hypothesis testing significance level (alpha)")
    parser.add_argument(
        "-r",
        "--resampling_repetition",
        type=int,
        default=10000,
        help="Bootstrap resampling repetition")
    parser.add_argument(
        "-n",
        "--display_not_found",
        action="store_true",
        default=False,
        help="Display question Ids that have no prediction")
    args = parser.parse_args()

    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json["version"] != expected_version):
            print("Expected dataset file version is v{}, but got v{}"
                  .format(expected_version, dataset_json["version"]),
                  file=sys.stderr)
        dataset = dataset_json["data"]

    with open(args.baseline_prediction_file) as baseline_prediction_file:
        baseline_predictions = json.load(baseline_prediction_file)

    with open(args.experimental_prediction_file) as experimental_prediction_file:
        experimental_predictions = json.load(experimental_prediction_file)

    test = StatsHypothesisTest(dataset=dataset,
                               baseline_predictions=baseline_predictions,
                               experimental_predictions=experimental_predictions,
                               test_repetition=args.test_repetition,
                               sample_size=args.sample_size,
                               significance_level=args.significance_level,
                               resampling_repetition=args.resampling_repetition,
                               display_not_found=args.display_not_found)
    print(json.dumps(test.results(), indent=4))

    # plt.hist(test.means)
    # plt.show()


if __name__ == '__main__':
    main()
