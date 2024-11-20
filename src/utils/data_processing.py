"""
Module for processing data related to transcription and statistical analysis.

"""

import re
import numpy as np
import json

from datasets import Dataset


def create_groups(
        dataset: Dataset,
        group_column: str, 
        metric_column: str,
        ) -> dict:
    """
    Groups data by a specified column and computes scores for a metric column.

    Args:
        dataset (datasets.Dataset): The dataset to process, expected to be a Hugging Face Dataset object.
        group_column (str): The name of the column to group by.
        metric_column (str): The name of the column containing the metric to compute scores for.

    Returns:
        dict: A dictionary where keys are unique group values and values are lists of metric scores for each group.
    """    
    group_scores = {}

    groups = dataset.unique(group_column)

    for group in groups:
        dataset_filtered = dataset.filter(lambda x: x[group_column] == group)
        dataset_filtered = dataset_filtered.filter(lambda x: x[metric_column] is not None)
        group_scores[group] = dataset_filtered[metric_column]

    return group_scores


def clean_eval_text(text: str) -> str:
    """
    Cleans and normalizes evaluation text for processing.

    This function performs the following steps:
      1. Strips leading and trailing whitespace.
      2. Converts the text to lowercase for case normalization.
      3. Removes all punctuation by replacing it with an empty string.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned and normalized text.
    """
    text = text.strip()  # Remove leading and trailing whitespace
    text = text.lower()  # Normalize case
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def compute_metric_per_example(
        batch: dict,
        metric,
        reference_field_name: str,
        transcription_field_name: str
    ) -> dict:
    """
    Computes a metric for each example in a batch and appends the metric results to the batch.

    This function evaluates model predictions against reference data using a specified metric.
    It standardizes the text inputs and calculates metric scores for all examples in the batch.

    Args:
        batch (dict): A dictionary containing the batched data. Each key represents a field 
                      (e.g., references, predictions) mapped to a list of values.
        metric (datasets.Metric): A metric object from the `datasets` library used for evaluation.
        reference_field_name (str): The key in the batch dictionary containing the reference text.
        transcription_field_name (str): The key in the batch dictionary containing the predicted text.

    Returns:
        dict: The batch dictionary with an additional field containing the computed metric scores. 
              The metric scores are replicated across all examples for consistency.
    """
    # Extract references and predictions from the batch
    references = batch[reference_field_name]
    transcriptions = batch[transcription_field_name]

    # Standardize text inputs
    references = [clean_eval_text(ref) for ref in references]
    transcriptions = [clean_eval_text(trans) for trans in transcriptions]

    # Compute the metric scores for the batch
    metric_scores = metric.compute(references=references, predictions=transcriptions)

    # Return a dictionary with all fields of the batch and the metric scores
    return {**batch, metric.name: [metric_scores] * len(references)}

class NpEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder to handle numpy data types.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)