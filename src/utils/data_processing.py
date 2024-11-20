"""
Module for processing data related to transcription and statistical analysis.

"""
import os
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
        
def save_results_to_json(results_data, output_dir, cfg):
    """Save results data to a JSON file.

    Args:
        results_data (dict): The data to be saved.
        output_dir (str): Directory where the results file will be stored.
        cfg (dict): Configuration dictionary containing optional filename.
        encoder_class (type, optional): JSON encoder class for custom serialization. Defaults to None.

    Returns:
        str: Path to the saved results file.
    """
    results_file = os.path.join(output_dir, cfg.get("results_fname", "results")) + ".json"

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=4, cls=NpEncoder)

    return results_file