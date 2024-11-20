import re
import statistics


def compute_stat(values: list) -> dict:
    """Calculate basic statistics for a list of values.

    Args:
        values (list): A list of numeric values.

    Returns:
        dict: A dictionary containing average, standard deviation, max, min, variance, and median.
    """
    return {
        "average": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "max": max(values),
        "min": min(values),
        "variance": statistics.variance(values) if len(values) > 1 else 0,
        "median": statistics.median(values),
    }


def compute_agg_statistics(data: dict) -> dict:
    """Compute statistics for each group and overall from a dictionary of lists.

    Args:
        data (dict): A dictionary where keys are group names and values are lists of numeric data.

    Returns:
        dict: A dictionary containing statistics for each group and overall.
    """
    # Compute stats for each group
    group_stats = {group: compute_stat(values) for group, values in data.items()}
    
    # Compute overall stats
    all_values = [val for values in data.values() for val in values]
    overall_stats = compute_stat(all_values)
    
    # Combine into a single dictionary
    return {
        "group_stats": group_stats,
        "overall_stats": overall_stats,
    }


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