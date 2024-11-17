"""
Speech Perception Bias Evaluation script for speech recognition models using Hugging Face Transformers.

This script evaluates a speech recognition model on a specified dataset.
It utilizes Hydra for configuration management, allowing for flexible
parameterization of the model, processor, pipeline, and dataset.

The script performs the following steps:
- Initializes the model, processor, and pipeline using the provided configurations.
- Loads the dataset specified in the configuration.
- Runs inference on the dataset efficiently using batch processing.
- Logs the transcription results.
- TODO: add multi-dataset support
- TODO: add bias evaluation modules

For examples of how to run this script, please refer to the `README.md` file.

"""

import logging
import os
import sys
import re

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

from datasets import (
    load_dataset,
    DatasetDict,
    Audio
)

from evaluate import load

from itertools import islice

log = logging.getLogger(__name__)

# Resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])


def create_groups(dataset, group_column, metric_column):
    group_scores = {}

    groups = dataset.unique(group_column)

    for group in groups:
        dataset_filtered = dataset.filter(lambda x: x[group_column] == group)
        dataset_filtered = dataset_filtered.filter(lambda x: x[metric_column] is not None)
        group_scores[group] = dataset_filtered[metric_column]

    return group_scores


def transcribe_batch(
        batch: dict, 
        asr_pipeline, 
        transcription_field_name: str
        ) -> dict:
    """
    Transcribes a batch of audio samples using the ASR pipeline.

    Args:
        batch (dict): A batch of data containing audio samples.
        asr_pipeline (transformers.pipelines.Pipeline): The ASR pipeline for inference.
        transcription_field_name (str): The key to store the transcribed text in the batch.

    Returns:
        dict: The batch with an added field for the transcribed text.
    """
    # Get the list of audio samples from the batch
    audio_list = batch["audio"]

    # Perform inference using the pipeline
    transcriptions = asr_pipeline(audio_list)

    # Extract the transcription text
    texts = [item["text"] for item in transcriptions]

    # Add the transcriptions to the batch
    batch[transcription_field_name] = texts
    return batch


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


@hydra.main(config_path="../config", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluates a speech recognition model using the configurations provided.

    This function initializes the model, processor, and pipeline based on
    the configurations specified via Hydra. It loads the dataset and performs
    inference on the dataset efficiently using batch processing.

    Args:
        cfg (DictConfig): Configuration composed by Hydra, containing parameters
            for the model, processor, pipeline, and dataset.

    Returns:
        None

    """
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    log.info(f"Device: {device}")

    # Instantiate the model
    model: AutoModelForSpeechSeq2Seq = hydra.utils.instantiate(cfg.model, _convert_="object")
    log.info("Model loaded.")

    model.to(device)

    # Instantiate the processor
    processor: AutoProcessor = hydra.utils.instantiate(cfg.processor, _convert_="object")
    log.info("Processor instantiated.")

    # Instantiate the pipeline
    asr_pipeline = hydra.utils.instantiate(
        cfg.pipeline,
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        _convert_="object",
    )
    log.info("Pipeline created.")

    # Load the dataset
    ds = hydra.utils.instantiate(cfg.dataset, _convert_="object")
    log.info("Dataset loaded.")

    # Ensure that the 'audio' column is decoded
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Set the batch size
    batch_size = cfg.get("batch_size", 8)

    # Run inference on the dataset using map with batched processing
    ds = ds.map(
        lambda batch: transcribe_batch(batch, asr_pipeline, cfg.predicted_transcription_field_name),
        batched=True,
        batch_size=batch_size,
    )
    log.info("Inference completed.")
    log.info(f"Dataset Info: {ds}.")

    # Save the transcriptions to a file
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"output_dir={output_dir}")
    output_file = os.path.join(output_dir, cfg.get("transcriptions_fname", "transcriptions"))
    ds.save_to_disk(output_file)
    log.info(f"Transcriptions saved to {output_file}.")

    # Evaluate transcriptions
    metric = load(cfg.metric.path)
    log.info(f"Metric loaded={metric.name}.")
    ds = ds.remove_columns(["audio"])
    ds = ds.map(
        lambda batch: compute_metric_per_example(
            batch, metric, cfg.actual_transcription_field_name, cfg.predicted_transcription_field_name),
        batched=True,
        batch_size=batch_size,
    )
    log.info("Evaluation completed.")

    grouped_scores = create_groups(ds, cfg.bias_field_name, metric.name)

    test_statistic, p_value  = hydra.utils.call(cfg.stats, *grouped_scores)

    # Perform the significance test
    # u_statistic, p_value = statistic(group1_wers, group2_wers, alternative='two-sided')
    print(f"Test Statistic: {test_statistic}")
    print(f"P-value: {p_value}")


if __name__ == "__main__":
    evaluate()