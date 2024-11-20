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

For examples of how to run this script, please refer to the `README.md` file.

"""

import logging
import os

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
    Dataset,
    Audio
)

from evaluate import load

from itertools import islice

from utils import (
    create_groups, 
    transcribe_batch, 
    compute_metric_per_example, 
    compute_agg_statistics,
    save_results_to_json
)


log = logging.getLogger(__name__)

# Resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])


@hydra.main(config_path="../config", config_name="analyze", version_base=None)
def analyze(cfg: DictConfig) -> None:
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

    # Group scores based on the tested group
    grouped_scores = create_groups(ds, cfg.bias_field_name, metric.name)

    # Call statistical test
    test_statistic, p_value  = hydra.utils.call(cfg.stats, *grouped_scores)

    # Perform the significance test
    log.info(f"Test Statistic: {test_statistic}; P-value: {p_value}")

    # Calculate aggreate descriptive statistics for each group and overall
    agg_statistics = compute_agg_statistics(grouped_scores)

    # Prep results data
    results_data = {
        "grouped_scores": grouped_scores,
        "agg_statistics": agg_statistics,
        "test_statistic": test_statistic,
        "p_value": p_value,
    }

    # Write the output data to a JSON file
    save_results_to_json(results_data, output_dir, cfg)

    # results_file = os.path.join(output_dir, cfg.get("results_fname", "results"))
    # results_file += ".json"

    # with open(results_file, "w") as f:
    #     json.dump(results_data, f, indent=4, cls=NpEncoder)
    
    log.info(f"Results saved to {output_dir}.")

if __name__ == "__main__":
    analyze()