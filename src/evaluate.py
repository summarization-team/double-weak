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
from itertools import islice

log = logging.getLogger(__name__)

# Resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])

def transcribe_batch(batch, asr_pipeline):
    """
    Transcribes a batch of audio samples using the ASR pipeline.

    Args:
        batch (dict): A batch of data containing audio samples.
        asr_pipeline (transformers.pipelines.Pipeline): The ASR pipeline for inference.

    Returns:
        dict: The batch with an added 'transcription' field containing the transcribed text.
    """
    # Get the list of audio samples from the batch
    audio_list = batch["audio"]

    # Perform inference using the pipeline
    transcriptions = asr_pipeline(audio_list)

    # Extract the transcription text
    texts = [item["text"] for item in transcriptions]

    # Add the transcriptions to the batch
    batch["transcription"] = texts
    return batch

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
        lambda batch: transcribe_batch(batch, asr_pipeline),
        batched=True,
        batch_size=batch_size,
        remove_columns=ds.column_names,
    )
    log.info("Inference completed.")

    # Save the transcriptions to a file
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"output_dir={output_dir}")
    output_file = os.path.join(output_dir, cfg.transcriptions_fname)
    ds.to_json(output_file)
    log.info(f"Transcriptions saved to {output_file}.")

    # TODO: Add error analysis / bias analysis


if __name__ == "__main__":
    evaluate()