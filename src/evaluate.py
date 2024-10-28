"""
Speech Perception Bias Evaluation script for speech recognition models using Hugging Face Transformers.

This script evaluates a speech recognition model on a specified dataset.
It utilizes Hydra for configuration management, allowing for flexible
parameterization of the model, processor, pipeline, and dataset.

The script performs the following steps:
- Initializes the model, processor, and pipeline using the provided configurations.
- Loads the dataset specified in the configuration.
- Runs inference on a sample from the dataset.
- Logs the transcription result.
- TODO: add multi-dataset support
- TODO: add bias eval modules

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
    DatasetDict
)
from itertools import islice


log = logging.getLogger(__name__)

# resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])


@hydra.main(config_path="../config", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluates a speech recognition model using the configurations provided.

    This function initializes the model, processor, and pipeline based on
    the configurations specified via Hydra. It loads the dataset and performs
    inference on a sample to produce a transcription.

    Args:
        cfg (DictConfig): Configuration composed by Hydra, containing parameters
            for the model, processor, pipeline, and dataset.

    Returns:
        None

    """
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    log.info(f"Device:\n{device}")

    model: AutoModelForSpeechSeq2Seq = hydra.utils.instantiate(cfg.model, _convert_="object")
    log.info(f"Model Loaded")

    model.to(device)

    processor: AutoProcessor = hydra.utils.instantiate(cfg.processor, _convert_="object")
    log.info(f"Processor Instantiated")

    pipe: pipeline = hydra.utils.instantiate(
        cfg.pipeline,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        _convert_="object",
    )

    #TODO Add handling for multiple datasets
    ds_dict: DatasetDict = hydra.utils.instantiate(cfg.dataset, _convert_="object")

    sample = ds_dict[0]["audio"]

    result = pipe(sample)
    log.info(f"{result['text']}")

    #TODO Add error aalysis / bias analysis


if __name__ == "__main__":
    evaluate()