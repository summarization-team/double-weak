"""Primary script for evaluation.

For examples of how to run, see README.md.
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

from datasets import load_dataset
from itertools import islice


log = logging.getLogger(__name__)

# resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])


@hydra.main(config_path="../config", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")


if __name__ == "__main__":
    evaluate()