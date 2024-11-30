# double-weak
A system for evaluating biases in Automatic Speech Recognition (ASR) tools

## Project Overview
This project aims to identify and measure potential biases in ASR systems. ASR technologies have become increasingly prevalent in our daily lives, from virtual assistants to transcription services. However, these systems may exhibit performance disparities across different demographic groups, accents, or speech patterns.

### Why This Matters
- **Fairness**: ASR systems should provide consistent performance across all user groups.
- **Accessibility**: Identify potential barriers for users with different accents or speech patterns.
- **Quality Assurance**: Help developers understand and address bias in their ASR models.

## Components
- **Bias Evaluation**: Measures performance differences across specified categories.
- **HuggingFace Integration**: Works with HuggingFace ASR models and datasets.
- **Configurable Analysis**: Supports different bias categories and datasets.

## Environment Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/summarization-team/double-weak.git
    cd double-weak
    ```

2. Create and activate a conda environment:
    For GPU support:
    ```bash
    conda env create -f gpu_environment.yml
    conda activate double-weak-gpu
    ```

    For CPU-only:
    ```bash
    conda env create -f cpu_environment.yml
    conda activate double-weak-cpu
    ```

3. Verify installation:
    ```bash
    python src/verify_setup.py
    ```

## How to Run

### Default Configuration
Run the analysis with default settings:
```bash
python src/analyze.py
```

### Custom Bias Analysis
Analyze specific bias categories using hydra overrides:
```bash
python src/analyze.py bias_field_name=accent
```

### Alternative Dataset
Use a different dataset for analysis:
```bash
python src/analyze.py dataset.path="hf-internal-testing/librispeech_asr_dummy" dataset.split=validation bias_field_name=chapter_id
```

## Output
The analysis produces metrics showing potential biases in the ASR system's performance across different categories within the specified bias field.

## Components Details

### `src/analyze.py`
The `analyze.py` script is the core of the bias evaluation system. This script evaluates a speech recognition model on a specified dataset. It utilizes Hydra for configuration management, allowing for flexible parameterization of the model, processor, pipeline, and dataset.

 It performs the following tasks:
- **Loads Configuration**: Uses Hydra to load configurations for the model, dataset, and analysis parameters.
- **Data Processing**: Prepares the dataset for analysis, including loading audio files and transcriptions.
- **Model Inference**: Runs the ASR model on the dataset to generate transcriptions.
- **Bias Evaluation**: Computes metrics to evaluate bias in the ASR system's performance.

### Metrics
The script computes several key metrics to evaluate bias:
- **Word Error Rate (WER)**: Measures the percentage of words that were incorrectly transcribed.
- **Statistical Tests**: Uses statistical tests like the Kruskal-Wallis H-test and Mann-Whitney U test to identify significant differences in performance across different categories.

### Model Configuration
The project uses a pretrained ASR model from HuggingFace by default. The model configuration is specified in `config/model/from_pretrained.yaml`. 

#### Swapping Out the Model
To use a different ASR model, you can easily modify the configuration file:
1. Open `config/model/from_pretrained.yaml`.
2. Change the `model_name_or_path` parameter to the desired model's name or path.

For example:
```yaml
model_name_or_path: "facebook/wav2vec2-large-960h"
```

You can also specify the model directly when running the analysis:
```bash
python src/analyze.py model.model_name_or_path="facebook/wav2vec2-large-960h"
```

### `src/bias/module.py`
Contains the `BiasModule` and `CategoricalBiasModule` classes which define the structure for evaluating bias in ASR models.

### `src/utils/data_processing.py`
Includes functions for processing data related to transcription and statistical analysis, such as `create_groups` and `save_results_to_json`.

### `src/utils/calculation.py`
Provides utility functions for statistical calculations, including `compute_stat`, `compute_agg_statistics`, and `compute_metric_per_example`.

### `src/utils/transcription.py`
Defines the `transcribe_batch` function for transcribing a batch of audio samples using the ASR pipeline.

### Configuration Files
- `config/analyze.yaml`: Main configuration file for the analysis.
- `config/model/from_pretrained.yaml`: Configuration for loading a pretrained model.
- `config/pipeline/automatic-speech-recognition.yaml`: Configuration for the ASR pipeline.
- `config/metric/wer.yaml`: Configuration for the Word Error Rate (WER) metric.
- `config/stats/scipy/kruskal.yaml`: Configuration for the Kruskal-Wallis H-test.
- `config/stats/scipy/mannwhitneyu.yaml`: Configuration for the Mann-Whitney U test.
- `config/data/edacc.yaml`: Configuration for the EDACC dataset.
- `config/data/librispeech_asr_dummy.yaml`: Configuration for the LibriSpeech ASR dummy dataset.

### Scripts
- `scripts/analyze.sh`: Shell script to run the analysis.
- `scripts/analyze_gpu.sbatch`: SLURM batch script for running the analysis on a GPU.
- `scripts/analyze_cpu.sbatch`: SLURM batch script for running the analysis on a CPU.

### Environment Files
- `gpu_environment.yml`: Conda environment file for GPU support.
- `cpu_environment.yml`: Conda environment file for CPU-only support.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

