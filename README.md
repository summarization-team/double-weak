# double-weak
A system for evaluating biases in Automatic Speech Recognition (ASR) tools

## Project Overview
This project aims to identify and measure potential biases in ASR systems. ASR technologies have become increasingly prevalent in our daily lives, from virtual assistants to transcription services. However, these systems may exhibit performance disparities across different demographic groups, accents, or speech patterns ([Qian et al., 2017](https://doi.org/10.21437/Interspeech.2017-250), [Tatman & Kasten, 2017](https://doi.org/10.21437/Interspeech.2017-1746), [Palanica et al., 2019](https://doi.org/10.1038/s41746-019-0133-x), [Koenecke et al., 2020](https://doi.org/10.1073/pnas.1915768117), [Wu et al., 2020](https://doi.org/10.1145/3379503.3403563), [Feng et al., 2021](https://arxiv.org/abs/2103.15122))
.

### Why This Matters
- **Fairness**: ASR systems should provide consistent performance across all user groups .
- **Accessibility**: Identify potential barriers for users with different accents or speech patterns.
- **Quality Assurance**: Help developers understand and address bias in their ASR models.

### How the System Works
`double-weak` is uses `hydra` configuration files to make it easy to switch out datasets, models, evaluation metrics, bias categories, and statistical tests. These components are customizable and instantiable using `hydra` configurations, meaning that no code changes should be required to expand beyond the base architecture. The base architecture supports the following, though most can be altered via `hydra` config:
-  Evaluating models based on their Word Error Rate ("WER")
-  Testing of categorical variable biases (e.g., accent, gender, race, etc.) only, as opposed to continuous variable biases
-  Use of HuggingFace datasets. You can load any HuggingFace dataset. Data must be in the format of a HuggingFace datsaset, regardless of whether it is hosted by HuggingFace or converted from another.
-  Statistical testing with the [Kruskal-Wallis H-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html). You may use add configurations for any _appropriate_ statistical test.

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

## Relevant Literature

### Bias in ASR
- Feng, S., Kudina, O., & Halpern, B. M. (2021). Quantifying bias in automatic speech recognition. *arXiv preprint arXiv:2103.15122*. https://arxiv.org/abs/2103.15122  
- Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M., Mengesha, Z., & Goel, S. (2020). Racial disparities in automated speech recognition. *Proceedings of the National Academy of Sciences, 117*(14), 7684–7689. https://doi.org/10.1073/pnas.1915768117  
- Palanica, A., Thommandram, A., Lee, A., & Li, M. (2019). Do you understand the words that are comin' outta my mouth? Voice assistant comprehension of medication names. *npj Digital Medicine, 2*, 55. https://doi.org/10.1038/s41746-019-0133-x  
- Qian, Y., Evanini, K., Wang, X., Lee, C. M., & Mulholland, M. (2017, August). Bidirectional LSTM-RNN for improving automated assessment of non-native children's speech. In *INTERSPEECH 2017: Proceedings of the 18th Annual Conference of the International Speech Communication Association* (pp. 1417–1421). [10.21437/Interspeech.2017-250](https://doi.org/10.21437/Interspeech.2017-250)
- Tatman, R., & Kasten, C. (2017). Effects of talker dialect, gender, and race on accuracy of Bing Speech and YouTube automatic captions. *Proceedings of INTERSPEECH 2017: Proceedings of the 18th Annual Conference of the International Speech Communication Association*, 934–938. https://doi.org/10.21437/Interspeech.2017-1746  
- Wu, Y., Rough, D., Bleakley, A., Edwards, J., Cooney, O., Doyle, P. R., Clark, L., & Cowan, B. R. (2020). See what I’m saying? Comparing intelligent personal assistant use for native and non-native language speakers. In *Proceedings of the 22nd International Conference on Human-Computer Interaction with Mobile Devices and Services (MobileHCI '20)* (Article 34, pp. 1–9). Association for Computing Machinery. https://doi.org/10.1145/3379503.3403563  


### Bias in Human Perception
- Abrahamsson, N., & Hyltenstam, K. (2009). Age of onset and nativelikeness in a second language: Listener perception versus linguistic scrutiny. *Language Learning, 59*(2), 249–306. https://doi.org/10.1111/j.1467-9922.2009.00507.x  
- Becker, K. (2014). The social motivations of reversal: Raised BOUGHT in New York City English. *Language in Society, 43*(4), 395–420. https://www.jstor.org/stable/43904579  
- Johnson, K., Strand, E. A., & D'Imperio, M. (1999). Auditory–visual integration of talker gender in vowel perception. *Journal of Phonetics, 27*(4), 359–384. https://doi.org/10.1006/jpho.1999.0100  
- Lotto, A. J., & Holt, L. L. (2016). Speech perception: The view from the auditory system. In G. Hickok & S. L. Small (Eds.), *Neurobiology of Language* (pp. 179–188). Academic Press.  
- Munson, B. (2007). The acoustic correlates of perceived masculinity, perceived femininity, and perceived sexual orientation. *Language and Speech, 50*(Pt 1), 125–142. https://doi.org/10.1177/00238309070500010601  
- Nearey, T. M. (1997). Speech perception as pattern recognition. *The Journal of the Acoustical Society of America, 101*(6), 3241–3254.  


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

