# double-weak
A system for evaluating biases in Automatic Speech Recognition tools


# How to Run
This project demonstrate running bias calculations for a HuggingFac ASR model against a bias category within HuggingFace audio transcription dataset.

## Envroment
TODO

## Run Default Configration
```
python src/analyze.py
```


## Run w/Alternative Dataset
Use `hydra` argument overides as follows or create a new dervatev config.
```
python src/analyze.py bias_field_name=accent
```

## Run w/Alternative Dataset

```
python src/analyze.py dataset.path="hf-internal-testing/librispeech_asr_dummy" dataset.split=validation bias_field_name=chapter_id
```