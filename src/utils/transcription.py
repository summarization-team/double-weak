"""
Moduel for transcription of utilities.
"""

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