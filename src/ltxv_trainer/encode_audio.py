import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def encode_audio(
    audio: torch.Tensor,
    processor: Wav2Vec2Processor,
    wav2vec_model: Wav2Vec2Model,
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(
        [a.cpu().numpy() for a in audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        latents = outputs.last_hidden_state
        return latents
