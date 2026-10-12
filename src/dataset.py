import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from transformers import Wav2Vec2FeatureExtractor

from src.config import SAMPLE_RATE, MAX_SAMPLES, MIN_SAMPLES


def augment_audio(audio: np.ndarray) -> np.ndarray:
    """
    Applies emotion-preserving acoustic augmentations:
    - Random gain adjustment (+/- 2 dB)
    - Mild additive Gaussian noise (SNR 25-40 dB)
    """
    if np.random.rand() < 0.35:
        gain = 10 ** (np.random.uniform(-2, 2) / 20)
        audio = audio * np.float32(gain)
        
    if np.random.rand() < 0.25:
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        noise_std = rms / 10 ** (np.random.uniform(25, 40) / 20)
        noise = np.random.normal(0, noise_std, len(audio)).astype(np.float32)
        audio = audio + noise
        
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


class SpeechEmotionDataset(Dataset):
    def __init__(
        self,
        frame,
        max_samples: int = MAX_SAMPLES,
        crops: Tuple[float, ...] = (0.0, 0.5, 1.0),
        training: bool = False,
        augment: bool = True,
        min_samples: int = MIN_SAMPLES,
    ):
        self.frame = frame.reset_index(drop=True)
        self.max_samples = max_samples
        self.training = training
        self.augment = augment
        self.min_samples = min_samples
        
        self.items = []
        for i, length in enumerate(self.frame["n_samples"]):
            excess = max(0, int(length) - self.max_samples)
            if training:
                self.items.append((i, None))
            else:
                starts = sorted(set(round(excess * float(c)) for c in crops))
                for s in starts:
                    self.items.append((i, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        row_idx, start = self.items[index]
        row = self.frame.iloc[row_idx]
        
        audio = np.load(row["cache_path"], allow_pickle=False)
        if start is None:
            # Random crop during training if longer than max_samples
            if len(audio) > self.max_samples:
                start = np.random.randint(0, len(audio) - self.max_samples + 1)
            else:
                start = 0
                
        audio = audio[start:start + self.max_samples].copy()
        
        if self.training and self.augment:
            audio = augment_audio(audio)
            
        if len(audio) < self.min_samples:
            audio = np.pad(audio, (0, self.min_samples - len(audio)))
            
        label = int(row["label"]) if "label" in row else -1
        return audio, row_idx, label


class AudioCollator:
    def __init__(self, extractor: Wav2Vec2FeatureExtractor):
        self.extractor = extractor

    def __call__(self, items):
        audios = [item[0] for item in items]
        indices = torch.tensor([item[1] for item in items], dtype=torch.long)
        labels = torch.tensor([item[2] for item in items], dtype=torch.long)
        
        encoded = self.extractor(
            audios,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_values": encoded.input_values,
            "attention_mask": encoded.attention_mask,
            "indices": indices,
            "labels": labels,
        }


def get_dataloader(
    df,
    extractor: Wav2Vec2FeatureExtractor,
    batch_size: int = 4,
    workers: int = 0,
    training: bool = False,
    crops: Tuple[float, ...] = (0.0, 0.5, 1.0),
    augment: bool = True,
    seed: int = 42,
) -> DataLoader:
    dataset = SpeechEmotionDataset(
        df,
        training=training,
        crops=crops,
        augment=augment,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=AudioCollator(extractor),
        generator=generator if training else None,
    )
