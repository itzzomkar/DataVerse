import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, WavLMModel

from src.config import NUM_CLASSES


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) calculates weighted mean and weighted standard deviation
    across the time dimension, capturing both average prosody and emotional variance/dynamics.
    """
    def __init__(self, hidden_size: int, attention_size: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, time_steps, hidden_size)
        # mask: (batch_size, time_steps) bool
        scores = self.attention(x).squeeze(-1).float()  # (batch_size, time_steps)
        scores = scores.masked_fill(~mask, -1e4)
        weights = scores.softmax(dim=1).unsqueeze(-1)   # (batch_size, time_steps, 1)
        
        x = x.float()
        mean = (weights * x).sum(dim=1)  # (batch_size, hidden_size)
        variance = ((weights * x.square()).sum(dim=1) - mean.square()).clamp_min(1e-5)
        std = variance.sqrt()
        return torch.cat([mean, std], dim=-1)  # (batch_size, hidden_size * 2)


class SpeechEmotionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = NUM_CLASSES,
        classifier_dim: int = 256,
        dropout: float = 0.25,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_hidden_layers + 1
        
        # Learnable softmax weights across transformer layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        
        # Attentive Statistics Pooling
        self.pool = AttentiveStatisticsPooling(hidden_size)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, classifier_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states  # tuple of (batch, time, hidden)
        
        # Layer-weighted aggregation
        weights = self.layer_weights.softmax(dim=0)
        mixed = hidden_states[0] * weights[0]
        for i in range(1, len(hidden_states)):
            mixed = mixed + hidden_states[i] * weights[i]
            
        # Downsample attention mask to match backbone CNN feature extractor stride
        lengths = attention_mask.long().sum(dim=-1)
        conv_kernels = getattr(self.config, "conv_kernel", [10, 3, 3, 3, 3, 2, 2])
        conv_strides = getattr(self.config, "conv_stride", [5, 2, 2, 2, 2, 2, 2])
        
        for k, s in zip(conv_kernels, conv_strides):
            lengths = torch.div(lengths - k, s, rounding_mode="floor") + 1
        lengths = lengths.clamp(min=1, max=mixed.shape[1])
        
        time_mask = torch.arange(mixed.shape[1], device=mixed.device)[None, :] < lengths[:, None]
        
        pooled = self.pool(mixed, time_mask)
        logits = self.classifier(pooled)
        return logits
