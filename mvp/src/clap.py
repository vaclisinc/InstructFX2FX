"""
CLAP Model Wrapper

Provides a clean interface for encoding audio and text using CLAP.
"""

import torch
import torchaudio


class CLAPWrapper:
    """
    Wrapper for CLAP model with simplified interface.
    """

    def __init__(self, model_name='laion/clap-htsat-unfused', device='cpu'):
        """
        Initialize CLAP model.

        Args:
            model_name: CLAP model checkpoint name
            device: Device to load model on
        """
        from transformers import ClapModel, ClapProcessor

        self.device = device
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.sample_rate = 48000  # CLAP expects 48kHz

        self.model.eval()
        print(f"✓ CLAP model loaded: {model_name}")

    @torch.no_grad()
    def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to CLAP embedding.

        Args:
            audio: Audio tensor [B, C, T] or [C, T]

        Returns:
            Audio embedding [B, D] where D is embedding dimension
        """
        # Ensure batch dimension
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)

        # Convert to mono if stereo
        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)
        elif audio.shape[1] == 1:
            pass
        else:
            raise ValueError(f"Expected 1 or 2 channels, got {audio.shape[1]}")

        # Resample to 48kHz if needed
        if hasattr(self, 'current_sr') and self.current_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                self.current_sr,
                self.sample_rate
            ).to(audio.device)
            audio = resampler(audio)

        # Move to CPU for processor (CLAP processor expects numpy/CPU)
        audio_np = audio.squeeze(1).cpu().numpy()

        # Process audio
        inputs = self.processor(
            audios=audio_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        # Move to model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        audio_embeds = self.model.get_audio_features(**inputs)

        return audio_embeds

    @torch.no_grad()
    def get_text_embedding(self, text: str | list[str]) -> torch.Tensor:
        """
        Encode text to CLAP embedding.

        Args:
            text: Single text string or list of strings

        Returns:
            Text embedding [B, D] where B is batch size
        """
        # Ensure list format
        if isinstance(text, str):
            text = [text]

        # Process text
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True
        )

        # Move to model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        text_embeds = self.model.get_text_features(**inputs)

        return text_embeds


def load_clap_model(device='cpu') -> CLAPWrapper:
    """
    Convenience function to load CLAP model.

    Args:
        device: Device to load model on

    Returns:
        CLAPWrapper instance
    """
    return CLAPWrapper(device=device)


# Alternative: Use MS-CLAP (from Text2FX paper)
# This is the CLAP variant used in the Text2FX paper

class MSCLAPWrapper:
    """
    Microsoft CLAP wrapper (as used in Text2FX).

    This uses the same CLAP model as the original Text2FX paper.
    """

    def __init__(self, device='cpu'):
        """Initialize MS-CLAP model."""
        try:
            import laion_clap
            self.device = device
            self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
            self.model.load_ckpt()  # Load pretrained checkpoint
            self.sample_rate = 48000
            print(f"✓ MS-CLAP model loaded")
        except ImportError:
            raise ImportError(
                "laion-clap not installed. Install with: pip install laion-clap"
            )

    @torch.no_grad()
    def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Get audio embeddings using MS-CLAP.

        Args:
            audio: Audio tensor [B, C, T]

        Returns:
            Audio embeddings [B, D]
        """
        # MS-CLAP expects audio as [B, T] (mono)
        if audio.ndim == 3:
            audio = audio.mean(dim=1)  # Convert to mono

        # Get embeddings
        audio_embeds = self.model.get_audio_embedding_from_data(
            x=audio,
            use_tensor=True
        )

        return torch.from_numpy(audio_embeds).to(self.device)

    @torch.no_grad()
    def get_text_embedding(self, text: str | list[str]) -> torch.Tensor:
        """
        Get text embeddings using MS-CLAP.

        Args:
            text: Text string or list of strings

        Returns:
            Text embeddings [B, D]
        """
        if isinstance(text, str):
            text = [text]

        text_embeds = self.model.get_text_embedding(text, use_tensor=True)

        return torch.from_numpy(text_embeds).to(self.device)
