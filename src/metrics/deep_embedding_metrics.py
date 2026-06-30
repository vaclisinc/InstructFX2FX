from .metric import Metric
import torch, librosa
from fxencoder_plusplus import load_model
from frechet_audio_distance import FrechetAudioDistance

# Using https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py for VGGish embeddings.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fxencoderplusplus = load_model('default', device=device)
vggish_embedder = FrechetAudioDistance(
    model_name='vggish',
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False,
)

def get_fx_embedding(audio_path):
    """Get the FX embedding for a given audio file."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wav, sr = librosa.load(audio_path, sr=44100, mono=False)
    wav = torch.from_numpy(wav).float()

    if wav.ndim == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)
    elif wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.ndim != 2:
        raise ValueError(f'Unexpected waveform shape: {tuple(wav.shape)}')

    wav = wav.unsqueeze(0).to(device)  # [1, 2, seq_len]

    return fxencoderplusplus.get_fx_embedding(wav)


def get_afx_rep_embedding(audio_path):
    """Get AFx-Rep style embedding for a given audio file.

    In this codebase, AFx-Rep is represented by FXEncoder++ embeddings.
    """
    return get_fx_embedding(audio_path)


def get_vggish_embedding(audio_path):
    """Get a VGGish embedding for a given audio file."""
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    emb = vggish_embedder.get_embeddings([wav], sr)
    return emb[0]


def cosine_similarity_from_audio(audio_path_a, audio_path_b):
    """Compute cosine similarity between FX embeddings of two audio files."""
    emb_a = get_fx_embedding(audio_path_a)
    emb_b = get_fx_embedding(audio_path_b)

    # Flatten to [D] so we can compute scalar cosine similarity.
    emb_a = emb_a.reshape(-1)
    emb_b = emb_b.reshape(-1)

    sim = torch.nn.functional.cosine_similarity(
        emb_a.unsqueeze(0), emb_b.unsqueeze(0), dim=1
    )
    return sim.item()

class FXEncCosineSimilarity(Metric):
    def compute(audio_path_a, audio_path_b):
        return cosine_similarity_from_audio(audio_path_a, audio_path_b)