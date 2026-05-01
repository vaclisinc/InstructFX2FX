## Future Directions

### 1. Ground-Truth Impulse Response Database (FIR Convolution)

Leverage existing databases of measured impulse responses (IRs) for real acoustic
environments — church, underwater, classroom, concert hall, etc.

**Core idea:** When the user prompt hits a keyword (e.g. "make it sound like a church"),
look up the corresponding ground-truth IR and convolve it with the dry audio via FIR.
The convolved audio can then serve as:

- **A target for CLAP-based optimization** — use CLAP(convolved, text) as the loss target
  so gradient descent drives the predicted parameters toward the real acoustic signature.
- **Context for the LLM** — provide the convolved audio (or its CLAP embedding) as a
  reference so the LLM has a concrete anchor for what "church-like" actually sounds like.

**Why this matters:** The main criticism is "you don't know how to tune, so how can you
teach a computer?" Physically measured IRs sidestep this entirely — the ground truth comes
from real-world acoustics, not human judgment. This is the fastest path to a convincing demo.

### 2. CLAP-in-the-Loop LLM (LLM ↔ CLAP Co-Reasoning)

Instead of using CLAP only as a loss function for gradient descent, let the LLM directly
use CLAP to validate and refine its own parameter proposals.

**Core idea:** LLM proposes FX parameters → render audio → compute CLAP embedding →
feed CLAP similarity / delta back to the LLM → LLM adjusts parameters → iterate.
CLAP becomes the "ears" of the LLM, not just the ears of the optimization loop.

**Key design question:** LLMs can't interpret raw embedding vectors. The feedback signal
needs to be translated into natural language (e.g. "current output is 0.15 cosine distance
from target 'bright' — still too warm") so the LLM can reason about what to change.

**Potential benefit:** This bridges the gap between the LLM's semantic understanding
(what "bright" means musically) and CLAP's perceptual grounding (how "bright" sounds),
enabling iterative self-correction without gradient descent.

### 3. Non-Linear Trajectories in CLAP Embedding Space

Inspired by Vision-Language-Action (VLA) models in robotics, where SE(3) flow matching
on Lie groups outperforms linear interpolation for policy learning.

**Core idea:** The current optimization assumes Euclidean geometry in CLAP space — L2 loss
implies linear interpolation is the optimal path between source and target embeddings. But
CLAP's learned manifold is likely curved: the perceptually meaningful path from "warm" to
"bright" may not be a straight line in embedding space.

**Possible approaches:**
- **Flow matching on the CLAP manifold** — learn a vector field that traces perceptually
  smooth transitions between adjective embeddings, rather than naive linear interpolation.
- **Geodesic loss** — replace L2 with a loss that respects the local curvature of the
  embedding manifold (e.g. pulling back a Riemannian metric from CLAP's latent space).
- **Empirical validation first** — linearly interpolate between two adjective embeddings
  (e.g. warm → bright), decode the midpoint to audio, and check whether it sounds
  reasonable. If midpoints are perceptually incoherent, that's evidence the space needs
  non-linear treatment.

**Why this is interesting:** This is the most research-oriented direction and could yield
novel contributions. The connection between Lie-group-aware optimization in robotics and
audio effect parameter spaces is unexplored.

---

## Exisiting Issue

1. DASP reverb is way slower than eq:
 The root cause is clear. The NoiseShapedReverb forward pass does a time-domain convolution of
   the full audio against a synthesized impulse response with:
  - num_samples=65536 (~1.5 s IR at 44100 Hz)
  - num_bandpass_taps=1023 (12 large FIR filters on white noise)

  That final vconv1d on 5-second audio × 65536-sample IR every gradient step is the bottleneck
  — versus EQ which just applies simple biquad IIR filters. The parameter count (18 vs 25) is
  irrelevant; it's the O(seq_len × num_samples) time-domain convolution running inside
  torch.autograd on every iteration.

  There are two levers to fix it:

  1. Pass a smaller num_samples to NoiseShapedReverb (e.g. 8192 instead of 65536) — reduces IR
  length from ~1.5 s to ~0.19 s, which cuts convolution time ~8×. This is where to start since
  dasp_pytorch does accept it as a constructor arg.
  2. Switch to FFT convolution — torchaudio.functional.fftconvolve or torch.fft-based
  overlap-add would make long IRs affordable, but requires patching the library or wrapping it.

-> solved but still slow (at least no OOM on my macbook)

