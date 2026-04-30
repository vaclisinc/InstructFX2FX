## Exisiting Issue

1. DASP reverb is way slower than eq on GD:
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

  Let me check if num_samples can be set at construction time: