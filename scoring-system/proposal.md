# Music DeepEncoder (MuDE) Proposal

Design notes for an audio-first analogue of DeepSeek-OCR, aimed at powering an “LLM-as-Music-Judge” pipeline with compact, information-rich music latents.

---

## TL;DR

- **Goal**: Convert raw audio into dense “music latent” tokens that an LLM judge can reason over without ingesting full waveforms.
- **Inspiration**: DeepSeek-OCR compresses documents into visual latents before decoding; we adopt the same staged compression for audio.
- **Stack**: EnCodec (local perception) → MERT (global semantics) → bridging adapter → frozen LLM judge.
- **Roadmap**: Train the bridge on music captioning/alignment, then fine-tune the judge on expert scores and vibe vectors.

---

## Motivation

Raw waveforms are too large for iterative judging loops, yet CLAP-style embeddings are too coarse for detailed tone evaluation. We need an intermediate representation—rich enough to capture harmony, rhythm, and timbre, but small enough for token budgets and iterative refinement.

DeepSeek-OCR achieved a 10× token reduction and better accuracy by splitting perception, compression, and reasoning. MuDE (Music DeepEncoder) adapts that recipe for audio so Charlotte’s ongoing MERT research slots directly into the judge stack.

---

## Architecture Overview

| Role                             | DeepSeek-OCR (vision) | MuDE (audio) suggestion | Rationale |
|----------------------------------|------------------------|-------------------------|-----------|
| Local perception / tokenizer     | SAM-base               | EnCodec                 | High-fidelity, locality-aware tokens that preserve timbre and micro-dynamics. |
| Global semantics / knowledge     | CLIP-large             | MERT                    | Music foundation model for deep structural understanding; outperforms CLAP on music tasks. |
| Bridge / compressor              | 16× conv compressor    | Q-Former + linear layers| Translate MERT features into compact latent packets with controllable length. |
| Decoder / judge                  | DeepSeek-3B-MoE        | Frozen LLM (e.g. Llama) | Reads compressed music latents to emit numeric scores or vibe vectors. |

Key ideas:

1. **Local perception** – Tokenize audio into short, perceptually weighted units (EnCodec), analogous to DeepSeek’s patch-wise SAM.
2. **Global semantics** – Feed those units into MERT to capture harmony, rhythm, and tone context (analogous to CLIP).
3. **Bridge** – Learn a lightweight adapter that distills MERT features into a sequence of latent tokens consumable by the judge LLM.
4. **Judge LLM** – Frozen base model to ensure stability; only LoRA or adapter layers are trained for scoring behaviour.

---

## Training Strategy

1. **Alignment Phase**  
   - Freeze EnCodec and MERT.  
   - Train only the bridge to map audio latents into the LLM token space.  
   - Objective: music captioning or retrieval (teach the LLM what each latent span represents).

2. **Judging Phase**  
   - Freeze the bridge; attach lightweight adapters to the LLM.  
   - Fine-tune on expert ratings, vibe labels, or preference data so the model emits scores and qualitative feedback.

3. **Evaluation**  
   - Compare against CLAP-based baselines on agreement with human judges, prompt fidelity, and compute cost.  
   - Track token usage and throughput to confirm DeepSeek-style efficiency gains.

---

## Related Work

- **DeepSeek-OCR**: Modular vision-language pipeline delivering 7–20× compression via local attention + convolutional downsampling before CLIP and a sparse multimodal decoder. Forms the blueprint for MuDE.
- **MusiLingo (2023)**: Applies MERT then a multimodal decoder to generate captions. Shows MERT’s strength but skips local chunking—an opportunity for MuDE to preserve finer timbral detail.
- **U-SAM (2025)**: Unified speech/audio/music model. Demonstrates the feasibility of broad audio-language models but still relies on global encoders without specialized latent compression.

---

## Next Steps

1. **Prototype latent pipeline**: EnCodec → MERT → simple projection into an LLM prompt; verify reconstruction fidelity and token savings.
2. **Build text embedding bank**: Combine curated descriptors with LLM-generated augmentations to support top-k retrieval + rewrite loops when needed.
3. **Data prep**: Collect or synthesize music captions, expert ratings, and vibe vectors to supervise both phases.
4. **Benchmark**: Evaluate against CLAP-only scoring on alignment, stability, and cost; document gaps for the next iteration.

For full context and references, cross-check with the research notes in the root `README.md` and keep issue [#17](https://github.com/vaclisinc/text2preset/issues/17) updated.
