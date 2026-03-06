/**
 * apply_eq.js — Audealize FX processor for SocialFX ground truth generation.
 *
 * Supports: eq (40-band Audealize Equalizer)
 * (Compression has no Audealize renderer — SocialFX comp data is parameter-only)
 *
 * Called from Python via subprocess:
 *   node apply_eq.js <fx_type> <input.wav> <output.wav> <params.json> [--range 1.0]
 *
 * Examples:
 *   node apply_eq.js eq    dry.wav wet.wav eq_curve.json
 *   node apply_eq.js 
 */

const fs = require("fs");
const { OfflineAudioContext, AudioBuffer } = require("node-web-audio-api");

// ═══════════════════════════════════════════════════════════════════════════
// EQ — 40 cascaded peaking biquads (exact Audealize port)
// ═══════════════════════════════════════════════════════════════════════════

const EQ_FREQS = [
  20, 50, 83, 120, 161, 208, 259, 318, 383, 455, 537, 628, 729, 843, 971,
  1114, 1273, 1452, 1652, 1875, 2126, 2406, 2719, 3070, 3462, 3901, 4392,
  4941, 5556, 6244, 7014, 7875, 8839, 9917, 11124, 12474, 13984, 15675,
  17566, 19682,
];
const EQ_Q = 4.31;

function eqCurveToGains(curve, range) {
  let maxEl = -Infinity, minEl = Infinity;
  for (let i = 0; i < 40; i++) {
    if (curve[i] > maxEl) maxEl = curve[i];
    if (curve[i] < minEl) minEl = curve[i];
  }
  return curve.map((v) => {
    const norm = maxEl !== minEl ? ((v - minEl) / (maxEl - minEl)) * 2 - 1 : 0;
    return range * 5 * norm;
  });
}

async function applyEq(samples, sampleRate, params, range) {
  const gainsDb = eqCurveToGains(params, range);
  const ctx = new OfflineAudioContext(1, samples.length, sampleRate);

  const buf = new AudioBuffer({ numberOfChannels: 1, length: samples.length, sampleRate });
  buf.copyToChannel(samples, 0);

  const src = ctx.createBufferSource();
  src.buffer = buf;

  const filters = [];
  for (let i = 0; i < 40; i++) {
    const f = ctx.createBiquadFilter();
    f.type = "peaking";
    f.frequency.value = EQ_FREQS[i];
    f.Q.value = EQ_Q;
    f.gain.value = gainsDb[i];
    filters.push(f);
    if (i > 0) filters[i - 1].connect(f);
  }
  src.connect(filters[0]);
  filters[39].connect(ctx.destination);
  src.start(0);

  const rendered = await ctx.startRendering();
  const out = new Float32Array(samples.length);
  rendered.copyFromChannel(out, 0);
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// REVERB — Noise-shaped convolution reverb (Audealize-style)
// SocialFX reverb has 40 params (RSC bands). We synthesize a noise-shaped IR
// from these params and convolve.
// ═══════════════════════════════════════════════════════════════════════════

const REVERB_DURATION = 3.0;

async function synthesizeIR(sampleRate, params, range) {
  const irLength = Math.floor(sampleRate * REVERB_DURATION);
  const ctx = new OfflineAudioContext(1, irLength, sampleRate);

  const noiseBuf = new AudioBuffer({ numberOfChannels: 1, length: irLength, sampleRate });
  const noiseData = noiseBuf.getChannelData(0);
  for (let i = 0; i < irLength; i++) {
    noiseData[i] = Math.random() * 2 - 1;
  }

  const decayRate = 6.0;
  for (let i = 0; i < irLength; i++) {
    const t = i / sampleRate;
    noiseData[i] *= Math.exp(-decayRate * t);
  }
  noiseBuf.copyToChannel(noiseData, 0);

  const src = ctx.createBufferSource();
  src.buffer = noiseBuf;

  const gainsDb = eqCurveToGains(params, range);
  const filters = [];
  for (let i = 0; i < 40; i++) {
    const f = ctx.createBiquadFilter();
    f.type = "peaking";
    f.frequency.value = EQ_FREQS[i];
    f.Q.value = EQ_Q;
    f.gain.value = gainsDb[i];
    filters.push(f);
    if (i > 0) filters[i - 1].connect(f);
  }
  src.connect(filters[0]);
  filters[39].connect(ctx.destination);
  src.start(0);

  const rendered = await ctx.startRendering();
  const ir = new Float32Array(irLength);
  rendered.copyFromChannel(ir, 0);

  let maxAbs = 0;
  for (let i = 0; i < ir.length; i++) {
    if (Math.abs(ir[i]) > maxAbs) maxAbs = Math.abs(ir[i]);
  }
  if (maxAbs > 0) {
    for (let i = 0; i < ir.length; i++) ir[i] /= maxAbs;
  }

  return ir;
}

async function applyReverb(samples, sampleRate, params, range) {
  const ir = await synthesizeIR(sampleRate, params, range);

  const outLength = samples.length + ir.length - 1;
  const ctx = new OfflineAudioContext(1, outLength, sampleRate);

  const inputBuf = new AudioBuffer({ numberOfChannels: 1, length: samples.length, sampleRate });
  inputBuf.copyToChannel(samples, 0);

  const irBuf = new AudioBuffer({ numberOfChannels: 1, length: ir.length, sampleRate });
  irBuf.copyToChannel(ir, 0);

  const src = ctx.createBufferSource();
  src.buffer = inputBuf;

  const convolver = ctx.createConvolver();
  convolver.normalize = true;
  convolver.buffer = irBuf;

  const wetGain = ctx.createGain();
  wetGain.gain.value = 0.7;
  const dryGain = ctx.createGain();
  dryGain.gain.value = 0.3;

  src.connect(convolver);
  convolver.connect(wetGain);
  wetGain.connect(ctx.destination);

  src.connect(dryGain);
  dryGain.connect(ctx.destination);

  src.start(0);

  const rendered = await ctx.startRendering();
  const out = new Float32Array(samples.length);
  rendered.copyFromChannel(out, 0);
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// WAV I/O
// ═══════════════════════════════════════════════════════════════════════════

function readWav(filepath) {
  const buf = fs.readFileSync(filepath);
  if (buf.toString("ascii", 0, 4) !== "RIFF") throw new Error("Not WAV");
  let offset = 12, fmt = null, data = null;
  while (offset < buf.length) {
    const id = buf.toString("ascii", offset, offset + 4);
    const sz = buf.readUInt32LE(offset + 4);
    if (id === "fmt ") {
      fmt = {
        format: buf.readUInt16LE(offset + 8),
        channels: buf.readUInt16LE(offset + 10),
        sampleRate: buf.readUInt32LE(offset + 12),
        bitsPerSample: buf.readUInt16LE(offset + 22),
      };
    } else if (id === "data") {
      data = { offset: offset + 8, size: sz };
    }
    offset += 8 + sz + (sz % 2);
  }
  if (!fmt || !data) throw new Error("Bad WAV");
  const bps = fmt.bitsPerSample / 8;
  const n = Math.floor(data.size / (bps * fmt.channels));
  const mono = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let ch = 0; ch < fmt.channels; ch++) {
      const off = data.offset + (i * fmt.channels + ch) * bps;
      if (fmt.bitsPerSample === 16) sum += buf.readInt16LE(off) / 32768;
      else if (fmt.bitsPerSample === 24) {
        const v = buf.readUIntLE(off, 3);
        sum += (v > 0x7fffff ? v - 0x1000000 : v) / 8388608;
      } else if (fmt.bitsPerSample === 32 && fmt.format === 3) sum += buf.readFloatLE(off);
      else if (fmt.bitsPerSample === 32) sum += buf.readInt32LE(off) / 2147483648;
    }
    mono[i] = sum / fmt.channels;
  }
  return { samples: mono, sampleRate: fmt.sampleRate };
}

function writeWav(filepath, samples, sampleRate) {
  const n = samples.length;
  const buf = Buffer.alloc(44 + n * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write("WAVE", 8);
  buf.write("fmt ", 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), 44 + i * 2);
  }
  fs.writeFileSync(filepath, buf);
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════

const FX_PROCESSORS = { eq: applyEq };
// NOTE: reverb path removed for now (EQ only).

async function main() {
  const args = process.argv.slice(2);
  if (args.length < 4) {
    console.error("Usage: node apply_eq.js <eq|reverb> <input.wav> <output.wav> <params.json> [--range 1.0]");
    process.exit(1);
  }

  const fxType = args[0];
  const inputPath = args[1];
  const outputPath = args[2];
  const paramsPath = args[3];
  let range = 1.0;
  if (args[4] === "--range") range = parseFloat(args[5]);

  if (!FX_PROCESSORS[fxType]) {
    console.error(`Unknown fx_type: '${fxType}'. Use 'eq'.`);
    process.exit(1);
  }

  const params = JSON.parse(fs.readFileSync(paramsPath, "utf-8"));
  if (!Array.isArray(params) || params.length !== 40) {
    console.error("params.json must be a JSON array of exactly 40 numbers");
    process.exit(1);
  }

  const { samples, sampleRate } = readWav(inputPath);
  const output = await FX_PROCESSORS[fxType](samples, sampleRate, params, range);
  writeWav(outputPath, output, sampleRate);

  console.log(JSON.stringify({ status: "ok", fx_type: fxType, input: inputPath, output: outputPath }));
}

main().catch((e) => { console.error(e); process.exit(1); });
