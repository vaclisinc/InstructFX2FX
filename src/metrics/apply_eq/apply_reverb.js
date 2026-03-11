/**
 * apply_reverb.js — SocialFX reverb (5 params: delay_time, decay, stereo_spread, cutoff_freq, wet_gain)
 *
 * Usage: node apply_reverb.js <input.wav> <output.wav> <params.json>
 *
 * params.json = JSON array of 5 floats: [delay_time, decay, stereo_spread, cutoff_freq, wet_gain]
 * Builds an IR with pre-delay, decay-shaped noise, and lowpass at cutoff_freq; convolves and mixes by wet_gain.
 */

const fs = require("fs");
const { OfflineAudioContext, AudioBuffer } = require("node-web-audio-api");

const IR_DURATION = 2.5;  // seconds (tail after pre-delay)

// param order: delay_time, decay, stereo_spread, cutoff_freq, wet_gain
function parseParams(arr) {
  if (!Array.isArray(arr) || arr.length < 5) return null;
  return {
    delayTime: Math.max(0, Number(arr[0])),
    decay: Math.max(0, Math.min(1, Number(arr[1]))),
    stereoSpread: Number(arr[2]),
    cutoffFreq: Math.max(20, Math.min(20000, Number(arr[3]))),
    wetGain: Math.max(0, Math.min(1, Number(arr[4]))),
  };
}

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

async function synthesizeIR(sampleRate, p) {
  const preDelaySamples = Math.floor(p.delayTime * sampleRate);
  const tailSamples = Math.floor(sampleRate * IR_DURATION);
  const irLength = preDelaySamples + tailSamples;

  const ctx = new OfflineAudioContext(1, irLength, sampleRate);
  const irData = new Float32Array(irLength);

  // Pre-delay: zeros
  // Tail: noise with exponential decay (high decay param = longer reverb = slower decay rate)
  const decayRate = 3 * (1 - p.decay) + 0.8;
  for (let i = 0; i < tailSamples; i++) {
    irData[preDelaySamples + i] = (Math.random() * 2 - 1) * Math.exp(-decayRate * (i / sampleRate));
  }

  const irBuf = new AudioBuffer({ numberOfChannels: 1, length: irLength, sampleRate });
  irBuf.copyToChannel(irData, 0);

  const src = ctx.createBufferSource();
  src.buffer = irBuf;

  const lowpass = ctx.createBiquadFilter();
  lowpass.type = "lowpass";
  lowpass.frequency.value = Math.min(p.cutoffFreq, sampleRate / 2 - 100);
  lowpass.Q.value = 0.7;

  src.connect(lowpass);
  lowpass.connect(ctx.destination);
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

async function main() {
  const args = process.argv.slice(2);
  if (args.length < 3) {
    console.error("Usage: node apply_reverb.js <input.wav> <output.wav> <params.json>");
    process.exit(1);
  }

  const inputPath = args[0];
  const outputPath = args[1];
  const paramsPath = args[2];

  const raw = JSON.parse(fs.readFileSync(paramsPath, "utf-8"));
  const params = Array.isArray(raw) ? parseParams(raw) : null;
  if (!params) {
    console.error("params.json must be a JSON array of at least 5 numbers [delay_time, decay, stereo_spread, cutoff_freq, wet_gain]");
    process.exit(1);
  }

  const { samples, sampleRate } = readWav(inputPath);
  const ir = await synthesizeIR(sampleRate, params);

  const convLen = samples.length + ir.length - 1;
  const ctx = new OfflineAudioContext(1, convLen, sampleRate);

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
  wetGain.gain.value = params.wetGain;
  const dryGain = ctx.createGain();
  dryGain.gain.value = 1 - params.wetGain;

  src.connect(convolver);
  convolver.connect(wetGain);
  wetGain.connect(ctx.destination);
  src.connect(dryGain);
  dryGain.connect(ctx.destination);
  src.start(0);

  const rendered = await ctx.startRendering();
  const wet = new Float32Array(samples.length);
  rendered.copyFromChannel(wet, 0);
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    out[i] = (1 - params.wetGain) * samples[i] + wet[i];
  }
  writeWav(outputPath, out, sampleRate);
  console.log(JSON.stringify({ status: "ok", output: outputPath }));
}

main().catch((e) => { console.error(e); process.exit(1); });
