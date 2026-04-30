window.TEST_PIPELINE_EXAMPLES = {
  "examples": [
    {
      "case": "A",
      "turn": 1,
      "prompt": "bright",
      "reprompt": "bright",
      "expected_fx": [
        "eq"
      ],
      "fx_chain": [
        "eq"
      ],
      "params_path": "outputs/A/turn1/params.json",
      "before_audio_path": "outputs/A/turn1/before.wav",
      "after_audio_path": "outputs/A/turn1/after.wav",
      "params": {
        "eq": {
          "b1_freq": 60.0,
          "b1_gain": 3.0,
          "b1_q": 0.8,
          "b2_freq": 250.0,
          "b2_gain": 1.5,
          "b2_q": 1.2,
          "b3_freq": 1000.0,
          "b3_gain": 0.0,
          "b3_q": 1.0,
          "b4_freq": 3000.0,
          "b4_gain": 4.0,
          "b4_q": 1.0,
          "b5_freq": 6000.0,
          "b5_gain": 6.0,
          "b5_q": 0.5,
          "b6_freq": 12000.0,
          "b6_gain": 8.0,
          "b6_q": 0.7
        }
      }
    },
    {
      "case": "A",
      "turn": 2,
      "prompt": "warmer",
      "reprompt": "warmer",
      "expected_fx": [
        "eq"
      ],
      "fx_chain": [
        "eq"
      ],
      "params_path": "outputs/A/turn2/params.json",
      "before_audio_path": "outputs/A/turn2/before.wav",
      "after_audio_path": "outputs/A/turn2/after.wav",
      "params": {
        "eq": {
          "b1_freq": 65.99485409684797,
          "b1_gain": 4.162433624267578,
          "b1_q": 0.7147089081844831,
          "b2_freq": 213.2367070907582,
          "b2_gain": 2.6753740310668945,
          "b2_q": 1.068768917197818,
          "b3_freq": 844.0079859897643,
          "b3_gain": 1.181081771850586,
          "b3_q": 1.119289236642945,
          "b4_freq": 2630.1292090467596,
          "b4_gain": 2.855924606323242,
          "b4_q": 1.119790446144892,
          "b5_freq": 5530.756209527171,
          "b5_gain": 6.971423149108887,
          "b5_q": 0.5533802988521856,
          "b6_freq": 12527.615008003708,
          "b6_gain": 6.974275588989258,
          "b6_q": 0.7757751742135692
        }
      }
    },
    {
      "case": "B",
      "turn": 1,
      "prompt": "make it sound like a church",
      "reprompt": "make it sound like a church",
      "expected_fx": [
        "rev"
      ],
      "fx_chain": [
        "rev"
      ],
      "params_path": "outputs/B/turn1/params.json",
      "before_audio_path": "outputs/B/turn1/before.wav",
      "after_audio_path": "outputs/B/turn1/after.wav",
      "params": {
        "rev": {
          "early_gain": 0.7,
          "early_delay": 0.02,
          "early_diffusion": 0.9,
          "early_width": 0.85,
          "early_lowcut": 150.0,
          "early_highcut": 6500.0,
          "early_mix": 0.6,
          "late_gain": 0.8,
          "decay_time": 4.5,
          "late_diffusion": 0.9,
          "density": 0.95,
          "mod_rate": 0.8,
          "mod_depth": 0.5,
          "late_lowcut": 100.0,
          "late_highcut": 7000.0,
          "late_width": 0.9,
          "late_mix": 0.7,
          "pre_delay": 0.05,
          "damping": 0.5,
          "lowcut": 100.0,
          "highcut": 8000.0,
          "wet": 0.7,
          "dry": 0.3,
          "width": 0.9,
          "mix": 0.6
        }
      }
    },
    {
      "case": "B",
      "turn": 2,
      "prompt": "add some grit",
      "reprompt": "add some grit",
      "expected_fx": [
        "dist"
      ],
      "fx_chain": [
        "rev",
        "dist"
      ],
      "params_path": "outputs/B/turn2/params.json",
      "before_audio_path": "outputs/B/turn2/before.wav",
      "after_audio_path": "outputs/B/turn2/after.wav",
      "params": {
        "rev": {
          "early_gain": 0.6916897892951965,
          "early_delay": 0.02032394260168076,
          "early_diffusion": 0.9062321782112122,
          "early_width": 0.8477765917778015,
          "early_lowcut": 159.87973949907993,
          "early_highcut": 6396.401203921254,
          "early_mix": 0.6062440872192383,
          "late_gain": 0.8089719414710999,
          "decay_time": 4.671956615854529,
          "late_diffusion": 0.8911781311035156,
          "density": 0.9544222354888916,
          "mod_rate": 0.7790344208478928,
          "mod_depth": 0.5111633539199829,
          "late_lowcut": 99.375976840892,
          "late_highcut": 6918.431075321204,
          "late_width": 0.9043819904327393,
          "late_mix": 0.6885766983032227,
          "pre_delay": 0.04958715736865998,
          "damping": 0.5048732161521912,
          "lowcut": 94.12829432080677,
          "highcut": 7653.768565554865,
          "wet": 0.7157725095748901,
          "dry": 0.2804076075553894,
          "width": 0.9023272395133972,
          "mix": 0.6139215230941772
        },
        "dist": {
          "drive_db": 8.5
        }
      }
    },
    {
      "case": "B",
      "turn": 3,
      "prompt": "too harsh, soften it",
      "reprompt": "too harsh, soften it",
      "expected_fx": [],
      "fx_chain": [
        "rev",
        "dist",
        "eq"
      ],
      "params_path": "outputs/B/turn3/params.json",
      "before_audio_path": "outputs/B/turn3/before.wav",
      "after_audio_path": "outputs/B/turn3/after.wav",
      "params": {
        "rev": {
          "early_gain": 0.2392972707748413,
          "early_delay": 0.05610524415969849,
          "early_diffusion": 0.46509888768196106,
          "early_width": 0.3535085916519165,
          "early_lowcut": 169.90959841178443,
          "early_highcut": 6613.581831678016,
          "early_mix": 0.48319536447525024,
          "late_gain": 0.45873862504959106,
          "decay_time": 0.7356757884180969,
          "late_diffusion": 0.5762675404548645,
          "density": 0.48070383071899414,
          "mod_rate": 2.519737482070923,
          "mod_depth": 0.7363624572753906,
          "late_lowcut": 159.78410784452728,
          "late_highcut": 6119.591561949636,
          "late_width": 0.8062594532966614,
          "late_mix": 0.5276663899421692,
          "pre_delay": 0.04447237849235535,
          "damping": 0.6819272637367249,
          "lowcut": 42.072936534616126,
          "highcut": 15923.753869507424,
          "wet": 0.8420076966285706,
          "dry": 0.5098164081573486,
          "width": 0.49809208512306213,
          "mix": 0.6242713928222656
        },
        "dist": {
          "drive_db": 8.5
        },
        "eq": {
          "b1_freq": 5244.459181393208,
          "b1_gain": 16.476754188537598,
          "b1_q": 6.313068717957664,
          "b2_freq": 14213.152836156036,
          "b2_gain": -16.91985297203064,
          "b2_q": 1.0405014027928345,
          "b3_freq": 367.1327575519939,
          "b3_gain": 2.7490367889404297,
          "b3_q": 6.538061469418405,
          "b4_freq": 2543.3381537710425,
          "b4_gain": -1.0604267120361328,
          "b4_q": 1.0321032247785198,
          "b5_freq": 333.7124178627981,
          "b5_gain": 4.721872329711914,
          "b5_q": 2.862254929701999,
          "b6_freq": 127.01359291102676,
          "b6_gain": 19.01997470855713,
          "b6_q": 1.785704026133007
        }
      }
    },
    {
      "case": "C",
      "turn": 1,
      "prompt": "distorted and crushed",
      "reprompt": "distorted and crushed",
      "expected_fx": [
        "dist"
      ],
      "fx_chain": [
        "dist",
        "bitcrush"
      ],
      "params_path": "outputs/C/turn1/params.json",
      "before_audio_path": "outputs/C/turn1/before.wav",
      "after_audio_path": "outputs/C/turn1/after.wav",
      "params": {
        "dist": {
          "drive_db": 12.0
        },
        "bitcrush": {
          "bit_depth": 4
        }
      }
    }
  ]
};
