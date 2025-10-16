"""Test fixtures for parameter validation tests.

Provides reusable test data for valid and invalid parameter cases.
"""

# Valid EQ Parameters
VALID_EQ_SINGLE_BAND = {
    "frequency": 1000.0,
    "gain": 3.0,
    "q": 1.0
}

VALID_EQ_MINIMUM_BANDS = {
    "bands": [
        {"frequency": 100.0, "gain": 2.0, "q": 0.7},
        {"frequency": 1000.0, "gain": -1.5, "q": 1.2},
        {"frequency": 8000.0, "gain": 3.0, "q": 0.9}
    ],
    "eq_type": "parametric"
}

VALID_EQ_MAXIMUM_BANDS = {
    "bands": [
        {"frequency": 20.0, "gain": 0.0, "q": 0.1},
        {"frequency": 100.0, "gain": 3.0, "q": 1.0},
        {"frequency": 250.0, "gain": -2.0, "q": 2.0},
        {"frequency": 500.0, "gain": 1.5, "q": 1.5},
        {"frequency": 1000.0, "gain": -3.0, "q": 0.8},
        {"frequency": 2000.0, "gain": 2.5, "q": 1.2},
        {"frequency": 4000.0, "gain": -1.0, "q": 0.9},
        {"frequency": 8000.0, "gain": 4.0, "q": 1.1},
        {"frequency": 12000.0, "gain": -2.5, "q": 0.7},
        {"frequency": 20000.0, "gain": 1.0, "q": 0.5}
    ],
    "eq_type": "parametric"
}

VALID_EQ_EDGE_CASES = {
    "bands": [
        {"frequency": 20.0, "gain": -12.0, "q": 0.1},      # Min frequency, min gain, min q
        {"frequency": 20000.0, "gain": 12.0, "q": 10.0},   # Max frequency, max gain, max q
        {"frequency": 1000.0, "gain": 0.0, "q": 1.0}       # Mid values
    ],
    "eq_type": "parametric"
}

# Invalid EQ Parameters
INVALID_EQ_TOO_FEW_BANDS = {
    "bands": [
        {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
        {"frequency": 2000.0, "gain": -2.0, "q": 0.8}
    ],
    "eq_type": "parametric"
}

INVALID_EQ_TOO_MANY_BANDS = {
    "bands": [
        {"frequency": 20.0 + i*2000, "gain": 1.0, "q": 1.0}
        for i in range(11)  # 11 bands (max is 10)
    ],
    "eq_type": "parametric"
}

INVALID_EQ_FREQUENCY_TOO_LOW = {
    "bands": [
        {"frequency": 10.0, "gain": 3.0, "q": 1.0},  # Below 20 Hz
        {"frequency": 1000.0, "gain": -2.0, "q": 0.8},
        {"frequency": 5000.0, "gain": 2.0, "q": 1.2}
    ],
    "eq_type": "parametric"
}

INVALID_EQ_FREQUENCY_TOO_HIGH = {
    "bands": [
        {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
        {"frequency": 5000.0, "gain": -2.0, "q": 0.8},
        {"frequency": 25000.0, "gain": 2.0, "q": 1.2}  # Above 20000 Hz
    ],
    "eq_type": "parametric"
}

INVALID_EQ_GAIN_TOO_LOW = {
    "bands": [
        {"frequency": 1000.0, "gain": -15.0, "q": 1.0},  # Below -12 dB
        {"frequency": 2000.0, "gain": 3.0, "q": 0.8},
        {"frequency": 4000.0, "gain": -2.0, "q": 1.2}
    ],
    "eq_type": "parametric"
}

INVALID_EQ_GAIN_TOO_HIGH = {
    "bands": [
        {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
        {"frequency": 2000.0, "gain": 15.0, "q": 0.8},  # Above 12 dB
        {"frequency": 4000.0, "gain": -2.0, "q": 1.2}
    ],
    "eq_type": "parametric"
}

INVALID_EQ_Q_TOO_LOW = {
    "bands": [
        {"frequency": 1000.0, "gain": 3.0, "q": 0.05},  # Below 0.1
        {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
        {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
    ],
    "eq_type": "parametric"
}

INVALID_EQ_Q_TOO_HIGH = {
    "bands": [
        {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
        {"frequency": 2000.0, "gain": -2.0, "q": 15.0},  # Above 10
        {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
    ],
    "eq_type": "parametric"
}

INVALID_EQ_MISSING_REQUIRED_FIELD = {
    "bands": [
        {"frequency": 1000.0, "gain": 3.0},  # Missing 'q'
        {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
        {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
    ],
    "eq_type": "parametric"
}

INVALID_EQ_WRONG_TYPE = {
    "bands": [
        {"frequency": "1000", "gain": 3.0, "q": 1.0},  # String instead of float
        {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
        {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
    ],
    "eq_type": "parametric"
}

# Valid Reverb Parameters
VALID_REVERB_MINIMAL = {
    "room_size": 0.5,
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": False
}

VALID_REVERB_EDGE_CASES = {
    "room_size": 0.0,     # Min
    "damping": 1.0,       # Max
    "wet_level": 0.0,     # Min
    "dry_level": 1.0,     # Max
    "width": 0.0,         # Min
    "freeze_mode": True
}

VALID_REVERB_ALL_MAX = {
    "room_size": 1.0,
    "damping": 1.0,
    "wet_level": 1.0,
    "dry_level": 1.0,
    "width": 1.0,
    "freeze_mode": True
}

VALID_REVERB_ALL_MIN = {
    "room_size": 0.0,
    "damping": 0.0,
    "wet_level": 0.0,
    "dry_level": 0.0,
    "width": 0.0,
    "freeze_mode": False
}

# Invalid Reverb Parameters
INVALID_REVERB_ROOM_SIZE_TOO_LOW = {
    "room_size": -0.1,    # Below 0
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_ROOM_SIZE_TOO_HIGH = {
    "room_size": 1.5,     # Above 1
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_DAMPING_TOO_LOW = {
    "room_size": 0.5,
    "damping": -0.1,      # Below 0
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_DAMPING_TOO_HIGH = {
    "room_size": 0.5,
    "damping": 1.5,       # Above 1
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_WET_LEVEL_TOO_LOW = {
    "room_size": 0.5,
    "damping": 0.5,
    "wet_level": -0.1,    # Below 0
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_WET_LEVEL_TOO_HIGH = {
    "room_size": 0.5,
    "damping": 0.5,
    "wet_level": 1.5,     # Above 1
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_DRY_LEVEL_TOO_LOW = {
    "room_size": 0.5,
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": -0.1,    # Below 0
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_DRY_LEVEL_TOO_HIGH = {
    "room_size": 0.5,
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": 1.5,     # Above 1
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_WIDTH_TOO_LOW = {
    "room_size": 0.5,
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": -0.1,        # Below 0
    "freeze_mode": False
}

INVALID_REVERB_WIDTH_TOO_HIGH = {
    "room_size": 0.5,
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": 1.5,         # Above 1
    "freeze_mode": False
}

INVALID_REVERB_MISSING_REQUIRED = {
    "room_size": 0.5,
    "damping": 0.5,
    # Missing wet_level, dry_level, width
    "freeze_mode": False
}

INVALID_REVERB_WRONG_TYPE = {
    "room_size": "0.5",   # String instead of float
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": False
}

INVALID_REVERB_FREEZE_WRONG_TYPE = {
    "room_size": 0.5,
    "damping": 0.5,
    "wet_level": 0.33,
    "dry_level": 0.67,
    "width": 1.0,
    "freeze_mode": "false"  # String instead of bool
}

# Valid Compressor Parameters
VALID_COMPRESSOR_MINIMAL = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 5.0,
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 3.0
}

VALID_COMPRESSOR_EDGE_CASES = {
    "threshold": -60.0,   # Min
    "ratio": 1.0,         # Min
    "attack": 0.1,        # Min
    "release": 10.0,      # Min
    "knee": 0.0,          # Min
    "makeup_gain": 0.0    # Min
}

VALID_COMPRESSOR_ALL_MAX = {
    "threshold": 0.0,
    "ratio": 20.0,
    "attack": 100.0,
    "release": 1000.0,
    "knee": 12.0,
    "makeup_gain": 24.0
}

VALID_COMPRESSOR_GENTLE = {
    "threshold": -10.0,
    "ratio": 2.0,
    "attack": 10.0,
    "release": 100.0,
    "knee": 3.0,
    "makeup_gain": 2.0
}

VALID_COMPRESSOR_AGGRESSIVE = {
    "threshold": -30.0,
    "ratio": 10.0,
    "attack": 1.0,
    "release": 20.0,
    "knee": 0.0,
    "makeup_gain": 12.0
}

# Invalid Compressor Parameters
INVALID_COMPRESSOR_THRESHOLD_TOO_LOW = {
    "threshold": -70.0,   # Below -60
    "ratio": 4.0,
    "attack": 5.0,
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_THRESHOLD_TOO_HIGH = {
    "threshold": 5.0,     # Above 0
    "ratio": 4.0,
    "attack": 5.0,
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_RATIO_TOO_LOW = {
    "threshold": -20.0,
    "ratio": 0.5,         # Below 1
    "attack": 5.0,
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_RATIO_TOO_HIGH = {
    "threshold": -20.0,
    "ratio": 25.0,        # Above 20
    "attack": 5.0,
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_ATTACK_TOO_LOW = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 0.05,       # Below 0.1
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_ATTACK_TOO_HIGH = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 150.0,      # Above 100
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_RELEASE_TOO_LOW = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 5.0,
    "release": 5.0,       # Below 10
    "knee": 6.0,
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_RELEASE_TOO_HIGH = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 5.0,
    "release": 1500.0,    # Above 1000
    "knee": 6.0,
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_KNEE_TOO_LOW = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 5.0,
    "release": 50.0,
    "knee": -1.0,         # Below 0
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_KNEE_TOO_HIGH = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 5.0,
    "release": 50.0,
    "knee": 15.0,         # Above 12
    "makeup_gain": 3.0
}

INVALID_COMPRESSOR_MAKEUP_GAIN_TOO_LOW = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 5.0,
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": -5.0   # Below 0
}

INVALID_COMPRESSOR_MAKEUP_GAIN_TOO_HIGH = {
    "threshold": -20.0,
    "ratio": 4.0,
    "attack": 5.0,
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 30.0   # Above 24
}

INVALID_COMPRESSOR_MISSING_REQUIRED = {
    "threshold": -20.0,
    "ratio": 4.0,
    # Missing attack, release, knee, makeup_gain
}

INVALID_COMPRESSOR_WRONG_TYPE = {
    "threshold": "-20",   # String instead of float
    "ratio": 4.0,
    "attack": 5.0,
    "release": 50.0,
    "knee": 6.0,
    "makeup_gain": 3.0
}

# Valid Effect Chains
VALID_EFFECT_CHAIN_SINGLE = {
    "description": "warm and intimate vocal sound",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 200.0, "gain": 3.0, "q": 0.7},
                {"frequency": 3000.0, "gain": -2.0, "q": 1.2},
                {"frequency": 8000.0, "gain": 1.0, "q": 0.9}
            ]
        }
    ],
    "order": ["eq"]
}

VALID_EFFECT_CHAIN_MULTIPLE = {
    "description": "bright and energetic guitar sound",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 5000.0, "gain": 4.0, "q": 1.0},
                {"frequency": 10000.0, "gain": 2.0, "q": 0.8},
                {"frequency": 1000.0, "gain": -1.0, "q": 1.2}
            ]
        },
        {
            "type": "compressor",
            "threshold": -20.0,
            "ratio": 4.0,
            "attack": 5.0,
            "release": 50.0,
            "knee": 3.0,
            "makeup_gain": 6.0
        }
    ],
    "order": ["eq", "compressor"]
}

VALID_EFFECT_CHAIN_ALL_EFFECTS = {
    "description": "full processing chain",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 100.0, "gain": 2.0, "q": 0.7},
                {"frequency": 1000.0, "gain": -1.0, "q": 1.2},
                {"frequency": 8000.0, "gain": 3.0, "q": 0.9}
            ]
        },
        {
            "type": "compressor",
            "threshold": -15.0,
            "ratio": 3.0,
            "attack": 10.0,
            "release": 100.0,
            "knee": 6.0,
            "makeup_gain": 4.0
        },
        {
            "type": "reverb",
            "room_size": 0.3,
            "damping": 0.7,
            "wet_level": 0.2,
            "dry_level": 0.8,
            "width": 1.0,
            "freeze_mode": False
        }
    ],
    "order": ["eq", "compressor", "reverb"]
}

# Invalid Effect Chains
INVALID_EFFECT_CHAIN_EMPTY_EFFECTS = {
    "description": "empty chain",
    "effects": [],
    "order": []
}

INVALID_EFFECT_CHAIN_MISSING_DESCRIPTION = {
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        }
    ],
    "order": ["eq"]
}

INVALID_EFFECT_CHAIN_MISSING_ORDER = {
    "description": "missing order",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        }
    ]
}

INVALID_EFFECT_CHAIN_MISMATCHED_ORDER = {
    "description": "order doesn't match effects",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        }
    ],
    "order": ["eq", "compressor"]  # compressor not in effects
}

INVALID_EFFECT_CHAIN_INVALID_EFFECT = {
    "description": "contains invalid effect parameters",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 1000.0, "gain": 50.0, "q": 1.0},  # Invalid gain (> 12)
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        }
    ],
    "order": ["eq"]
}
