import eval.config as cfg
import os
from typing import List
import glob

def _dry_paths(instrument: str) -> List[str]:
    d = os.path.join(cfg.DRY_AUDIO_DIR, instrument)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".wav"))


def _seq_gt_paths(word_A: str, word_B: str, instrument: str) -> List[str]:
    pattern = os.path.join(cfg.GT_BANK_DIR_SEQUENTIAL, f"{word_A}_to_{word_B}", instrument, "*.wav")
    return sorted(glob.glob(pattern))


def _latest_experiment_dir(word_A: str, word_B: str, instrument: str):
    pair_inst_dir = os.path.join(cfg.SYSTEM_RESULTS_DIR, f"{word_A}_to_{word_B}", instrument)
    candidates = sorted(glob.glob(os.path.join(pair_inst_dir, "experiment_*")))
    print('[DEBUG] Looking for experiment dirs in:', pair_inst_dir, 'Found:', candidates)
    return candidates[-1] if candidates else None