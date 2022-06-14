from pathlib import Path
from collections import defaultdict
import json
import random

data_dir = Path("../data/train-clean-360-asv")
manifest_dict = defaultdict(lambda: {})
full_json = "train_full.json"
train_json = "train.json"
val_json = "val.json"
val_ratio = 0.1
train_ratio = 1 - val_ratio

with open(data_dir / "wav.scp", 'r') as f:
    for line in f.readlines():
        items = line.split()
        utt_id = items[0]
        audio_path = items[5]
        manifest_dict[utt_id]["wav"] = f"{{data_root}}/{audio_path}"

with open(data_dir / "utt2dur", 'r') as f:
    for line in f.readlines():
        items = line.split()
        utt_id = items[0]
        dur = items[1]
        manifest_dict[utt_id]["length"] = float(dur)

with open(data_dir / "utt2spk", 'r') as f:
    for line in f.readlines():
        items = line.split()
        utt_id = items[0]
        spk_id = items[1]
        manifest_dict[utt_id]["spk_id"] = spk_id

utt_ids = list(manifest_dict.keys())
random.shuffle(utt_ids)
num_trains = int(train_ratio * len(utt_ids))
train_dict = {}
val_dict = {}
for utt_id in utt_ids[:num_trains]:
    train_dict[utt_id] = manifest_dict[utt_id]
for utt_id in utt_ids[num_trains:]:
    val_dict[utt_id] = manifest_dict[utt_id]

with open(train_json, "w") as f:
    json.dump(train_dict, f, indent=4)
with open(val_json, "w") as f:
    json.dump(val_dict, f, indent=4)
with open(full_json, "w") as f:
    json.dump(manifest_dict, f, indent=4)
