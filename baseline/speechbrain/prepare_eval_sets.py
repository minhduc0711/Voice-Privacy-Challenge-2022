from pathlib import Path
from collections import defaultdict
import json
import shutil
import pandas as pd
from tqdm.auto import tqdm

enroll_trial_pairs_df = pd.read_csv("./enroll_trial_pairs.csv")
# print(enroll_trial_pairs_df.sort_values(by=["enroll_set"]))
# for row in enroll_trial_pairs_df.iterrows():
#     a, b = row[1]
#     print(a,b)
#     # print(row[1]["enroll_set"])
#     import sys
#     sys.exit(0)

enroll_dir_list = [
    Path("../data/") / subset_name
    for subset_name in enroll_trial_pairs_df["enroll_set"].unique()
]
trial_dir_list = [
    Path("../data/") / subset_name
    for subset_name in enroll_trial_pairs_df["trial_set"].unique()
]

results_dir = Path("dataset_metadata/")

# enroll_dir = Path("../data/libri_dev_enrolls")
# trial_dir = Path("../data/libri_dev_trials_f")
for enroll_dir in tqdm(enroll_dir_list, desc="Generating metadata for enroll sets"):
    utt_2_dur = {}
    utt_2_spkid = {}
    enroll_dict = defaultdict(lambda: defaultdict(lambda: {}))

    with open(enroll_dir / "utt2dur", 'r') as f:
        for line in f.readlines():
            items = line.split()
            utt_id = items[0]
            dur = items[1]
            utt_2_dur[utt_id] = float(dur)

    with open(enroll_dir / "utt2spk", 'r') as f:
        for line in f.readlines():
            items = line.split()
            utt_id = items[0]
            spk_id = items[1]
            utt_2_spkid[utt_id] = spk_id

    with open(enroll_dir / "wav.scp", 'r') as f:
        for line in f.readlines():
            # Scuffed extraction rules that differs between anon & orig data dir
            # 1. anon dir use full paths while orig dir use relative paths
            # 2. the paths are also on a different column if we intepret wav.scp as space-separated values
            path_col_idx = 1 if enroll_dir.name.find("anon") == -1 else 2
            items = line.split()
            utt_id = items[0]
            spk_id = utt_2_spkid[utt_id]
            audio_path = items[path_col_idx]

            if not audio_path.startswith("/"):
                audio_path = f"{{data_root}}/{audio_path}"

            enroll_dict[spk_id][utt_id]["wav"] = audio_path
            enroll_dict[spk_id][utt_id]["spk_id"] = spk_id
            enroll_dict[spk_id][utt_id]["length"] = utt_2_dur[utt_id]

    results_subdir = results_dir / enroll_dir.name
    if results_subdir.exists():
        shutil.rmtree(results_subdir)
    results_subdir.mkdir(exist_ok=True, parents=True)
    for spk_id in enroll_dict.keys():
        with open(results_subdir / f"{spk_id}.json", "w") as f:
            json.dump(enroll_dict[spk_id], f, indent=4)

for trial_dir in tqdm(trial_dir_list, desc="Generating metadata for trial sets"):
    trial_dict = defaultdict(lambda: {})
    with open(trial_dir / "wav.scp", 'r') as f:
        for line in f.readlines():
            # Scuffed extraction rules that differs between anon & orig data dir
            # 1. anon dir use full paths while orig dir use relative paths
            # 2. the paths are also on a different column if we intepret wav.scp as space-separated values
            path_col_idx = 1 if trial_dir.name.find("anon") == -1 else 2
            items = line.split()
            utt_id = items[0]

            audio_path = items[path_col_idx]
            if not audio_path.startswith("/"):
                audio_path = f"{{data_root}}/{audio_path}"

            trial_dict[utt_id]["wav"] = audio_path

    with open(trial_dir / "utt2spk", 'r') as f:
        for line in f.readlines():
            items = line.split()
            utt_id = items[0]
            spk_id = items[1]
            trial_dict[utt_id]["spk_id"] = spk_id

    with open(trial_dir / "utt2dur", 'r') as f:
        for line in f.readlines():
            items = line.split()
            utt_id = items[0]
            dur = items[1]
            trial_dict[utt_id]["length"] = float(dur)

    results_subdir = results_dir / trial_dir.name
    if results_subdir.exists():
        shutil.rmtree(results_subdir)
    results_subdir.mkdir(exist_ok=True, parents=True)
    with open(results_subdir / f"{trial_dir.name}.json", "w") as f:
        json.dump(trial_dict, f, indent=4)
    shutil.copy(trial_dir / "trials", results_subdir)
