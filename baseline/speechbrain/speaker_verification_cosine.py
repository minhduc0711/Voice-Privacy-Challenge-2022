#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
from pathlib import Path
import os
import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from tqdm.auto import tqdm
import pandas as pd


# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
        embeddings = params["mean_var_norm_emb"](
            embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        )
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        if len(data_loader) > 20:
            data_loader = tqdm(data_loader)
        for batch in data_loader:
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            emb = compute_embedding(wavs, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def get_verification_scores(veri_test):
    """ Computes positive and negative scores given the verification split.
    """
    f = open(veri_test, 'r')
    scores = []
    positive_scores = []
    negative_scores = []

    save_file = Path(params["output_folder"]) / trial_name / "scores.txt"
    save_file.parent.mkdir(exist_ok=True, parents=True)
    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # # creating cohort for score normalization
    if "score_norm" in params:
        train_cohort = torch.stack(list(train_dict.values()))
    
    for line in tqdm(list(f.readlines()), desc="Scoring each trial"):
        # Reading verification file (enrol_file test_file label)
        spk_id, trial_utt_id, label = line.strip().split(" ")

        enroll_vec = enroll_dict[spk_id]
        trial_vec = trial_dict[trial_utt_id]

        if "score_norm" in params:
            # Getting norm stats for enrol impostors
            enrol_rep = enroll_vec.repeat(train_cohort.shape[0], 1, 1)
            score_e_c = similarity(enrol_rep, train_cohort)
            if "cohort_size" in params:
                score_e_c = torch.topk(
                    score_e_c, k=params["cohort_size"], dim=0
                )[0]
            mean_e_c = torch.mean(score_e_c, dim=0)
            std_e_c = torch.std(score_e_c, dim=0)
            # Getting norm stats for test impostors
            test_rep = trial_vec.repeat(train_cohort.shape[0], 1, 1)
            score_t_c = similarity(test_rep, train_cohort)
            if "cohort_size" in params:
                score_t_c = torch.topk(
                    score_t_c, k=params["cohort_size"], dim=0
                )[0]
            mean_t_c = torch.mean(score_t_c, dim=0)
            std_t_c = torch.std(score_t_c, dim=0)

        # Compute the score for the given sentence
        score = similarity(enroll_vec, trial_vec)[0]

        # Perform score normalization
        if "score_norm" in params:
            if params["score_norm"] == "z-norm":
                score = (score - mean_e_c) / std_e_c
            elif params["score_norm"] == "t-norm":
                score = (score - mean_t_c) / std_t_c
            elif params["score_norm"] == "s-norm":
                score_e = (score - mean_e_c) / std_e_c
                score_t = (score - mean_t_c) / std_t_c
                score = 0.5 * (score_e + score_t)

        # write score file
        s_file.write(f"{spk_id} {trial_utt_id} {label} {float(score)}\n")
        scores.append(score)

        if label == "target":
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    f.close()
    s_file.close()
    return positive_scores, negative_scores


def dataio_prep(json_paths, hparams, is_train=False):
    "Creates the dataloaders and their data processing pipelines."
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    dataloaders = {}
    if isinstance(json_paths, Path) or isinstance(json_paths, str):
        json_paths = [json_paths]
    for json_path in tqdm(json_paths):
        json_path = Path(json_path)
        ds = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": hparams["data_folder"]},
            # dynamic_items=[audio_pipeline],
            # output_keys=["id", "sig"],
        )
        if is_train:
            ds = ds.filtered_sorted(
                sort_key="length", select_n=params["n_train_snts"]
            )
        sb.dataio.dataset.add_dynamic_item([ds], audio_pipeline)
        sb.dataio.dataset.set_output_keys([ds], ["id", "sig"])

        dataloaders[json_path.stem] = sb.dataio.dataloader.make_dataloader(
            ds, **params["enrol_dataloader_opts"])

    if len(dataloaders) == 1:
        return dataloaders[json_path.stem]
    else:
        return dataloaders


if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    eer_file = open(Path(params["output_folder"]) / "EER.txt", "w")
    enroll_trial_pairs_df = pd.read_csv("./enroll_trial_pairs.csv")
    enroll_trial_pairs_df.sort_values(by=["enroll_set"], inplace=True)
    prev_enroll_set = None
    for row in enroll_trial_pairs_df.iterrows():
        # here we create the datasets objects as well as tokenization and encoding
        enroll_set, trial_set = row[1]
        trial_name = f"ASV-{enroll_set}-{trial_set}"
        print("============================")
        print(f"Evaluating trial {trial_name}")

        if enroll_set != prev_enroll_set:
            enroll_dir = Path("dataset_metadata/") / enroll_set
            enroll_loaders = dataio_prep(enroll_dir.iterdir(), params)

            logger.info("Computing enroll embeddings...")
            enroll_dict = {}
            for spk_id, loader in tqdm(enroll_loaders.items(), 
                                       desc="Computing mean embeddings for enroll utts"):
                embeddings = torch.stack(list(compute_embedding_loop(loader).values()), dim=0)
                enroll_dict[spk_id] = torch.mean(embeddings, dim=0)

            prev_enroll_set = enroll_set

        trial_loader = dataio_prep(f"dataset_metadata/{trial_set}/{trial_set}.json", params)
        logger.info("Computing trial embeddings...")
        trial_dict = compute_embedding_loop(trial_loader)

        if "score_norm" in params:
            train_loader = dataio_prep("train.json", params, is_train=True)
            logger.info("Computing train embeddings...")
            train_dict = compute_embedding_loop(train_loader)

        positive_scores, negative_scores = get_verification_scores(f"dataset_metadata/{trial_set}/trials")
        # del enrol_dict, test_dict

        eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
        print(f"EER: {eer * 100:.2f}")
        eer_file.write(f"{trial_name} {eer * 100:.2f}\n")
    eer_file.close()

    # # First run
    # enrol_dict = compute_embedding_loop(enrol_dataloader)
    # test_dict = compute_embedding_loop(test_dataloader)

    # # Second run (normalization stats are more stable)
    # enrol_dict = compute_embedding_loop(enrol_dataloader)
    # test_dict = compute_embedding_loop(test_dataloader)

    # if "score_norm" in params:
    #     train_dict = compute_embedding_loop(train_dataloader)

    # # Compute the EER
    # logger.info("Computing EER..")
    # # Reading standard verification split
    # with open(veri_file_path) as f:
    #     veri_test = [line.rstrip() for line in f]

    # positive_scores, negative_scores = get_verification_scores(veri_test)
    # del enrol_dict, test_dict

    # eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    # logger.info("EER(%%)=%f", eer * 100)

    # min_dcf, th = minDCF(
    #     torch.tensor(positive_scores), torch.tensor(negative_scores)
    # )
    # logger.info("minDCF=%f", min_dcf * 100)
