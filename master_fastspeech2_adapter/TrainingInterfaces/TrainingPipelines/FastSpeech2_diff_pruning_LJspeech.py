import os
import time

import torch
import wandb
import random
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2_diff_pruning_train_loop import diff_pruning_fastspeeech2
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_ljspeech
from Utility.storage_config import MODELS_DIR, PREPROCESSING_DIR



def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "FastSpeech2_diff_pruning_LJspeech")  # KEEP THE 'FastSpeech2_' BUT CHANGE THE MODEL ID TO SOMETHING MEANINGFUL FOR YOUR DATA
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()
    datasets.append(prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_ljspeech(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "LJSpeech"),
                                              lang="en"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS


    train_set = ConcatDataset(datasets)
    model = FastSpeech2()
    train_loop_model = diff_pruning_fastspeeech2()


    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop_model.train_loop(
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               batch_size=8,  # YOU MIGHT GET OUT OF MEMORY ISSUES ON SMALL GPUs, IF SO, DECREASE THIS
               lang="en",  # CHANGE THIS TO THE LANGUAGE YOU'RE TRAINING ON
               lr=0.001,
               epochs_per_save=1,
               warmup_steps=10,
               # DOWNLOAD THIS INITIALIZATION MODELS FROM THE RELEASE PAGE OF THE GITHUB
               path_to_checkpoint=os.path.join(MODELS_DIR, "FastSpeech2_libri_all_clean_2", "best.pt"),
               path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
               fine_tune=True,
               resume=resume,
               phase_1_steps=50000,
               phase_2_steps=50000,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()
