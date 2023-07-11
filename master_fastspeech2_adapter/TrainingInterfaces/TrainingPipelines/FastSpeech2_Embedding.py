"""
This is the setup with which the embedding model is trained. After the embedding model has been trained, it is only used in a frozen state.
"""

import os
import time
import random
import torch
import wandb
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Spectrogram_to_Embedding.embedding_function_train_loop import train_loop
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_libritts_all_clean
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
        save_dir = os.path.join(MODELS_DIR, "FastSpeech2_libri_all_clean_2")
    os.makedirs(save_dir, exist_ok=True)

    datasets = list()


    train_set = prepare_fastspeech_corpus(transcript_dict=build_path_to_transcript_dict_libritts_all_clean(),
                                          corpus_dir=os.path.join(PREPROCESSING_DIR, "libri"),
                                          lang="en",
                                          save_imgs=True)





    model = FastSpeech2()
    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)
    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               batch_size=8,
               lang="en",
               lr=0.001,
               epochs_per_save=1,
               warmup_steps=10,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume,
               use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()
