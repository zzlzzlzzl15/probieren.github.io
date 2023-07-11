import os

import torch

from InferenceInterfaces.FastSpeech2Interface import InferenceFastSpeech2

all_models = [
"libri_all_clean_adapter_LJspeech",
"finetuning_LJspeech",
"BitFit",
"diff_pruning_LJspeech"
]
"""
all_models = [
"libri_all_clean_adapter_VCTK_252_200data",
"diff_pruning_vctk_p252_200data",
"BitFit_vctk_p252_200data",
"finetuning_vctk_p252_200data",
"libri_all_clean_adapter_VCTK_252_150data",
"BitFit_vctk_p252_150data",
"diff_pruning_vctk_p252_150data",
"finetuning_vctk_p252_150data",
"libri_all_clean_adapter_VCTK_252_100data",
"BitFit_vctk_p252_100data",
"diff_pruning_vctk_p252_100data",
"finetuning_vctk_p252_100data",
"libri_all_clean_adapter_VCTK_252_50data",
"BitFit_vctk_p252_50data",
"diff_pruning_vctk_p252_50data",
"finetuning_vctk_p252_50data",
"libri_all_clean_adapter_VCTK_254_200data",
"diff_pruning_vctk_p254_200data",
"BitFit_vctk_p254_200data",
"finetuning_vctk_p254_200data",
"libri_all_clean_adapter_VCTK_254_150data",
"BitFit_vctk_p254_150data",
"diff_pruning_vctk_p254_150data",
"finetuning_vctk_p254_150data",
"libri_all_clean_adapter_VCTK_254_100data",
"BitFit_vctk_p254_100data",
"diff_pruning_vctk_p254_100data",
"finetuning_vctk_p254_100data",
"libri_all_clean_adapter_VCTK_254_50data",
"BitFit_vctk_p254_50data",
"diff_pruning_vctk_p254_50data",
"finetuning_vctk_p254_50data",
"libri_all_clean_adapter_VCTK_230_200data",
"diff_pruning_vctk_p230_200data",
"BitFit_vctk_p230_200data",
"finetuning_vctk_p230_200data",
"libri_all_clean_adapter_VCTK_230_150data",
"BitFit_vctk_p230_150data",
"diff_pruning_vctk_p230_150data",
"finetuning_vctk_p230_150data",
"libri_all_clean_adapter_VCTK_230_100data",
"BitFit_vctk_p230_100data",
"diff_pruning_vctk_p230_100data",
"finetuning_vctk_p230_100data",
"libri_all_clean_adapter_VCTK_230_50data",
"BitFit_vctk_p230_50data",
"diff_pruning_vctk_p230_50data",
"finetuning_vctk_p230_50data"]
"""

def read_texts(model_id, sentence, filename, device="cpu", language="en", speaker_reference=None):
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def read_texts_as_ensemble(model_id, sentence, filename, device="cpu", language="en", amount=10):
    """
    for this function, the filename should NOT contain the .wav ending, it's added automatically
    """
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language(language)
    if type(sentence) == str:
        sentence = [sentence]
    for index in range(amount):
        tts.default_utterance_embedding = torch.zeros(704).float().random_(-40, 40).to(device)
        tts.read_to_file(text_list=sentence, file_location=filename + f"_{index}" + ".wav")


def read_harvard_sentences(model_id, device):
    tts = InferenceFastSpeech2(device=device, model_name=model_id)

    with open("Utility/test_sentences_combined_3.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_03_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

    with open("Utility/test_sentences_combined_6.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_06_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


def read_contrastive_focus_sentences(model_id, device):
    tts = InferenceFastSpeech2(device=device, model_name=model_id)

    with open("Utility/contrastive_focus_test_sentences.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/focus_{}".format(model_id)
    os.makedirs(output_dir, exist_ok=True)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("audios", exist_ok=True)
    for name in all_models:
        read_texts(model_id=name,
                   sentence="and though the famous family of Aldus restored its technical excellence, rejecting battered letters,",
                   filename="audios/LJspeech/FastSpeech2_"+name+"_0050.wav",
                   #speaker_reference="/mount/resources/speech/corpora/VCTK/wav48_silence_trimmed/p251/p251_023_mic2.flac",
                   device=exec_device,)
