import argparse
import sys

from TrainingInterfaces.Spectrogram_to_Embedding.finetune_embeddings_to_tasks import finetune_model_emotion
from TrainingInterfaces.Spectrogram_to_Embedding.finetune_embeddings_to_tasks import finetune_model_speaker
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Controllable import run as control
from TrainingInterfaces.TrainingPipelines.FastSpeech2_Embedding import run as gst
from TrainingInterfaces.TrainingPipelines.FastSpeech2_IntegrationTest import run as integration_test
from TrainingInterfaces.TrainingPipelines.FastSpeech2_MetaCheckpoint import run as meta_fast
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_example import run as fine_ger
from TrainingInterfaces.TrainingPipelines.HiFiGAN_Avocodo import run as hifigan_combined
from TrainingInterfaces.TrainingPipelines.HiFiGAN_Avocodo_low_RAM import run as hifigan_combined_low_ram
from TrainingInterfaces.TrainingPipelines.pretrain_aligner import run as aligner
from TrainingInterfaces.TrainingPipelines.FastSpeech2_libri_all_clean import run as libri
from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_LJspeech import run as LJspeech_adapter
from TrainingInterfaces.TrainingPipelines.FastSpeech2_LJspeech import run as LJSpeech
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit import run as BitFit
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_LJspeech import run as finetuning_LJspeech
from TrainingInterfaces.TrainingPipelines.FastSpeech2_pruning_LJspeech import run as pruning_LJspeech
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_LJspeech import run as diff_pruning_LJspeech
from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p226 import run as vctk_p226_adapter
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p226 import run as BitFit_vctk_p226
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p226 import run as diff_pruning_vctk_p226
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p226 import run as finetuning_vctk_p226
from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p251 import run as adapter_finetuning_vctk_p251
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p251 import run as finetuning_vctk_p251
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p251 import run as BitFit_vctk_p251
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p251 import run as diff_pruning_vctk_p251

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p252_200data import run as adapter_finetuning_vctk_p252_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p252_200data import run as diff_pruning_vctk_p252_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p252_200data import run as BitFit_vctk_p252_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p252_200data import run as finetuning_vctk_p252_200data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p252_100data import run as adapter_finetuning_vctk_p252_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p252_100data import run as BitFit_vctk_p252_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p252_100data import run as diff_pruning_vctk_p252_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p252_100data import run as finetuning_vctk_p252_100data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p252_150data import run as adapter_finetuning_vctk_p252_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p252_150data import run as BitFit_vctk_p252_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p252_150data import run as diff_pruning_vctk_p252_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p252_150data import run as finetuning_vctk_p252_150data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p252_50data import run as adapter_finetuning_vctk_p252_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p252_50data import run as BitFit_vctk_p252_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p252_50data import run as diff_pruning_vctk_p252_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p252_50data import run as finetuning_vctk_p252_50data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p252_35data import run as adapter_finetuning_vctk_p252_35data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p252_35data import run as BitFit_vctk_p252_35data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p252_35data import run as diff_pruning_vctk_p252_35data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p252_35data import run as finetuning_vctk_p252_35data

################################################################################################################################################
from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p254_200data import run as adapter_finetuning_vctk_p254_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p254_200data import run as diff_pruning_vctk_p254_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p254_200data import run as BitFit_vctk_p254_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p254_200data import run as finetuning_vctk_p254_200data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p254_150data import run as adapter_finetuning_vctk_p254_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p254_150data import run as BitFit_vctk_p254_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p254_150data import run as diff_pruning_vctk_p254_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p254_150data import run as finetuning_vctk_p254_150data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p254_100data import run as adapter_finetuning_vctk_p254_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p254_100data import run as BitFit_vctk_p254_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p254_100data import run as diff_pruning_vctk_p254_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p254_100data import run as finetuning_vctk_p254_100data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p254_50data import run as adapter_finetuning_vctk_p254_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p254_50data import run as BitFit_vctk_p254_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p254_50data import run as diff_pruning_vctk_p254_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p254_50data import run as finetuning_vctk_p254_50data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p254_35data import run as adapter_finetuning_vctk_p254_35data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p254_35data import run as BitFit_vctk_p254_35data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p254_35data import run as diff_pruning_vctk_p254_35data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p254_35data import run as finetuning_vctk_p254_35data
###################################################################################################################################################

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p230_200data import run as adapter_finetuning_vctk_p230_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p230_200data import run as diff_pruning_vctk_p230_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p230_200data import run as BitFit_vctk_p230_200data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p230_200data import run as finetuning_vctk_p230_200data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p230_150data import run as adapter_finetuning_vctk_p230_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p230_150data import run as BitFit_vctk_p230_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p230_150data import run as diff_pruning_vctk_p230_150data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p230_150data import run as finetuning_vctk_p230_150data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p230_100data import run as adapter_finetuning_vctk_p230_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p230_100data import run as BitFit_vctk_p230_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p230_100data import run as diff_pruning_vctk_p230_100data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p230_100data import run as finetuning_vctk_p230_100data

from TrainingInterfaces.TrainingPipelines.FastSpeech2_adapter_finetuning_vctk_p230_50data import run as adapter_finetuning_vctk_p230_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_BitFit_vctk_p230_50data import run as BitFit_vctk_p230_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_diff_pruning_vctk_p230_50data import run as diff_pruning_vctk_p230_50data
from TrainingInterfaces.TrainingPipelines.FastSpeech2_finetuning_vctk_p230_50data import run as finetuning_vctk_p230_50data



pipeline_dict = {
    "meta":             meta_fast,
    "hificodo":         hifigan_combined,
    "aligner":          aligner,
    "fine_ger":         fine_ger,
    "integration_test": integration_test,
    "gst":              gst,
    "spk":              finetune_model_speaker,
    "emo":              finetune_model_emotion,
    "control":          control,
    "low_ram_avocodo":  hifigan_combined_low_ram,
    "libri":            libri,
    "LJspeech_adapter": LJspeech_adapter,
    "LJSpeech":         LJSpeech,
    "BitFit":           BitFit,
    "finetuning_LJspeech": finetuning_LJspeech,
    "pruning_LJspeech": pruning_LJspeech,
    "diff_pruning_LJspeech": diff_pruning_LJspeech,
    "vctk_p226_adapter": vctk_p226_adapter,
    "BitFit_vctk_p226" : BitFit_vctk_p226,
    "diff_pruning_vctk_p226" : diff_pruning_vctk_p226,
    "finetuning_vctk_p226" : finetuning_vctk_p226,
    "adapter_finetuning_vctk_p251": adapter_finetuning_vctk_p251,
    "finetuning_vctk_p251": finetuning_vctk_p251,
    "BitFit_vctk_p251": BitFit_vctk_p251,
    "diff_pruning_vctk_p251": diff_pruning_vctk_p251,
    "adapter_finetuning_vctk_p252_200data": adapter_finetuning_vctk_p252_200data,
    "diff_pruning_vctk_p252_200data": diff_pruning_vctk_p252_200data,
    "BitFit_vctk_p252_200data": BitFit_vctk_p252_200data,
    "finetuning_vctk_p252_200data": finetuning_vctk_p252_200data,
    "adapter_finetuning_vctk_p252_150data": adapter_finetuning_vctk_p252_150data,
    "BitFit_vctk_p252_150data": BitFit_vctk_p252_150data,
    "diff_pruning_vctk_p252_150data": diff_pruning_vctk_p252_150data,
    "finetuning_vctk_p252_150data": finetuning_vctk_p252_150data,
    "adapter_finetuning_vctk_p252_100data": adapter_finetuning_vctk_p252_100data,
    "BitFit_vctk_p252_100data": BitFit_vctk_p252_100data,
    "diff_pruning_vctk_p252_100data": diff_pruning_vctk_p252_100data,
    "finetuning_vctk_p252_100data": finetuning_vctk_p252_100data,
    "adapter_finetuning_vctk_p252_50data": adapter_finetuning_vctk_p252_50data,
    "BitFit_vctk_p252_50data": BitFit_vctk_p252_50data,
    "diff_pruning_vctk_p252_50data": diff_pruning_vctk_p252_50data,
    "finetuning_vctk_p252_50data": finetuning_vctk_p252_50data,
    "adapter_finetuning_vctk_p252_35data": adapter_finetuning_vctk_p252_35data,
    "BitFit_vctk_p252_35data": BitFit_vctk_p252_35data,
    "diff_pruning_vctk_p252_35data": diff_pruning_vctk_p252_35data,
    "finetuning_vctk_p252_35data": finetuning_vctk_p252_35data,
    "adapter_finetuning_vctk_p254_200data": adapter_finetuning_vctk_p254_200data,
    "diff_pruning_vctk_p254_200data": diff_pruning_vctk_p254_200data,
    "BitFit_vctk_p254_200data": BitFit_vctk_p254_200data,
    "finetuning_vctk_p254_200data": finetuning_vctk_p254_200data,
    "adapter_finetuning_vctk_p254_150data": adapter_finetuning_vctk_p254_150data,
    "BitFit_vctk_p254_150data": BitFit_vctk_p254_150data,
    "diff_pruning_vctk_p254_150data": diff_pruning_vctk_p254_150data,
    "finetuning_vctk_p254_150data": finetuning_vctk_p254_150data,
    "adapter_finetuning_vctk_p254_100data": adapter_finetuning_vctk_p254_100data,
    "BitFit_vctk_p254_100data": BitFit_vctk_p254_100data,
    "diff_pruning_vctk_p254_100data": diff_pruning_vctk_p254_100data,
    "finetuning_vctk_p254_100data": finetuning_vctk_p254_100data,
    "adapter_finetuning_vctk_p254_50data": adapter_finetuning_vctk_p254_50data,
    "BitFit_vctk_p254_50data": BitFit_vctk_p254_50data,
    "diff_pruning_vctk_p254_50data": diff_pruning_vctk_p254_50data,
    "finetuning_vctk_p254_50data": finetuning_vctk_p254_50data,
    "adapter_finetuning_vctk_p254_35data": adapter_finetuning_vctk_p254_35data,
    "BitFit_vctk_p254_35data": BitFit_vctk_p254_35data,
    "diff_pruning_vctk_p254_35data": diff_pruning_vctk_p254_35data,
    "finetuning_vctk_p254_35data": finetuning_vctk_p252_35data,
    "adapter_finetuning_vctk_p230_200data": adapter_finetuning_vctk_p230_200data,
    "diff_pruning_vctk_p230_200data": diff_pruning_vctk_p230_200data,
    "BitFit_vctk_p230_200data": BitFit_vctk_p230_200data,
    "finetuning_vctk_p230_200data": finetuning_vctk_p230_200data,
    "adapter_finetuning_vctk_p230_150data": adapter_finetuning_vctk_p230_150data,
    "BitFit_vctk_p230_150data": BitFit_vctk_p230_150data,
    "diff_pruning_vctk_p230_150data": diff_pruning_vctk_p230_150data,
    "finetuning_vctk_p230_150data": finetuning_vctk_p230_150data,
    "adapter_finetuning_vctk_p230_100data": adapter_finetuning_vctk_p230_100data,
    "BitFit_vctk_p230_100data": BitFit_vctk_p230_100data,
    "diff_pruning_vctk_p230_100data": diff_pruning_vctk_p230_100data,
    "finetuning_vctk_p230_100data": finetuning_vctk_p230_100data,
    "adapter_finetuning_vctk_p230_50data": adapter_finetuning_vctk_p230_50data,
    "BitFit_vctk_p230_50data": BitFit_vctk_p230_50data,
    "diff_pruning_vctk_p230_50data": diff_pruning_vctk_p230_50data,
    "finetuning_vctk_p230_50data": finetuning_vctk_p230_50data,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IMS Speech Synthesis Toolkit - Call to Train')

    parser.add_argument('pipeline',
                        choices=list(pipeline_dict.keys()),
                        help="Select pipeline to train.")

    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU to run on. If not specified runs on CPU, but other than for integration tests that doesn't make much sense.",
                        default="cpu")

    parser.add_argument('--resume_checkpoint',
                        type=str,
                        help="Path to checkpoint to resume from.",
                        default=None)

    parser.add_argument('--resume',
                        action="store_true",
                        help="Automatically load the highest checkpoint and continue from there.",
                        default=False)

    parser.add_argument('--finetune',
                        action="store_true",
                        help="Whether to fine-tune from the specified checkpoint.",
                        default=False)

    parser.add_argument('--model_save_dir',
                        type=str,
                        help="Directory where the checkpoints should be saved to.",
                        default=None)

    parser.add_argument('--wandb',
                        action="store_true",
                        help="Whether to use weigths and biases to track training runs. Requires you to run wandb login and place your auth key before.",
                        default=False)

    parser.add_argument('--wandb_resume_id',
                        type=str,
                        help="ID of a stopped wandb run to continue tracking",
                        default=None)

    args = parser.parse_args()

    if args.finetune and args.resume_checkpoint is None:
        print("Need to provide path to checkpoint to fine-tune from!")
        sys.exit()

    pipeline_dict[args.pipeline](gpu_id=args.gpu_id,
                                 resume_checkpoint=args.resume_checkpoint,
                                 resume=args.resume,
                                 finetune=args.finetune,
                                 model_dir=args.model_save_dir,
                                 use_wandb=args.wandb,
                                 wandb_resume_id=args.wandb_resume_id)
