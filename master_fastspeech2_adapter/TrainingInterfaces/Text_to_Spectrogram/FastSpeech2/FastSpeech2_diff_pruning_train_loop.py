import os
import time
import math
import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torch.multiprocessing
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from enum import Enum, auto
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2

from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
from Utility.WarmupScheduler import WarmupScheduler
from Utility.utils import cumsum_durations
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint

from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize

from typing import Union, Callable, List, Dict, Tuple, Optional
from Layers.diff_param import DiffWeight, DiffWeightFixmask
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class ModelState(Enum):
    FINETUNING = auto()
    DIFFPRUNING = auto()
    FIXMASK = auto()

class diff_pruning_fastspeeech2():
    def __init__(self):
        self.net = FastSpeech2()
        self.structured_diff_pruning = True
        self.alpha_init = 5
        self.concrete_lower = -1.5
        self.concrete_upper = 1.5
        self.gradient_accumulation_steps = 1
        self.diff_pruning = True
        self.num_epochs_finetune = 1
        self.num_epochs_fixmask = 1
        self.weight_decay = 0.0
        self.learning_rate = 5e-5
        self.learning_rate_alpha = 0.1
        self.adam_epsilon = 1e-8
        self.warmup_steps = 4000
        self.sparsity_pen = 1.25e-5
        self.max_grad_norm = 1.0
        self.fixmask_pct = 0.1
        self.logging_step = 5
        self._model_state = ModelState.FINETUNING
        self.model_pre = self.net.encoder

    @property
    def device(self) -> torch.device:
        return next(self.net.encoder.parameters()).device

    @property
    def model_type(self) -> str:
        return self.net.encoder.config.model_type

    @property
    def model_name(self) -> str:
        return self.net.encoder.config._name_or_path

    def total_layers(self) -> int:
        num_layer = 0
        for k in self.net.encoder.named_modules():
            num_layer += 1
        return num_layer

    def _parametrized(self) -> bool:
        return (self._model_state == ModelState.DIFFPRUNING or self._model_state == ModelState.FIXMASK)

    @staticmethod
    # TODO log ratio could be removed if only symmetric concrete distributions are possible
    def get_log_ratio(concrete_lower: float, concrete_upper: float) -> int:
        # calculate regularization term in objective
        return 0 if (concrete_lower == 0) else math.log(-concrete_lower / concrete_upper)

    def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
        return torch.sigmoid(alpha - log_ratio).sum()

    def get_encoder_base_modules(self, return_names: bool = False):
        if self._parametrized:
            check_fn = lambda m: hasattr(m, "parametrizations")
        else:
            check_fn = lambda m: len(m._parameters) > 0
        return [(n, m) if return_names else m for n, m in self.net.encoder.named_modules()]

    """
    def get_layer_idx_from_module(self, n: str) -> int:
        # get layer index based on module name
        num_layer = 0
        for k in self.net.encoder.named_modules():
            num_layer += 1
        return num_layer
    """






    @torch.no_grad()
    def plot_progress_spec(self, net, device, save_dir, step, lang, default_emb):
        tf = ArticulatoryCombinedTextFrontend(language=lang)
        sentence = ""
        if lang == "en":
            sentence = "This is a complex sentence, it even has a pause!"
        elif lang == "de":
            sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
        elif lang == "el":
            sentence = "Αυτή είναι μια σύνθετη πρόταση, έχει ακόμη και παύση!"
        elif lang == "es":
            sentence = "Esta es una oración compleja, ¡incluso tiene una pausa!"
        elif lang == "fi":
            sentence = "Tämä on monimutkainen lause, sillä on jopa tauko!"
        elif lang == "ru":
            sentence = "Это сложное предложение, в нем даже есть пауза!"
        elif lang == "hu":
            sentence = "Ez egy összetett mondat, még szünet is van benne!"
        elif lang == "nl":
            sentence = "Dit is een complexe zin, er zit zelfs een pauze in!"
        elif lang == "fr":
            sentence = "C'est une phrase complexe, elle a même une pause !"
        elif lang == "pt":
            sentence = "Esta é uma frase complexa, tem até uma pausa!"
        elif lang == "pl":
            sentence = "To jest zdanie złożone, ma nawet pauzę!"
        elif lang == "it":
            sentence = "Questa è una frase complessa, ha anche una pausa!"
        elif lang == "cmn":
            sentence = "这是一个复杂的句子，它甚至包含一个停顿。"
        elif lang == "vi":
            sentence = "Đây là một câu phức tạp, nó thậm chí còn chứa một khoảng dừng."
        phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
        spec, durations, pitch, energy = net.inference(text=phoneme_vector,
                                                       return_duration_pitch_energy=True,
                                                       utterance_embedding=default_emb,
                                                       lang_id=get_language_id(lang).to(device))
        spec = spec.transpose(0, 1).to("cpu").numpy()
        duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
        if not os.path.exists(os.path.join(save_dir, "spec")):
            os.makedirs(os.path.join(save_dir, "spec"))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        lbd.specshow(spec,
                     ax=ax,
                     sr=16000,
                     cmap='GnBu',
                     y_axis='mel',
                     x_axis=None,
                     hop_length=256)
        ax.yaxis.set_visible(False)
        ax.set_xticks(duration_splits, minor=True)
        ax.xaxis.grid(True, which='minor')
        ax.set_xticks(label_positions, minor=False)
        phones = tf.get_phone_string(sentence, for_plot_labels=True)
        ax.set_xticklabels(phones)
        word_boundaries = list()
        for label_index, word_boundary in enumerate(phones):
            if word_boundary == "|":
                word_boundaries.append(label_positions[label_index])
        ax.vlines(x=duration_splits, colors="green", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
        ax.vlines(x=word_boundaries, colors="orange", linestyles="dotted", ymin=0.0, ymax=8000, linewidth=1.0)
        pitch_array = pitch.cpu().numpy()
        for pitch_index, xrange in enumerate(zip(duration_splits[:-1], duration_splits[1:])):
            if pitch_array[pitch_index] > 0.001:
                ax.hlines(pitch_array[pitch_index] * 1000, xmin=xrange[0], xmax=xrange[1], color="blue", linestyles="solid",
                          linewidth=0.5)
        ax.set_title(sentence)
        plt.savefig(os.path.join(os.path.join(save_dir, "spec"), str(step) + ".png"))
        plt.clf()
        plt.close()
        return os.path.join(os.path.join(save_dir, "spec"), str(step) + ".png")


    def collate_and_pad(self, batch):
        # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id
        return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
                torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
                pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
                torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
                pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
                pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
                pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
                None,
                torch.stack([datapoint[8] for datapoint in batch]))


    def train_loop(self,
                   train_dataset,
                   device,
                   save_directory,
                   batch_size=32,
                   epochs_per_save=1,
                   lang="en",
                   lr=0.0001,
                   warmup_steps=4000,
                   path_to_checkpoint=None,
                   path_to_embed_model="Models/Embedding/embedding_function.pt",
                   fine_tune= False,
                   resume=False,
                   phase_1_steps=100000,
                   phase_2_steps=100000,
                   use_wandb=False):
        """
        Args:
            resume: whether to resume from the most recent checkpoint
            warmup_steps: how long the learning rate should increase before it reaches the specified value
            lr: The initial learning rate for the optimiser
            path_to_checkpoint: reloads a checkpoint to continue training from there
            fine_tune: whether to load everything from a checkpoint, or only the model parameters
            lang: language of the synthesis
            net: Model to train
            train_dataset: Pytorch Dataset Object for train data
            device: Device to put the loaded tensors on
            save_directory: Where to save the checkpoints
            batch_size: How many elements should be loaded at once
            epochs_per_save: how many epochs to train in between checkpoints
            phase_1_steps: how many steps to train before using any of the cycle objectives
            phase_2_steps: how many steps to train using the cycle objectives
            path_to_embed_model: path to the pretrained embedding function
        """


        """
        Args of diff_pruning
        """





        fine_tune = not self.diff_pruning

        style_embedding_function = StyleEmbedding().to(device)
        check_dict = torch.load(path_to_embed_model, map_location=device)
        style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        style_embedding_function.eval()
        style_embedding_function.requires_grad_(False)

        torch.multiprocessing.set_sharing_strategy('file_system')
        train_loader = DataLoader(batch_size=batch_size,
                                  dataset=train_dataset,
                                  drop_last=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  shuffle=True,
                                  prefetch_factor=8,
                                  collate_fn=self.collate_and_pad,
                                  persistent_workers=True)

        """calculate the num_epochs adn num_epochs_fixmask"""
        """num_epochs_total = self.num_epochs_finetune + self.num_epochs_fixmask

        steps = phase_1_steps * num_epochs_total + phase_2_steps * num_epochs_total
        train_steps_finetune_phase_1 = len(train_loader) // self.gradient_accumulation_steps * self.num_epochs_finetune
        train_steps_fixmask_phase_1 =
        train_steps_finetune_phase_2 =
        train_steps_fixmask_phase_2 =
        """
        steps = phase_1_steps+ phase_2_steps
        log_ratio = self.get_log_ratio(self.concrete_lower, self.concrete_upper)

        if self.diff_pruning:

            #self._init_sparsity_pen(self.sparsity_pen)
            self._add_diff_parametrizations(
                self.alpha_init,
                self.concrete_lower,
                self.concrete_upper,
                self.structured_diff_pruning
            )
            '''
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.items()):
                print(n)
            '''
        self._init_optimizer_and_schedule(
            phase_1_steps,
            self.learning_rate,
            self.weight_decay,
            self.adam_epsilon,
            warmup_steps,
            self.learning_rate_alpha,
        )



        step_counter = 0
        loss_fn = lambda x, y: torch.nn.BCEWithLogitsLoss()(x.flatten(), y)



        scaler = GradScaler()
        epoch = 0
        if resume:
            path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
        if path_to_checkpoint is not None:
            check_dict = torch.load(path_to_checkpoint, map_location=device)
            self.net.load_state_dict(check_dict["model"])
            '''
            if not fine_tune:
                self.optimizer[0].load_state_dict(check_dict["optimizer"])
                self.scheduler.load_state_dict(check_dict["scheduler"])
                step_counter = check_dict["step_counter"]
                scaler.load_state_dict(check_dict["scaler"])
            '''
        self.net.to(device)
        self.model_pre = self.net.to(device)
        start_time = time.time()
        while True:
            self.net.train()
            epoch += 1
            self.optimizer.zero_grad()
            train_losses_this_epoch = list()
            cycle_losses_this_epoch = list()






            for batch in tqdm(train_loader):
                with autocast():
                    if step_counter == phase_1_steps:
                        self._finetune_to_fixmask(self.fixmask_pct)
                        self._init_optimizer_and_schedule(
                            phase_1_steps + phase_2_steps,
                            self.learning_rate,
                            self.weight_decay,
                            self.adam_epsilon,
                            warmup_steps
                        )
                    if step_counter <= phase_1_steps:



                        # ===============================================
                        # =        PHASE 1: no cycle objective          =
                        # ===============================================
                        style_embedding = style_embedding_function(batch_of_spectrograms=batch[2].to(device),
                                                                   batch_of_spectrogram_lengths=batch[3].to(device))

                        train_loss = self.net(text_tensors=batch[0].to(device),
                                         text_lengths=batch[1].to(device),
                                         gold_speech=batch[2].to(device),
                                         speech_lengths=batch[3].to(device),
                                         gold_durations=batch[4].to(device),
                                         gold_pitch=batch[6].to(device),  # mind the switched order
                                         gold_energy=batch[5].to(device),  # mind the switched order
                                         utterance_embedding=style_embedding,
                                         lang_ids=batch[8].to(device),
                                         return_mels=False)

                        train_loss_real = train_loss
                        if self._model_state == ModelState.DIFFPRUNING:
                            l0_pen = 0.
                            for module_name, base_module in self.get_encoder_base_modules(return_names=True):
                                #layer_idx = self.get_layer_idx_from_module(module_name)
                                sparsity_pen = self.sparsity_pen
                                module_pen = 0.
                                for par_list in list(base_module.parametrizations.values()):
                                    for a in par_list[0].alpha_weights:
                                        module_pen += self.get_l0_norm_term(a, log_ratio)
                                l0_pen += (module_pen * sparsity_pen)
                            train_loss += l0_pen

                        train_losses_this_epoch.append(train_loss_real.item())

                    else:
                        # ================================================
                        # = PHASE 2:     cycle objective is added        =
                        # ================================================
                        style_embedding_function.eval()
                        style_embedding_of_gold, out_list_gold = style_embedding_function(
                            batch_of_spectrograms=batch[2].to(device),
                            batch_of_spectrogram_lengths=batch[3].to(device),
                            return_all_outs=True)

                        train_loss, output_spectrograms = self.net(text_tensors=batch[0].to(device),
                                                              text_lengths=batch[1].to(device),
                                                              gold_speech=batch[2].to(device),
                                                              speech_lengths=batch[3].to(device),
                                                              gold_durations=batch[4].to(device),
                                                              gold_pitch=batch[6].to(device),  # mind the switched order
                                                              gold_energy=batch[5].to(device),  # mind the switched order
                                                              utterance_embedding=style_embedding_of_gold.detach(),
                                                              lang_ids=batch[8].to(device),
                                                              return_mels=True)
                        style_embedding_function.train()
                        style_embedding_of_predicted, out_list_predicted = style_embedding_function(
                            batch_of_spectrograms=output_spectrograms,
                            batch_of_spectrogram_lengths=batch[3].to(device),
                            return_all_outs=True)

                        cycle_dist = 0
                        for out_gold, out_pred in zip(out_list_gold, out_list_predicted):
                            # essentially feature matching, as is often done in vocoder training,
                            # since we're essentially dealing with a discriminator here.
                            cycle_dist = cycle_dist + torch.nn.functional.l1_loss(out_pred, out_gold.detach())

                        train_loss_real = train_loss
                        if self._model_state == ModelState.DIFFPRUNING:
                            l0_pen = 0.
                            for module_name, base_module in self.get_encoder_base_modules(return_names=True):
                                #layer_idx = self.get_layer_idx_from_module(module_name)
                                sparsity_pen = self.sparsity_pen
                                module_pen = 0.
                                for par_list in list(base_module.parametrizations.values()):
                                    for a in par_list[0].alpha_weights:
                                        module_pen += self.get_l0_norm_term(a, log_ratio)
                                l0_pen += (module_pen * sparsity_pen)
                            train_loss += l0_pen

                        train_losses_this_epoch.append(train_loss_real.item())
                        cycle_losses_this_epoch.append(cycle_dist.item())
                        train_loss = train_loss + cycle_dist


                train_loss.backward()

                del train_loss
                del train_loss_real
                step_counter += 1



                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0, error_if_nonfinite=False)
                self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            self.net.eval()
            if epoch % epochs_per_save == 0:
                default_embedding = style_embedding_function(
                    batch_of_spectrograms=train_dataset[0][2].unsqueeze(0).to(device),
                    batch_of_spectrogram_lengths=train_dataset[0][3].unsqueeze(0).to(device)).squeeze()
                torch.save({
                    "model":        self.net.state_dict(),
                    "optimizer":    self.optimizer.state_dict(),
                    "step_counter": step_counter,
                    "scaler":       scaler.state_dict(),
                    "scheduler":    self.scheduler.state_dict(),
                    "default_emb":  default_embedding,
                }, os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
                delete_old_checkpoints(save_directory, keep=5)
                path_to_most_recent_plot = self.plot_progress_spec(self.net,
                                                              device,
                                                              save_dir=save_directory,
                                                              step=step_counter,
                                                              lang=lang,
                                                              default_emb=default_embedding)
                if use_wandb:
                    wandb.log({
                        "progress_plot": wandb.Image(path_to_most_recent_plot)
                    })
            print(self._model_state)
            print("Epoch:              {}".format(epoch))
            print("Spectrogram Loss:   {}".format(sum(train_losses_this_epoch) / len(train_losses_this_epoch)))
            if len(cycle_losses_this_epoch) != 0:
                print("Cycle Loss:         {}".format(sum(cycle_losses_this_epoch) / len(cycle_losses_this_epoch)))
            print("Time elapsed:       {} Minutes".format(round((time.time() - start_time) / 60)))
            print("Steps:              {}".format(step_counter))
            if use_wandb:
                wandb.log({
                    "spectrogram_loss": sum(train_losses_this_epoch) / len(train_losses_this_epoch),
                    "cycle_loss":       sum(cycle_losses_this_epoch) / len(cycle_losses_this_epoch) if len(
                        cycle_losses_this_epoch) != 0 else 0.0,
                    "epoch":            epoch,
                    "steps":            step_counter,
                })
            if step_counter > steps and epoch % epochs_per_save == 0:
                # DONE
                return


    def _init_optimizer_and_schedule(
            self,
            num_training_steps: int,
            learning_rate: float,
            weight_decay: float,
            adam_epsilon: float,
            num_warmup_steps: int = 0,
            learning_rate_alpha: Optional[float] = None
    ) -> None:

        if self._model_state == ModelState.DIFFPRUNING:
            optimizer_params = [
                {
                    # diff params (last dense layer is in no_decay list)
                    # TODO needs to be changed when original weight is set to fixed pre trained
                    "params": [p for n, p in self.net.encoder.named_parameters() if n[-8:] == "finetune"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                },
                {
                    "params": [p for n, p in self.net.encoder.named_parameters() if
                               n[-5:] == "alpha" or n[-11:] == "alpha_group"],
                    "lr": learning_rate_alpha
                },

            ]
        else:
            optimizer_params = [{
                "params": [p for n, p in self.net.encoder.named_parameters()],
                "lr": learning_rate
            }]

        self.optimizer = AdamW(optimizer_params, eps=adam_epsilon)

        self.scheduler = WarmupScheduler(self.optimizer, warmup_steps=num_warmup_steps)




    """
    def _init_sparsity_pen(self, sparsity_pen: Union[float, List[float]]) -> None:
        if isinstance(sparsity_pen, list):
            self.sparsity_pen = sparsity_pen
            assert len(sparsity_pen) == self.total_layers, "invalid sparsity penalty per layer: # of layers mismatch"
        else:
            self.sparsity_pen = [sparsity_pen] * self.total_layers
    """

    def _add_diff_parametrizations(self, *args) -> None:
        print("special)(**&&")
        #assert not self._parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules():

            for _,(n, p) in enumerate(list(base_module.named_parameters())):
                print(p)
                parametrize.register_parametrization(base_module, n, DiffWeight(base_module, n, p, *args),unsafe = False)
        self._model_state = ModelState.DIFFPRUNING

    @torch.no_grad()
    def _finetune_to_fixmask(self, pct: float) -> None:

        def _get_cutoff(values, pct):
            k = int(len(values) * pct)
            return torch.topk(torch.abs(values), k, largest=True, sorted=True)[-1]

        if self._model_state == ModelState.DIFFPRUNING:
            diff_weights = torch.tensor([])
            for base_module in self.get_encoder_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    par_list[0].eval()
                    diff_weight = (getattr(base_module, n) - par_list.original).detach().cuda()
                    diff_weights = torch.cat([diff_weights, diff_weight.flatten()])
            cutoff = _get_cutoff(diff_weights, pct)
            for base_module in self.get_encoder_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    pre_trained_weight = torch.clone(par_list.original)
                    parametrize.remove_parametrizations(base_module, n)
                    p = base_module._parameters[n].detach()
                    diff_weight = (p - pre_trained_weight)
                    diff_mask = (torch.abs(diff_weight) >= cutoff)
                    base_module._parameters[n] = Parameter(diff_weight * diff_mask)
                    parametrize.register_parametrization(base_module, n,
                                                         DiffWeightFixmask(self.net, n, pre_trained_weight, diff_mask))

        elif self._model_state == ModelState.FINETUNING:
            diff_weights = torch.tensor([]).to(self.device)
            pre_trained = self.model_pre.encoder
            for p, p_pre in zip(self.net.encoder.parameters(), pre_trained.parameters()):
                diff_weight = (p.cuda() - p_pre).flatten().detach().to(self.device)
                diff_weights = torch.cat([diff_weights, diff_weight])
            cutoff = _get_cutoff(diff_weights, pct)
            base_modules = dict(self.get_encoder_base_modules(return_names=True))
            for (n, p), p_pre,base_module in zip(list(self.net.encoder.named_parameters()), pre_trained.parameters(),self.get_encoder_base_modules()):
                n_parts = n.split(".")
                module_name, p_name = ".".join(n_parts[:-1]), n_parts[-1]
                #base_module = base_modules[module_name]
                diff_weight = (p - p_pre)
                diff_mask = (torch.abs(diff_weight) >= cutoff)
                base_module._parameters[n] = Parameter(diff_weight * diff_mask)
                parametrize.register_parametrization(base_module, p_name,
                                                     DiffWeightFixmask(self.net, n, p_pre, diff_mask))

        self._model_state = ModelState.FIXMASK

    @torch.no_grad()
    def _count_non_zero_params(self) -> Tuple[int, int]:
        assert self._parametrized, "Function only implemented for diff pruning"
        self.eval()
        n_p = 0
        n_p_zero = 0
        n_p_one = 0
        for base_module in self.get_encoder_base_modules():
            for par_list in list(base_module.parametrizations.values()):
                if isinstance(par_list[0], DiffWeightFixmask):
                    n_p_ = par_list[0].mask.numel()
                    n_p_zero_ = (~par_list[0].mask).sum()
                    n_p += n_p_
                    n_p_zero += n_p_zero_
                    n_p_one += (n_p_ - n_p_zero_)
                else:
                    z = par_list[0].z.detach()
                    n_p += z.numel()
                    n_p_zero += (z == 0.).sum()
                    n_p_one += (z == 1.).sum()
        self.train()
        n_p_between = n_p - (n_p_zero + n_p_one)
        return n_p, n_p_zero, n_p_between

    def _remove_diff_parametrizations(self) -> None:
        for module in self.get_encoder_base_modules():
            for n in list(module.parametrizations):
                parametrize.remove_parametrizations(module, n)
        self._model_state = ModelState.FINETUNING
