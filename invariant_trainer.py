import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import transformers
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.trainer_callback import TrainerState
from transformers.utils import logging

from tqdm import tqdm

import math
import os
import numpy as np
import pandas as pd
from typing import List, Optional

logger = logging.get_logger(__name__)


class InvariantTrainer(transformers.Trainer):

    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        optimizer, lr_scheduler = None, None
        # if self.optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        if self.sharded_dpp:
            optimizer = OSS(
                params=optimizer_grouped_parameters,
                optim=optimizer_cls,
                **optimizer_kwargs,
            )
        else:
            optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler

    def remove_dataparallel_wrapper(self):
        if hasattr(self.model, 'module'):
            self.model = self.model.module

    def invariant_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            resume_from_checkpoint: Optional[str] = None,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str`, `optional`):
                Local path to a saved checkpoint as saved by a previous instance of :class:`~transformers.Trainer`. If
                present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])

        if nb_steps is not None:
            max_steps = nb_steps
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            num_train_epochs = max(1, math.floor(max_steps / num_update_steps_per_epoch))
        else:
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            max_steps = num_update_steps_per_epoch * num_train_epochs

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        self.state = TrainerState()

        if self.args.n_gpu > 0:
            self.model.to('cuda')

        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0

        log_every = 100  # frequence de log
        csv_total_loss_path = os.path.join(self.args.output_dir, "train_total_loss.csv")
        csv_heads_loss_path = os.path.join(self.args.output_dir, "train_env_losses.csv")
        train_loss_log = []
        heads_loss_log = []

        for epoch in range(num_train_epochs):
            logger.info(f" Epoch: {epoch}")

            # make all dataloader iterateable
            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)

            for steps_trained_in_current_epoch in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                env_step_losses = {"step": total_trained_steps}
                total_loss_step = 0.0
                for env_name in training_set.keys():
                    logger.info(f" Update on environement {env_name}")
                    optimizer.zero_grad()
                    optimizers[env_name].zero_grad()

                    inputs = next(iter_loaders[env_name])
                    if self.args.n_gpu > 0:
                        inputs = inputs.to('cuda')

                    loss = self.training_step(self.model, inputs)

                    total_loss_step += loss.item()
                    env_step_losses[env_name] = loss.item()

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.use_amp:
                            self.scaler.unscale_(optimizer)
                            self.scaler.unscale_(optimizers[env_name])

                        if hasattr(optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            optimizer.clip_grad_norm(self.args.max_grad_norm)
                            optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm,
                            )

                    if self.use_amp:
                        self.scaler.step(optimizer)
                        self.scaler.step(optimizers[env_name])
                        self.scaler.update()
                    else:
                        optimizer.step()
                        optimizers[env_name].step()

                    lr_scheduler.step() # attention car environnement 1 aura toujours plus d'impact que les autres
                    lr_schedulers[env_name].step()

                    total_trained_steps += 1
                    if saving_heads:
                        if total_trained_steps % nb_steps_heads_saving == 0:
                            self.save_heads(total_trained_steps)
                    if saving_intermediary_models:
                        if total_trained_steps % nb_steps_model_saving == 0:
                            self.save_intermediary_model(total_trained_steps)

                train_loss_log.append({"step": total_trained_steps, "loss": total_loss_step / len(training_set)})
                heads_loss_log.append(env_step_losses)

                if total_trained_steps % log_every == 0:
                    pd.DataFrame(train_loss_log).to_csv(csv_total_loss_path, index=False)
                    pd.DataFrame(heads_loss_log).to_csv(csv_heads_loss_path, index=False)

        if train_loss_log:
            pd.DataFrame(train_loss_log).to_csv(csv_total_loss_path, index=False)
        if heads_loss_log:
            pd.DataFrame(heads_loss_log).to_csv(csv_heads_loss_path, index=False)


        return {
            "metrics": {
                "final_loss": loss.item(),
                "nb_steps": max_steps,
                "global_step": total_trained_steps
            }
        }


    def ensemble_train(
            self,
            training_set,
            nb_steps: Optional[int] = None,
            nb_steps_heads_saving: Optional[int] = 0,
            resume_from_checkpoint: Optional[str] = None,
            num_train_epochs: Optional[int] = 1,
            nb_steps_model_saving: Optional[int] = 0,
            **kwargs,
    ):
        """
        Training the heads as en ensemble instead of following the IRM-games dynamic

        Args:
            resume_from_checkpoint (:obj:`str`, `optional`):
                Local path to a saved checkpoint as saved by a previous instance of :class:`~transformers.Trainer`. If
                present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])

        if nb_steps is not None:
            max_steps = nb_steps
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            num_train_epochs = max(1, math.floor(max_steps / num_update_steps_per_epoch))
        else:
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            max_steps = num_update_steps_per_epoch * num_train_epochs

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        self.state = TrainerState()

        if self.args.n_gpu > 0:
            self.model.to('cuda')

        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0

        print("Num train epoch: ", num_train_epochs)
        print("Batch size: ", total_train_batch_size)
        print("Train data size: ", min_train_set_size)
        print("num_update_steps_per_epoch: ", num_update_steps_per_epoch)

        for epoch in range(num_train_epochs):
            logger.info(f" Epoch: {epoch}")
            print("epoch: ", epoch)
            # make all dataloader iterateable
            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)

            for steps_trained_in_current_epoch in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                for env_name in training_set.keys():
                    logger.info(f" Update on environement {env_name}")
                    # get a batch
                    optimizer.zero_grad()
                    for e_n in training_set.keys():
                        optimizers[e_n].zero_grad()

                    batch = next(iter_loaders[env_name])
                    # uncomment it, for CPU only run
                    if self.args.n_gpu > 0:
                        batch = batch.to('cuda')

                    # loss.backward() is done inside training step
                    loss = self.training_step(self.model, batch)

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(optimizer)
                            for env_name in training_set.keys():
                                self.scaler.unscale_(optimizers[env_name])

                        if hasattr(optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            optimizer.clip_grad_norm(self.args.max_grad_norm)
                            for e_n in training_set.keys():
                                optimizers[e_n].clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm,
                            )

                    if self.use_amp:
                        self.scaler.step(optimizer)
                        for e_n in training_set.keys():
                            self.scaler.step(optimizers[e_n])
                        self.scaler.update()
                    else:
                        optimizer.step()
                        for e_n in training_set.keys():
                            optimizers[e_n].step()

                    lr_scheduler.step()
                    for e_n in training_set.keys():
                        lr_schedulers[e_n].step()

                    total_trained_steps += 1
                    if saving_heads:
                        if total_trained_steps % nb_steps_heads_saving == 0:
                            self.save_heads(total_trained_steps)
                    if saving_intermediary_models:
                        if total_trained_steps % nb_steps_model_saving == 0:
                            self.save_intermediary_model(total_trained_steps)

        return {
            "metrics": {
                "final_loss": loss.item(),
                "nb_steps": max_steps,
                "global_step": total_trained_steps
            }
        }

    
    def invariant_train_games(
        self,
        training_set,
        nb_steps: Optional[int] = None,
        nb_steps_heads_saving: Optional[int] = 0,
        resume_from_checkpoint: Optional[str] = None,
        num_train_epochs: Optional[int] = 1,
        nb_steps_model_saving: Optional[int] = 0,
        **kwargs,
    ):

        head_updates_per_encoder_update = getattr(self.args, "head_updates_per_encoder_update", 1)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` instead.",
                FutureWarning,
            )

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])

        if nb_steps is not None:
            max_steps = nb_steps
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            num_train_epochs = max(1, math.floor(max_steps / num_update_steps_per_epoch))
        else:
            num_update_steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size))
            max_steps = num_update_steps_per_epoch * num_train_epochs

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        optimizer, lr_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        self.state = TrainerState()

        if self.args.n_gpu > 0:
            self.model.to('cuda')

        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0

        log_every = 100
        csv_total_loss_path = os.path.join(self.args.output_dir, "train_total_loss.csv")
        csv_heads_loss_path = os.path.join(self.args.output_dir, "train_env_losses.csv")
        train_loss_log = []  # logs for phase 2 (encoder) total loss
        heads_loss_log = []  # logs for phase 1 head losses

        for epoch in range(num_train_epochs):
            logger.info(f" Epoch: {epoch}")

            iter_loaders = {}
            for env_name in training_set.keys():
                train_loader = dataloaders[env_name]
                iter_loaders[env_name] = iter(train_loader)

            for steps_trained_in_current_epoch in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                # === Phase 1: update each environment-specific head ===
                env_step_losses = {"step": total_trained_steps}
                
                for _ in range(head_updates_per_encoder_update):
                    self.model.encoder.requires_grad_(False)
                    for env_name in training_set.keys():
                        logger.info(f" Update on environement {env_name}")
                        self.model.lm_heads[env_name].requires_grad_(True)
                        optimizers[env_name].zero_grad()

                        inputs = next(iter_loaders[env_name])
                        if self.args.n_gpu > 0:
                            inputs = inputs.to('cuda')

                        loss = self.training_step(self.model, inputs)

                        env_step_losses[env_name] = loss.item()

                        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                            if self.use_amp:
                                self.scaler.unscale_(optimizers[env_name])
                            
                            if hasattr(optimizers[env_name], "clip_grad_norm"):
                                optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.lm_heads[env_name].parameters(),
                                    self.args.max_grad_norm,
                                )
                        
                        if self.use_amp:
                            self.scaler.step(optimizers[env_name])
                        else:
                            optimizers[env_name].step()

                        lr_schedulers[env_name].step()

                        total_trained_steps += 1

                        if saving_heads and total_trained_steps % nb_steps_heads_saving == 0:
                            self.save_heads(total_trained_steps)
                        if saving_intermediary_models and total_trained_steps % nb_steps_model_saving == 0:
                            self.save_intermediary_model(total_trained_steps)

                    heads_loss_log.append(env_step_losses)

                # === Phase 2: update shared encoder ===
                self.model.encoder.requires_grad_(True)
                for env_name in training_set.keys():
                    self.model.lm_heads[env_name].requires_grad_(False)

                optimizer.zero_grad()
                total_loss = 0.0
                for env_name in training_set.keys():
                    inputs = next(iter_loaders[env_name])
                    if self.args.n_gpu > 0:
                        inputs = inputs.to('cuda')
                    loss = self.training_step(self.model, inputs)
                    total_loss += loss

                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)

                if hasattr(optimizer, "clip_grad_norm"):
                    optimizer.clip_grad_norm(self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.encoder.parameters(),
                        self.args.max_grad_norm,
                    )

                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()

                total_loss_val = total_loss.item()
                train_loss_log.append({"step": total_trained_steps, "loss": total_loss_val/len(training_set)})

                if total_trained_steps % log_every == 0:
                    pd.DataFrame(train_loss_log).to_csv(csv_total_loss_path, index=False)
                    pd.DataFrame(heads_loss_log).to_csv(csv_heads_loss_path, index=False)

        if train_loss_log:
            pd.DataFrame(train_loss_log).to_csv(csv_total_loss_path, index=False)
        if heads_loss_log:
            pd.DataFrame(heads_loss_log).to_csv(csv_heads_loss_path, index=False)

        return {
            "metrics": {
                "eval_loss": total_loss_val,
                "nb_steps": max_steps,
                "global_step": total_trained_steps
            }
        }


    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        print("saving-heads")
        if not os.path.exists("lm_heads"):
            os.makedirs("lm_heads")

        for env, lm_head in self.model.lm_heads.items():
            filepath = os.path.join("lm_heads", f"{env}-{step_count}.npy")
            np.save(filepath, lm_head.vocab_projector.weight.data.cpu().numpy())

    def get_single_train_dataloader(self, env_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # if is_tpu_available():
        #     train_sampler = get_tpu_sampler(train_dataset)
        # else:
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator
        )
