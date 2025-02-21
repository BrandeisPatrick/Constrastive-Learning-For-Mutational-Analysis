# my_project/trainer/trainer.py

import os
import logging
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from .trainer_utils import save_model_checkpoint, safe_barrier
from my_project.utils.load_model import load_model

import wandb  # Import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,  # Explicitly type hint wandb_run
        **kwargs
    ):
        """
        Initializes the Trainer with model, optimizer, scheduler, and training arguments.
        """
        self.device = args.default
        # Ensure model is moved to device before optimizer initialization
        self.model = model.to(self.device)
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')

        # Assign wandb_run if provided
        self.wandb_run = wandb_run

        # Control logging frequency (log every `log_interval` steps)
        self.log_interval = getattr(args, 'log_interval', 10)  # Default to 10 if not set

        # -------------------- Debug Logging Start --------------------
        # Log optimizer parameter groups and details of each parameter
        logger.debug(f"Optimizer parameter groups: {self.optimizer.param_groups}")
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                logger.debug(f"Optimizer group {i} includes parameter with shape {param.shape}, "
                             f"requires_grad={param.requires_grad}, device={param.device}")
        # -------------------- Debug Logging End --------------------

        # -------------------- Debug Logging Start --------------------
        # Log model parameter details: check requires_grad and device
        logger.debug("Model parameters and their properties:")
        for name, param in self.model.named_parameters():
            logger.debug(f"Parameter '{name}': requires_grad={param.requires_grad}, device={param.device}")
        # -------------------- Debug Logging End --------------------

        # -------------------- Added Feature Start --------------------
        # Log parameters that do not require gradients
        logger.debug("Checking for parameters that do not require gradients:")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                logger.debug(f"Parameter '{name}' does not require gradients!")
        # -------------------- Added Feature End --------------------

    def get_current_lr(self) -> List[float]:
        """
        Retrieves the current learning rate(s) from the optimizer.
        """
        return [group['lr'] for group in self.optimizer.param_groups]

    def _move_to_device(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src1, src2, labels = batch
        return src1.to(self.device), src2.to(self.device), labels.to(self.device)

    def _log_metrics(
        self,
        epoch: int,
        step: Optional[int] = None,
        mode: str = 'train',
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        avg_loss: Optional[float] = None,
        avg_accuracy: Optional[float] = None,
        current_lrs: Optional[List[float]] = None
    ):
        """
        Logs metrics to both logger and W&B.
        """
        if step is not None and loss is not None and accuracy is not None:
            # Per-step logging
            logger.info(
                f"Epoch [{epoch+1}/{self.args.max_epochs}], "
                f"Step [{step}], "
                f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            if self.wandb_run:
                self.wandb_run.log({
                    "train/step": step,
                    "train/loss": loss,
                    "train/accuracy": accuracy
                })
        elif avg_loss is not None and avg_accuracy is not None:
            # Epoch-level logging
            log_message = (
                f"Epoch [{epoch+1}/{self.args.max_epochs}], "
                f"{mode.capitalize()} Loss: {avg_loss:.4f}, {mode.capitalize()} Accuracy: {avg_accuracy:.4f}"
            )
            if current_lrs is not None:
                # Log each learning rate separately
                for i, lr in enumerate(current_lrs):
                    log_dict_key = f"{mode}/learning_rate_group_{i}"
                    log_dict_value = lr
                    if self.wandb_run:
                        self.wandb_run.log({log_dict_key: log_dict_value})
                    log_message += f", Learning Rate Group {i}: {lr:.6f}"
            
            logger.info(log_message)
            if self.wandb_run:
                log_dict = {
                    f"{mode}/avg_loss_epoch": avg_loss,
                    f"{mode}/avg_accuracy_epoch": avg_accuracy,
                    f"{mode}/epoch": epoch + 1
                }
                self.wandb_run.log(log_dict)

    def _save_checkpoint(self, epoch: int, loss: float, accuracy: float, prefix: str = 'TRAIN'):
        checkpoint_filename = f"{prefix}_epoch{epoch+1}_loss{loss:.4f}_accuracy{accuracy:.4f}.pt"
        checkpoint_path = os.path.join(self.args.save_dir, checkpoint_filename)
        save_model_checkpoint(self.model, self.args, checkpoint_path)
        logger.info(f"Model saved at {checkpoint_path}")

        if self.wandb_run:
            self.wandb_run.save(checkpoint_path)

    def _run_epoch(
        self,
        data_loader: DataLoader,
        epoch: int,
        mode: str = 'train'
    ) -> Tuple[float, float]:
        is_train = (mode == 'train')
        self.model.train() if is_train else self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        is_main_process = (self.args.rank == 0)
        phase = "Training" if is_train else "Validation"
        logger.info(f"Starting {phase} for Epoch {epoch+1}/{self.args.max_epochs}")

        # -------------------- Debug Logging Start --------------------
        # Log the current learning rates at the start of the epoch
        current_lrs = self.get_current_lr()
        logger.debug(f"Epoch {epoch+1}: Current learning rates before epoch: {current_lrs}")
        for idx, lr in enumerate(current_lrs):
            if lr == 0:
                logger.warning(f"Learning rate for group {idx} is 0.")
        # -------------------- Debug Logging End --------------------

        for batch_idx, batch in enumerate(data_loader):
            src1, src2, labels = self._move_to_device(batch)

            with torch.set_grad_enabled(is_train):
                logits = self.model(src1, src2)
                loss = self.criterion(logits, labels)
                loss_val = loss.item()

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()

                    # -------------------- Debug Logging Start --------------------
                    # Log learning rates immediately after gradient computation
                    current_lrs = self.get_current_lr()
                    logger.debug(f"After backward pass: Current learning rates: {current_lrs}")
                    # Log gradient statistics for each parameter
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_mean = param.grad.mean().item()
                            grad_max = param.grad.max().item()
                            # logger.debug(f"Gradient of '{name}': Mean={grad_mean:.6f}, Max={grad_max:.6f}")
                    # -------------------- Debug Logging End --------------------

                    # ---------------------------------------------------------
                    # 1) Clone parameters BEFORE optimizer step
                    before_params = {}
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            before_params[name] = param.clone()

                    # -------------------- Debug Logging Start --------------------
                    # Log parameter norms before the optimizer step
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            param_norm = param.data.norm().item()
                            # logger.debug(f"Before optimizer step: Norm of parameter '{name}': {param_norm:.6f}")
                    # -------------------- Debug Logging End --------------------

                    # 2) Perform the optimizer step
                    logger.debug("Executing optimizer step...")
                    self.optimizer.step()
                    logger.debug("Optimizer step completed.")

                    # -------------------- Debug Logging Start --------------------
                    # Log current learning rates after optimizer step
                    current_lrs = self.get_current_lr()
                    logger.debug(f"After optimizer step: Current learning rates: {current_lrs}")
                    # -------------------- Debug Logging End --------------------

                    # 3) Compare parameters AFTER optimizer step and count changes
                    changed_count = 0
                    unchanged_count = 0
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            after = param.clone()
                            # Tolerance-based comparison for numerical stability
                            if not torch.allclose(before_params[name], after, atol=1e-8):
                                changed_count += 1
                                diff_norm = (after - before_params[name]).norm().item()
                                logger.debug(f"Parameter '{name}' changed. Norm of difference: {diff_norm:.6f}")
                            else:
                                unchanged_count += 1

                    total_params = changed_count + unchanged_count
                    logger.debug(
                        f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(data_loader)}]: "
                        f"Parameters changed: {changed_count}/{total_params}, "
                        f"Parameters unchanged: {unchanged_count}/{total_params}"
                    )
                    # ---------------------------------------------------------

                    # If using a non-ReduceLROnPlateau scheduler, step it here
                    if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                trues = labels.cpu().numpy()

                total_correct += (preds == trues).sum()
                total_samples += trues.size
                total_loss += loss_val * trues.size

                # Optional per-step logging
                if is_main_process and is_train and (batch_idx + 1) % self.log_interval == 0:
                    current_step = batch_idx + 1 + epoch * len(data_loader)
                    avg_loss = total_loss / total_samples
                    accuracy = total_correct / total_samples
                    self._log_metrics(
                        epoch,
                        step=current_step,
                        mode='train',
                        loss=loss_val,
                        accuracy=accuracy
                    )
                    # Reset counters to log rolling metrics
                    total_loss, total_correct, total_samples = 0.0, 0, 0

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # If using ReduceLROnPlateau, step it with the final epoch-level loss
        if not is_train and self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        # Retrieve current learning rates after epoch
        current_lrs = self.get_current_lr()

        # Log epoch-level metrics
        self._log_metrics(
            epoch,
            mode=mode,
            avg_loss=avg_loss,
            avg_accuracy=accuracy,
            current_lrs=current_lrs
        )

        return avg_loss, accuracy

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        avg_loss, accuracy = self._run_epoch(train_loader, epoch, mode='train')

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0 and self.args.rank == 0:
            self._save_checkpoint(epoch, avg_loss, accuracy, prefix='TRAIN')
        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader, epoch: int):
        avg_loss, accuracy = self._run_epoch(val_loader, epoch, mode='val')

        # Save the best model based on validation loss
        if avg_loss < self.best_val_loss and self.args.rank == 0:
            self.best_val_loss = avg_loss
            self._save_checkpoint(epoch, avg_loss, accuracy, prefix='BEST')
            logger.info("Best model updated and saved.")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        if self.args.rank == 0:
            logger.info("#### Training Started ####")
        for epoch in range(self.args.max_epochs):
            if self.args.rank == 0:
                logger.info(f"### Starting Epoch {epoch+1}/{self.args.max_epochs} ###")
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.validate(val_loader, epoch)
            safe_barrier()  # Ensure all processes have completed the epoch

        if self.wandb_run:
            self.wandb_run.finish()  # Finalize the W&B run

    def test(self, test_loader: DataLoader, best_checkpoint_path: str) -> Tuple[float, float, List[int], List[int]]:
        if self.args.rank == 0:
            logger.info("#### Testing Started ####")

        try:
            self.model = load_model(self.args, checkpoint_path=best_checkpoint_path).to(self.device)
            self.model.eval()
            if self.args.rank == 0:
                logger.info(f"Loaded model from {best_checkpoint_path}")
        except Exception as e:
            if self.args.rank == 0:
                logger.error(f"Failed to load model from {best_checkpoint_path}: {e}")
            return 0.0, 0.0, [], []

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_trues = []

        is_main_process = (self.args.rank == 0)
        if is_main_process:
            logger.info("Starting Testing Phase")

        for batch_idx, batch in enumerate(test_loader):
            src1, src2, labels = self._move_to_device(batch)
            logits = self.model(src1, src2)
            loss = self.criterion(logits, labels)

            loss_val = loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            trues = labels.cpu().numpy()
            total_correct += (preds == trues).sum()
            total_samples += trues.size
            total_loss += loss_val * trues.size

            all_preds.extend(preds)
            all_trues.extend(trues)

            if is_main_process:
                logger.info(f"Batch {batch_idx+1}/{len(test_loader)} - Loss: {loss_val:.4f}")

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        if is_main_process:
            logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
            cm = confusion_matrix(all_trues, all_preds)
            logger.info(f"Confusion Matrix:\n{cm}")

        if self.wandb_run and is_main_process:
            self.wandb_run.log({
                "test/loss": avg_loss,
                "test/accuracy": accuracy
            })

        return avg_loss, accuracy, all_trues, all_preds