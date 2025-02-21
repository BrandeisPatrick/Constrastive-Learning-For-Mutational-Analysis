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

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    
    For a pair with embeddings e1 and e2 (with Euclidean distance d) and a binary label y 
    (1 for similar, 0 for dissimilar), the loss is computed as:
    
        L = 0.5 * [ y * d^2 + (1 - y) * max(margin - d, 0)^2 ]
    """
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_similar = labels * distances.pow(2)
        loss_dissimilar = (1 - labels) * torch.clamp(self.margin - distances, min=0.0).pow(2)
        loss = 0.5 * (loss_similar + loss_dissimilar)
        return loss.mean()

class Trainer_stt4_enc:
    def __init__(
        self,
        model: nn.Module,
        args,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        **kwargs
    ):
        """
        Trainer for contrastive learning.
        
        The model is expected to return two embeddings (one for each input sequence).
        The trainer computes the Euclidean distance between the embeddings and applies a contrastive loss
        to minimize the distance for similar pairs and maximize it for dissimilar pairs.
        
        Args:
            model: A model that returns a pair of embeddings for inputs (src1, src2).
            args: An object containing training hyperparameters. Expected attributes include:
                  - default: the device to use.
                  - max_epochs: maximum number of epochs.
                  - rank: for multi-process training (0 indicates the main process).
                  - save_dir: directory for saving checkpoints.
                  - contrastive_margin: margin value for the contrastive loss.
                  - log_interval: logging frequency (in batches).
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            wandb_run: Optional wandb run for logging.
        """
        self.device = args.default
        self.model = model.to(self.device)
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler

        margin = getattr(args, 'contrastive_margin', 1.0)
        self.criterion = ContrastiveLoss(margin=margin)
        self.best_val_loss = float('inf')
        self.wandb_run = wandb_run
        self.log_interval = getattr(args, 'log_interval', 10)

        # Debug logging for optimizer and model parameters.
        logger.debug(f"Optimizer parameter groups: {self.optimizer.param_groups}")
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                logger.debug(f"Group {i}: param shape: {param.shape}, requires_grad: {param.requires_grad}")
        for name, param in self.model.named_parameters():
            logger.debug(f"Model parameter '{name}': requires_grad: {param.requires_grad}, device: {param.device}")

    def get_current_lr(self) -> List[float]:
        """Retrieve the current learning rate(s) from the optimizer."""
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
        avg_loss: Optional[float] = None,
        current_lrs: Optional[List[float]] = None
    ):
        """
        Logs metrics to both logger and wandb.
        """
        if step is not None and loss is not None:
            logger.info(f"Epoch [{epoch+1}/{self.args.max_epochs}], Step [{step}], Loss: {loss:.4f}")
            if self.wandb_run:
                self.wandb_run.log({f"{mode}/step": step, f"{mode}/loss": loss})
        elif avg_loss is not None:
            log_message = f"Epoch [{epoch+1}/{self.args.max_epochs}], {mode.capitalize()} Loss: {avg_loss:.4f}"
            if current_lrs is not None:
                for i, lr in enumerate(current_lrs):
                    log_message += f", LR Group {i}: {lr:.6f}"
                    if self.wandb_run:
                        self.wandb_run.log({f"{mode}/learning_rate_group_{i}": lr})
            logger.info(log_message)
            if self.wandb_run:
                self.wandb_run.log({f"{mode}/avg_loss_epoch": avg_loss, f"{mode}/epoch": epoch+1})

    def _save_checkpoint(self, epoch: int, loss: float, prefix: str = 'TRAIN'):
        checkpoint_filename = f"{prefix}_epoch{epoch+1}_loss{loss:.4f}.pt"
        checkpoint_path = os.path.join(self.args.save_dir, checkpoint_filename)
        save_model_checkpoint(self.model, self.args, checkpoint_path)
        logger.info(f"Model saved at {checkpoint_path}")
        if self.wandb_run:
            self.wandb_run.save(checkpoint_path)

    def _compute_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Euclidean distance between two embeddings.
        
        Args:
            emb1, emb2: Tensors of shape (batch, embd_dim)
            
        Returns:
            distances: Tensor of shape (batch,)
        """
        return torch.norm(emb1 - emb2, p=2, dim=1)

    def _run_epoch(self, data_loader: DataLoader, epoch: int, mode: str = 'train') -> float:
        is_train = (mode == 'train')
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        total_samples = 0
        is_main_process = (self.args.rank == 0)
        phase = "Training" if is_train else "Validation"
        logger.info(f"Starting {phase} for Epoch {epoch+1}/{self.args.max_epochs}")
        current_lrs = self.get_current_lr()
        logger.debug(f"Epoch {epoch+1}: Learning rates: {current_lrs}")

        for batch_idx, batch in enumerate(data_loader):
            src1, src2, labels = self._move_to_device(batch)
            labels = labels.float()  # Expected: 0 (dissimilar) or 1 (similar)
            with torch.set_grad_enabled(is_train):
                # Note: the model's forward accepts src1 and src2 and returns (emb1, emb2)
                emb1, emb2 = self.model(src1, src2)
                distances = self._compute_distance(emb1, emb2)  # (batch,)
                loss = self.criterion(distances, labels)
                loss_val = loss.item()

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

            batch_size = src1.size(0)
            total_loss += loss_val * batch_size
            total_samples += batch_size

            if is_main_process and is_train and (batch_idx + 1) % self.log_interval == 0:
                current_step = batch_idx + 1 + epoch * len(data_loader)
                avg_loss = total_loss / total_samples
                self._log_metrics(epoch, step=current_step, mode='train', loss=loss_val)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        if not is_train and self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        current_lrs = self.get_current_lr()
        self._log_metrics(epoch, mode=mode, avg_loss=avg_loss, current_lrs=current_lrs)
        return avg_loss

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        avg_loss = self._run_epoch(train_loader, epoch, mode='train')
        if (epoch + 1) % 50 == 0 and self.args.rank == 0:
            self._save_checkpoint(epoch, avg_loss, prefix='TRAIN')
        return avg_loss

    def validate(self, val_loader: DataLoader, epoch: int):
        avg_loss = self._run_epoch(val_loader, epoch, mode='val')
        if avg_loss < self.best_val_loss and self.args.rank == 0:
            self.best_val_loss = avg_loss
            self._save_checkpoint(epoch, avg_loss, prefix='BEST')
            logger.info("Best model updated and saved.")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        if self.args.rank == 0:
            logger.info("#### Contrastive Training (Embeddings) Started ####")
        for epoch in range(self.args.max_epochs):
            if self.args.rank == 0:
                logger.info(f"### Starting Epoch {epoch+1}/{self.args.max_epochs} ###")
            self.train_epoch(train_loader, epoch)
            self.validate(val_loader, epoch)
            safe_barrier()  # Ensure all processes complete the epoch
        if self.wandb_run:
            self.wandb_run.finish()

    def test(self, test_loader: DataLoader, best_checkpoint_path: str) -> Tuple[float, List[float], List[float]]:
        if self.args.rank == 0:
            logger.info("#### Contrastive Testing (Embeddings) Started ####")
        try:
            self.model = load_model(self.args, checkpoint_path=best_checkpoint_path).to(self.device)
            self.model.eval()
            if self.args.rank == 0:
                logger.info(f"Loaded model from {best_checkpoint_path}")
        except Exception as e:
            if self.args.rank == 0:
                logger.error(f"Failed to load model from {best_checkpoint_path}: {e}")
            return 0.0, [], []

        total_loss = 0.0
        total_samples = 0
        all_labels = []
        all_distances = []
        is_main_process = (self.args.rank == 0)
        if is_main_process:
            logger.info("Starting Testing Phase")

        for batch_idx, batch in enumerate(test_loader):
            src1, src2, labels = self._move_to_device(batch)
            labels = labels.float()
            with torch.no_grad():
                emb1, emb2 = self.model(src1, src2)
                distances = self._compute_distance(emb1, emb2)
                loss = self.criterion(distances, labels)
            loss_val = loss.item()
            batch_size = src1.size(0)
            total_loss += loss_val * batch_size
            total_samples += batch_size
            all_labels.extend(labels.cpu().numpy().tolist())
            all_distances.extend(distances.cpu().numpy().tolist())
            if is_main_process:
                logger.info(f"Batch {batch_idx+1}/{len(test_loader)} - Loss: {loss_val:.4f}")

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        if is_main_process:
            logger.info(f"Test Loss: {avg_loss:.4f}")
            threshold = getattr(self.args, 'similarity_threshold', 0.5)
            preds = [1 if d < threshold else 0 for d in all_distances]
            cm = confusion_matrix(all_labels, preds)
            logger.info(f"Confusion Matrix:\n{cm}")
            if self.wandb_run:
                self.wandb_run.log({"test/loss": avg_loss})

        return avg_loss, all_labels, all_distances