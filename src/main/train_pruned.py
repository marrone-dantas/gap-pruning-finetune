import logging
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from train import TrainModel
from typing import Optional
from torch.utils.data import DataLoader
import numpy as np

global_logger = logging.getLogger(__name__)

class TrainPrunedModel(TrainModel):
    """
    Trainer for a pruned model: 
    - Initially freezes all Conv2d layers and trains only classifier & BatchNorm.
    - Optionally applies differential learning rates.
    - Optionally unfreezes Conv layers after a number of epochs.
    """
    def __init__(self, is_gpu: bool = True):
        super().__init__(is_gpu=is_gpu)
        self.scaler = GradScaler()

    def train_pruned(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 30,
        freeze_conv: bool = True,
        unfreeze_after: int = 5,
        differential_lr: bool = False,
        lr_head: float = 1e-2,
        lr_conv: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler=None,
        unfreeze_scheduler_step: bool = False,
        pct_val: float = 0.1,
        generate_log: bool = True,
        path_log: Optional[str] = None,
        prefix: str = "",
    ):
        """
        Fine-tune the pruned model.
        
        Args:
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            epochs: total number of epochs.
            freeze_conv: if True, freeze Conv2d weights at start.
            unfreeze_after: epoch at which to unfreeze all Conv2d layers.
            differential_lr: if True, use separate lrs for head and conv.
            lr_head: learning rate for classifier & BatchNorm.
            lr_conv: learning rate for Conv2d (only if differential_lr).
            weight_decay: weight decay for all optim params.
            scheduler: LR scheduler instance (optional).
            unfreeze_scheduler_step: if True, reset scheduler when unfreezing.
        """
        if val_loader is None:
            from sklearn.model_selection import StratifiedShuffleSplit
            from torch.utils.data import Subset

            ds = train_loader.dataset

            try:
                # Try to get labels from common dataset attributes
                labels = getattr(ds, 'targets', None) or getattr(ds, 'labels', None)
                labels = np.array(labels)

                sss = StratifiedShuffleSplit(n_splits=1, test_size=pct_val, random_state=42)
                train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
            except Exception as e:
                global_logger.info(f"Failed to get labels directly: {e}")
                global_logger.info("Trying to extract labels manually by iterating over the dataset...")

                labels = []
                try:
                    # Disable gradient calculation
                    with torch.no_grad():
                        for data in tqdm(ds):
                            # Assuming each item is (input, label) or {'input': ..., 'label': ...}
                            if isinstance(data, (tuple, list)) and len(data) > 1:
                                labels.append(data[1])
                            elif isinstance(data, dict) and ('label' in data or 'target' in data):
                                labels.append(data.get('label', data.get('target')))
                            else:
                                raise ValueError("Unsupported dataset format for label extraction.")
                    labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else np.array(labels)

                    sss = StratifiedShuffleSplit(n_splits=1, test_size=pct_val, random_state=42)
                    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
                except Exception as e2:
                    global_logger.info(f"Failed to manually extract labels as well: {e2}")
                    global_logger.info("Falling back to random split.")

                    indices = np.arange(len(ds))
                    np.random.seed(42)
                    np.random.shuffle(indices)
                    val_size = int(len(indices) * pct_val)
                    val_idx = indices[:val_size]
                    train_idx = indices[val_size:]

            batch_size   = train_loader.batch_size
            num_workers  = getattr(train_loader, 'num_workers', 0)
            pin_memory   = getattr(train_loader, 'pin_memory', False)

            train_loader = DataLoader(
                Subset(ds, train_idx),
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory
            )
            val_loader = DataLoader(
                Subset(ds, val_idx),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )

            global_logger.info(
                f"Created stratified split: {len(train_idx)} train / "
                f"{len(val_idx)} val (pct_val={pct_val})"
            )
        
        
        self.model.to(self.device)
        # 1) Freeze conv layers if requested
        if freeze_conv:
            global_logger.info("Freezing all Conv2d layers for initial fine-tune.")
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    for p in module.parameters():
                        p.requires_grad = False

        # 2) Build parameter groups
        head_params, conv_params, bn_params = [], [], []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_params += list(m.parameters())
            elif isinstance(m, nn.Conv2d):
                conv_params += list(m.parameters())
            elif isinstance(m, nn.Linear):
                head_params += list(m.parameters())

        if differential_lr:
            param_groups = [
                {'params': head_params + bn_params, 'lr': lr_head,  'weight_decay': weight_decay},
                {'params': conv_params,             'lr': lr_conv,  'weight_decay': weight_decay},
            ]
            global_logger.info(f"Using differential LR: head={lr_head}, conv={lr_conv}")
        else:
            param_groups = [
                {'params': head_params + bn_params + conv_params, 'lr': lr_head, 'weight_decay': weight_decay}
            ]
            global_logger.info(f"Training all trainable params at LR={lr_head}")

        self.optimizer = optim.SGD(param_groups, momentum=0.9)
        self.scheduler = scheduler

        best_val_acc = -float('inf')
        no_improve = 0
        history = []

        for epoch in range(1, epochs + 1):
            # Unfreeze conv layers at specified epoch
            if freeze_conv and epoch == unfreeze_after:
                global_logger.info(f"Epoch {epoch}: unfreezing all Conv2d layers.")
                for module in self.model.modules():
                    if isinstance(module, nn.Conv2d):
                        for p in module.parameters():
                            p.requires_grad = True
                # rebuild optimizer to include conv params if they were frozen
                if not differential_lr:
                    for group in self.optimizer.param_groups:
                        group['params'] = head_params + bn_params + conv_params
                if unfreeze_scheduler_step and self.scheduler is not None:
                    self.scheduler.last_epoch = -1  # reset scheduler
                    global_logger.info("Scheduler reset after unfreeze.")

            # Training loop
            self.model.train()
            train_loss, correct, total = 0.0, 0, 0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for x, y in train_bar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                with autocast():
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                preds = logits.argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
                train_loss += loss.item() * y.size(0)
                train_bar.set_postfix({
                    'loss': f"{train_loss/total:.4f}",
                    'acc':  f"{correct/total:.4f}"
                })

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss, val_acc = self.evaluate_model(val_loader, generate_log=False)
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Validation
            val_loss, val_acc = self.evaluate_model(val_loader, generate_log=generate_log, path_log=path_log, prefix=prefix)
            history.append({
                'epoch': epoch,
                'train_loss': train_loss/total,
                'train_acc': correct/total,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            global_logger.info(
                f"Epoch {epoch} summary: train_loss={train_loss/total:.4f}, "
                f"train_acc={correct/total:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            # Early stopping on validation acc stagnation
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_val.pth")
                global_logger.info(f"New best val_acc: {best_val_acc:.4f}. Model saved.")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 3:
                    global_logger.info(f"No improvement in val_acc for {no_improve} epochs. Stopping early.")
                    break

        global_logger.info("Fine-tuning complete.")
        return history
