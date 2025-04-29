from torchvision import datasets
import models as models
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import dataset as dataset
from tqdm import tqdm 
import csv
from torch.cuda.amp import autocast, GradScaler
from collections import deque
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from collections import deque
from tqdm import tqdm


class TrainModel():

    def __init__(self, is_gpu=True):
        self.model = None
        self.arr_dataset = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.is_gpu = is_gpu
        self.device = torch.device('cuda' if is_gpu else 'cpu')
        self.hist = []
        self.best_loss = float('inf')
        self.best_acc = -1 * float('inf')
        self.scaler = GradScaler()
        self.verbose = True

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)


    def reset_metrics(self):
        self.best_loss = float('inf')
        self.best_acc = -1 * float('inf')

    def set_model(self, id_model='alexnet', num_classes=10):
        self.model = models.get_model(name_model=id_model, num_classes=num_classes)

    def load_weights(self, weights):
        print('Loading weights:', weights)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))

    def set_dataset(self, id_dataset='cifar100', batch_size=1024, num_train=45000, num_val=5000, download=False):
        print('Dataset Selected:',id_dataset,'| Batch:',batch_size,'|')
        self.arr_dataset = dataset.get_dataset(id_dataset=id_dataset, batch_size=batch_size, num_train=num_train, num_val=num_val, download=download)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def save_hist(self,path,fmt="%1.3f",delimiter=','):

        np.savetxt(path,np.asarray(self.hist),fmt=fmt,delimiter=delimiter,header='train_loss,train_acc,val_loss,val_acc')

    def evaluate_model(self, loader, generate_log=True, path_log='output', prefix=''):
        self.model.to(self.device)
        self.model.eval()
        arr_true = []
        arr_pred = []
        running_loss = 0
        test_acc = 0
        total = 0

        # Initialize progress bar
        progress_bar = tqdm(loader, desc='Evaluating')

        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            

            arr_pred.extend(predicted.cpu().detach().numpy())
            arr_true.extend(target.cpu().detach().numpy())
            total += target.size(0)
            test_acc += predicted.eq(target).sum().item()
                
            # Update progress bar description with current loss and accuracy
            progress_bar.set_description(f'Loss: {running_loss / total:.4f}, Acc: {test_acc / total:.4f}')

        # Calculate final metrics
        final_loss = running_loss / len(loader)
        final_acc = test_acc / total
        
        # Export results to CSV if logging is enabled
        if generate_log:
            log_file_path = f'{path_log}/{prefix}evaluation_log.csv'
            with open(log_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['True Label', 'Predicted Label'])
                for true, pred in zip(arr_true, arr_pred):
                    writer.writerow([true, pred])
                    
                # Optionally, you could add another section for summary statistics
                writer.writerow([])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Loss', final_loss])
                writer.writerow(['Accuracy', final_acc])

        return final_loss, final_acc

    def count_parameters(self) -> dict:
        """
        Conta parâmetros de um modelo PyTorch.

        Retorna um dict com:
        - total: todos os parâmetros
        - trainable: parâmetros com requires_grad=True
        - non_trainable: parâmetros com requires_grad=False
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = total - trainable

        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': non_trainable
        }

    def compare_model_sizes(self, stats_x,
                            stats_y) -> dict:
        """
        Compara o número de parâmetros de dois modelos PyTorch:
        - percent_smaller: quanto (%) o modelo Y é menor que X
        - times_smaller: quantas vezes o modelo Y cabe em X
        """

        total_x = stats_x['total']
        total_y = stats_y['total']

        # evita divisão por zero
        if total_x == 0 or total_y == 0:
            raise ValueError("Um dos modelos não tem parâmetros.")

        # quantas vezes X é maior que Y
        times_smaller = total_x / total_y

        # quão menor (em %) Y é que X
        percent_smaller = (1 - (total_y / total_x)) * 100

        return {
            'total_x': total_x,
            'total_y': total_y,
            'times_smaller': times_smaller,
            'percent_smaller': percent_smaller
        }
    def partial_train_model(self,
                            epochs: int = 100,
                            unfreeze_patience: int = 5,
                            variation_patience: int = 5,
                            variation_threshold: float = 0.01,
                            pct_train: float = 1.0,
                            pct_eval: float = 0.2,
                            freeze_pre_last_conv: bool = False,
                            freeze_conv_only: bool = False,
                            num_workers: int = 4,
                            pin_memory: bool = True,
                            train_ckpt_path: str = 'best_train.pth',
                            val_ckpt_path: str = 'best_val.pth'):
        """
        Train model on a stratified subset with options to freeze layers,
        saving model weights at the epochs with the best train loss and best val loss.

        Args:
            epochs: max epochs.
            unfreeze_patience: epochs w/o val improvement to unfreeze.
            variation_patience: window size for variation check.
            variation_threshold: min val_acc variation to continue.
            target_acc: stop if reached.
            pct_train: fraction of train split to use.
            pct_eval: fraction for internal eval.
            freeze_pre_last_conv: freeze layers up to last Conv2d.
            freeze_conv_only: freeze only Conv2d layers.
            num_workers: DataLoader workers.
            pin_memory: DataLoader pin_memory flag.
            train_ckpt_path: filepath to save weights at best train loss.
            val_ckpt_path: filepath to save weights at best val loss.
        """
        import copy
        self.model.to(self.device)
        train_loader_orig, _, _ = self.arr_dataset
        dataset = train_loader_orig.dataset
        labels = np.array(getattr(dataset, 'targets', getattr(dataset, 'labels', [y for _, y in dataset])))
        # Stratified split train/eval
        sss = StratifiedShuffleSplit(1, test_size=pct_eval, random_state=42)
        train_idx, eval_idx = next(sss.split(np.zeros(len(labels)), labels))
        # Subsample train
        if pct_train < 1.0:
            train_lbls = labels[train_idx]
            sss2 = StratifiedShuffleSplit(1, train_size=pct_train, random_state=42)
            train_idx = train_idx[next(sss2.split(np.zeros(len(train_lbls)), train_lbls))[0]]
        # Build DataLoaders
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=train_loader_orig.batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        eval_loader = DataLoader(Subset(dataset, eval_idx), batch_size=train_loader_orig.batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        if self.verbose:
            print(f"[INFO] {len(train_idx)} train samples, {len(eval_idx)} eval samples.")
        # Freeze only conv layers if requested
        if freeze_conv_only:
            if self.verbose: print("[INFO] Freezing only Conv2d layers...")
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    for p in m.parameters(): p.requires_grad = False
        # Or freeze pre-last-conv
        if freeze_pre_last_conv:
            if self.verbose: print("[INFO] Freezing up to last Conv2d...")
            mods = list(self.model.named_modules())
            convs = [(n, m) for n, m in mods if isinstance(m, nn.Conv2d)]
            if convs:
                last_name, _ = convs[-1]
                idx = next(i for i, (n, _) in enumerate(mods) if n == last_name)
                for p in self.model.parameters(): p.requires_grad = False
                for _, mod in mods[idx:]:
                    for p in mod.parameters(): p.requires_grad = True
        # Optimizer builder
        def reset_optimizer():
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            return optim.SGD(params, lr=0.001, momentum=0.9)
        self.optimizer = reset_optimizer()
        self.model.train()

        # Initialize best loss trackers
        best_train_loss = float('inf')
        best_val_loss = float('inf')

        best_acc = -float('inf')
        no_improve = 0
        unfrozen = False
        history = deque(maxlen=variation_patience)

        for epoch in range(1, epochs+1):
            train_loss, correct, total = 0.0, 0, 0
            bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=not self.verbose)
            for x, y in bar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                with autocast(): out = self.model(x)
                loss = self.criterion(out, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                train_loss += loss.item() * y.size(0)
                preds = out.argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
                if self.verbose:
                    bar.set_postfix({'loss': train_loss/total, 'acc': correct/total})
            avg_train_loss = train_loss / total

            # Save weights for best train loss
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(self.model.state_dict(), train_ckpt_path)
                if self.verbose: print(f"[INFO] New best train loss: {best_train_loss:.4f}. Saving to {train_ckpt_path}.")

            val_loss, val_acc = self.evaluate_model(eval_loader, generate_log=False)
            # Save weights for best val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), val_ckpt_path)
                if self.verbose: print(f"[INFO] New best val loss: {best_val_loss:.4f}. Saving to {val_ckpt_path}.")

            if self.verbose: print(f"[INFO] Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # Staged unfreeze logic applies to both freeze modes
            if (freeze_conv_only or freeze_pre_last_conv) and not unfrozen:
                if val_acc <= best_acc:
                    no_improve += 1
                else:
                    best_acc, no_improve = val_acc, 0
                if no_improve >= unfreeze_patience:
                    if self.verbose: print(f"[INFO] No improvement for {unfreeze_patience} epochs. Unfreezing all layers.")
                    for p in self.model.parameters(): p.requires_grad = True
                    self.optimizer = reset_optimizer()
                    unfrozen = True
                    history.clear()
                    continue
            else:
                history.append(val_acc)
                if len(history) == variation_patience and (max(history)-min(history) < variation_threshold):
                    if self.verbose: print("[INFO] Accuracy variation below threshold. Stopping.")
                    break
                if val_acc <= best_acc:
                    no_improve += 1
                else:
                    best_acc, no_improve = val_acc, 0
                if no_improve >= unfreeze_patience:
                    if self.verbose: print(f"[INFO] No improvement for {unfreeze_patience} epochs. Stopping.")
                    break

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Append full epoch stats
            self.hist.append([avg_train_loss, correct/total, val_loss, val_acc])

        if self.verbose: print("[INFO] Training complete.")