import copy
import numpy as np
from typing import Any, Callable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from tqdm.notebook import tqdm
from utils.early_stopping import EarlyStopping
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model, loader: DataLoader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for inputs, outputs in loader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            predictions, _ = model(inputs)

            _, predicted = torch.max(predictions.data, 1)

            # total += outputs.size(0)
            # correct += (predicted == outputs).sum().item()

        # accuracy = 100 * correct / total
        accuracy = 0
        print("Accuracy: {:.2f} %".format(accuracy))


def validate_model(model, criterion, loader: DataLoader):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        total = 0
        correct = 0

        # per window
        for inputs, outputs in loader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            predictions = model(inputs, target_length=outputs.size(1))

            loss = criterion(predictions, outputs)
            val_loss += loss.item()
            # _, predicted = torch.max(predictions.data, 1)
            # total += outputs.size(0)
            # correct += (predicted == outputs).sum().item()

        # val_acc = correct / total
        val_acc = 0

    return val_loss, val_acc


def infer(model, loader: DataLoader):
    all_inputs = []
    all_outputs = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        # per window
        for inputs, outputs in loader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            predictions = model(inputs, target_length=outputs.size(1))

            all_inputs.extend(inputs.cpu().detach().numpy())
            all_outputs.extend(outputs.cpu().detach().numpy())
            all_predictions.extend(predictions.cpu().detach().numpy())

    return np.array(all_inputs), np.array(all_outputs), np.array(all_predictions)


class Trainer:
    def __init__(
        self,
        name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        early_stop: EarlyStopping | None,
        scheduler: Callable[[Any], None] | None = None,
        teacher_forcing_phase: int = 0,
    ):
        self.name = name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stop = early_stop
        self.scheduler = scheduler
        self.teacher_forcing_phase = teacher_forcing_phase

        self.epochs = 0
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def fit(self, num_epochs: int):
        for epoch in tqdm(range(self.epochs, self.epochs + num_epochs)):
            epoch_loss = 0
            epoch_acc = 0
            num_batches = len(self.train_loader)
            model_kwargs = {}

            if self.teacher_forcing_phase > 0:
                model_kwargs["teacher_forcing_ratio"] = max(
                    0.5, 1 - (self.epochs / self.teacher_forcing_phase)
                )

            self.model.train()
            for _, (inputs, outputs) in enumerate(self.train_loader):
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                predictions = self.model(inputs, outputs, **model_kwargs)
                loss = self.criterion(predictions, outputs)

                epoch_loss += loss.item()
                # epoch_acc += (predictions.argmax(1) == outputs).float().mean().item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = epoch_loss / num_batches
            # avg_acc = epoch_acc / num_batches
            self.train_losses.append(avg_loss)
            # self.train_accs.append(avg_acc)

            val_loss, val_acc = validate_model(
                self.model, self.criterion, self.val_loader
            )
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if self.early_stop is not None:
                self.early_stop(val_loss, self.model)

            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch+1}/{num_epochs}]  LR: {lr:.5f}  T_Loss: {avg_loss:.4f}   "
                f"V_Loss: {val_loss:.4f}"
            )

            if self.scheduler is not None:
                self.scheduler(val_loss)

            if self.early_stop is not None and self.early_stop.should_stop:
                print("Early stopping")
                break

    def plot(self):
        plt.title(f"{self.name} - Loss")
        plt.plot(self.train_losses, label="Training")
        plt.plot(self.val_losses, label="Validation")
        plt.legend()
        plt.xlabel(r"Epochs")
        plt.ylabel(r"Loss")
        plt.savefig(f"docs/plots/{self.name}-loss.pdf", format="pdf", dpi=300)
        plt.show()

        plt.title(f"{self.name} - Accuracy")
        plt.plot(self.train_accs, label="Training")
        plt.plot(self.val_accs, label="Validation")
        plt.legend()
        plt.xlabel(r"Epochs")
        plt.ylabel(r"Accuracy %")
        plt.savefig(f"docs/plots/{self.name}-acc.pdf", format="pdf", dpi=300)
        plt.show()

    def get_best_model(self):
        best_checkpoint = torch.load(self.early_stop.path)
        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(best_checkpoint)
        return best_model

    # def validate_best_checkpoint(self):
    #     best_checkpoint = torch.load(self.early_stop.path)
    #     val_loss, val_acc = validate_model(
    #         best_checkpoint, self.criterion, self.val_loader
    #     )
    #     print(f"Best checkpoint - Loss: {val_loss:.4f}   Accuracy: {val_acc:.2f}")
    #     return val_loss, val_acc

    # def final_test(self, test_loader: DataLoader):
    #     best_checkpoint = torch.load(self.early_stop.path)
    #     test_model(best_checkpoint, test_loader)

    def infer(self, loader: DataLoader):
        return infer(self.get_best_model(), loader)
