import torch
import copy
from datetime import datetime


class Approach:
    def __init__(self, device, model, nepochs, lr, logger):
        self.device = device
        self.model = model.to(device)
        self.nepochs = nepochs
        self.lr = lr
        self.logger = logger
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self._time = datetime.now().strftime("%H:%M")
        self._best_state = None

    def train(self, trn_loader, val_loader):
        best_loss = 999.
        for epoch in range(self.nepochs):
            t_loss = self.train_epoch(trn_loader)
            loss = self.eval(val_loader)

            self.logger.add_scalar(tag="Loss/train", scalar_value=t_loss, global_step=epoch)
            self.logger.add_scalar(tag="Loss/val", scalar_value=loss, global_step=epoch)
            print(f"Epoch {epoch} | Val loss: {loss:.4f} | Train loss: {t_loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                self._best_state = copy.deepcopy(self.model.state_dict())

    def train_epoch(self, trn_loader):
        self.model.train()
        total_loss = 0.
        for batch_id, data in enumerate(trn_loader):
            self.optimizer.zero_grad()

            loss = self._forward(data)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

        final_loss = total_loss / len(trn_loader)
        return final_loss

    def eval(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.
            for batch_id, data in enumerate(val_loader):
                loss = self._forward(data)
                total_loss += loss.item()

        final_loss = total_loss / len(val_loader)
        return final_loss

    def _forward(self, data):
        pass
