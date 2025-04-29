import pytorch_lightning as pl

class TFTLightningModule(pl.LightningModule):
    def __init__(self, tft):
        super().__init__()
        # self.tft = tft
        self.model = tft
        self.train_losses_per_epoch = []
        self.validation_losses_per_epoch = []
        # self.save_hyperparameters(ignore=["model", "loss", "logging_metrics"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.model.loss(y_hat["prediction"][:, :, 0], y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True,
                 batch_size=x['encoder_target'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        # print(y_hat["prediction"][:, :, 0])
        # print(y)
        # loss = MeanSquaredError().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))(y_hat["prediction"][:, :, 0], y[0])
        loss = self.model.loss(y_hat["prediction"][:, :, 0], y)
        self.log("val_loss", loss, prog_bar=True, batch_size=x['encoder_target'].shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        # Perform a single test step
        x, y = batch
        out = self.model(x)
        y_pred = out[0]  # The model output is a tuple, 'y_pred' is the first element
        # loss = MeanSquaredError().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))(y_pred, y)
        loss = self.model.loss(y_pred["prediction"][:, :, 0], y)
        self.log("test_loss", loss, prog_bar=True, batch_size=x['encoder_target'].shape[0])
        return loss

    def on_train_epoch_end(self):
        self.train_losses_per_epoch.append(self.trainer.callback_metrics['train_loss'].item())
        self.validation_losses_per_epoch.append(self.trainer.callback_metrics['val_loss'].item())

    def configure_optimizers(self):
        return self.model.configure_optimizers()


class ModelLightningModule(pl.LightningModule):
    def __init__(self, tft):
        super().__init__()
        # self.tft = tft
        self.model = tft
        self.train_losses_per_epoch = []
        self.validation_losses_per_epoch = []
        # self.save_hyperparameters(ignore=["model", "loss", "logging_metrics"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.model.loss(y_hat["prediction"], y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True,
                 batch_size=x['encoder_target'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        # print(y_hat["prediction"][:, :, 0])
        # print(y)
        # loss = MeanSquaredError().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))(y_hat["prediction"][:, :, 0], y[0])
        loss = self.model.loss(y_hat["prediction"], y)
        self.log("val_loss", loss, prog_bar=True, batch_size=x['encoder_target'].shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        # Perform a single test step
        x, y = batch
        out = self.model(x)
        y_pred = out[0]  # The model output is a tuple, 'y_pred' is the first element
        # loss = MeanSquaredError().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))(y_pred, y)
        loss = self.model.loss(y_pred["prediction"], y)
        self.log("test_loss", loss, prog_bar=True, batch_size=x['encoder_target'].shape[0])
        return loss

    def on_train_epoch_end(self):
        self.train_losses_per_epoch.append(self.trainer.callback_metrics['train_loss'].item())
        self.validation_losses_per_epoch.append(self.trainer.callback_metrics['val_loss'].item())

    def configure_optimizers(self):
        return self.model.configure_optimizers()
