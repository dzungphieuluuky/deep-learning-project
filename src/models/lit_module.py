import lightning as L
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.optim import AdamW

class LitTransformerModule(L.LightningModule):
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2, learning_rate: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log("test_loss", loss)
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)