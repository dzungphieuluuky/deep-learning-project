import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class Evaluator:
    @staticmethod
    def compute_metrics(predictions: np.ndarray, labels: np.ndarray):
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    @staticmethod
    def predict(model, dataloader, device: str = "cpu"):
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        return np.array(predictions)