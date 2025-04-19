from torcheval.metrics.functional import binary_auprc, binary_accuracy, binary_precision, binary_recall, binary_f1_score, binary_auroc
from dataclasses import dataclass

@dataclass
class MetricResults:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0 
    
    def measure(self, mse_losses, pred_labels, labels, adjust=False, avoid_curves = False):
            
        self.accuracy = binary_accuracy(pred_labels, labels)
        self.precision = binary_precision(pred_labels, labels)
        self.recall = binary_recall(pred_labels, labels)
        self.f1_score = binary_f1_score(pred_labels, labels)
        
        if not avoid_curves:
            self.auroc = binary_auroc(mse_losses.float(), labels.int())
            self.auprc =  binary_auprc(mse_losses.float(), labels.int())
        
    def get_dict(self, avoid_curves = False):
        return {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1_score,
            "AUROC": self.auroc,
            "AUPRC": self.auprc
        } if not avoid_curves else {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1_score,
        }
        
    def __repr__(self):
        return ', '.join([f"{key}: {value:.4f}" for key, value in self.get_dict().items()])

class Metrics():
    def __init__(self, avoid_curves = False):
        self.avoid_curves = avoid_curves
        self._ongoing_results = []

    def reinit(self):
        self._ongoing_results = []

    def update(self, mse_losses, pred_labels, labels, adjust=False):
        new = MetricResults()
        new.measure(mse_losses, pred_labels, labels, adjust, self.avoid_curves)
        self._ongoing_results.append(new)
    
    def compute(self):
        if len(self._ongoing_results) == 0:
            return None
            
        result = MetricResults()
        
        for field in MetricResults.__dataclass_fields__.keys():
            if self.avoid_curves and field in ['auroc', 'auprc']:
                continue
            
            setattr(result, field, 
                    sum([getattr(res, field) for res in self._ongoing_results]) / len(self._ongoing_results))
        
        return result
    


if __name__ == "__main__":
    import torch

    mse_losses = torch.tensor([0.1, 0.4, 0.35, 0.8])
    pred_labels = torch.tensor([0, 0, 0, 1])
    labels = torch.tensor([0, 1, 0, 1])

    
    metrics = Metrics(avoid_curves=True)
    metrics.update(mse_losses, pred_labels, labels, adjust=False)
    average_results = metrics.compute()
    print(average_results)