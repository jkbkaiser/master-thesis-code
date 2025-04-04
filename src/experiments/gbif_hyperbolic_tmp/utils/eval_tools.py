import torch
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
from ..embeddings.base import BaseEmbedding

def distortion(H1,H2,n):
    dist_score = 0
    for i in range(n):
        if H1[i] != 0 and H2[i] != 0:
            dist_score += abs(H1[i] - H2[i])/H1[i]
    # for i in range(n):
    #     for j in range(n):
    #         if H1[i,j] != 0 and H2[i,j] != 0:
    #             dist_score += abs(H1[i,j] - H2[i,j])/H1[i,j]
    
    return dist_score / (n*n) 

def evaluate_edge_predictions(model: BaseEmbedding, dataloader: DataLoader) -> None:
    device = model.weight.device

    scores, labels = torch.empty(0).to(device), torch.empty(0).to(device)

    with torch.no_grad():
        for batch in dataloader:
            edges = batch["edges"].to(device)
            edge_label_targets = batch["edge_label_targets"].to(device)

            batch_scores = model.score(edges)
            flattened_scores = batch_scores.flatten()
            scores = torch.cat([scores, flattened_scores])

            flattened_targets = edge_label_targets.flatten()
            labels = torch.cat([labels, flattened_targets])

    precisions, recalls, thresholds = precision_recall_curve(
        y_true=labels.cpu(),
        probas_pred=scores.cpu(),
    )
    # print(precisions, recalls)
    # print(torch.unique(labels.cpu()), scores.cpu().shape)
    # map = average_precision_score(y_true=labels.cpu(), y_score=scores.cpu())
    # maps = np.average(precisions)
    # print(f"Mean average precision: {map:.2%}, {maps:.2%}")
    # print(labels.shape, scores.shape)
    # hdist = distortion(labels.cpu().numpy(), scores.cpu().numpy(), len(labels))
    # print(f"Distortion: {hdist:.5%}")
    max_f1_score = -1

    for precision, recall, threshold in zip(precisions, recalls, thresholds):
        if precision + recall == 0:
            continue
        f1_score = 2 * precision * recall / (precision + recall)
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            max_recall = recall
            max_precision = precision
            opt_threshold = threshold

    print(f"Maximal F1-score of {max_f1_score:.2%} at threshold {opt_threshold:.4}")
    print(f"Recall and precision at maximal F1-score: {max_recall:.2%} & {max_precision:.2%}")