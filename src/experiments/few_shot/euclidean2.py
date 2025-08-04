import argparse
import ast
import os
import random
from collections import defaultdict
from pathlib import Path

import datasets
import mlflow
import torch
import torchvision.transforms as transforms
import tqdm
from datasets import Dataset
from dotenv import load_dotenv
from geoopt import PoincareBallExact

from src.constants import DEVICE
from src.experiments.bioscan_hyperbolic.lighting import LightningGBIF
from src.shared.datasets import DatasetVersion
from src.shared.datasets.bioscan import BioscanDataset

load_dotenv()

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]


mlflow.set_tracking_uri(MLFLOW_SERVER)
client = mlflow.tracking.MlflowClient()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def sample_few_shot_task(dataset: Dataset, label_field: str, n_way: int, n_shot: int, n_query: int):
    class_to_samples = defaultdict(list)
    for example in dataset:
        class_to_samples[example[label_field]].append(example)

    # Filter to classes with enough samples
    eligible_classes = [cls for cls, samples in class_to_samples.items() if len(samples) >= n_shot + n_query]
    chosen_classes = random.sample(eligible_classes, n_way)

    support, query = [], []
    for cls in chosen_classes:
        examples = random.sample(class_to_samples[cls], n_shot + n_query)
        for example in examples:
            image_tensor = transform(example["image"])
            example["image"] = image_tensor

        support.extend(examples[:n_shot])
        query.extend(examples[n_shot:])

    return support, query


def compute_prototypes(model, support_set, label_field: str, ball: PoincareBallExact):
    label_to_embeddings = defaultdict(list)

    for item in support_set:
        image = item['image'].to(DEVICE)
        label = item[label_field]
        with torch.no_grad():
            embedding = model.embed(image.unsqueeze(0))[-1]
        label_to_embeddings[label].append(embedding.squeeze(0))

    prototypes = {}
    for label, embeds in label_to_embeddings.items():
        embeds_tensor = torch.stack(embeds, dim=0)
        weights = torch.ones(embeds_tensor.size(0), device=embeds_tensor.device)
        midpoint = ball.weighted_midpoint(embeds_tensor, weights=weights)
        prototypes[label] = midpoint
    return prototypes


def load_model(run_id, prototypes):
    artifact_path = "epoch=19/epoch=19.ckpt"
    local_dir = Path("mlruns_cache") / run_id
    local_ckpt_path = local_dir / artifact_path

    if not local_ckpt_path.exists():
        print(f"Downloading checkpoint to: {local_ckpt_path}")

        local_dir.mkdir(parents=True, exist_ok=True)
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=local_dir,
        )
    else:
        print(f"Using cached checkpoint at: {local_ckpt_path}")

    run = client.get_run(run_id)
    params = run.data.params

    model_hparams = {
        "backbone_name": params["backbone_name"],
        "freeze_backbone": params["freeze_backbone"],
        "prototypes": prototypes,
        "architecture": ast.literal_eval(params["architecture"]),
    }

    ds = BioscanDataset(DatasetVersion.BIOSCAN)
    ds.load(batch_size=16, use_torch=True)

    model = LightningGBIF.load_from_checkpoint(
        checkpoint_path=local_ckpt_path,
        model_hparams=model_hparams,
        optimizer_name=None,
        optimizer_hparams=None,
        ds=ds,
    ).to(DEVICE)
    model.eval()

    return model


def evaluate_query_set(model, query_set, prototypes, label_field, ball):
    proto_labels = list(prototypes.keys())
    proto_tensors = torch.stack([prototypes[label] for label in proto_labels])  # shape: [N_way, D]

    correct = 0
    total = len(query_set)
    average_precisions = []

    for example in query_set:
        image = example["image"].to(DEVICE)
        true_label = example[label_field]

        query_embed = model.embed(image.unsqueeze(0))[-1]  # [D]
        dists = ball.dist(query_embed.unsqueeze(0), proto_tensors).squeeze(0)  # shape: [N_way]

        pred_idx = torch.argmin(dists).item()
        pred_label = proto_labels[pred_idx]
        if pred_label == true_label:
            correct += 1

        sorted_indices = torch.argsort(dists)  # low distance = high rank
        sorted_labels = [proto_labels[idx] for idx in sorted_indices]

        if true_label in sorted_labels:
            rank = sorted_labels.index(true_label)
            average_precisions.append(1 / (rank + 1))  # Precision at the true label
        else:
            average_precisions.append(0.0)

    acc = correct / total
    mean_ap = sum(average_precisions) / total
    return acc, mean_ap


def run(args):
    run_id = args.run_id
    prototypes = args.prototypes
    n_way = args.n_way
    n_shot = args.n_shot
    n_query = args.n_query

    mlflow.set_experiment("few-shot")

    dataset_dict = datasets.load_dataset("jkbkaiser/clibdb_unseen")
    full_dataset = datasets.concatenate_datasets([dataset_dict["validation"], dataset_dict["test"]])
    ball = PoincareBallExact(c=1.5)

    model = load_model(run_id, prototypes)

    accuracies = []
    aps = []

    progress = tqdm.tqdm(range(500))

    with mlflow.start_run():
        mlflow.log_params({
            "prototypes": prototypes,
            "dataset": "clibdb",
            "run_id": run_id,
            "n_way": n_way,
            "n_shot": n_shot,
            "n_query": n_query,
        })

        for _ in progress:
            support, query = sample_few_shot_task(full_dataset, "species", n_way=n_way, n_shot=n_shot, n_query=n_query)
            prototypes = compute_prototypes(model, support, "species", ball)
            acc, ap = evaluate_query_set(model, query, prototypes, "species", ball)

            aps.append(ap)
            accuracies.append(acc)

            avg_acc = sum(accuracies) / len(accuracies)
            avg_ap = sum(aps) / len(aps)

            mlflow.log_metrics({
                "acc": acc,
                "ap": ap,
                "avg_acc": avg_acc,
                "avg_ap": avg_ap,
            })

            progress.set_description(f"Avg Acc: {avg_acc:.4f} Avg AP: {avg_ap:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot evaluation with hyperbolic prototypes")

    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID to load the model from")
    parser.add_argument("--prototypes", type=str, required=True, help="Name or path to prototype file")
    parser.add_argument("--n_way", type=int, default=5, help="Number of classes per few-shot task")
    parser.add_argument("--n_shot", type=int, default=1, help="Number of support examples per class")
    parser.add_argument("--n_query", type=int, default=15, help="Number of query examples per class")
    parser.add_argument("--n_tasks", type=int, default=500, help="Number of few-shot tasks to evaluate")
    parser.add_argument("--dataset", type=str, default="jkbkaiser/clibdb_unseen", help="Dataset name or path")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)

