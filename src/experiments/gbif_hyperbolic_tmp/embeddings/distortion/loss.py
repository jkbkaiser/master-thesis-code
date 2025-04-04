import numpy as np
import torch
from geoopt.manifolds import PoincareBallExact


def distortion_loss(
    embeddings: torch.Tensor, dist_targets: torch.Tensor, ball: PoincareBallExact, epoch:int, max_epoch:int
) -> torch.Tensor:
    print(dist_targets, embeddings)
    embedding_dists = ball.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
    dist_loss = (embedding_dists - dist_targets).abs() / dist_targets 

    norm_loss = compute_norm_loss(embeddings, dist_targets, ball, epoch, max_epoch)

    if dist_loss.isnan().any():
        print("break")

    return dist_loss.mean() + 0.01 * norm_loss.mean()
    

def compute_norm_loss(
        embeddings: torch.Tensor, dist_targets: torch.Tensor, ball: PoincareBallExact, epoch:int, max_epoch:int
        ) -> torch.Tensor:
    # tangent_vecs = ball.logmap0(embeddings)
    embedding_norm = ball.dist0(embeddings,keepdim=True) 
    # embedding_norm = ball.norm(embeddings,tangent_vecs,keepdim=True) 
    unique_even_dists = torch.unique(dist_targets[dist_targets % 2 == 0])
    all_even_embedding_norms = [embedding_norm[dist_targets == i] for i in unique_even_dists]
    mean_even_embedding_norms = [norm.mean() for norm in all_even_embedding_norms]
    # print(mean_even_embedding_norms)
    even_embedding_loss = torch.cat([(even_embedding_norms - mean_even_embedding_norms) for even_embedding_norms, mean_even_embedding_norms in zip(all_even_embedding_norms, mean_even_embedding_norms)])
    return (epoch/max_epoch) * even_embedding_loss.abs()
