import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import optax

from src.experiments.gbif_jax.dataset import load_dataloader


class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, num=4)

        self.layers = [
            eqx.nn.Conv2d(3, 2, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            jax.nn.relu,
            eqx.nn.Conv2d(2, 1, kernel_size=4, key=key2),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(3364, 1024, key=key2),
            jax.nn.swish,
            eqx.nn.Linear(1024, 512, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(512, 663, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@eqx.filter_jit
def loss_fn(model, x, y):
    logits = jax.vmap(model)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@eqx.filter_jit
def compute_accuracy(model, x, y):
    logits = jax.vmap(model)(x)
    y_pred = jnp.argmax(logits, axis=1)
    return jnp.mean(y_pred == y)


def evaluate(model, testloader):
    avg_loss = jnp.array(0)
    avg_acc = jnp.array(0)

    for imgs, labels in testloader:
        avg_loss += loss_fn(model, imgs, labels)
        avg_acc += compute_accuracy(model, imgs, labels)

    return avg_loss / len(testloader), avg_acc / len(testloader)


def train(model, trainloader, testloader, steps, learning_rate, print_every=5):
    optim = optax.adamw(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            updates=grads, state=opt_state, params=eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        mlflow.log_metric("train_loss", train_loss.item())

        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            mlflow.log_metric("test_accuracy", test_accuracy.item())
            mlflow.log_metric("test_loss", test_loss.item())

            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Training script for gbif datasets",
        description="Train various models using different configurations using gbif datasets",
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, required=False, type=float
    )
    parser.add_argument("-stps", "--steps", default=100, required=False, type=int)
    parser.add_argument("-pe", "--print-every", default=5, required=False, type=int)
    parser.add_argument("-sd", "--seed", default=42, required=False, type=int)
    parser.add_argument("-bs", "--batch-size", default=16, required=False, type=int)
    parser.add_argument(
        "-name", "--experiment-name", default="gbif", required=False, type=str
    )
    return parser.parse_args()


def run(args):
    trainloader = load_dataloader("train", batch_size=args.batch_size)
    testloader = load_dataloader("test", batch_size=args.batch_size)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(args.experiment_name)

    key = jax.random.key(seed=args.seed)
    model = CNN(key)

    with mlflow.start_run():
        params = {
            key: value
            for key, value in vars(args).items()
            if key not in ["print_every", "experiment_name"]
        }

        for k, v in params.items():
            mlflow.log_param(k, v)

        train(
            model,
            trainloader,
            testloader,
            learning_rate=args.learning_rate,
            steps=args.steps,
            print_every=args.print_every,
        )


if __name__ == "__main__":
    args = parse_args()
    run(args)
