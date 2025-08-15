import os, json
from pathlib import Path
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Optional
import mlflow
from flwr.common import ndarrays_to_parameters
from common.model import get_model  # you have get_model(input_shape=...)

INIT_DIR = Path(os.getenv("INIT_MODEL_DIR", "/models/baseline"))

def load_initial_parameters():
    """Build the same model, load centralized weights, convert to Flower Parameters."""
    weights_path = INIT_DIR / "model.weights.h5"  # saved by your centralized script
    meta_path = INIT_DIR / "meta.json"

    if not weights_path.exists() or not meta_path.exists():
        print(f"[warm-start] No baseline found at {INIT_DIR}. Starting from random init.")
        return None

    meta = json.load(open(meta_path))
    input_dim = int(meta["input_dim"])

    # rebuild the exact architecture
    model = get_model(input_shape=(input_dim,))
    # (Keras models with InputLayer are already built; if you ever change that, uncomment:)
    # model.build((None, input_dim))
    model.load_weights(str(weights_path))

    nds = [w.astype(np.float32) for w in model.get_weights()]
    print(f"[warm-start] Loaded {len(nds)} weight tensors from {weights_path.name}")
    return ndarrays_to_parameters(nds)

# ✅ Set up MLflow for server aggregation
mlflow.set_tracking_uri("http://mlflow:5000")  # internal service name from docker-compose
mlflow.set_experiment("server_aggregation")

# ✅ Custom strategy with MLflow logging
class FedAvgWithMLflow(FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ):
        # Aggregate using default FedAvg
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(rnd, results, failures)

        # Collect metrics from clients
        accs = []
        losses = []
        for _, fit_res in results:
            if "accuracy" in fit_res.metrics:
                accs.append(fit_res.metrics["accuracy"])
            if "loss" in fit_res.metrics:
                losses.append(fit_res.metrics["loss"])

        # Compute global metrics
        global_acc = sum(accs) / len(accs) if accs else 0.0
        global_loss = sum(losses) / len(losses) if losses else 0.0

        print(f"[Server] ✅ Round {rnd} global accuracy: {global_acc:.4f}, global loss: {global_loss:.4f}")

        # ✅ Log to MLflow
        with mlflow.start_run(run_name=f"server_round_{rnd}", nested=True):
            mlflow.log_metric("global_accuracy", float(global_acc))
            mlflow.log_metric("global_loss", float(global_loss))
            mlflow.log_param("round", rnd)

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException]
    ):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Collect evaluation metrics
        eval_accs = []
        eval_losses = []
        for _, eval_res in results:
            if "accuracy" in eval_res.metrics:
                eval_accs.append(eval_res.metrics["accuracy"])
            if "loss" in eval_res.metrics:
                eval_losses.append(eval_res.metrics["loss"])

        # Compute global evaluation metrics
        global_eval_acc = sum(eval_accs) / len(eval_accs) if eval_accs else 0.0
        global_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0.0

        print(f"[Server] 📊 Round {rnd} eval accuracy: {global_eval_acc:.4f}, eval loss: {global_eval_loss:.4f}")

        # ✅ Log evaluation metrics
        with mlflow.start_run(run_name=f"server_eval_round_{rnd}", nested=True):
            mlflow.log_metric("global_eval_accuracy", float(global_eval_acc))
            mlflow.log_metric("global_eval_loss", float(global_eval_loss))
            mlflow.log_param("round", rnd)

        return aggregated_loss, aggregated_metrics


def main():

    initial_params = load_initial_parameters()

    # ✅ Use our custom strategy
    strategy = FedAvgWithMLflow(
        initial_parameters=initial_params,
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

    # ✅ Start Flower server
    print("[Server] 🚀 Starting Flower server with MLflow logging...")
    fl.server.start_server(
        server_address="[::]:8085",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
