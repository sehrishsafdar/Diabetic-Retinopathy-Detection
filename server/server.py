import flwr as fl
import csv
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

# ✅ List to store evaluation metrics per round
round_logs = []

# ✅ Save evaluation metrics to CSV after training
def write_metrics_to_csv():
    with open("training_metrics.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy", "Precision", "Recall", "F1"])
        for row in round_logs:
            writer.writerow(row)

# ✅ Aggregation function for evaluation metrics
def aggregate_eval_metrics(metrics):
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_examples = 0

    for num_examples, metric in metrics:
        if isinstance(metric, dict):
            total_accuracy += metric.get("accuracy", 0.0) * num_examples
            total_precision += metric.get("precision", 0.0) * num_examples
            total_recall += metric.get("recall", 0.0) * num_examples
            total_f1 += metric.get("f1", 0.0) * num_examples
            total_examples += num_examples
        else:
            print(f"[Warning] Skipping non-dict metric: {metric}")

    if total_examples == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy": total_accuracy / total_examples,
        "precision": total_precision / total_examples,
        "recall": total_recall / total_examples,
        "f1": total_f1 / total_examples,
    }

# ✅ Aggregation function for training (fit) metrics
def aggregate_fit_metrics(metrics):
    total_accuracy = 0.0
    total_loss = 0.0
    total_examples = 0

    for num_examples, metric in metrics:
        if isinstance(metric, dict):
            total_accuracy += metric.get("accuracy", 0.0) * num_examples
            total_loss += metric.get("loss", 0.0) * num_examples
            total_examples += num_examples
        else:
            print(f"[Warning] Skipping non-dict fit metric: {metric}")

    if total_examples == 0:
        return {"accuracy": 0.0, "loss": 0.0}

    return {
        "accuracy": total_accuracy / total_examples,
        "loss": total_loss / total_examples,
    }

# ✅ Custom FedAvg Strategy with logging
class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated is not None:
            _, metrics = aggregated
            if metrics is not None:
                acc = metrics.get("accuracy", 0.0)
                prec = metrics.get("precision", 0.0)
                rec = metrics.get("recall", 0.0)
                f1 = metrics.get("f1", 0.0)
                print(f"[Server] Round {server_round} - Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
                round_logs.append([server_round, acc, prec, rec, f1])
            else:
                print(f"[Server] Round {server_round} - No metrics returned.")
        else:
            print(f"[Server] Round {server_round} - No aggregation result.")

        return aggregated


    def configure_fit(self, server_round, parameters, client_manager):
        config = {"epochs": 1}
        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.min_fit_clients)
        return [(client, fit_ins) for client in clients]

# ✅ Start the FL server
def start_server():
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,  # ✅ Added to remove warning
    )

    config = ServerConfig(num_rounds=5)

    fl.server.start_server(
        server_address="localhost:8080",
        config=config,
        strategy=strategy,
    )

    write_metrics_to_csv()

# ✅ Entry point
if __name__ == "__main__":
    start_server()
