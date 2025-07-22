import csv
import matplotlib.pyplot as plt

rounds = []
accuracies = []
losses = []

with open("training_metrics.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rounds.append(int(row["Round"]))
        accuracies.append(float(row["Accuracy"]))
        losses.append(float(row["Loss"]))

plt.figure(figsize=(10, 4))

# ✅ Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(rounds, accuracies, marker='o', color='green')
plt.title("Federated Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")

# ✅ Loss plot
plt.subplot(1, 2, 2)
plt.plot(rounds, losses, marker='x', color='red')
plt.title("Federated Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("metrics_plot.png")
plt.show()
