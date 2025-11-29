import numpy as np
import matplotlib.pyplot as plt

baseline = np.load("baseline_queue.npy")
trained = np.load("trained_queue.npy")
masac = np.load("masac_graph_queue.npy")

plt.figure(figsize=(10,5))
plt.plot(baseline, label="Baseline (Fixed Cycle)", linewidth=2)
plt.plot(trained, label="Trained MARL Policy", linewidth=2)
plt.plot(masac,label ="MA-SAC Graph Critic")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("Traffic Congestion Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("queue_comparison.png")
plt.show()

print("Plot saved as queue_comparison.png")
