import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Fungsi Fitness (Sphere Function)
# ==============================
def fitness(position):
    return position[0]**2 + position[1]**2

# ==============================
# Parameter PSO
# ==============================
num_particles = 30
max_iter = 100

w = 0.7
c1 = 2.0
c2 = 2.0

dim = 2

# ==============================
# Inisialisasi Partikel
# ==============================
positions = np.random.uniform(-10, 10, (num_particles, dim))
velocities = np.random.uniform(-1, 1, (num_particles, dim))

pbest_positions = positions.copy()
pbest_scores = np.array([fitness(p) for p in positions])

gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index].copy()
gbest_score = pbest_scores[gbest_index]

history = []

# ==============================
# Proses PSO
# ==============================
for iteration in range(max_iter):

    for i in range(num_particles):

        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )

        positions[i] = positions[i] + velocities[i]

        score = fitness(positions[i])

        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i].copy()

            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()

    history.append(gbest_score)

# ==============================
# Menampilkan Hasil
# ==============================
print("=" * 40)
print(" PARTICLE SWARM OPTIMIZATION ")
print("=" * 40)

print("\nBest Position")
print("X =", round(gbest_position[0], 6))
print("Y =", round(gbest_position[1], 6))

print("\nBest Fitness")
print(gbest_score)

print("=" * 40)

# ==============================
# Visualisasi
# ==============================
plt.figure(figsize=(8,5))
plt.plot(history, linewidth=2)
plt.title("PSO Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()