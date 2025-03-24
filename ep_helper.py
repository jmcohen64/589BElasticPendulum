import elastic_pendulum as ep
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

g, k, m, L0, epsilon = 9.81, 1.0, 1.0, 10.0, -1

initial_conditions = np.array([
    [1.0, np.pi/6, 0.0, 0.0],
    [1.2, np.pi/4, 0.0, 0.0],   
    [0.8, np.pi/3, 0.5, 0.0],
    [1.1, np.pi/2, 0.0, -0.5]]).transpose()

ic_dict = {1: "r = 1.0, Î = π/6" , 1: "r = 1.2, Î = π/4", 1: "r = 0.8, Î = π/3", 1: "r = 1.1, Î = π/2"}

perturbations = np.array([
 [1, 1, 1, 1],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]])

def poincare_event(t, Y, g, k, m, L0, epsilon):
    """
    Event function for Poincaré section: Detects zero crossings of theta (Y[1])
    only when dtheta/dt (Y[3]) is positive.
    """
    theta = Y[1]   # Extract theta
    dtheta = Y[3]  # Extract dtheta/dt

    return theta if dtheta > 0 else -theta  # Only consider upward crossings
     
# Ensure the solver detects only crossings where dz/dt > 0
poincare_event.terminal = False  # Do not stop integration at crossings
poincare_event.direction = 1  # Detect only upward crossings

# Check if dimensions are consistent for vectorized solver
assert ep.elastic_pendulum(1, initial_conditions, g, k, m, L0, epsilon).shape \
    == initial_conditions.shape

tmax = 100
n_eval = 1000
t_span, t_eval = (0, tmax), np.linspace(0, tmax, n_eval)
sol = solve_ivp(ep.elastic_pendulum,
                t_span,
                initial_conditions.flatten(),
                args=(g, k, m, L0, epsilon),
                t_eval=t_eval,
                method='RK45',
                atol=1e-12,
                rtol=1e-12,
                vectorized=True)

n_ic = initial_conditions.shape[1]
Y = sol.y.reshape(4, n_ic, -1)

E = ep.energy(Y, n_ic, g, k, m, L0, epsilon)

labels = ["Case 1" , "Case 2", "Case 3", "Case 4"]

#plot energy
plt.figure(figsize=(10, 5))
for i in range(n_ic):
    plt.plot(sol.t, E[i], label = ic_dict.get(i) )  # Plot energy for each IC

plt.xlabel("Time (s)")
plt.ylabel("Total Energy (J)")
plt.title("Total Energy of the Elastic Pendulum Over Time")
plt.legend(labels)
plt.grid()
#plt.show()

#for i in range(n_ic):
#    pass
#    #assert np.max(abs(E[i] - E[i][0])) < 1e-8

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

#Plot trajectory
for j in range(n_ic):
    axes[0].plot(sol.t, Y[0][j], label = ic_dict.get(j) )
    axes[1].plot(sol.t, Y[1][j], label = ic_dict.get(j) )

axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Radial Distance r (m)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Angle \u03B8¸ (rad)")

axes[0].grid()
axes[1].grid()
plt.suptitle(
    "Elastic Pendulum Motion for Different Initial Conditions (Vectorized)"
)
plt.legend(labels)
#plt.show()

initial_conditions=np.array([
            [1],
            [0.52359878],
            [0],
            [0]])

tmax = 5000
n_eval = 50000
t_span, t_eval = (0, tmax), np.linspace(0, tmax, n_eval)

sol1 = solve_ivp(ep.elastic_pendulum,
                t_span,
                initial_conditions.flatten(),
                args=(g, k, m, L0, epsilon),
                t_eval=t_eval,
                method='RK45',
                atol=1e-12,
                rtol=1e-12,
                events=poincare_event,
                vectorized= True)

# Extract Poincar\'e section points
print(sol1.y_events)
poincare_r = sol1.y_events[0][:, 0]
poincare_dr = sol1.y_events[0][:, 2]

# Plot improved Poincar\'e section
plt.figure(figsize=(8, 6))
plt.scatter(poincare_r, poincare_dr, s=1, color='blue', alpha=0.7)
plt.xlabel("r")
plt.ylabel("dr/dt")
plt.title("Poincare Section of the Elastic Pendulum (\u03B8 = 0, d\u03B8/dt > 0)")
plt.grid(True)
plt.show()
#plt.savefig("lorenz_poincare_event.png", dpi=300, bbox_inches='tight')
#plt.close()


"""
def plot_perturbation_growth(sol, n_ic):
    
    Plots the norm of perturbation growth over time for each initial condition.
    
    Parameters:
        sol : OdeSolution object from solve_ivp
        n_ic : int, Number of initial conditions
    
    t = sol.t  # Time points
    dy_sol = sol.y[4:].reshape(4, n_ic, 4, -1)  # Reshape perturbation part

    # Compute norm of perturbation over time (Frobenius norm)
    norm_dy = np.linalg.norm(dy_sol, axis=(0, 2))  # Shape: (n_ic, n_eval)

    # Plot perturbation norm evolution for each initial condition
    plt.figure(figsize=(10, 6))
    for i in range(n_ic):
        plt.plot(t, norm_dy[i], label=f"IC {i+1}")

    plt.yscale("log")  # Log scale to see exponential growth
    plt.xlabel("Time (s)")
    plt.ylabel("Perturbation Norm ||δY||")
    plt.title("Perturbation Growth Over Time")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage after calling compute_lyapunov_exponent:
plot_perturbation_growth(sol, n_ic)"
#plot energy
plt.figure(figsize=(10, 5))
for i in range(n_ic):
    plt.plot(sol.t, E[i], label=f'Case {i+1}')  # Plot energy for each IC

plt.xlabel("Time (s)")
plt.ylabel("Total Energy (J)")
plt.title("Total Energy of the Elastic Pendulum Over Time")
plt.legend()
plt.grid()
plt.show()

for i in range(n_ic):
    pass
    #assert np.max(abs(E[i] - E[i][0])) < 1e-8
#Plot trajectory
for j in range(n_ic):
    axes[0].plot(sol.t, Y[0][j])
    axes[1].plot(sol.t, Y[1][j])

axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Radial Distance r (m)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Angle Î¸ (rad)")

axes[0].grid()
axes[1].grid()
plt.suptitle(
    "Elastic Pendulum Motion for Different Initial Conditions (Vectorized)"
)
plt.show()
"""


"""



# Example usage with initial conditions
initial_conditions = np.array([
    [1.0, np.pi/6, -0.1, np.pi/2]
]).T  # (4,1) shape

verify_jacobian(initial_conditions, g, k, m, L0, epsilon)


initial_conditions=np.array([
            [1, 1],
            [0.52359878, 1.04719755],
            [0, 0],
            [0, 0]])
perturbations=np.array([
 [1, 1],
 [0, 0],
 [0, 0],
 [0, 0]])
"""
initial_conditions=np.array([
            [1],
            [0.52359878],
            [0],
            [0]])
perturbations=np.array([
 [1],
 [0],
 [0],
 [0]])


# Generate perturbations dynamically for the number of initial conditions
if len(initial_conditions.shape) > 1:
    n_ic = initial_conditions.shape[1]  # Number of initial conditions
else:
    n_ic = 1

# Compute Lyapunov exponents for all initial conditions
#lyap_exps = ep.compute_lyapunov_exponent(initial_conditions, perturbations, (g, k, m, L0, epsilon))
#print(debug_log)
#print(f"Lyapunov Exponents for {n_ic} cases: {lyap_exps}")