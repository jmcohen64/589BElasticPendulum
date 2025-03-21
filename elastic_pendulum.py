import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def elastic_pendulum(t, Y, g, k, m, L0, epsilon):
    r, theta, dr, dtheta = Y.reshape(4, -1)
    d2r = r * dtheta**2 + g * np.cos(theta) - (k/m) * (r - L0) + epsilon / r**2
    d2theta = -2 * dr * dtheta / r - (g / r) * np.sin(theta)
    dY = np.array([dr, dtheta, d2r, d2theta])
    return dY

def energy(Y, n_ic, g, k, m, L0, epsilon):
    r, theta, dr, dtheta = Y.reshape(4, n_ic, -1)
    T = 0.5 * m * (dr**2 + r**2 * dtheta**2)
    U = -m * g * r * np.cos(theta) + 0.5 * k * (r - L0)**2 + epsilon / r
    E = T + U
    return E

def elastic_pendulum_jacobian(t, Y, g, k, m, L0, epsilon):
    """
    given state variable Y
    returns jacobian matrix where each element is a numpy row vector; ie a (4 x 4 x n_ic) matrix
    """
    #convert state variable into a 4 x n_ic matrix; each row is associated with one basis vector and the ith element is the value of that basis for the ith IC
    r, theta, dr, dtheta = Y.reshape(4, -1)
    
    ddr_dr = np.array(dtheta**2 - k/m - 2*epsilon/(r**3))
    ddr_dtheta = np.array(-g*np.sin(theta))
    ddr_drdot = np.zeros_like(r)
    ddrd_thetadot = np.array(2*r*theta)
    
    zero = np.zeros_like(r)
    one = np.ones_like(r)

    ddtheta_dr = np.array(2*dr*dtheta / r**2)
    ddtheta_dtheta = np.array(-g/r*np.cos(theta))
    ddtheta_drdot = np.array(-2*dtheta/r) 
    ddtheta_dthetadot = np.array(-2*dr/r)

    DfY = np.array([[zero, zero, one, zero], [zero, zero, zero, one], [ddr_dr, ddr_dtheta, ddr_drdot, ddrd_thetadot], [ddtheta_dr, ddtheta_dtheta, ddtheta_drdot, ddtheta_dthetadot]])
    return DfY

def variational_IVP(t, Y, g, k, m, L0, epsilon):
    """
    Computes the variational equations for the elastic pendulum.
    Y contains both the system state and perturbation.
    Computes the variational equations for multiple initial conditions.
    """
    #n_ic = Y.shape[1]  # Number of initial conditions
    #y = Y[:4]  # Extract system state (4, n_ic)
    #dy = Y[4:].reshape(4, n_ic, n_ic)  # Extract perturbations (4, n_ic, n_ic)
    #print("Shape of dy:", dy.shape)

    Y = Y.reshape(8, -1)
    n_ic = Y.shape[1]
    y = Y[:4]
    dy = Y[4:]

    #print(y.shape)
    y_dot = elastic_pendulum(t, y, g, k, m, L0, epsilon)
    J = elastic_pendulum_jacobian(t, y, g, k, m, L0, epsilon)  # (4, 4, n_ic)

    #reshape dy to match J
    dy = dy.reshape(4,1,n_ic)
    dy_dot = np.zeros_like(dy)
    #print("J shape:", J.shape)
    #print("dy shape:", dy.shape)
    for i in range(dy.shape[2]):  # Loop over the third dimension or each condition
        dy_dot[:, :, i] = np.dot(J[:, :, i], dy[:, :, i])  # Matrix-vector multiplication at each step
    #print(dy_dot.shape)
    #collapse into 2d matrix
    dy_dot = dy_dot.reshape(4,n_ic)

    return np.vstack([y_dot, dy_dot]) 

def compute_lyapunov_exponent(initial_conditions, perturbations, params, t_max=100, dt=0.1):
    """
    Compute the largest Lyapunov exponent for an elastic pendulum.
    
    parameters:
    Initial_condition : array -like
    Initial conditions [r, theta , dr, dtheta].
    perturbation : array -like
    Small perturbation applied to the initial condition ,
    [delta_r , delta_theta , delta_dr , delta_dtheta]
    params : tuple
    System parameters (g, k, m, L0, epsilon).
    t_max : float , optional
    Maximum integration time (default is 100).
    dt : float , optional
    Time step for numerical integration (default is 0.1).
    
    Returns:
    float
    Estimated largest Lyapunov exponent.
    """
    # TODO: Implement numerical integration and track divergence of nearby trajectories.
            
    g, k, m, L0, epsilon = params
    n_ic = initial_conditions.shape[1]  # Number of initial conditions
    n_eval = int(t_max / dt)
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, n_eval)

    #print("Perturbations shape before reshaping:", perturbations.shape)

    Y0 = np.vstack([initial_conditions, perturbations])

    #print("Y0 shape before flattening:", Y0.shape)
    #print("Y0 shape for variational:", Y0.flatten().shape)
    #print("Y0 shape for elastic:", Y0[:4].flatten().shape)

    # Solve variational equation
    sol = solve_ivp(variational_IVP, t_span, Y0.flatten(), args=params, t_eval=t_eval,
                    method='RK45', atol=1e-12, rtol=1e-12, vectorized=True)
    
    #sol2 = solve_ivp(elastic_pendulum, t_span, Y0[:4].flatten(), args=params, t_eval=t_eval,
    #                method='RK45', atol=1e-12, rtol=1e-12, vectorized=True)

    # Extract perturbation vector evolution: (4, n_ic, 4, n_eval)

    #y_sol = sol.y[:8].reshape(4, n_ic, -1)
    dy_sol = sol.y[8:].reshape(4, n_ic, -1)
    #y_sol2 = sol2.y.reshape(4, n_ic, -1)
    #.reshape(4, n_ic, 4, -1)
    #print("elastic y:", y_sol2.shape)
    #print("variational y:", y_sol.shape)
    #print("variational dy:",dy_sol.shape)
    
    #transpose matrix to get the initial condition back as a row
    #print(dy_sol.shape)
    #print(dy_sol[0][1])
    dy_sol = np.swapaxes(dy_sol, 0 , 1)
    #print(dy_sol.shape)
    #print(dy_sol[1][0])
    
    #dy0 = dy_sol[:,:,0]
    #dyT = dy_sol[:,:,-1]

    lyap_exponents = []
    for i in range(n_ic):
        #compute the starting norm
        #print(dy_sol[i, :, 0], dy_sol[i, :,-1])
        norm_dy0 = np.linalg.norm(dy_sol[i,:,0])
        norm_dyT = np.linalg.norm(dy_sol[i,:,-1])
        l_exponent = (1/t_max)*np.log(norm_dy0 / norm_dyT)
        lyap_exponents.append(l_exponent)


    # Compute the norm of perturbation over time: (n_ic, n_eval)
    #norm_dy = np.linalg.norm(dy_sol, axis=(0, 2))  # Frobenius norm over (4,4)

    #norm_dy = np.linalg.norm(dy_sol, axis=0)  # Sum across state variables
    #print(sol.t.shape)
    #print(norm_dy.shape)
    # Compute Lyapunov exponent for each initial condition
    #lyap_exponents = np.polyfit(sol.t, np.log(norm_dy), 1)[0]

    return lyap_exponents  # Array of Lyapunov exponents for all `n_ic`

g, k, m, L0, epsilon = 9.81, 1.0, 1.0, 10.0, -1
"""
initial_conditions = np.array([
    [1.0, np.pi/6, 0.0, 0.0],
    [1.2, np.pi/4, 0.0, 0.0]
]).transpose()

[0.8, np.pi/3, 0.0, 0.0],
    [1.1, np.pi/2, 0.0, -0.5],
    [1, 0, -10.0, 0.1]
# Check if dimensions are consistent for vectorized solver
assert elastic_pendulum(1, initial_conditions, g, k, m, L0, epsilon).shape \
    == initial_conditions.shape

tmax = 100
n_eval = 1000
t_span, t_eval = (0, tmax), np.linspace(0, tmax, n_eval)
sol = solve_ivp(elastic_pendulum,
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

E = energy(Y, n_ic, g, k, m, L0, epsilon)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
labels = ["Case 1", "Case 2", "Case 3", "Case 4"]


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

# Generate perturbations dynamically for the number of initial conditions
n_ic = initial_conditions.shape[1]
#perturbations = np.random.uniform(-1e-6, 1e-6, size=(4, n_ic))  # Perturb each state a little
#print(perturbations.shape)

# Compute Lyapunov exponents for all initial conditions
lyap_exps = compute_lyapunov_exponent(initial_conditions, perturbations, (g, k, m, L0, epsilon))

print(f"Lyapunov Exponents for {n_ic} cases: {lyap_exps}")
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
"""