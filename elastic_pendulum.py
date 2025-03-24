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
    ddrd_thetadot = np.array(2*r*dtheta)
    
    zero = np.zeros_like(r)
    one = np.ones_like(r)

    ddtheta_dr = np.array(2*dr*dtheta / r**2 + g*np.sin(theta) / r**2)
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
    #reshape st first 4 rows are state vars for state and last 4 are state vars for perturbation
    Y = Y.reshape(8, -1)
    #n_ic = Y.shape[1]
    if len(Y.shape) > 1:
        n_ic = Y.shape[1]  # Number of initial conditions
    else:
        n_ic = 1

    y = Y[:4]
    dy = Y[4:]

    y_dot = elastic_pendulum(t, y, g, k, m, L0, epsilon)
    J = elastic_pendulum_jacobian(t, y, g, k, m, L0, epsilon)  # (4, 4, n_ic)

    #reshape dy to match J
    dy = dy.reshape(4,1,n_ic)
    dy_dot = np.zeros_like(dy)
    for i in range(dy.shape[2]):  # Loop over the third dimension, which is the number of initial conditions
        dy_dot[:, :, i] = np.dot(J[:, :, i], dy[:, :, i])  # Matrix-vector multiplication at each step
    #return dy to being a 4 x n_ic matrix
    dy_dot = dy_dot.reshape(4,n_ic)

    return np.vstack([y_dot, dy_dot]) 

def numerical_jacobian(func, t, Y, g, k, m, L0, epsilon, delta=1e-6):
    """
    Compute the Jacobian numerically using central differences.
    
    Parameters:
    func - The function whose Jacobian we want to compute (elastic_pendulum)
    t - Current time
    Y - State vector (4 x n_ic)
    delta - Small perturbation for finite differences
    
    Returns:
    J_num - Numerical Jacobian (4 x 4 x n_ic)
    """
    #n_vars, n_ic = Y.shape  # (4, n_ic)

    if len(Y.shape) > 1:
        n_ic = Y.shape[1]  # Number of initial conditions
    else:
        n_ic = 1


    J_num = np.zeros((4, 4, n_ic))  # Shape (4, 4, n_ic)

    for i in range(4):  # Loop over each state variable
        perturb = np.zeros_like(Y)
        perturb[i, :] = delta  # Perturb only one variable at a time

        f_plus = func(t, Y + perturb, g, k, m, L0, epsilon)
        f_minus = func(t, Y - perturb, g, k, m, L0, epsilon)

        J_num[:, i, :] = (f_plus - f_minus) / (2 * delta)

    return J_num

# Test function to compare numerical and analytical Jacobians
def verify_jacobian(Y, g, k, m, L0, epsilon, delta=1e-6):
    t = 0  # Arbitrary time, since it's not explicitly time-dependent

    # Compute analytical Jacobian
    J_analytic = elastic_pendulum_jacobian(t, Y, g, k, m, L0, epsilon)

    # Compute numerical Jacobian
    J_numeric = numerical_jacobian(elastic_pendulum, t, Y, g, k, m, L0, epsilon, delta)

    # Compute the error between the two
    error = np.abs(J_analytic - J_numeric)

    print("Analytical Jacobian:\n", J_analytic)
    print("\nNumerical Jacobian:\n", J_numeric)
    print("\nAbsolute Error:\n", error)

    # Check if the error is within acceptable bounds
    max_error = np.max(error)
    if max_error < 1e-4:
        print("\n✅ The Jacobians match within tolerance.")
    else:
        print("\n❌ Significant difference detected. Check your analytical Jacobian.")


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

    #reshape ic and perturbations to force n_ic as columns
    initial_conditions.reshape(4,-1)
    perturbations.reshape(4,-1)

    if len(initial_conditions.shape) > 1:
        n_ic = initial_conditions.shape[1]  # Number of initial conditions
    else:
        n_ic = 1
    
    n_eval = int(t_max / dt)
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, n_eval)

    Y0 = np.vstack([initial_conditions, perturbations])

    # Solve variational equation
    sol = solve_ivp(variational_IVP, t_span, Y0.flatten(), args=params, t_eval=t_eval,
                    method='RK45', atol=1e-12, rtol=1e-12, vectorized=True)
    
    print(n_ic)
    print(sol.y.shape)
    
    if n_ic ==1:
        dy_sol = sol.y[4:]
    else:
        dy_sol = sol.y[n_ic*4:].reshape(4, n_ic, 1000)
    
    #transpose matrix to get the initial condition back as a row
    dy_sol = np.swapaxes(dy_sol, 0 , 1)

    lyap_exponents = []
    for i in range(n_ic):
        if n_ic == 1:
            #compute t=0 norm
            norm_dy0 = np.linalg.norm(dy_sol[0])
            #computing t = tmax norm
            norm_dyT = np.linalg.norm(dy_sol[-1])
        else:
            #compute t=0 norm
            norm_dy0 = np.linalg.norm(dy_sol[i,:,0])
            #computing t = tmax norm
            norm_dyT = np.linalg.norm(dy_sol[i,:,-1])

        l_exponent = (1/t_max)*np.log(norm_dyT / norm_dy0)
        lyap_exponents.append(l_exponent)

    return lyap_exponents  # Array of Lyapunov exponents for all `n_ic`

initial_conditions=np.array([
            [1],
            [0.52359878],
            [0],
            [0]])

g, k, m, L0, epsilon = 9.81, 1.0, 1.0, 10.0, -1
n_eval = 1000
t_span = (0, 100)
t_eval = np.linspace(0, 100, n_eval)
sol = solve_ivp(elastic_pendulum, t_span, initial_conditions.flatten(), args=(g, k, m, L0, epsilon), t_eval=t_eval,
                    method='RK45', atol=1e-12, rtol=1e-12, vectorized = True)
