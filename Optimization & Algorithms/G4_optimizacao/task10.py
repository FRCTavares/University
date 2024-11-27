import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# Load the .mat file
def load_data(file_path):
    data = sio.loadmat(file_path)
    X = data['x']  # Input data (X values)
    Y = data['y']  # Target data (Y values)
    U = data['u']  # ...
    V = data['v']  # ...
    S = data['s']  # ...
    R = data['r']  # ...
    return X.flatten(), Y.flatten(), U.flatten(), V.flatten(), S.flatten(), R.flatten()



# Mixture of Linear Models
def mixture_model(x, u, v, s, r, K, N):
    y_pred = np.zeros(N)
    W = np.zeros((N, K))

    alphas = np.zeros((N, K))
    for k in range(K-1):
        alphas[:, k] = u[k] * x + v[k]  

    # Centering to prevent overflow
    max_alpha = np.max(alphas,axis=1,keepdims=True)
    exp_alphas = np.exp(alphas - max_alpha)
    
    # Normalizing weights 
    for k in range(K):
        W[:, k] = exp_alphas[:, k] / np.sum(exp_alphas, axis=1)

    # Combine predictions from all K models
    for k in range(K):
        y_pred += W[:, k] * (s[k] * x + r[k])
   
    return y_pred, W



# Compute partial derivatives (Jacobian elements)
def compute_jacobian_partial(x, u, v, s, r, W, k, K, N, param):
    if param == 's':
        return W[:, k] * x  
    elif param == 'r':
        return W[:, k]      
    elif param == 'u':
        J_partial_u = np.zeros(N)
        for i in range(N):

            k_term = W[i, k] * (1 - W[i, k]) * x[i]*(s[k] * x[i] + r[k])# derivative of Wk*Yk
            
            j_term = W[i, k]*x[i]*sum(W[i, j] * (s[j] * x[i] + r[j]) for j in range(K) if j != k) # sum of derivative of Wj*Yj

            J_partial_u[i] = k_term - j_term 
            

        return J_partial_u 
    
    elif param == 'v':
        J_partial_v = np.zeros(N)
        for i in range(N):

            W_term = W[i, k] * (1 - W[i, k])*(s[k] * x[i] + r[k])# derivative of Wj*Yj
            
            sum_term = W[i, k]*sum(W[i, j] * (s[j] * x[i] + r[j]) for j in range(K) if j != k) #sum of derivative Wj*Yj

            J_partial_v[i] = W_term - sum_term 

        return J_partial_v 
    else:
        raise ValueError(f"Invalid parameter: {param}")

# Compute the gradient and Jacobian
def compute_Jacobian(x, u, v, s, r, W, K, N):
    J = np.zeros((N, 4 * K - 2))  # Jacobian 

    # Compute partial derivatives for uk and vk 
    for k in range(K - 1):
        J[:, k] = compute_jacobian_partial(x, u, v, s, r, W, k, K, N, 'u')  # Jacobian for u_k
        J[:, (K - 1) + k] = compute_jacobian_partial(x, u, v, s, r, W, k, K, N, 'v')  # Jacobian for v_k

    # Compute partial derivatives for sk and rk 
    for k in range(K):
        J[:, 2*(K - 1)+ k] = compute_jacobian_partial(x, u, v, s, r, W, k, K, N, 's')  # Jacobian for s_k
        J[:, 2*(K - 1)+ K + k] = compute_jacobian_partial(x, u, v, s, r, W, k, K, N, 'r')  # Jacobian for r_k

    return J


# Levenberg-Marquardt optimization
def levenberg_marquardt(x, y, u, v, s, r, K, N, max_iter=5000, epsilon=1e-4, lambda_init=1.0):
    lambda_ = lambda_init  # Damping parameter
    residuals_list = []  # To store sum of squared residuals for ploting
    grad_obj_func_list = [] # To store gradient of objective function for ploting
   
    for iter in range(max_iter):
        
        # Compute model prediction and weights
        y_pred, W = mixture_model(x, u, v, s, r, K, N)
        # Combine the u, v, s, r vectors into a single column vector
        param_vector = np.concatenate([u, v, s, r]).reshape(-1, 1)
        
        # Compute the residual and objective function
        residual = y_pred - y
        obj_func = np.sum(residual ** 2)
        residuals_list.append(obj_func)
        # Compute the full Jacobian matrix (gradients)
        J = compute_Jacobian(x, u, v, s, r, W, K, N)
        
        # Construct the A matrix (Jacobian and regularization term)
        sqrt_lambda = np.sqrt(lambda_)

        # Create a (4*K-2)x(4*K-2) identity matrix
        identity_matrix = np.identity(4*K-2)
       
        # Multiply the identity matrix by sqrt_lambda to get the final sqrt_lambda matrix
        sqrt_lambda_matrix = sqrt_lambda * identity_matrix

        # Combine the jacobian with the square root of the scalar
        A = np.vstack([J, sqrt_lambda_matrix])

        
        J_param = J @ param_vector  # Multiply jacobian by parameters vector
        b_top = J_param - residual.reshape(-1, 1)  # Subtract residual from each row

        # Bottom part: sqrt_lambda * param_vector
        b_bottom = sqrt_lambda * identity_matrix @ param_vector  # Shape will be 14 x 1

        # Combine the top and bottom parts
        b = np.vstack([b_top, b_bottom])
       
        # Solve the least-squares problem for min
        minimized_parameters,_ , _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Compute the candidate parameters with min
        u_new, v_new, s_new, r_new = update_parameters(u, v, s, r, minimized_parameters, K)
        # Compute new prediction and objective function with updated parameters
        y_pred_new, W_new = mixture_model(x, u_new, v_new, s_new, r_new, K, N)
        obj_func_new = np.sum((y_pred_new - y) ** 2)

      
        gradient = 2 * J.T @ residual
        # Check stopping criterion
        grad_obj_func_list.append(np.linalg.norm(gradient))
        if np.abs(np.linalg.norm(gradient))< epsilon:
            print(f"Converged at iteration {iter}")
            break
        # Check if the step is valid

        if obj_func_new < obj_func:  # Valid step
            u, v, s, r = u_new, v_new, s_new, r_new
            lambda_ *= 0.7  # Decrease lambda
            
            
        else:  # Null step
            lambda_ *= 2.0  # Increase lambda


    # Plot the results
    
    plot_results(x, y, u, v, s, r, y_pred, W,residuals_list,grad_obj_func_list)
    return u, v, s, r

# Update the parameters (u, v, s, r) with the delta step from LM
def update_parameters(u, v, s, r, minimized_parameters, K):

    # Ensure the deltas are treated as 1D vectors instead of 2D
    u_new = minimized_parameters[:K - 1].flatten()  # Flatten in case it's higher dimensional
    v_new = minimized_parameters[K - 1:2 * K - 2].flatten()
    s_new  = minimized_parameters[2 * K - 2:3 * K - 2].flatten()
    r_new  = minimized_parameters[3 * K - 2:].flatten()

    # Update the parameters with the corresponding deltas
   
    
    return u_new , v_new , s_new , r_new 

# Plot the fitted signal and the weights
def plot_results(x, y, u, v, s, r, y_pred, W,residuals_list,grad_obj_func_list):
    
    plt.figure() 
    plt.plot(x, y, 'ko', markersize=3, label='Original Data')
    plt.plot(x, y_pred, 'o', markerfacecolor='none', markeredgecolor='blue', markersize=3, label='Fitted Signal')
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.title('Fitted Signal using Mixture of Linear Models')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    for k in range(W.shape[1]):
        plt.plot(x, W[:, k], label=f'Weight {k+1}')
    plt.title('Weights of each Linear Model')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(len(residuals_list)), residuals_list, 'b-', label='Objective funtion')
    plt.xlabel('Iterations')
    plt.ylabel('Objective funtion')
    plt.title('Objective funtion Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()

    plt.figure()
    plt.plot(range(len(grad_obj_func_list)), grad_obj_func_list, 'b-', label='Gradient of objective funtion')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient of objective funtion')
    plt.title('Gradient of objective funtion Over Iterations')
    plt.legend()
    plt.ylim(0.00001, 100000)
    plt.grid(True)
    plt.yscale('log')
    plt.show()

    


# Example usage
file_path = 'lm_dataset_task.mat'  # Replace with actual path
# Initial values for u, v, s, r
x, y, u, v, s, r = load_data(file_path)


K = len(s)  # Number of models
N = len(x)  # Number of data points

# Run the LM optimization
u_opt, v_opt, s_opt, r_opt = levenberg_marquardt(x, y, u, v, s, r, K, N)
#print results
print(u_opt)
print(v_opt)
print(s_opt)
print(r_opt)
