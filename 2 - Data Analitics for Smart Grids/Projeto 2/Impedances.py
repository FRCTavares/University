def estimate_line_impedances(topo, s, vr, el, ni, nBUS, z_true, num_time_steps=5):
    """
    Estimates the complex impedance of each line in a LV network using multiple time steps.
    
    Args:
        topo: Network topology as list of edges [[parent, child], ...]
        s: Power consumption dataset (time steps × loads)
        vr: Voltage reference
        el: Voltage exponent for power flow
        ni: Number of iterations for power flow
        nBUS: Number of buses in the network
        z_true: True impedances - used to simulate "measured" voltages
        num_time_steps: Number of time steps to use for estimation (default: 5)
        
    Returns:
        z_est: Estimated complex impedances for each line
    """
    from scipy.optimize import minimize
    import numpy as np
    
    # Calculate phase angle
    al = np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi))
    
    # Use first num_time_steps time steps for estimation
    time_steps = min(num_time_steps, s.shape[0])
    
    # Generate "measured" voltages at node 4 for each time step 
    v4_measured = []
    
    for t in range(time_steps):
        # Build injected power matrix for current time step
        si = np.zeros((3, 4), dtype=complex)
        si[2,1] = s[t,0]  # Phase C, Bus 1
        si[0,2] = s[t,1]  # Phase A, Bus 2  
        si[1,2] = s[t,2]  # Phase B, Bus 2
        si[2,3] = s[t,3]  # Phase C, Bus 3
        
        # Compute "measured" voltage at node 4 
        try:
            mvp, _ = pf3ph(topo, z_true, si, vr, el, ni, al, nBUS)
            v4_measured.append(mvp[:,3])  # Store voltage at terminal node
        except Exception as e:
            print(f"Warning: Power flow calculation failed for time step {t}: {e}")
            continue
    
    print(f"Generated {len(v4_measured)} valid measurements for estimation.")
    if len(v4_measured) == 0:
        raise ValueError("No valid power flow calculations. Cannot proceed with estimation.")
    
    # Error function to minimize
    def error_function(z_params):
        # Convert optimization vector to complex impedances
        z_real = z_params[:3]
        z_imag = z_params[3:]
        z_complex = [complex(r, i) for r, i in zip(z_real, z_imag)]
        
        total_error = 0
        
        # Calculate error for each time step
        for t in range(len(v4_measured)):
            # Build injected power matrix for current time step
            si = np.zeros((3, 4), dtype=complex)
            si[2,1] = s[t,0]  # Phase C, Bus 1
            si[0,2] = s[t,1]  # Phase A, Bus 2
            si[1,2] = s[t,2]  # Phase B, Bus 2
            si[2,3] = s[t,3]  # Phase C, Bus 3
            
            # Compute voltage with current impedance estimate
            try:
                mvp, _ = pf3ph(topo, z_complex, si, vr, el, ni, al, nBUS)
                v4 = mvp[:,3]
                
                # Calculate error between simulated and "measured" voltages
                error = np.sum(np.abs(v4 - v4_measured[t])**2)
                total_error += error
            except Exception:
                # Penalize heavily if power flow fails for these parameters
                return 1e6
        
        # Add regularization term to favor smaller impedance values
        regularization = 1e-5 * np.sum(z_params**2)
        
        return total_error + regularization
    
    # Use reasonable initial guess (slightly perturbed values)
    z_init = [complex(0.03, 0.01), complex(0.04, 0.02), complex(0.05, 0.03)]
    
    # Initial parameter vector (reshape complex impedances to real vector)
    z0 = np.concatenate([np.real(z_init), np.imag(z_init)])
    
    # Minimize the error function with improved settings
    result = minimize(
        error_function, 
        z0, 
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6},
        bounds=[(0.001, 0.5) for _ in range(6)]  # Reasonable bounds for R and X values
    )
    
    # Convert optimized parameters back to complex impedances
    z_params_opt = result.x
    z_real_opt = z_params_opt[:3]
    z_imag_opt = z_params_opt[3:]
    z_est = [complex(r, i) for r, i in zip(z_real_opt, z_imag_opt)]
    
    return z_est