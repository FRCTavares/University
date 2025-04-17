###############################################################################################
# Projeto 2 - Estimação das Correntes de Carga e das Impedâncias das Linhas numa Rede de BT   #
#                                                                                             #
# Grupo 13                                                                                    #
#                                                                                             #
# Membros:                                                                                    #
#   Francisco Tavares (103402)                                                                #
#   Marta Valente (103574)                                                                    #
###############################################################################################

# =============================================================================================
# Importação de bibliotecas
# =============================================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================================
# Parâmetros globais
# =============================================================================================
cosPhi = 0.95
m = 12                # Número de períodos (ex.: 12 períodos de 15 min → 3 horas)
netFactor = 0.25      # Fator da rede

# =============================================================================================
# Dados 
# =============================================================================================
def load_data():
    # Consumption dataset: cada linha corresponde a uma medição (a cada 15 minutos)
    s = [[0.0450, 0.0150, 0.0470, 0.0330],
         [0.0250, 0.0150, 0.2480, 0.0330],
         [0.0970, 0.0250, 0.3940, 0.0330],
         [0.0700, 0.0490, 0.0200, 0.4850],
         [0.1250, 0.0460, 0.0160, 0.1430],
         [0.2900, 0.0270, 0.0160, 0.0470],
         [0.2590, 0.0150, 0.0170, 0.0200],
         [0.2590, 0.0160, 0.0280, 0.0160],
         [0.4420, 0.0160, 0.0500, 0.0170],
         [0.2010, 0.0230, 0.0460, 0.0160],
         [0.2060, 0.0490, 0.0220, 0.0240],
         [0.1300, 0.0470, 0.0160, 0.0490],
         [0.0460, 0.0260, 0.0170, 0.0480]]
    s = np.array(s)

    # Topologia da rede: rede radial com 4 nós
    topo = [[1, 2], [2, 3], [3, 4]]
    nBUS = np.max(topo)

    # Impedâncias das linhas (já escaladas pelo netFactor)
    z = np.multiply([complex(0.1, 0.05), complex(0.15, 0.07), complex(0.2, 0.1)], netFactor)



    vr = 1    # Tensão de referência
    el = 1    # Expoente do cálculo de tensão (usado na fórmula de potência)
    ni = 20   # Número de iterações para o cálculo do Power Flow

    return s, topo, nBUS, z, vr, el, ni

# =============================================================================================
# Função de Power Flow
# =============================================================================================
def pf3ph(t, z, si, vr, el, ni, al, nBUS):
    """
    Cálculo do fluxo de potência trifásico para uma rede radial.
    
    Args:
        t: topologia da rede (lista de pares [pai, filho])
        z: vetor de impedâncias de cada linha
        si: matriz de potência injetada em cada fase e em cada bus (3x4)
        vr: tensão de referência
        el: expoente de tensão
        ni: número de iterações
        al: ângulo de fase (usualmente exp(-j*2π/3))
        nBUS: número de barras (nós)
        
    Returns:
        mvp: matriz de tensões de fase em cada nó (3 x número de nós)
        ip: matriz de correntes de fase (3 x número de nós)
    """
    t = np.array(t)
    p = t[:, 0]
    f = t[:, 1]
    w = len(p) + 1  # número de nós = número de ramos + 1
    
    vp = np.zeros((nBUS - 1, w), dtype=complex)
    vn = np.zeros((nBUS - 1, w), dtype=complex)
    
    # Inicializa a tensão de referência na primeira barra
    vp[0, 0:w] = vr

    # Propaga a tensão pelos nós usando a rotação (assumindo tensões iguais em todas as barras)
    for h in range(2, nBUS):
        vp[h - 1, :] = vp[h - 2, :] * al

    va = vp - vn  # Tensão auxiliar
    ia = np.conjugate(np.divide(np.multiply(si, np.abs(va)**el), va))
    
    for it_iter in range(ni):
        va = vp - vn
        ip = np.conjugate(np.divide(np.multiply(si, np.abs(va)**el), va))
        inn = -np.sum(ip, 0)  # Corrente do neutro

        for k in range(w - 1, 0, -1):  # Ciclo Backward
            n_ = f[k - 1]
            m_ = p[k - 1]
            ip[:, m_ - 1] = ip[:, m_ - 1] + ip[:, n_ - 1]
            inn = -np.sum(ip, 0)

        eps = np.linalg.norm(np.max(np.abs(ia - ip), 0))
        if eps <= 1e-4:
            mvp = vp - vn
            return mvp, ip
        else:
            ia = ip

        for k in range(w - 1):  # Ciclo Forward
            n_ = f[k]
            m_ = p[k]
            vn[:, n_ - 1] = vn[:, m_ - 1] - z[k] * inn[n_ - 1]
            vp[:, n_ - 1] = vp[:, m_ - 1] - z[k] * ip[:, n_ - 1]
        ia = ip

    # Caso não converja, retorna as últimas estimativas
    return (vp - vn), ip

# =============================================================================================
# Função de Estimativa de Correntes com Regularização
# =============================================================================================
def base_estimation():
    """
    A função calcula a estimativa de corrente utilizando o método de mínimos quadrados.
    """
    # Carregar dados
    s, topo, nBUS, z, vr, el, ni = load_data()

    al=np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi)) #Phase Angle
    sp=np.mean(s[0:m,:], axis=0) #Average power in each phase (i0)

    si=[[0, 0, sp[2], 0],[0, 0, sp[1], 0],[0, sp[0],  0, sp[3]]] #Power in each bus and in each phase

    mvp, ip=pf3ph(topo, z, si, vr, el, ni, al, nBUS)  #Compute the power flow

    zv=mvp[:,3]    #Voltage Measurement in Node 4 

    ds=s[0:m,:]-sp              #Difference from the average

    scale=1e-9*np.abs(1/sp)**2  #Measurements Accuracy

    #Matrices Creation
    it=np.zeros((3,m))
    ie=np.zeros((3,m))
    dx = np.zeros((4,m))
    A=np.zeros((3,4), dtype=complex)

    # Matriz A
    # Gamma matrix for the 4 phases
    gamma1 = np.array([0, 0, al**2])
    gamma2 = np.array([0, al, 0])
    gamma3 = np.array([1, 0, 0])

    W = np.array([[2,1,1], [1,2,1], [1,1,2]])
    
    z_accumulated = [
    z[0],                 # Node 1: z[0] (impedância entre source e nó 1)
    z[0] + z[1],          # Node 2: z[0] + z[1] (impedância acumulada até nó 2)
    z[0] + z[1],          # Node 3: z[0] + z[1] (impedância acumulada até nó 3, igual ao nó 2)
    z[0] + z[1] + z[2]    # Node 4: z[0] + z[1] + z[2] (impedância acumulada até nó 4)
    ]  

    Wk1 = np.zeros((3, 3), dtype=complex)
    Wk2 = np.zeros((3, 3), dtype=complex)
    Wk3 = np.zeros((3, 3), dtype=complex)
    Wk4 = np.zeros((3, 3), dtype=complex)

    Wk1 = z_accumulated[0] * W
    Wk2 = z_accumulated[1] * W
    Wk3 = z_accumulated[2] * W
    Wk4 = z_accumulated[3] * W

    #Compute matrix A
    # Phase a (row 0)
    A[0,0] = np.dot(Wk1[0], gamma1)
    A[0,1] = np.dot(Wk2[0], gamma2)
    A[0,2] = np.dot(Wk3[0], gamma3)
    A[0,3] = np.dot(Wk4[0], gamma1)

    # Phase b (row 1)
    A[1,0] = np.dot(Wk1[1], gamma1)
    A[1,1] = np.dot(Wk2[1], gamma2)
    A[1,2] = np.dot(Wk3[1], gamma3)
    A[1,3] = np.dot(Wk4[1], gamma1)

    # Phase c (row 2)
    A[2,0] = np.dot(Wk1[2], gamma1)
    A[2,1] = np.dot(Wk2[2], gamma2)
    A[2,2] = np.dot(Wk3[2], gamma3)
    A[2,3] = np.dot(Wk4[2], gamma1)


    for i in range(m):
        #Power in each instant 
        si=np.zeros(shape=[3,4]) # Three lines one for each phase and four buses
        si[2,1]=s[i,0]
        si[0,2]=s[i,1]
        si[1,2]=s[i,2]
        si[2,3]=s[i,3]
        # Power Flow
        mvp, ip = pf3ph(topo, z, si, vr, el, ni, al, nBUS) # Power Flow
        v4 = mvp[:,3] # Voltage at bus 4
        dv=v4-zv # Voltage difference

        Atrans=np.transpose(A)
        di=np.dot(np.dot(np.linalg.inv(np.dot(Atrans,A)+np.diag(scale)), Atrans),-dv) #Min-norm solution

        dx[:,i]=di # Store the result in the matrix dx

    yy1 = sp
    yy2 = sp+np.transpose(dx)
    yy3 = ds
    yy4 = ds-np.transpose(dx)

    #Graph 1
    for i in range (1,m):
        plt.plot(yy1, yy2[i,:], 'C0o')

    plt.xlabel("Pseudo-medição [pu]")
    plt.ylabel("Estimação [pu]")
    plt.show()

    #Graph 2
    for i in range (1,m):
        plt.plot(yy3[i,:], yy4[i,:], 'C0o')

    plt.xlabel("Erro de pseudo-medição [pu]")
    plt.ylabel("Erro de estimação [pu]")
    plt.show()

    return dx # Return the matrix with the results of the current estimation
 

# =============================================================================================
# Desafio para o projeto 2
# =============================================================================================
def estimate_line_impedances(topo, s, vr, el, ni, nBUS, z_true, num_time_steps=5):
    """
    Estimates the complex impedance of each line in a radial LV network using multiple time steps.
    
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
    
    # Generate "measured" voltages at node 4 for each time step using true impedances
    v4_measured = []
    
    for t in range(time_steps):
        # Build injected power matrix for current time step
        si = np.zeros((3, 4), dtype=complex)
        si[2,1] = s[t,0]  # Phase C, Bus 1
        si[0,2] = s[t,1]  # Phase A, Bus 2  
        si[1,2] = s[t,2]  # Phase B, Bus 2
        si[2,3] = s[t,3]  # Phase C, Bus 3
        
        # Compute "measured" voltage at node 4 using true impedances
        try:
            mvp, _ = pf3ph(topo, z_true, si, vr, el, ni, al, nBUS)
            v4_measured.append(mvp[:,3])  # Store voltage at terminal node
        except Exception as e:
            print(f"Warning: Power flow calculation failed for time step {t}: {e}")
            continue
    
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

def compare_load_current_estimation(z_true, z_est):
    """
    Compares load current estimation using original (true) impedances versus estimated ones.
    
    Args:
        z_true: Original impedances (list of 3 complex values)
        z_est: Estimated impedances (list of 3 complex values)
        
    Returns:
        dx_true: Current estimation with true impedances
        dx_est: Current estimation with estimated impedances
        mse: Mean squared error between dx_true and dx_est
    """
    # Modified base_estimation function to accept impedances as parameter
    def base_estimation_with_z(z_values):
        # Load other data
        s, topo, nBUS, _, vr, el, ni = load_data()
        
        al = np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi))
        sp = np.mean(s[0:m,:], axis=0)
        
        si = [[0, 0, sp[2], 0], [0, 0, sp[1], 0], [0, sp[0], 0, sp[3]]]
        mvp, ip = pf3ph(topo, z_values, si, vr, el, ni, al, nBUS)
        
        zv = mvp[:,3]
        scale = 1e-9*np.abs(1/sp)**2
        ds=s[0:m,:]-sp              #Difference from the average
        
        # Matrix creation
        dx = np.zeros((4,m), dtype=complex)
        A = np.zeros((3,4), dtype=complex)
        
        # Gamma matrices
        gamma1 = np.array([0, 0, al**2])
        gamma2 = np.array([0, al, 0])
        gamma3 = np.array([1, 0, 0])
        
        W = np.array([[2,1,1], [1,2,1], [1,1,2]])
        
        # Calculate accumulated impedances
        z_accumulated = [
            z_values[0],
            z_values[0] + z_values[1],
            z_values[0] + z_values[1],
            z_values[0] + z_values[1] + z_values[2]
        ]
        
        # Calculate W matrices
        Wk1 = z_accumulated[0] * W
        Wk2 = z_accumulated[1] * W
        Wk3 = z_accumulated[2] * W
        Wk4 = z_accumulated[3] * W
        
        # Compute matrix A
        # Phase a (row 0)
        A[0,0] = np.dot(Wk1[0], gamma1)
        A[0,1] = np.dot(Wk2[0], gamma2)
        A[0,2] = np.dot(Wk3[0], gamma3)
        A[0,3] = np.dot(Wk4[0], gamma1)
        
        # Phase b (row 1)
        A[1,0] = np.dot(Wk1[1], gamma1)
        A[1,1] = np.dot(Wk2[1], gamma2)
        A[1,2] = np.dot(Wk3[1], gamma3)
        A[1,3] = np.dot(Wk4[1], gamma1)
        
        # Phase c (row 2)
        A[2,0] = np.dot(Wk1[2], gamma1)
        A[2,1] = np.dot(Wk2[2], gamma2)
        A[2,2] = np.dot(Wk3[2], gamma3)
        A[2,3] = np.dot(Wk4[2], gamma1)
        
        for i in range(m):
            # Power in each instant
            si = np.zeros(shape=[3,4])
            si[2,1] = s[i,0]
            si[0,2] = s[i,1]
            si[1,2] = s[i,2]
            si[2,3] = s[i,3]
            
            # Power Flow
            mvp, ip = pf3ph(topo, z_values, si, vr, el, ni, al, nBUS)
            v4 = mvp[:,3]
            dv = v4 - zv
            
            # Min-norm solution
            Atrans = np.transpose(A)
            di = np.dot(np.dot(np.linalg.inv(np.dot(Atrans,A)+np.diag(scale)), Atrans), -dv)
            
            dx[:,i] = di

           
        return dx, sp

    
    # Get estimations with both impedance sets
    print("Computing current estimation with true impedances...")
    dx_true, sp_true = base_estimation_with_z(z_true)
    
    print("Computing current estimation with estimated impedances...")
    dx_est, sp_true = base_estimation_with_z(z_est)

    print("Computing current estimation with estimated impedances...")
    dx_est, sp_est = base_estimation_with_z(z_est)
    
    # Add comparison plots between true and estimated current estimations
    plt.figure(figsize=(10, 7))
    
    for i in range(4):  # For each load
        abs_error = np.abs(dx_true[i,:] - dx_est[i,:])
        plt.plot(range(1, m+1), abs_error, 'o-', label=f'Carga {i+1}')
    
    plt.xlabel("Período de Tempo")
    plt.ylabel("Erro Absoluto [pu]")
    plt.title("Erro Absoluto na Estimação de Corrente")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compute MSE between true and estimated current variations
    mse = np.mean(np.abs(dx_true - dx_est)**2)
    mse_per_load = [np.mean(np.abs(dx_true[i,:] - dx_est[i,:])**2) for i in range(4)]
    
   # Print MSE results
    print(f"\nErro Quadrático Médio Global: {mse:.6e}")
    print("\nMSE por carga:")
    for i, mse_load in enumerate(mse_per_load):
        print(f"  Carga {i+1}: {mse_load:.6e}")
    
    # Analysis of results
    print("\nAnálise:")
    if mse < 1e-6:
        print("As estimativas são muito semelhantes, sugerindo uma estimativa precisa da impedância da linha.")
    elif mse < 1e-4:
        print("As estimativas mostram boa concordância com diferenças mínimas.")
    elif mse < 1e-2:
        print("As estimativas mostram diferenças moderadas, indicando alguma imprecisão na estimativa da impedância.")
    else:
        print("As estimativas diferem significativamente, sugerindo erros substanciais na estimativa da impedância.")
    
    return dx_true, dx_est, mse

def analyze_impedance_estimation():
    """
    Analyzes and compares load current estimation with true vs estimated impedances.
    Prints detailed comparison of initial guess, estimated impedances, and true impedances.
    """
    # Load data
    s, topo, nBUS, z_true, vr, el, ni = load_data()
    
    # Calculate average power from first m measurements
    sp = np.mean(s[0:m, :], axis=0)
    
    # Create better initial guess by perturbing true values (for demonstration only)
    # In a real scenario, we wouldn't know the true values
    z_init_demo = [complex(np.real(z)*1.2, np.imag(z)*0.8) for z in z_true]
    
    # Standard initial guess that would be used in practice
    z_init = [complex(0.03, 0.01), complex(0.04, 0.02), complex(0.05, 0.03)]
    
    # Print initial impedance values
    print("\nInitial Impedance Values:")
    for i in range(len(z_init)):
        print(f"Line {i+1}: {z_init[i]:.6f}")
    
    # Estimate impedances
    print("\nEstimating line impedances...")
    z_est = estimate_line_impedances(topo, s, vr, el, ni, nBUS, z_true, num_time_steps=5)
    
    # Create comparison table
    print("\n" + "="*70)
    print(f"{'Line':^10}|{'True Z':^20}|{'Initial Z':^20}|{'Estimated Z':^20}")
    print("="*70)
    
    for i in range(len(z_true)):
        true_z = f"{z_true[i]:.6f}"
        init_z = f"{z_init[i]:.6f}"
        est_z = f"{z_est[i]:.6f}"
        print(f"{i+1:^10}|{true_z:^20}|{init_z:^20}|{est_z:^20}")
    
    print("="*70)
    
    # Print detailed analysis
    print("\nDetailed Impedance Analysis:")
    for i in range(len(z_true)):
        print(f"\nLine {i+1}:")
        print(f"  True:      {z_true[i]:.6f}")
        print(f"  Initial:   {z_init[i]:.6f}")
        print(f"  Estimated: {z_est[i]:.6f}")
        
        # Calculate errors
        rel_error = abs(z_est[i] - z_true[i]) / abs(z_true[i]) * 100
        rel_error_real = abs(z_est[i].real - z_true[i].real) / abs(z_true[i].real) * 100
        rel_error_imag = abs(z_est[i].imag - z_true[i].imag) / abs(z_true[i].imag) * 100
        
        print(f"  Rel Error: {rel_error:.2f}% (Magnitude)")
        print(f"  Rel Error Real Part: {rel_error_real:.2f}%")
        print(f"  Rel Error Imag Part: {rel_error_imag:.2f}%")
    
    # Compare load current estimations
    print("\nComparing load current estimation with true vs estimated impedances...")
    dx_true, dx_est, mse = compare_load_current_estimation(z_true, z_est)

    
    return dx_true, dx_est, mse

def noise_impact_analysis(num_time_steps=5, num_trials=3):
    """
    Analyzes the impact of different noise levels on line impedance estimation.
    
    Args:
        num_time_steps: Number of time steps to use for each estimation (default: 5)
        num_trials: Number of estimation trials per noise level for statistical analysis (default: 3)
        
    Returns:
        results_dict: Dictionary containing all analysis results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from tabulate import tabulate  # If not installed, run: pip install tabulate
    
    # Load data
    s, topo, nBUS, z_true, vr, el, ni = load_data()
    
    # Define noise levels to test (percentage of voltage magnitude)
    noise_levels = [0, 0.005, 0.01, 0.02, 0.05]  # 0%, 0.5%, 1%, 2%, 5%
    
    # Calculate phase angle
    al = np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi))
    
    # Prepare results storage
    results = {
        'noise_levels': noise_levels,
        'z_est_all': [],
        'rel_errors': np.zeros((len(noise_levels), len(z_true))),
        'rel_errors_real': np.zeros((len(noise_levels), len(z_true))),
        'rel_errors_imag': np.zeros((len(noise_levels), len(z_true))),
        'rel_errors_std': np.zeros((len(noise_levels), len(z_true))),
        'mse_values': np.zeros(len(noise_levels)),
        'mse_std': np.zeros(len(noise_levels))
    }
    
    # Use first num_time_steps time steps for all estimations
    time_steps = min(num_time_steps, s.shape[0])
    
    # Generate "clean" voltages for all selected time steps
    v4_clean_all = []
    si_all = []
    
    print("\nGenerating clean voltage measurements...")
    for t in range(time_steps):
        # Build power matrix for current time step
        si = np.zeros((3, 4), dtype=complex)
        si[2,1] = s[t,0]  # Phase C, Bus 1
        si[0,2] = s[t,1]  # Phase A, Bus 2  
        si[1,2] = s[t,2]  # Phase B, Bus 2
        si[2,3] = s[t,3]  # Phase C, Bus 3
        
        si_all.append(si.copy())
        
        # Compute voltage with true impedances (clean voltage)
        try:
            mvp, _ = pf3ph(topo, z_true, si, vr, el, ni, al, nBUS)
            v4_clean_all.append(mvp[:,3])  # Store voltage at node 4
        except Exception as e:
            print(f"Warning: Power flow calculation failed for time step {t}: {e}")
            continue
    
    if len(v4_clean_all) == 0:
        raise ValueError("No valid power flow calculations. Cannot proceed with analysis.")
    
    # Function to estimate impedances with noisy measurements
    def estimate_with_noise(noise_level, trial_idx):
        # Add noise to voltage measurements
        v4_noisy_all = []
        for v4_clean in v4_clean_all:
            # Calculate average voltage magnitude for proper noise scaling
            avg_mag = np.mean(np.abs(v4_clean))
            # Generate complex Gaussian noise
            noise = noise_level * avg_mag * (np.random.randn(3) + 1j * np.random.randn(3))
            # Add noise to clean voltage
            v4_noisy = v4_clean + noise
            v4_noisy_all.append(v4_noisy)
        
        # Define error function to minimize
        def error_function(z_params):
            # Convert optimization vector to complex impedances
            z_real = z_params[:3]
            z_imag = z_params[3:]
            z_complex = [complex(r, i) for r, i in zip(z_real, z_imag)]
            
            total_error = 0
            
            # Calculate error for each time step
            for t in range(len(v4_noisy_all)):
                # Compute voltage with current impedance estimate
                try:
                    mvp, _ = pf3ph(topo, z_complex, si_all[t], vr, el, ni, al, nBUS)
                    v4 = mvp[:,3]
                    
                    # Calculate error between simulated and noisy voltages
                    error = np.sum(np.abs(v4 - v4_noisy_all[t])**2)
                    total_error += error
                except Exception:
                    # Penalize heavily if power flow fails for these parameters
                    return 1e6
            
            # Add regularization term to favor smaller impedance values
            regularization = 1e-5 * np.sum(z_params**2)
            
            return total_error + regularization
        
        # Use reasonable initial guess
        z_init = [complex(0.03, 0.01), complex(0.04, 0.02), complex(0.05, 0.03)]
        
        # Initial parameter vector
        z0 = np.concatenate([np.real(z_init), np.imag(z_init)])
        
        # Minimize the error function
        result = minimize(
            error_function, 
            z0, 
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6},
            bounds=[(0.001, 0.5) for _ in range(6)]
        )
        
        # Convert optimized parameters back to complex impedances
        z_params_opt = result.x
        z_real_opt = z_params_opt[:3]
        z_imag_opt = z_params_opt[3:]
        z_est = [complex(r, i) for r, i in zip(z_real_opt, z_imag_opt)]
        
        return z_est
    
    # Analyze each noise level
    for nl_idx, noise_level in enumerate(noise_levels):
        print(f"\nAnalyzing noise level: {noise_level*100:.1f}%")
        
        # Storage for multiple trials
        z_est_trials = []
        rel_error_trials = np.zeros((num_trials, len(z_true)))
        rel_error_real_trials = np.zeros((num_trials, len(z_true)))
        rel_error_imag_trials = np.zeros((num_trials, len(z_true)))
        mse_trials = np.zeros(num_trials)
        
        for trial in range(num_trials):
            print(f"  Trial {trial+1}/{num_trials}...")
            
            # Estimate impedances with noisy measurements
            z_est = estimate_with_noise(noise_level, trial)
            z_est_trials.append(z_est)
            
            # Calculate relative errors for each line
            for i, (z_t, z_e) in enumerate(zip(z_true, z_est)):
                rel_error = abs(z_e - z_t) / abs(z_t) * 100
                rel_error_real = abs(z_e.real - z_t.real) / abs(z_t.real) * 100
                rel_error_imag = abs(z_e.imag - z_t.imag) / abs(z_t.imag) * 100
                
                rel_error_trials[trial, i] = rel_error
                rel_error_real_trials[trial, i] = rel_error_real
                rel_error_imag_trials[trial, i] = rel_error_imag
            
            # Calculate MSE in current estimations
            if noise_level > 0:  # Skip MSE calculation for reference case
                _, _, mse = compare_load_current_estimation(z_true, z_est)
                mse_trials[trial] = mse
            
        # Store average results across trials
        results['z_est_all'].append(z_est_trials)
        results['rel_errors'][nl_idx] = np.mean(rel_error_trials, axis=0)
        results['rel_errors_real'][nl_idx] = np.mean(rel_error_real_trials, axis=0)
        results['rel_errors_imag'][nl_idx] = np.mean(rel_error_imag_trials, axis=0)
        results['rel_errors_std'][nl_idx] = np.std(rel_error_trials, axis=0)
        
        if noise_level > 0:
            results['mse_values'][nl_idx] = np.mean(mse_trials)
            results['mse_std'][nl_idx] = np.std(mse_trials)
    
    # Visualize results
# Plot 1: Impedance estimation error vs. noise level
    plt.figure(figsize=(12, 6))
    for i in range(len(z_true)):
        plt.errorbar(
            [n*100 for n in noise_levels], 
            results['rel_errors'][:, i],
            yerr=results['rel_errors_std'][:, i],
            marker='o', 
            label=f'Linha {i+1}'
        )
    
    plt.xlabel('Nível de Ruído (%)')
    plt.ylabel('Erro Relativo (%)')
    plt.title('Impacto do Ruído na Precisão da Estimação de Impedância de Linha')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot 2: MSE in current variations vs. noise level
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        [n*100 for n in noise_levels[1:]], 
        results['mse_values'][1:],
        yerr=results['mse_std'][1:],
        marker='s',
        color='red'
    )
    plt.yscale('log')
    plt.xlabel('Nível de Ruído (%)')
    plt.ylabel('MSE nas Variações de Corrente')
    plt.title('Impacto do Ruído na Precisão da Estimação de Corrente')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    headers = ['Nível de Ruído', 'Linha', 'Z Verdadeiro', 'Z Médio Est.', 'Erro (%)', 'Erro R (%)', 'Erro X (%)']
    table_data = []
    
    for nl_idx, noise_level in enumerate(noise_levels):
        for i in range(len(z_true)):
            # Calculate mean estimated Z across trials
            z_est_mean = np.mean([z_est_trials[i] for z_est_trials in results['z_est_all'][nl_idx]])
            
            row = [
                f"{noise_level*100:.1f}%",
                f"Linha {i+1}",
                f"{z_true[i]:.6f}",
                f"{z_est_mean:.6f}",
                f"{results['rel_errors'][nl_idx, i]:.2f}",
                f"{results['rel_errors_real'][nl_idx, i]:.2f}",
                f"{results['rel_errors_imag'][nl_idx, i]:.2f}"
            ]
            table_data.append(row)
    
    print("\nResumo da Estimação de Impedância com Diferentes Níveis de Ruído:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print MSE summary
    print("\nImpacto do Ruído na Estimação de Corrente (MSE):")
    for nl_idx, noise_level in enumerate(noise_levels):
        if noise_level > 0:  # Skip reference case
            print(f"  {noise_level*100:.1f}%: {results['mse_values'][nl_idx]:.2e} ± {results['mse_std'][nl_idx]:.2e}")
    
    return results

# =============================================================================================
# Função do Menu e Principal
# =============================================================================================
def show_menu():
    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 80)
    print("                             Projeto 2 - Grupo 13")
    print("=" * 80)
    print("\nEscolha uma opção (deve ser por ordem):")
    print("0 - Sair")
    print("1 - Rede LV")
    print("2 - Análise de Impedâncias")
    print("3 - Análise do Impacto de Ruído")
    print("=" * 80)
    try:
        option_input = input("\nOpção: ").strip()
        show_plots = False
        if option_input.endswith('p'):
            show_plots = True
            option_input = option_input[:-1]
        option = int(option_input)
        return option, show_plots
    except ValueError:
        return -1, False

def main():
    # Carregar dados
    s, topo, nBUS, z, vr, el, ni = load_data()
    # Conexões de fase não são necessárias explicitamente, pois já estão codificadas em gamma
    # Calcular pseudo-medições a partir dos m primeiros instantes (média de s)
    sp = np.mean(s[0:m, :], axis=0)  # vetor de tamanho 4
    # Construir potência média em cada fase (si) com base na média sp
    si = [[0, 0, sp[2], 0],
          [0, 0, sp[1], 0],
          [0, sp[0], 0, sp[3]]]
    
    while True:
        option, show_plots = show_menu()
        if option == 0:
            break
        elif option == 1:
            print("Rede LV selecionada.")
            # Estimar as variações de corrente usando regularização
            di = base_estimation()
            print("\nDelta i (variação de corrente estimada):")
            print(di)
            input("\nPressione Enter para continuar...")
        elif option == 2:
            print("Analyzing impedance estimation impact on current estimation...")
            dx_true, dx_est, mse = analyze_impedance_estimation()
            input("\nPressione Enter para continuar...")
        elif option == 3:
            print("Analyzing the impact of noise on impedance estimation...")
            results = noise_impact_analysis()
            input("\nPressione Enter para continuar...")
        else:
            print("Opção inválida. Tente novamente.")
            input("Pressione Enter para continuar...")

if __name__ == "__main__":
    main()
