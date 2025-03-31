###############################################################################################
# Laboratório 6 - Identificação de fase                                                       #
#                                                                                             #
# Grupo 13                                                                                    #
#                                                                                             #
# Membros:                                                                                    #
#   Francisco Tavares (103402)                                                                #
#   Marta Valente (103574)                                                                    #
###############################################################################################

# ============================================================================================================================================
# Importação de bibliotecas
# ============================================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================================================================================
# Parâmetros Globais
# ============================================================================================================================================
cosPhi=0.95             # Fator de potência (0.95 = 95% de potência ativa)
time=48                 # Tempo total de simulação (48 medições de 15 minutos = 12 horas)
m=12                    # Número de medições (12 medições de 15 minutos = 3 horas)
netFactor=0.25          # Fator de escala da rede (0.25 = 25% da rede original)
noiseFactor=0.00        # Fator de ruído (0.00 = sem ruído, 0.05 = 5% de ruído)
vr=1.0                  # Tensão de referência (1.0 pu)

# ============================================================================================================================================
# Dados de consumo
# ============================================================================================================================================
def data():
    # Conjunto de dados de consumo
    s=  [[0.0450,    0.0150,    0.0470,    0.0330],
        [0.0250,    0.0150,    0.2480,    0.0330],
        [0.0970,    0.0250,    0.3940,    0.0330],
        [0.0700,    0.0490,    0.0200,    0.4850],
        [0.1250,    0.0460,    0.0160,    0.1430],
        [0.2900,    0.0270,    0.0160,    0.0470],
        [0.2590,    0.0150,    0.0170,    0.0200],
        [0.2590,    0.0160,    0.0280,    0.0160],
        [0.4420,    0.0160,    0.0500,    0.0170],
        [0.2010,    0.0230,    0.0460,    0.0160],
        [0.2060,    0.0490,    0.0220,    0.0240],
        [0.1300,    0.0470,    0.0160,    0.0490],
        [0.0460,    0.0260,    0.0170,    0.0480]]
    s = np.array(s)

    # Topologia
    topo = [[1, 2],[2,3],[3,4]]
    nBUS=np.max(topo)

    # Impedância
    z=np.multiply([complex(0.1,0.05),complex(0.15,0.07),complex(0.2,0.1)],netFactor)

    el=1
    ni=20 # Número de iterações para o Power Flow

    return s, topo, nBUS, z, vr, el, ni

# ============================================================================================================================================
# Função para cálculo do Power Flow
# ============================================================================================================================================
def pf3ph(t, z, si, vr, el, ni, al, s, nBUS):
    # Criação de matrizes
    t=np.array(t)
    p=t[:,0]
    f=t[:,1]
    w=len(p)+1
    vp=np.zeros((nBUS-1,w), dtype=complex)
    vn=np.zeros((nBUS-1,w), dtype=complex)
    vp[0,0:w]=vr
    
    for h in range (2,nBUS):
        vp[h-1,:]=vp[h-2,:]*al  # Criar um sistema trifásico de tensões
                                # As tensões serão iguais em todos os BUS

    va=vp-vn                                                     # Tensão auxiliar
    ia=np.conj(np.divide(np.multiply(si,np.abs(va)**el),va))     # Corrente auxiliar 
    
    for it in range(ni):                                         # Iterações do Power Flow
        va=vp-vn
        ip=np.conj(np.divide(np.multiply(si,np.abs(va)**el),va)) # Corrente de fase 
        inn=-np.sum(ip,0)                                        # Corrente de neutro 
        for k in range(w-1,0,-1):                                # Ciclo backward
            n=f[k-1]
            m=p[k-1]
            ip[:,m-1]=ip[:,m-1]+ip[:,n-1]                        # Corrente de fase
            inn=-np.sum(ip,0)                                    # Corrente de neutro

        eps= np.linalg.norm(np.max(np.abs(ia-ip),0))             # Erro, comparando as novas correntes com as antigas (iteração anterior)

        if eps>1e-4:
            ia=ip
            mvp=0
            mvn=0
            eps=np.inf
        else:                      # Se o erro for menor que o limite, podemos retornar os resultados 
            mvp=(vp-vn)            # Tensões de fase a retornar
            mvn=vn[0,:]            # Tensão de neutro a retornar
            #return mvp, mvn, eps, ip, inn;
            return mvp;
        for k in range (w-1):                    # Ciclo forward
            n=f[k]                                
            m=p[k]
            vn[:,n-1]=vn[:,m-1]-z[k]*inn[n-1]    # Tensão de neutro 
            vp[:,n-1]=vp[:,m-1]-z[k]*ip[:,n-1]   # Tensão de fase
        ia=ip             # Guardar a corrente da iteração anterior
    
    return vp-vn  # Retornar tensão se o número máximo de iterações for atingido

# ============================================================================================================================================
# Função para criar e obter as medições
# ============================================================================================================================================
def get_measurements(node_index):
    """
    Obtém as medições de tensão de um nó específico da rede.
    
    Parâmetros:
    measured_node_index - índice do nó onde são realizadas as medições (padrão: 3, que corresponde ao nó 4)
    
    Retorna:
    Y - vetor de tensões medidas (3M)
    X - matriz auxiliar
    v - matriz auxiliar
    dv_abs - variações de tensão para visualização
    al - ângulo de fase
    voltages - tensões medidas para visualização
    loads - cargas dos clientes
    """
    # Obter dados
    s, topo, nBUS, z, vr, el, ni = data()
    
    # Validar o índice do nó de medição
    if node_index < 0 or node_index >= nBUS:
        print(f"Aviso: Índice de nó inválido ({node_index+1}). Usando nó 4 como padrão.")

    # Ângulo de fase
    al = np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi))
    
    # Criação de matrizes
    Y = np.zeros((3*m), dtype=complex)          # Matriz de tensões medidas
    X = np.zeros((3*m,m), dtype=complex)        # Matriz auxiliar
    v = np.zeros((m,3))                         # Matriz auxiliar
    dv_abs = np.zeros((m,3))                    # Variações de tensão para plotar
    
    # Arrays para visualização
    voltages = np.zeros((m, 3))
    loads = s[:m, :]  # Primeiras m linhas e todas as colunas de s

    for i in range(m):
        si = [[0, 0, s[i,2], 0],[0, 0, s[i,1], 0],[0, s[i,0], 0, s[i,3]]] # Ligação dos consumidores por
                                                                          # nó e por fase
                                                                          # Consumidor 1 (s[i,0]) está 
                                                                          # ligado ao Bus 2 na Fase 3
        mvp = pf3ph(topo, z, si, vr, el, ni, al, s, nBUS)
        noise = 1 + noiseFactor * np.random.randn(3)
        # Usar o nó especificado pelo parâmetro measured_node_index
        mvp[:,node_index] = np.multiply(mvp[:,node_index], noise)  # Adicionar ruído às tensões
        Y[3*(i):3*(i)+3] = mvp[:,node_index]                                # Guardar as tensões na matriz Y
        dv_abs[i,:] = vr - np.abs(mvp[:,node_index])                        # Variações de tensão (para plotar)
        
        # Armazenar valores para visualização
        voltages[i, :] = np.abs(mvp[:,node_index])  # Magnitudes das tensões do nó especificado

    Volt = np.reshape(Y, (m,3))   

    #print(f'As tensões medidas nos PMUs do nó {measured_node_index+1} são:\n', Volt)

    return Y, X, v, dv_abs, al, voltages, loads

# ============================================================================================================================================
# Função para criar a matriz X e o vetor y para o problema de mínimos quadrados
# ============================================================================================================================================
def build_regression_matrices(Y, al, s, z, node_index):
    """
    Constrói a matriz X (3M x 3N) e o vetor y (3M x 1) para o problema de mínimos quadrados
    
    Parâmetros:
    Y - vetor de tensões medidas (3M)
    al - ângulo de fase
    s - matriz de consumo (M x N)
    z - impedâncias dos troços
    node_index - índice do nó onde são feitas as medições
    
    Retorna:
    X - matriz de regressão (3M x 3N)
    y - vetor de diferenças de tensão (3M x 1)
    """
    # Número de amostras de tempo e número de clientes
    M = m  # m é o parâmetro global (12)
    N = s.shape[1]  # N é o número de colunas em s (4)
    
    # Compute dVk for each time period m   
    vz = np.zeros(3*m, dtype=complex)
    vz_abs = np.zeros(3*m, dtype=complex)
    for i in range(m):
        vz[3*(i):3*(i)+3] = np.multiply(vr,[1, al, al**2]) - Y[3*(i):3*(i)+3]                   # With PMUs info
        vz_abs[3*(i):3*(i)+3] = np.multiply(vr - np.absolute(Y[3*(i):3*(i)+3]),[1, al, al**2])   # With RTUs info
 
    # Matriz base de impedância/deslocamento
    Z_base = np.array([[2, al, al**2],
                      [1, 2*al, al**2],
                      [1, al, 2*al**2]])
    
    # Calcular as impedâncias acumuladas para cada nó
    if node_index == 3:  # Indicador no nó 4
        z_accumulated = [
            z[0],                 # Node 1: z[0] (impedância entre source e nó 1)
            z[0] + z[1],          # Node 2: z[0] + z[1] (impedância acumulada até nó 2)
            z[0] + z[1],          # Node 3: z[0] + z[1] (impedância acumulada até nó 3, igual ao nó 2)
            z[0] + z[1] + z[2]    # Node 4: z[0] + z[1] + z[2] (impedância acumulada até nó 4)
        ]   
    else:
        raise ValueError("Índice de nó inválido. Deve ser 1 ou 3.")

    # Inicialização da matriz X com dimensão 3M x 3N
    X = np.zeros((3*M, 3*N), dtype=complex)
    
    # Inicialização do vetor y com dimensão 3M
    y = np.zeros(3*M, dtype=complex)
    
    # Construção da matriz X e do vetor y
    for i in range(M):
        # Para cada amostra de tempo i
        
        # Vetor y: usa as medições do nó especificado por node_index 
        y[3*i:3*i+3] = vz[3*i:3*i+3]  
        # y[3*i:3*i+3] = vr - vz_abs[3*i:3*i+3]  # Usar vz_abs para tensões RTU
        
        # Para cada cliente j
        for j in range(N):
            # Aplicar a impedância acumulada correta para cada nó
            # Bloco X_ij = Z_base * s_ij * z_accumulated[j]
            X[3*i:3*i+3, 3*j:3*j+3] = Z_base * s[i, j] * z_accumulated[j]
    
    return X, y

# ============================================================================================================================================
# Função para resolução do sistema de mínimos quadrados e identificação de fase
# ============================================================================================================================================
def solve_and_identify(X, y, n_clients):
    """
    Resolve o sistema de mínimos quadrados β = X†y e identifica as fases dos clientes
    
    Parâmetros:
    X - matriz de regressão (3M x 3N)
    y - vetor de diferenças de tensão (3M x 1)
    n_clients - número de clientes
    
    Retorna:
    beta_matrix - matriz de coeficientes (N x 3)
    phases - fases identificadas para cada cliente (N)
    """
    # Resolução do sistema via pseudo-inversa: β = X†y
    X_pinv = np.linalg.pinv(X)
    beta = np.matmul(X_pinv, y)
    
    # Reestruturação do vetor β em uma matriz N x 3
    beta_matrix = np.zeros((n_clients, 3), dtype=complex)
    for i in range(n_clients):
        beta_matrix[i, :] = beta[3*i:3*i+3]
    
    # Cálculo das magnitudes dos coeficientes para visualização
    beta_abs = np.abs(beta_matrix)
    beta_real = np.real(beta_matrix)
    
    # Identificação das fases: argmax da parte real
    phases = np.argmax(beta_real, axis=1) + 1
    
    print("\nCoeficientes beta (parte real) para cada cliente:")
    phase_names = ["A", "B", "C"]
    for i in range(n_clients):
        print(f"Cliente {i+1}: [{beta_real[i, 0]:.4f}, {beta_real[i, 1]:.4f}, {beta_real[i, 2]:.4f}] -> Fase {phases[i]} ({phase_names[phases[i]-1]})")
    
    return beta_matrix, phases

# ============================================================================================================================================
# Funções para cálculo de fase usando mínimos quadrados e visualização dos resultados
# ============================================================================================================================================
def phase_id_least_squares():
    """
    Identificação de fase usando o método dos mínimos quadrados
    
    Parâmetros:
    normalize - se True, normaliza as colunas da matriz X (padrão: True)
    
    Retorna:
    beta_matrix - matriz de coeficientes (N x 3)
    beta_abs - magnitudes dos coeficientes (N x 3)
    phases - fases identificadas para cada cliente (N)
    voltages - tensões medidas (M x 3)
    loads - cargas dos clientes (M x N)
    """
    # Obter dados com escalamento opcional
    s, topo, nBUS, z, vr, el, ni = data()
    
    # Obter medições
    Y, _, v, dv_abs, al, voltages, loads = get_measurements(node_index=3)  # Índice 3 = nó 4
    
    # Número de clientes
    n_clients = loads.shape[1]
    
    # Construir a matriz X e o vetor y 
    X, y = build_regression_matrices(Y, al, loads, z, node_index=3)  # Índice 3 = nó 4
    
    # Resolver o sistema e identificar as fases
    beta_matrix, phases = solve_and_identify(X, y, n_clients)

    # Calcular magnitudes para fins de visualização
    beta_abs = np.abs(beta_matrix)
    
    return beta_matrix, beta_abs, phases, voltages, loads

def plot_results(voltages, loads, node_name="node_k"):
    """
    Gera dois gráficos em subplots verticais com estilo de step plot:
    1. Variação das tensões terminais por fase
    2. Leituras dos consumidores
    
    Parâmetros:
    voltages - array de dimensão (M, 3) com valores de tensão para cada fase
    loads - array de dimensão (M, 4) com valores de potência para cada consumidor
    node_name - identificador do nó para o nome do arquivo (default: "node_k")
    """
    # Criar figura com dois subplots verticais
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Configurar eixo X comum - use integers for x-axis
    x = np.arange(len(voltages))
    
    # Plot 1: Variação das tensões terminais por fase (estilo step plot)
    ax1.step(x, 1 - voltages[:, 0], where='post', linewidth=2, color='blue', label='Fase A')
    ax1.step(x, 1 - voltages[:, 1], where='post', linewidth=2, color='orange', label='Fase B')
    ax1.step(x, 1 - voltages[:, 2], where='post', linewidth=2, color='green', label='Fase C')
    
    ax1.set_title(f'Variação das tensões terminais por fase ({node_name})', fontsize=14)
    ax1.set_ylabel('Queda de Tensão [pu]', fontsize=12)
    ax1.set_xlim(-0.5, len(voltages)-0.5)
    
    # Set up proper ticks at integer positions
    ax1.set_xticks(x)
    # Convert to time labels (each step is 15 minutes)
    time_labels = [f"{i*15}" for i in x]
    ax1.set_xticklabels(time_labels)
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    
    # Plot 2: Leituras dos consumidores (estilo step plot)
    ax2.step(x, loads[:, 0], where='post', linewidth=2, color='blue', label='Carga 1')
    ax2.step(x, loads[:, 1], where='post', linewidth=2, color='orange', label='Carga 2')
    ax2.step(x, loads[:, 2], where='post', linewidth=2, color='green', label='Carga 3')
    ax2.step(x, loads[:, 3], where='post', linewidth=2, color='red', label='Carga 4')
    
    ax2.set_title('Leituras dos consumidores', fontsize=14)
    ax2.set_xlabel('Tempo [min]', fontsize=12)
    ax2.set_ylabel('Potência [pu]', fontsize=12)
    
    # Same x-axis setup as the first plot (handled by sharex=True)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    
    # Ajustar layout e mostrar gráfico
    plt.tight_layout()
    plt.show()

    # Criar diretório para guardar os gráficos
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Guardar gráficos em formato PNG com nome específico para o nó
    fig.savefig(f'plots/phase_id_results_{node_name}.png')

# ============================================================================================================================================
# DESAFIO EXTRA 1 - Função para cálculo de fase alterando o indicador para o nó 1
# ============================================================================================================================================
def phase_id_node_m():
    """
    Identificação de fase usando o método dos mínimos quadrados com medições no nó m (nó 2)
    
    Retorna:
    beta_matrix - matriz de coeficientes (N x 3)
    beta_abs - magnitudes dos coeficientes (N x 3)
    phases - fases identificadas para cada cliente (N)
    voltages - tensões medidas (M x 3)
    loads - cargas dos clientes (M x N)
    """
    # Obter dados
    s, topo, nBUS, z, vr, el, ni = data()
    
    # Obter medições no nó m (índice 1 = nó 2)
    Y, _, v, dv_abs, al, voltages, loads = get_measurements(node_index=1)
    print("\nMedições realizadas no nó 2 (nó m)!")
    
    # Número de clientes
    n_clients = loads.shape[1]
    
    # Construir matriz X e vetor y para o nó m
    # Número de amostras de tempo e número de clientes
    M = m  # m é o parâmetro global (12)
    N = s.shape[1]  # N é o número de colunas em s (4)
    
    # Calcular dVk para cada período de tempo m
    vz = np.zeros(3*m, dtype=complex)
    for i in range(m):
        vz[3*(i):3*(i)+3] = np.multiply(vr,[1, al, al**2]) - Y[3*(i):3*(i)+3]
 
    # Matriz base de impedância/deslocamento
    Z_base = np.array([[2, al, al**2],
                      [1, 2*al, al**2],
                      [1, al, 2*al**2]])
    
    # Para o nó m (índice 1), só precisamos da impedância z[0]
    # que é a impedância entre a fonte e o nó m
    z_accumulated = [z[0] for _ in range(N)]
    
    # Inicialização da matriz X com dimensão 3M x 3N
    X = np.zeros((3*M, 3*N), dtype=complex)
    
    # Inicialização do vetor y com dimensão 3M
    y = np.zeros(3*M, dtype=complex)
    
    # Construção da matriz X e do vetor y
    for i in range(M):
        # Para cada amostra de tempo i
        
        # Vetor y: usa as medições do nó m
        y[3*i:3*i+3] = vz[3*i:3*i+3]
        
        # Para cada cliente j
        for j in range(N):
            # Aplicar a impedância z[0] para todos os clientes
            # Bloco X_ij = Z_base * s_ij * z_accumulated[j]
            X[3*i:3*i+3, 3*j:3*j+3] = Z_base * s[i, j] * z_accumulated[j]
    
    # Resolver o sistema e identificar as fases
    X_pinv = np.linalg.pinv(X)
    beta = np.matmul(X_pinv, y)
    
    # Reestruturação do vetor β em uma matriz N x 3
    beta_matrix = np.zeros((n_clients, 3), dtype=complex)
    for i in range(n_clients):
        beta_matrix[i, :] = beta[3*i:3*i+3]
    
    # Cálculo das magnitudes dos coeficientes para visualização
    beta_abs = np.abs(beta_matrix)
    beta_real = np.real(beta_matrix)
    
    # Identificação das fases: argmax da parte real
    phases = np.argmax(beta_real, axis=1) + 1
    
    print("\nCoeficientes beta (parte real) para cada cliente (medição no nó 2):")
    phase_names = ["A", "B", "C"]
    for i in range(n_clients):
        print(f"Cliente {i+1}: [{beta_real[i, 0]:.4f}, {beta_real[i, 1]:.4f}, {beta_real[i, 2]:.4f}] -> Fase {phases[i]} ({phase_names[phases[i]-1]})")
    
    return beta_matrix, beta_abs, phases, voltages, loads


# ============================================================================================================================================
# DESAFIO EXTRA 2 - Função para implementar diferentes valores de ruído no método dos mínimos quadrados
# ============================================================================================================================================

def compare_noise_impact_on_ls_both(iterations=100):
    """
    Avalia a robustez do modelo LS a diferentes níveis de ruído para duas configurações:
    - Medição no nó k (node_index=3)
    - Medição no nó m (node_index=1)
    
    Mede a percentagem de acertos (accuracy) comparando com o caso sem ruído para cada método.
    
    Retorna:
      noise_values - lista de valores de ruído testados
      ls_means_k - precisão média para cada nível de ruído (medição no nó k)
      ls_means_m - precisão média para cada nível de ruído (medição no nó m)
    """
    print("\n--- Comparação de Robustez do LS para nó k e nó m ---")
    
    # Valores de ruído a testar
    noise_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    
    # Arrays para armazenar resultados (cada linha para um noise level, cada coluna for a iteração)
    ls_accuracy_k = np.zeros((len(noise_values), iterations))
    ls_accuracy_m = np.zeros((len(noise_values), iterations))
    
    # Obter os dados
    s, topo, nBUS, z, vr, el, ni = data()
    global noiseFactor
    original_noise = noiseFactor  # salvar o valor original
    
    # Ângulo de fase
    al = np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi))
    
    # Número de clientes
    n_clients = s[:m, :].shape[1]
    
    # Definir ruído para zero e obter "verdadeiras" fases para cada medição
    
    # Para nó k (node_index=3)
    noiseFactor = 0.0
    Y_k, _, _, _, _, _, loads = get_measurements(node_index=3)
    X_k, y_k = build_regression_matrices(Y_k, al, loads, z, node_index=3)
    X_pinv = np.linalg.pinv(X_k)
    beta = np.matmul(X_pinv, y_k)
    beta_matrix_k = np.zeros((n_clients, 3), dtype=complex)
    for i in range(n_clients):
        beta_matrix_k[i, :] = beta[3*i:3*i+3]
    true_phases_k = np.argmax(np.real(beta_matrix_k), axis=1) + 1
    print(f"Fases verdadeiras para medição no nó k (sem ruído): {true_phases_k}")
    
    # Para nó m (node_index=1) usando a nova função phase_id_node_m()
    noiseFactor = 0.0
    beta_matrix_m, _, true_phases_m, _, _ = phase_id_node_m()
    print(f"Fases verdadeiras para medição no nó m (sem ruído): {true_phases_m}")
    
    # Agora, para cada nível de ruído, rodamos várias iterações para cada configuração
    for n_idx, noise_value in enumerate(noise_values):
        print(f"Processando nível de ruído: {noise_value:.3f}", end="")
        accuracies_k = []
        accuracies_m = []
        for it in range(iterations):
            # Ajusta o noiseFactor para o nível atual
            noiseFactor = noise_value
            
            # --- Método para nó k (original) ---
            Y_k, _, _, _, _, _, loads = get_measurements(node_index=3)
            X_k, y_k = build_regression_matrices(Y_k, al, loads, z, node_index=3)
            X_pinv = np.linalg.pinv(X_k)
            beta = np.matmul(X_pinv, y_k)
            beta_matrix_k = np.zeros((n_clients, 3), dtype=complex)
            for i in range(n_clients):
                beta_matrix_k[i, :] = beta[3*i:3*i+3]
            predicted_phases_k = np.argmax(np.real(beta_matrix_k), axis=1) + 1
            accuracy_k = np.sum(predicted_phases_k == true_phases_k) / len(true_phases_k) * 100
            ls_accuracy_k[n_idx, it] = accuracy_k
            accuracies_k.append(accuracy_k)
            
            # --- Método para nó m (novo) ---
            _, _, predicted_phases_m, _, _ = phase_id_node_m()
            accuracy_m = np.sum(predicted_phases_m == true_phases_m) / len(true_phases_m) * 100
            ls_accuracy_m[n_idx, it] = accuracy_m
            accuracies_m.append(accuracy_m)
            
            if (it + 1) % 5 == 0:
                print(".", end="")
        print(f" Média nó k: {np.mean(accuracies_k):.2f}%, Média nó m: {np.mean(accuracies_m):.2f}%")
    
    noiseFactor = original_noise  # restaurar o valor original
    ls_means_k = np.mean(ls_accuracy_k, axis=1)
    ls_means_m = np.mean(ls_accuracy_m, axis=1)
    
    print("\nResultados finais:")
    for n_idx, noise_value in enumerate(noise_values):
        print(f"Ruído: {noise_value:.3f} -> Nó k: {ls_means_k[n_idx]:.2f}%, Nó m: {ls_means_m[n_idx]:.2f}%")
    
    return noise_values, ls_means_k, ls_means_m

def plot_noise_vs_accuracy_both(noise_values, ls_means_k, ls_means_m):
    """
    Plota a relação entre o nível de ruído e a precisão da identificação de fase para:
      - Medição no nó k (original)
      - Medição no nó m (novo)
    Os resultados são sobrepostos para comparação.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot para nó k
    plt.plot(noise_values, ls_means_k, 'o-', color='blue', linewidth=2, markersize=8, label='Nó k (node_index=3)')
    # Plot para nó m
    plt.plot(noise_values, ls_means_m, 's-', color='red', linewidth=2, markersize=8, label='Nó m (node_index=1)')
    
    plt.title("Robustez do LS vs Nível de Ruído (Comparação Nó k vs Nó m)", fontsize=14)
    plt.xlabel("Fator de Ruído", fontsize=12)
    plt.ylabel("Precisão (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Salvar e mostrar o gráfico
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig("plots/noise_vs_ls_accuracy_both.png")
    plt.show()
    
    return None


# ============================================================================================================================================
# Função Principal
# ============================================================================================================================================
def show_menu():
    """Exibe o menu principal e retorna a opção selecionada pelo utilizador."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa o ecrã
    print("=" * 80)
    print("                         LABORATÓRIO 6 - ID de Fase                 ")
    print("=" * 80)
    print("\nEscolha uma opção:")
    print("0 - Sair")
    print("1 - Método dos Mínimos Quadrados")
    print("2 - Alteração do nó de medição de k para m")
    print("3 - Variação do Ruído")

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
    # Variáveis para armazenar resultados entre opções do menu
    Y = None
    
    # Método dos mínimos quadrados
    beta_matrix = None
    beta_abs = None
    ls_phases = None
    
    # Dados para visualização
    voltages = None
    loads = None

    while True:
        option, show_plots = show_menu()
        if option == 0:
            break

        elif option == 1:
            # Calcular tensões medidas pelos PMUs
            Y, X, v, dv_abs, al, voltages, loads = get_measurements(node_index=3)
            print("\nAs tensões foram calculadas com sucesso!")
            
            # Calcular betas e identificar fases (método dos mínimos quadrados)
            beta_matrix, beta_abs, ls_phases, voltages, loads = phase_id_least_squares()
            print("\nOs betas e fases (método dos mínimos quadrados) foram calculados com sucesso!")

            # Gerar gráficos
            plot_results(voltages, loads, "node_k")
            print("\nOs gráficos foram gerados com sucesso!")
            
        elif option == 2:
            # Calcular betas e identificar fases com medições no nó m
            beta_matrix_m, beta_abs_m, ls_phases_m, voltages_m, loads_m = phase_id_node_m()
            print("\nOs betas e fases (com medições no nó 2) foram calculados com sucesso!")
            
            # Gerar gráficos para medições no nó m
            plot_results(voltages_m, loads_m, "node_m")
            print("\nOs gráficos foram gerados com sucesso!")

        elif option == 3:
            # Comparar robustez ao ruído
            noise_values, ls_means_k, ls_means_m = compare_noise_impact_on_ls_both()
            print("\nOs resultados foram calculados com sucesso!")

            # Gerar gráfico
            plot_noise_vs_accuracy_both(noise_values, ls_means_k, ls_means_m)
            print("\nO gráfico foi gerado com sucesso!")

        else:
            print("Opção inválida. Tente novamente.")
            input("\nPrima ENTER para continuar...")

if __name__ == "__main__":
    main()