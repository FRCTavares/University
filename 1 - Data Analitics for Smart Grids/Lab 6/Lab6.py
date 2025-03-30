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
cosPhi=0.95
time=48
m=12
netFactor=0.25
noiseFactor=0.00

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
    topo=[[1, 2],[2,3],[3,4]]
    nBUS=np.max(topo)

    # Impedância
    z=np.multiply([complex(0.1,0.05),complex(0.15,0.07),complex(0.2,0.1)],netFactor)

    vr=1 # Tensão de referência
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
def get_measurements(measured_node_index=3):
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
    if measured_node_index < 0 or measured_node_index >= nBUS:
        print(f"Aviso: Índice de nó inválido ({measured_node_index+1}). Usando nó 4 como padrão.")
        measured_node_index = 3  # Valor padrão: nó 4 (índice 3)
    
    # Ângulo de fase
    al = np.exp(np.multiply(np.multiply(complex(0,-1),2/3),np.pi))
    
    # Criação de matrizes
    Y = np.zeros((3*m), dtype=complex)
    X = np.zeros((3*m,m), dtype=complex)
    v = np.zeros((m,3))
    dv_abs = np.zeros((m,3))
    
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
        mvp[:,measured_node_index] = np.multiply(mvp[:,measured_node_index], noise)  # Adicionar ruído às tensões
        Y[3*(i):3*(i)+3] = mvp[:,measured_node_index]                                # Guardar as tensões na matriz Y
        dv_abs[i,:] = vr - np.abs(mvp[:,measured_node_index])                        # Variações de tensão (para plotar)
        
        # Armazenar valores para visualização
        voltages[i, :] = np.abs(mvp[:,measured_node_index])  # Magnitudes das tensões do nó especificado

    Volt = np.reshape(Y, (m,3))   

    #print(f'As tensões medidas nos PMUs do nó {measured_node_index+1} são:\n', Volt)

    return Y, X, v, dv_abs, al, voltages, loads

# ============================================================================================================================================
# Função para criar a matriz X e o vetor y para o problema de mínimos quadrados
# ============================================================================================================================================
def build_regression_matrices(Y, al, s, normalize=False):
    """
    Constrói a matriz X (3M x 3N) e o vetor y (3M x 1) para o problema de mínimos quadrados
    
    Parâmetros:
    Y - vetor de tensões medidas (3M)
    al - ângulo de fase
    s - matriz de consumo (M x N)
    normalize - se True, normaliza as colunas da matriz X (padrão: False)
    
    Retorna:
    X - matriz de regressão (3M x 3N)
    y - vetor de diferenças de tensão (3M x 1)
    """
    # Número de amostras de tempo e número de clientes
    M = m  # m é o parâmetro global (12)
    N = s.shape[1]  # N é o número de colunas em s (4)
    
    # Matriz Z (matriz de impedância/deslocamento)
    Z = np.array([[2, al, al**2],
                  [1, 2*al, al**2],
                  [1, al, 2*al**2]])
    
    # Inicialização da matriz X com dimensão 3M x 3N
    X = np.zeros((3*M, 3*N), dtype=complex)
    
    # Inicialização do vetor y com dimensão 3M, dtype=complex)
    y = np.zeros(3*M, dtype=complex)
    
    # Tensão de referência
    vr = 1
    
    # Construção da matriz X e do vetor y
    for i in range(M):
        # Para cada amostra de tempo i
        
        # Vetor y: diferença entre tensão de referência e tensões medidas
        # y_m = [V_ref,a, V_ref,b, V_ref,c]^T - [V_m,a, V_m,b, V_m,c]^T
        y[3*i:3*i+3] = np.multiply(vr, [1, al, al**2]) - Y[3*i:3*i+3]
        
        # Para cada cliente j
        for j in range(N):
            # Bloco X_ij = Z * s_ij
            # Onde Z é a matriz de impedância/deslocamento e s_ij é o consumo do cliente j no tempo i
            X[3*i:3*i+3, 3*j:3*j+3] = Z * s[i, j]
    
    # Normalização das colunas da matriz X (opcional)
    if normalize:
        # Para cada cliente j
        for j in range(N):
            # Normalizar cada bloco 3x3 da matriz X
            for k in range(3):
                col_idx = 3*j + k
                # Normalização L2 da coluna
                col_norm = np.linalg.norm(X[:, col_idx])
                if col_norm > 0:
                    X[:, col_idx] = X[:, col_idx] / col_norm
    
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
def phase_id_least_squares(normalize=True):
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
    Y, _, v, dv_abs, al, voltages, loads = get_measurements()
    
    # Número de clientes
    n_clients = loads.shape[1]
    
    # Construir a matriz X e o vetor y com normalização opcional
    X, y = build_regression_matrices(Y, al, loads, normalize)
    
    # Resolver o sistema e identificar as fases
    beta_matrix, phases = solve_and_identify(X, y, n_clients)

    # Calcular magnitudes para fins de visualização
    beta_abs = np.abs(beta_matrix)
    
    return beta_matrix, beta_abs, phases, voltages, loads

def plot_results(voltages, loads):
    """
    Gera dois gráficos em subplots verticais com estilo de step plot:
    1. Variação das tensões terminais por fase
    2. Leituras dos consumidores
    
    Parâmetros:
    voltages - array de dimensão (M, 3) com valores de tensão para cada fase
    loads - array de dimensão (M, 4) com valores de potência para cada consumidor
    """
    # Criar figura com dois subplots verticais
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Configurar eixo X comum - use integers for x-axis
    x = np.arange(len(voltages))
    
    # Plot 1: Variação das tensões terminais por fase (estilo step plot)
    ax1.step(x, 1 - voltages[:, 0], where='post', linewidth=2, color='blue', label='Fase A')
    ax1.step(x, 1 - voltages[:, 1], where='post', linewidth=2, color='orange', label='Fase B')
    ax1.step(x, 1 - voltages[:, 2], where='post', linewidth=2, color='green', label='Fase C')
    
    ax1.set_title('Variação das tensões terminais por fase', fontsize=14)
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
    
    # Guardar gráficos em formato PNG
    fig.savefig('plots/phase_id_results.png')

# ============================================================================================================================================
# DESAFIO EXTRA 1 - Função para cálculo de fase alterando o indicador para o nó 1
# ============================================================================================================================================
def switch_indicator_position(node_index=0, normalize=True):
    '''
    Altera a posição do indicador de fase para o nó especificado e recalcula as fases dos clientes.
    
    Argumentos:
    - node_index: índice do nó a usar como indicador (0, 1, 2 ou 3)
    - normalize: se True, normaliza as colunas da matriz X (padrão: True)
    - scale_factor: fator de escala para as cargas (padrão: 20.0)
    
    Retorna:
    - beta_matrix: matriz dos coeficientes do modelo LS
    - beta_abs: magnitudes dos coeficientes (|beta|)
    - phases: fases identificadas para os clientes
    - voltages: tensões medidas
    - loads: cargas dos clientes
    '''
    # Obter dados com escalamento opcional
    s, topo, nBUS, z, vr, el, ni = data()

    # Obter medições com novo indicador
    Y, _, v, dv_abs, al, voltages, loads = get_measurements(node_index)

    # Número de clientes
    n_clients = loads.shape[1]

    # Construir a matriz X e o vetor y com normalização opcional
    X, y = build_regression_matrices(Y, al, loads, normalize)

    # Resolver o sistema e identificar as fases
    beta_matrix, phases = solve_and_identify(X, y, n_clients)

    # Calcular magnitudes para visualização
    beta_abs = np.abs(beta_matrix)

    return beta_matrix, beta_abs, phases, voltages, loads
# ============================================================================================================================================
# DESAFIO EXTRA 2 - Função para implementar diferentes valores de ruído no método dos mínimos quadrados
# ============================================================================================================================================
def compare_noise_impact_on_ls(seeds=[0, 1, 2, 3, 4]):
    """
    Avalia a robustez do modelo LS a diferentes níveis de ruído.
    Mede a percentagem de acertos comparando com o caso sem ruído.
    """
    print("\n--- Robustez do modelo LS a ruído ---")
    noise_values = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
    al_ref = np.exp(np.multiply(np.multiply(complex(0, -1), 2 / 3), np.pi))

    reference_phases = None  # Fases com ruído 0 (referência)
    ls_means_1 = []
    ls_stds_1 = []
    ls_means_4 = []
    ls_stds_4 = []

    for noise in noise_values:
        globals()['noiseFactor'] = noise
        print(f"\n>> Noise = {noise}")

        accs = []

        # Nó = 4
        for seed in seeds:
            np.random.seed(seed)

            _, _, ls_phases, voltages, loads = phase_id_least_squares()

            if noise == 0.0 and reference_phases is None:
                reference_phases = ls_phases  # Save baseline
                accs.append(100.0)
                continue

            correct = [int(ls_phases[i]) == int(reference_phases[i]) for i in range(4)]
            acc = 100 * sum(correct) / 4
            accs.append(acc)

        ls_mean_4 = np.mean(accs)
        ls_std_4 = np.std(accs)

        ls_means_4.append(ls_mean_4)
        ls_stds_4.append(ls_std_4) 

        print(f"LS Accuracy (node 4) vs no-noise: {ls_mean_4:.2f}% ± {ls_std_4:.2f}%")

        # Nó = 1
        for seed in seeds:
            np.random.seed(seed)

            _, _, ls_phases, voltages, loads = switch_indicator_position()

            if noise == 0.0 and reference_phases is None:
                reference_phases = ls_phases  # Save baseline
                accs.append(100.0)
                continue

        correct = [int(ls_phases[i]) == int(reference_phases[i]) for i in range(4)]
        acc = 100 * sum(correct) / 4
        accs.append(acc)

        ls_mean_1 = np.mean(accs)
        ls_std_1 = np.std(accs)

        ls_means_1.append(ls_mean_1)
        ls_stds_1.append(ls_std_1)

        print(f"LS Accuracy (node 1) vs no-noise: {ls_mean_1:.2f}% ± {ls_std_1:.2f}%")

    return noise_values, ls_means_1, ls_means_4

def plot_noise_vs_accuracy(noise_values, ls_means_1, ls_means_4):
    plt.figure(figsize=(10, 6))

    plt.errorbar(noise_values, ls_means_1, fmt='o-', label='Nó 1', color='blue', capsize=5)
    plt.errorbar(noise_values, ls_means_4, fmt='o-', label='Nó 4', color='red', capsize=5)

    plt.title("Robustez do LS vs Nível de Ruído")
    plt.xlabel("Fator de Ruído")
    plt.ylabel("Precisão (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/noise_vs_ls_accuracy.png")
    plt.show()

    # Plot 

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
    print("2 - Alteração do Indicador de Fase")
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
            Y, _, _, _, al, voltages, loads = get_measurements()
            print("\nAs tensões foram calculadas com sucesso!")
            
            # Calcular betas e identificar fases (método dos mínimos quadrados)
            beta_matrix, beta_abs, ls_phases, voltages, loads = phase_id_least_squares()
            print("\nOs betas e fases (método dos mínimos quadrados) foram calculados com sucesso!")

            # Gerar gráficos
            plot_results(voltages, loads)
            print("\nOs gráficos foram gerados com sucesso!")
            
        elif option == 2:
            # Alterar a posição do indicador de fase para o nó 1
            beta_matrix, beta_abs, ls_phases, voltages, loads = switch_indicator_position(1)  # Índice 0 = nó 1
            print("\nAs fases foram recalculadas com sucesso!")
            
            # Gerar gráficos
            plot_results(voltages, loads)
            print("\nOs gráficos foram gerados com sucesso!")

        elif option == 3:
            # Comparar robustez ao ruído
            noise_values, ls_means_1, ls_means_4 = compare_noise_impact_on_ls()
            print("\nOs resultados foram calculados com sucesso!")

            # Gerar gráfico
            plot_noise_vs_accuracy(noise_values, ls_means_1, ls_means_4)
            print("\nO gráfico foi gerado com sucesso!")

        else:
            print("Opção inválida. Tente novamente.")
            input("\nPrima ENTER para continuar...")

if __name__ == "__main__":
    main()