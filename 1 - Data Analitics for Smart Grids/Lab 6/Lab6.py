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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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
def get_measurements():
    # Obter dados
    s, topo, nBUS, z, vr, el, ni = data()
    
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
        mvp[:,3] = np.multiply(mvp[:,3], noise)                       # Adicionar ruído às tensões
        Y[3*(i):3*(i)+3] = mvp[:,3]                                  # Guardar as tensões na matriz Y
        dv_abs[i,:] = vr - np.abs(mvp[:,3])                            # Variações de tensão (apenas para plotar)
        
        # Armazenar valores para visualização
        voltages[i, :] = np.abs(mvp[:,3])  # Magnitudes das tensões

    Volt = np.reshape(Y, (m,3))   

    print('As tensões medidas nos PMUs são:\n', Volt)

    return Y, X, v, dv_abs, al, voltages, loads

# ============================================================================================================================================
# Função para criar a matriz X e o vetor y para o problema de mínimos quadrados
# ============================================================================================================================================
def build_regression_matrices(Y, al, s):
    """
    Constrói a matriz X (3M x 3N) e o vetor y (3M x 1) para o problema de mínimos quadrados
    
    Parâmetros:
    Y - vetor de tensões medidas (3M)
    al - ângulo de fase
    s - matriz de consumo (M x N)
    
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
    
    print("Matriz X construída com dimensão:", X.shape)
    print("Vetor y construído com dimensão:", y.shape)
    
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
    
    
    # Impressão dos resultados
    print("\nCoeficientes beta (complexos) para cada cliente:")
    print(beta_matrix)
    
    print("\nParte real dos coeficientes beta:")
    print(beta_real)
    
    print("\nMagnitudes dos coeficientes beta:")
    print(beta_abs)
    
    print("\nFases identificadas para cada cliente:")
    for i in range(n_clients):
        phase_name = ["A", "B", "C"][phases[i]-1]
        print(f"Cliente {i+1}: Fase {phases[i]} (Fase {phase_name})")
    
    return beta_matrix, phases

# ============================================================================================================================================
# Função para cálculo de fase usando mínimos quadrados
# ============================================================================================================================================
def phase_id_least_squares():
    """
    Identificação de fase usando o método dos mínimos quadrados
    
    Retorna:
    beta_matrix - matriz de coeficientes (N x 3)
    beta_abs - magnitudes dos coeficientes (N x 3)
    phases - fases identificadas para cada cliente (N)
    voltages - tensões medidas (M x 3)
    loads - cargas dos clientes (M x N)
    """
    # Obter dados
    s, topo, nBUS, z, vr, el, ni = data()
    
    # Obter medições
    Y, _, v, dv_abs, al, voltages, loads = get_measurements()
    
    # Número de clientes
    n_clients = loads.shape[1]
    
    # Construir a matriz X e o vetor y
    X, y = build_regression_matrices(Y, al, loads)
    
    # Resolver o sistema e identificar as fases
    beta_matrix, phases = solve_and_identify(X, y, n_clients)
    
    # Calcular magnitudes para fins de visualização
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
    noise_values = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    al_ref = np.exp(np.multiply(np.multiply(complex(0, -1), 2 / 3), np.pi))

    reference_phases = None  # Fases com ruído 0 (referência)
    ls_means = []
    ls_stds = []

    for noise in noise_values:
        globals()['noiseFactor'] = noise
        print(f"\n>> Noise = {noise}")

        accs = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            _, _, ls_phases, voltages, loads = phase_id_least_squares()

            if noise == 0.0 and reference_phases is None:
                reference_phases = ls_phases  # Save baseline
                accs.append(100.0)
                continue

            correct = [int(ls_phases[i]) == int(reference_phases[i]) for i in range(4)]
            acc = 100 * sum(correct) / 4
            accs.append(acc)

        ls_mean = np.mean(accs)
        ls_std = np.std(accs)

        ls_means.append(ls_mean)
        ls_stds.append(ls_std)

        print(f"LS Accuracy vs no-noise: {ls_mean:.2f}% ± {ls_std:.2f}%")

    print("\n=== RESUMO FINAL ===")
    for i, noise in enumerate(noise_values):
        print(f"Noise: {noise:.2f} | LS Accuracy: {ls_means[i]:.2f}% ± {ls_stds[i]:.2f}%")

    return noise_values, ls_means, ls_stds


def plot_noise_vs_accuracy_combined(noise_values, ls_acc, nn_acc, ls_std=None, nn_std=None):
    plt.figure(figsize=(10, 6))

    # Plot LS
    plt.errorbar(noise_values, ls_acc, yerr=ls_std, fmt='o-', label='Least Squares (LS)', capsize=4)

    # Plot NN
    plt.errorbar(noise_values, nn_acc, yerr=nn_std, fmt='s--', label='Neural Network (NN)', capsize=4)

    plt.title("Robustez dos métodos vs Nível de Ruído")
    plt.xlabel("Fator de Ruído")
    plt.ylabel("Precisão (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Guardar e mostrar
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/noise_vs_combined_accuracy.png")
    plt.show()

    return noise_values, ls_acc, nn_acc, ls_std, nn_std


def plot_noise_vs_accuracy(noise_values, ls_acc, _, ls_std=None):
    plt.figure(figsize=(10, 6))
    plt.errorbar(noise_values, ls_acc, yerr=ls_std, fmt='o-', label='Least Squares (LS)', capsize=4)

    plt.title("Robustez do LS vs Nível de Ruído")
    plt.xlabel("Fator de Ruído")
    plt.ylabel("Precisão (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/noise_vs_ls_accuracy.png")
    plt.show()

# ============================================================================================================================================
# Função para plotar os resultados
# ============================================================================================================================================
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
    
    # Configurar eixo X comum
    x = range(len(voltages))
    
    # Plot 1: Variação das tensões terminais por fase (estilo step plot)
    ax1.step(x, 1 - voltages[:, 0], where='mid', linewidth=2, color='blue', label='Fase A')
    ax1.step(x, 1 - voltages[:, 1], where='mid', linewidth=2, color='orange', label='Fase B')
    ax1.step(x, 1 - voltages[:, 2], where='mid', linewidth=2, color='green', label='Fase C')
    
    ax1.set_title('Variação das tensões terminais por fase', fontsize=14)
    ax1.set_ylabel('Tensão [pu]', fontsize=12)
    ax1.set_xlim(-0.5, len(voltages)-0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i}" for i in x])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    
    # Plot 2: Leituras dos consumidores (estilo step plot)
    ax2.step(x, loads[:, 0], where='mid', linewidth=2, color='blue', label='Carga 1')
    ax2.step(x, loads[:, 1], where='mid', linewidth=2, color='orange', label='Carga 2')
    ax2.step(x, loads[:, 2], where='mid', linewidth=2, color='green', label='Carga 3')
    ax2.step(x, loads[:, 3], where='mid', linewidth=2, color='red', label='Carga 4')
    
    ax2.set_title('Leituras dos consumidores', fontsize=14)
    ax2.set_xlabel('Marca temporal [15min]', fontsize=12)
    ax2.set_ylabel('Potência [pu]', fontsize=12)
    ax2.set_xlim(-0.5, len(voltages)-0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{i*15}" for i in x])  # Marca temporal [15min]
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
# Função para comparação das implementações
# ============================================================================================================================================
def phase_id_least_squares_with_data(voltages, loads, al):
    """Version of phase_id_least_squares that accepts pre-generated data"""
    # Get other necessary data
    s, topo, nBUS, z, vr, el, ni = data()
    
    # Reconstruct Y from voltages
    Y = np.zeros((3*m), dtype=complex)
    for i in range(m):
        # Convert real voltages back to complex with phase angles
        Y[3*i] = voltages[i, 0]  # Phase A (magnitude only)
        Y[3*i+1] = voltages[i, 1] * al  # Phase B with 120° phase shift
        Y[3*i+2] = voltages[i, 2] * al**2  # Phase C with 240° phase shift
    
    # Number of clients
    n_clients = loads.shape[1]
    
    # Build regression matrices and solve
    X, y = build_regression_matrices(Y, al, loads)
    beta_matrix, phases = solve_and_identify(X, y, n_clients)
    beta_abs = np.abs(beta_matrix)
    
    return beta_matrix, beta_abs, phases
   
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
    print("2 - Mudar onde o indicador esta na rede LV po lo mais a meio SPOILER: vai ser pior porque não tem tanta informação")
    print("3 - Precuisão do modelo para diferentes valores de ruído")

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

            print("\nA Mudar onde o indicador esta na rede LV po lo mais a meio SPOILER: vai ser pior porque não tem tanta informação")


        elif option == 3:
            # Comparar robustez ao ruído
            noise_values, ls_means, ls_stds = compare_noise_impact_on_ls()
            plot_noise_vs_accuracy(noise_values, ls_means, None, ls_stds)
            print("\nGráfico gerado com sucesso!")

        else:
            print("Opção inválida. Tente novamente.")
            input("\nPrima ENTER para continuar...")

if __name__ == "__main__":
    main()

# Outra opção é em vez de só ter um medidor k, ter dois na rede LV!
# Outra ainda é aumentar a potência num cliente e ver se isso melhora a precisão ou não!