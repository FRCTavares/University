###############################################################################################
# Projeto 1                                                                                   #
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
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Definir a semente aleatória para reprodutibilidade
np.random.seed(42)

# ============================================================================================================================================
# Parâmetros globais
# ============================================================================================================================================
noiseFactor = 0.0025     # ruído
networkFactor = 100      # para alterar as características da rede (Y)
PtestFactor = 3          # para obter perdas semelhantes aos dados de treino

# ============================================================================================================================================
# Funções Iniciais e Globais
# ============================================================================================================================================
def apply_temporal_smoothing(P, alpha=0.8):
    """
    Applies exponential smoothing to simulate temporal correlation.
    
    Parameters:
        P (numpy.ndarray): Power injection matrix of shape (time_steps, num_buses)
        alpha (float): Smoothing factor (0-1). Higher values = more smoothing
        
    Returns:
        numpy.ndarray: Smoothed power injection matrix
    """
    P_smooth = np.copy(P)
    for t in range(1, P.shape[0]):
        P_smooth[t] = alpha * P_smooth[t - 1] + (1 - alpha) * P[t]
    return P_smooth

def load_data(file_path='DATA.xlsx', apply_smoothing=False, alpha=0.8):
    """
    Carrega os dados do arquivo Excel e retorna as informações necessárias.
    
    Parameters:
        file_path (str): Path to the Excel file containing the data
        apply_smoothing (bool): Whether to apply temporal smoothing
        alpha (float): Smoothing parameter (higher = more smoothing)
        
    Returns:
        tuple: (SlackBus, Net_Info, P, Ptest) or (None, None, None, None) if error occurs
    """
    try:
        Info = np.array(pd.read_excel(file_path, sheet_name='Info', header=None))
        SlackBus = Info[0, 1]  # Informação acerca do bus slack

        Net_Info = np.array(pd.read_excel(file_path, sheet_name='Y_Data'))
        Power_Info = np.array(pd.read_excel(file_path, sheet_name='Load(t,Bus)'))
        Power_Info = np.delete(Power_Info, [0], 1)
        Power_Test = np.array(pd.read_excel(file_path, sheet_name='Test_Load(t,Bus)'))
        Power_Test = np.delete(Power_Test, [0], 1)

        time = Power_Info.shape[0]
        P = Power_Info
        Ptest = Power_Test * PtestFactor
        
        # Apply temporal smoothing if requested
        if apply_smoothing:
            print("✔️ Applying temporal smoothing to introduce correlation...")
            P = apply_temporal_smoothing(P, alpha)
            Ptest = apply_temporal_smoothing(Ptest, alpha)

        return SlackBus, Net_Info, P, Ptest
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None, None, None
    except KeyError as e:
        print(f"Error: Sheet {e} not found in Excel file.")
        return None, None, None, None
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return None, None, None, None
   
def ensure_plot_directories():
    """Create plot directories if they don't exist."""
    for dir_path in ['Plots/Outer_Product', 'Plots/Edge_Reduced', 'Plots/Squares_Only', 
                     'Plots/Squares_Reduced', 'Plots/Noise', 'Plots/Comp']:
        os.makedirs(dir_path, exist_ok=True)

def compute_PL2(P_mat, B, Cl, Gv, noise_factor=0.0025):
    """
    Calcula as perdas físicas (PL2) para cada instante de tempo utilizando a Eq (13).
    
    Parameters:
      P_mat: Matriz de potência, com dimensão (n_instantes, nBus)
      B: Matriz de susceptância, dimensão (nBus-1, nBus-1)
      Cl: Matriz de incidência (com slack removido), dimensão (nBus-1, nLines)
      Gv: Vetor de condutância das linhas, dimensão (1, nLines)
      noise_factor: Fator de ruído multiplicativo (default: 0.0025)
      
    Returns:
      PL2_vec: Vetor das perdas para cada instante (valores escalares)
    """
    n_instantes = P_mat.shape[0]
    PL2_vec = np.zeros(n_instantes)
    invB = inv(B)  # Pré-computa a inversa de B
    for m in range(n_instantes):
        teta = invB @ P_mat[m, :].T
        grau = Cl.T @ teta
        PL2_m = (2 * Gv) @ (1 - np.cos(grau))
        PL2_scalar = PL2_m.item()
        PL2_vec[m] = PL2_scalar * (1 + np.random.normal(0, noise_factor))
    return PL2_vec

def build_network_matrices(SlackBus, Net_Info, network_factor=100):
    """
    Constrói as matrizes da rede (Y, G, B, C, Cl, Gv, Gd) e retorna os valores.
    
    Parameters:
        SlackBus: The slack bus index
        Net_Info: Network information array
        network_factor: Factor to adjust network characteristics (default: 100)
        
    Returns:
        tuple: Network matrices and related values
    """
    nBus = int(max(np.max(Net_Info[:, 0]), np.max(Net_Info[:, 1])))
    nLines = Net_Info.shape[0]
    Y = np.zeros((nBus, nBus), dtype=complex)

    for i in range(Net_Info.shape[0]):
        y_aux = Net_Info[i, 2].replace(",", ".").replace("i", "j")
        Y[Net_Info[i, 0]-1, Net_Info[i, 0]-1] += complex(y_aux) * networkFactor
        Y[Net_Info[i, 1]-1, Net_Info[i, 1]-1] += complex(y_aux) * networkFactor
        Y[Net_Info[i, 0]-1, Net_Info[i, 1]-1] -= complex(y_aux) * networkFactor
        Y[Net_Info[i, 1]-1, Net_Info[i, 0]-1] -= complex(y_aux) * networkFactor

    # Remover o bus slack da matriz Y
    Yl = np.delete(Y, SlackBus-1, axis=0)
    Yl = np.delete(Yl, SlackBus-1, axis=1)

    G = Yl.real  # Matriz de condutância
    B = Yl.imag  # Matriz de susceptância

    # Construção da matriz de incidência C
    C = np.zeros((nBus, nLines))
    linha_indices = []  # Armazenar os índices dos barramentosconectados por cada linha
    nLine_Aux = 0
    for i in range(nBus):
        for j in range(i+1, nBus):
            if abs(Y[i, j]) != 0:
                C[i, nLine_Aux] = 1
                C[j, nLine_Aux] = -1
                linha_indices.append((i, j))  # Guarda os barramentosconectados por esta linha
                nLine_Aux += 1

    # Remover o bus slack da matriz C para obter Cl
    Cl = np.delete(C, SlackBus-1, axis=0)

    # Construção dos vetores de condutância das linhas
    Gv = np.zeros((1, nLines))
    Gd = np.zeros((nLines, nLines))
    nLine_Aux = 0
    for i in range(nBus):
        for j in range(i+1, nBus):
            if abs(Y[i, j]) != 0:
                Gv[0, nLine_Aux] = -np.real(Y[i, j])
                Gd[nLine_Aux, nLine_Aux] = -np.real(Y[i, j])
                nLine_Aux += 1
                
    return nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices

def plot_initial_data():
    """Visualizes the initial power injection data."""
    print("\n=== Visualizando Dados Iniciais ===\n")
    
    # Load data (both original and smoothed)
    SlackBus, Net_Info, P, Ptest = load_data(apply_smoothing=False)
    _, _, P_smooth, Ptest_smooth = load_data(apply_smoothing=True)
    
    # Construct network matrices to get basic info
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Create visualizations
    
    # 1. Time series plot of power injections for each bus (using subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_range = min(100, P.shape[0])  # Show first 100 time steps or all if fewer
    time_steps = np.arange(time_range)
    
    for i in range(P.shape[1]):
        row, col = i // 2, i % 2
        axes[row, col].step(time_steps, P[:time_range, i], where='post', 
                          label=f'Bus {i+1}')
        axes[row, col].set_title(f'Injeções de Potência - Bus {i+1}')
        axes[row, col].set_xlabel('Tempo [15 min]')
        axes[row, col].set_ylabel('Potência [p.u.]')
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig('Plots/Comp/injecoes_potencia_por_barramento.png', dpi=300)
    plt.show()
    
    # 2. Comparison between original and smoothed data
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i in range(P.shape[1]):
        row, col = i // 2, i % 2
        axes[row, col].step(time_steps, P[:time_range, i], where='post', 
                           label='Dados Originais')
        axes[row, col].step(time_steps, P_smooth[:time_range, i], where='post', 
                           label='Dados com Suavização')
        axes[row, col].set_title(f'Bus {i+1} - Comparação de Dados')
        axes[row, col].set_xlabel('Tempo [15 min]')
        axes[row, col].set_ylabel('Potência [p.u.]')
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig('Plots/Comp/comparacao_dados_originais_suavizados.png', dpi=300)
    plt.show()
    
    return

# ============================================================================================================================================
# Função para o modelo Outer-Product (Outer-Product)
# ============================================================================================================================================
def run_outer_product_model(noise_factor=0.0025, network_factor=100, ptest_factor=3, apply_smoothing=False):
    print("\n=== Executando o Modelo com Outer-Product ===\n")
    
    # Ensure plot directories exist
    ensure_plot_directories()
    
    # Carregar dados (com ou sem suavização temporal)
    SlackBus, Net_Info, P, Ptest = load_data(apply_smoothing=apply_smoothing)
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino (PL2)
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    
    # Cálculo da matriz X para os dados de treino (termos de segundo grau)
    X = np.column_stack((
        P[:, 0]**2,              # P1^2
        2 * P[:, 0] * P[:, 1],   # 2*P1*P2
        2 * P[:, 0] * P[:, 2],   # 2*P1*P3
        2 * P[:, 0] * P[:, 3],   # 2*P1*P4
        P[:, 1]**2,              # P2^2
        2 * P[:, 1] * P[:, 2],   # 2*P2*P3
        2 * P[:, 1] * P[:, 3],   # 2*P2*P4
        P[:, 2]**2,              # P3^2
        2 * P[:, 2] * P[:, 3],   # 2*P3*P4
        P[:, 3]**2               # P4^2
    ))
    
    # Cálculo do novo beta utilizando OLS
    beta_novo = inv(X.T @ X) @ (X.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred = X @ beta_novo
    
    # Construção da matriz X_test para os dados de teste
    X_test = np.column_stack((
        Ptest[:, 0]**2,
        2 * Ptest[:, 0] * Ptest[:, 1],
        2 * Ptest[:, 0] * Ptest[:, 2],
        2 * Ptest[:, 0] * Ptest[:, 3],
        Ptest[:, 1]**2,
        2 * Ptest[:, 1] * Ptest[:, 2],
        2 * Ptest[:, 1] * Ptest[:, 3],
        Ptest[:, 2]**2,
        2 * Ptest[:, 2] * Ptest[:, 3],
        Ptest[:, 3]**2
    ))
    
    # Predição para os dados de teste utilizando o modelo de regressão
    PL2_test_pred = X_test @ beta_novo
    
    # Cálculo das perdas físicas para os dados de teste
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # Cálculo das métricas de erro
    train_rmse = np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred))
    test_rmse = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred))
    
    print("Modelo com Outer-Product:")
    print("Erro de treino - RMSE:", train_rmse)
    print("Erro de teste - RMSE:", test_rmse)
    
    # Visualização dos resultados
    time_intervals_train = np.arange(len(PL2_train_true))
    time_intervals_test = np.arange(len(PL2_test_true))
    
    # Calculate errors
    train_errors = 100 * np.abs(PL2_train_true - PL2_train_pred) / np.abs(PL2_train_true)
    test_errors = 100 * np.abs(PL2_test_true - PL2_test_pred) / np.abs(PL2_test_true)
    
    # Create a 2x2 subplot grid
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Top-left: Training data comparison
    ax[0, 0].step(time_intervals_train, PL2_train_true, where='post', label='Perdas Reais')
    ax[0, 0].step(time_intervals_train, PL2_train_pred, where='post', label='Perdas Estimadas')
    ax[0, 0].set_title('Comparação de Perdas - Treino')
    ax[0, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência [p.u.]')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Training error
    ax[1, 0].step(time_intervals_train, train_errors , where='post', label='Erro Percentual', color='blue')
    ax[1, 0].set_title('Erro Percentual - Treino')
    ax[1, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[1, 0].set_ylabel('Erro Percentual (%)')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].grid(True)
    
    # Bottom-left: Test data comparison
    ax[0, 1].step(time_intervals_test, PL2_test_true, where='post', label='Perdas Reais')
    ax[0, 1].step(time_intervals_test, PL2_test_pred, where='post', label='Perdas Estimadas')
    ax[0, 1].set_title('Comparação de Perdas - Teste')
    ax[0, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 1].set_ylabel('Perdas de Potência [p.u.]')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].grid(True)
    
    # Bottom-right: Test error
    ax[1, 1].step(time_intervals_test, test_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 1].set_title('Erro Percentual - Teste')
    ax[1, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[1, 1].set_ylabel('Erro Percentual (%)')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].grid(True)

    # Update y-axis limits for error plots
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[1, 0].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    
    plt.tight_layout()
    plt.savefig('Plots/Outer_Product/resultados_modelo_original.png', dpi=300)
    plt.show()

# ============================================================================================================================================
# Funções para o modelo com edge-reduced 
# ============================================================================================================================================
def edge_reduced_X(P_mat, linha_indices, SlackBus):
    """
    Constrói a matriz X' considerando apenas termos P_i^2 para todos os barramentos 
    e termos cruzados 2*P_i*P_j apenas quando existe uma linha ligando os barramentosi e j.
    
    Parâmetros:
      P_mat: Matriz de potência, com dimensão (n_instantes, nBus_sem_slack)
      linha_indices: Lista de tuplas (i,j) com os índices dos barramentosligados por linhas
      SlackBus: Índice do bus slack (1-based)
      
    Retorna:
      X_topo: Matriz de features considerando a Edge-Reduced da rede
    """
    n_instantes = P_mat.shape[0]
    nBus_sem_slack = P_mat.shape[1]
    
    # Lista para armazenar todas as features (quadráticas e cruzadas)
    features = []
    
    # 1. Adicionar termos quadráticos (squares only) P_i^2 para cada bus
    for i in range(nBus_sem_slack):
        features.append(('quad', i))
    
    # 2. Adicionar termos cruzados 2*P_i*P_j apenas para linhas existentes
    slack_0based = SlackBus - 1  # Converter para 0-based
    
    for i, j in linha_indices:
        # Ignorar se algum dos barramentosfor o slack
        if i == slack_0based or j == slack_0based:
            continue
        
        # Mapear índices de bus para colunas em P_mat (que não tem o slack)
        i_p = i if i < slack_0based else i - 1
        j_p = j if j < slack_0based else j - 1
        
        # Verificar se ambos os índices são válidos para P_mat
        if 0 <= i_p < nBus_sem_slack and 0 <= j_p < nBus_sem_slack:
            features.append(('cross', i_p, j_p))
    
    # Construir matriz X_topo
    X_topo = np.zeros((n_instantes, len(features)))
    
    for t in range(n_instantes):
        for col, feat in enumerate(features):
            if feat[0] == 'quad':
                i = feat[1]
                X_topo[t, col] = P_mat[t, i]**2
            else:  # Termo cruzado
                i, j = feat[1], feat[2]
                X_topo[t, col] = 2 * P_mat[t, i] * P_mat[t, j]
    
    return X_topo

def run_edge_reduced_model(apply_smoothing=False):
    """Executa o modelo que considera a Edge-Reduced da rede para o cálculo de perdas."""
    print("\n=== Executando o Modelo com Edge-Reduced ===\n")
    
    # Carregar dados (com ou sem suavização temporal)
    SlackBus, Net_Info, P, Ptest = load_data(apply_smoothing=apply_smoothing)
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    
    # Construir matriz X' baseada na Edge-Reduced para os dados de treino
    X_topo_train = edge_reduced_X(P, linha_indices, SlackBus)
    
    # Cálculo do beta utilizando OLS
    beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_topo = X_topo_train @ beta_topo
    
    # Construir matriz X' baseada na Edge-Reduced para os dados de teste
    X_topo_test = edge_reduced_X(Ptest, linha_indices, SlackBus)
    
    # Predição para os dados de teste
    PL2_test_pred_topo = X_topo_test @ beta_topo
    
    # Cálculo das perdas físicas para os dados de teste
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # Cálculo das métricas de erro
    train_rmse_topo = np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_topo))
    test_rmse_topo = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_topo))
    
    print("Modelo com Edge-Reduced:")
    print("Erro de treino - RMSE:", train_rmse_topo)
    print("Erro de teste - RMSE:", test_rmse_topo)
    print(f"Número de parâmetros: {len(beta_topo)}")
    
    # Visualização dos resultados
    time_intervals_train = np.arange(len(PL2_train_true))
    time_intervals_test = np.arange(len(PL2_test_true))
    
    # Calculate errors
    train_errors = 100 * np.abs(PL2_train_true - PL2_train_pred_topo) / np.abs(PL2_train_true)
    test_errors = 100 * np.abs(PL2_test_true - PL2_test_pred_topo) / np.abs(PL2_test_true)
    
    # Create a 2x2 subplot grid
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Top-left: Training data comparison
    ax[0, 0].step(time_intervals_train, PL2_train_true, where='post', label='Perdas Reais (PL2 com ruído)')
    ax[0, 0].step(time_intervals_train, PL2_train_pred_topo, where='post', label='Perdas Estimadas')
    ax[0, 0].set_title('Comparação de Perdas - Treino')
    ax[0, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Training error
    ax[0, 1].step(time_intervals_train, train_errors, where='post', label='Erro Percentual', color='blue')
    ax[0, 1].set_title('Erro Percentual - Treino')
    ax[0, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 1].set_ylabel('Erro Percentual (%)')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].grid(True)
    
    # Bottom-left: Test data comparison
    ax[1, 0].step(time_intervals_test, PL2_test_true, where='post', label='Perdas Reais')
    ax[1, 0].step(time_intervals_test, PL2_test_pred_topo, where='post', label='Perdas Estimadas')
    ax[1, 0].set_title('Comparação de Perdas - Teste')
    ax[1, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[1, 0].set_ylabel('Perdas de Potência')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].grid(True)
    
    # Bottom-right: Test error
    ax[1, 1].step(time_intervals_test, test_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 1].set_title('Erro Percentual - Teste')
    ax[1, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[1, 1].set_ylabel('Erro Percentual (%)')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].grid(True)
    
    # Update y-axis limits for percentage error plots
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[0, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    
    plt.tight_layout()
    plt.savefig('Plots/Edge_Reduced/resultados_modelo_topologia.png', dpi=300)
    plt.show()
    
    # Mostrar os coeficientes beta e sua relação com a estrutura da rede
    print("\nCoeficientes do Modelo com Edge-Reduced")
    
    # Termos quadráticos (squares only)
    print("\nTermos Quadráticos:")
    for i in range(P.shape[1]):  # Número de barramentos sem o slack
        print(f"P{i+1}²: beta = {beta_topo[i]:.6f}")
    
    # Termos cruzados
    print("\nTermos Cruzados:")
    for idx, (bus1, bus2) in enumerate(linha_indices):
        if bus1 != (SlackBus-1) and bus2 != (SlackBus-1):  # Verifica se não envolve o slack
            coef_idx = P.shape[1] + idx  # termos quadráticos (squares only) + índice atual
            if coef_idx < len(beta_topo):
                print(f"2*P{bus1+1}*P{bus2+1}: beta = {beta_topo[coef_idx]:.6f}")

# ============================================================================================================================================
# Funções para o modelo squares only
# ============================================================================================================================================
def squares_only_X(P_mat):
    """
    Constrói a matriz X_sq considerando apenas os termos quadráticos (squares only) P_i² para cada bus.
    
    Parâmetros:
      P_mat: Matriz de potência, com dimensão (n_instantes, nBus_sem_slack)
      
    Retorna:
      X_sq: Matriz de features contendo apenas os termos quadráticos (squares only)
    """
    n_instantes, nBus_sem_slack = P_mat.shape
    X_sq = np.zeros((n_instantes, nBus_sem_slack))
    
    for i in range(nBus_sem_slack):
        X_sq[:, i] = P_mat[:, i]**2
        
    return X_sq

def run_squared_model(apply_smoothing=False):
    """Executa o modelo simplificado que considera apenas termos quadráticos (squares only) P_i²."""
    print("\n=== Executando o Modelo Simplificado com Squares-Only)) ===\n")
    
    # Carregar dados (com ou sem suavização temporal)
    SlackBus, Net_Info, P, Ptest = load_data(apply_smoothing=apply_smoothing)
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    
    # Construir matriz X_sq para os dados de treino
    X_sq_train = squares_only_X(P)
    
    # Cálculo do beta utilizando OLS
    beta_sq = inv(X_sq_train.T @ X_sq_train) @ (X_sq_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_sq = X_sq_train @ beta_sq
    
    # Construir matriz X_sq para os dados de teste
    X_sq_test = squares_only_X(Ptest)
    
    # Predição para os dados de teste
    PL2_test_pred_sq = X_sq_test @ beta_sq
    
    # Cálculo das perdas físicas para os dados de teste
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # Cálculo das métricas de erro
    train_rmse_sq = np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_sq))
    test_rmse_sq = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_sq))
    
    print("Modelo Simplificado com Squares Only:")
    print("Erro de treino - RMSE:", train_rmse_sq)
    print("Erro de teste - RMSE:", test_rmse_sq)
    print(f"Número de parâmetros: {len(beta_sq)}")
    
    # Visualização dos resultados
    time_intervals_train = np.arange(len(PL2_train_true))
    time_intervals_test = np.arange(len(PL2_test_true))
    
    # Calculate errors
    train_errors = 100 * np.abs(PL2_train_true - PL2_train_pred_sq) / np.abs(PL2_train_true)
    test_errors = 100 * np.abs(PL2_test_true - PL2_test_pred_sq) / np.abs(PL2_test_true)
    
    # Create a 2x2 subplot grid
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Top-left: Training data comparison
    ax[0, 0].step(time_intervals_train, PL2_train_true, where='post', label='Treino: Perdas Reais')
    ax[0, 0].step(time_intervals_train, PL2_train_pred_sq, where='post', label='Treino: Perdas Estimadas (Squares-Only)')
    ax[0, 0].set_title('Comparação de Perdas - Treino (Modelo Simplificado)')
    ax[0, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Training error
    ax[0, 1].step(time_intervals_train, train_errors, where='post', label='Erro Percentual', color='blue')
    ax[0, 1].set_title('Erro Percentual - Treino')
    ax[0, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 1].set_ylabel('Erro Percentual (%)')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].grid(True)
    
    # Bottom-left: Test data comparison
    ax[1, 0].step(time_intervals_test, PL2_test_true, where='post', label='Teste: Perdas Reais')
    ax[1, 0].step(time_intervals_test, PL2_test_pred_sq, where='post', label='Teste: Perdas Estimadas Squares-Only', color='orange')
    ax[1, 0].set_title('Comparação de Perdas - Teste (Modelo Simplificado)')
    ax[1, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[1, 0].set_ylabel('Perdas de Potência')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].grid(True)
    
    # Bottom-right: Test error
    ax[1, 1].step(time_intervals_test, test_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 1].set_title('Erro Percentual - Teste')
    ax[1, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[1, 1].set_ylabel('Erro Percentual (%)')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].grid(True)
    
    # Update y-axis limits for percentage error plots
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[0, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    
    plt.tight_layout()
    plt.savefig('Plots/Squares_Only/resultados_modelo_quadratico.png', dpi=300)
    plt.show()
    
    # Mostrar os coeficientes beta
    print("\nCoeficientes do Modelo com Squares-Only")
    for i in range(P.shape[1]):  # Número de barramentos sem o slack
        print(f"P{i+1}²: beta = {beta_sq[i]:.6f}")

# ============================================================================================================================================
# Funções para o modelo squares_reduced
# ============================================================================================================================================
def squares_reduced_X(P_mat, linha_indices, SlackBus):
    """
    Redução da dimensionalidade de X através da soma das injeções de potência em barramentos eletricamente próximos antes de as utilizar 
    como variáveis explicativas das perdas. 
    No caso da rede em "papagaio" pode ser feito definindo P_sum = P1 + P2 + P3 e:
    
    X = [P_sum^2  2P_sumP4    P4^2], k = 1, . . . ,M
        | ...       ...       ... |
        [P_sum^2  2P_sumP4    P4^2]  
    """
    # Soma das variaveis eletricamente proximas	
    P_sum = P_mat[:, 0] + P_mat[:, 1] + P_mat[:, 2]

    # Construir matriz X_nearby
    X_nearby = np.zeros((P_mat.shape[0], 3))
    X_nearby[:, 0] = P_sum**2
    X_nearby[:, 1] = 2 * P_sum * P_mat[:, 3]
    X_nearby[:, 2] = P_mat[:, 3]**2

    return X_nearby

def run_nearby_model(apply_smoothing=False):
    """Executa o modelo simplificado que considera a soma de injeções eletricamente próximas."""
    print("\n=== Executando o Modelo Simplificado com Squares-Reduced ===\n")
    
    # Carregar dados (com ou sem suavização temporal)
    SlackBus, Net_Info, P, Ptest = load_data(apply_smoothing=apply_smoothing)
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    
    # Construir matriz X_nearby para os dados de treino
    X_nearby_train = squares_reduced_X(P, linha_indices, SlackBus)
    
    # Cálculo do beta utilizando OLS
    beta_nearby = inv(X_nearby_train.T @ X_nearby_train) @ (X_nearby_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_nearby = X_nearby_train @ beta_nearby
    
    # Construir matriz X_nearby para os dados de teste
    X_nearby_test = squares_reduced_X(Ptest, linha_indices, SlackBus)
    
    # Predição para os dados de teste
    PL2_test_pred_nearby = X_nearby_test @ beta_nearby
    
    # Cálculo das perdas físicas para os dados de teste
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # Cálculo das métricas de erro
    train_rmse_nearby = np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_nearby))
    test_rmse_nearby = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_nearby))
    
    print("Modelo Simplificado com Squares-Rduced:")
    print("Erro de treino - RMSE:", train_rmse_nearby)
    print("Erro de teste - RMSE:", test_rmse_nearby)

    # Visualização dos resultados
    time_intervals_train = np.arange(len(PL2_train_true))
    time_intervals_test = np.arange(len(PL2_test_true))

    # Calculate errors as percentage (consistente com run_outer_product_model)
    train_errors = 100 * np.abs(PL2_train_true - PL2_train_pred_nearby) / np.abs(PL2_train_true)
    test_errors = 100 * np.abs(PL2_test_true - PL2_test_pred_nearby) / np.abs(PL2_test_true)

    # Plotagem dos resultados
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    # Top-left: Training data comparison
    ax[0, 0].step(time_intervals_train, PL2_train_true, where='post', label='Perdas Reais')
    ax[0, 0].step(time_intervals_train, PL2_train_pred_nearby, where='post', label='Perdas Estimadas com Squares-Reduced')
    ax[0, 0].set_title('Comparação de Perdas - Treino')
    ax[0, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência [p.u.]')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Test data comparison
    ax[0, 1].step(time_intervals_test, PL2_test_true, where='post', label='Perdas Reais')
    ax[0, 1].step(time_intervals_test, PL2_test_pred_nearby, where='post', label='Perdas Estimadas com Squares-Reduced')
    ax[0, 1].set_title('Comparação de Perdas - Teste')
    ax[0, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 1].set_ylabel('Perdas de Potência [p.u.]')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].grid(True)
    
    # Bottom-left: Training error
    ax[1, 0].step(time_intervals_train, train_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 0].set_title('Erro Percentual - Treino')
    ax[1, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[1, 0].set_ylabel('Erro Percentual (%)')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].grid(True)
    
    # Bottom-right: Test error
    ax[1, 1].step(time_intervals_test, test_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 1].set_title('Erro Percentual - Teste')
    ax[1, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[1, 1].set_ylabel('Erro Percentual (%)')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].grid(True)

    # Update y-axis limits for percentage error plots (consistente com run_outer_product_model)
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[1, 0].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)

    plt.tight_layout()
    plt.savefig('Plots/Squares_Reduced/resultados_modelo_injeções_proximas.png', dpi=300)
    plt.show()


# ============================================================================================================================================
# Funções para o modelo de teste de diferentes valores de ruído
# ============================================================================================================================================
def compare_temporal_correlation_different_noise():
    """Compares model performance with and without temporal correlation at different noise levels."""
    print("\n=== Comparando Desempenho dos Modelos Com e Sem Correlação Temporal em Diferentes Níveis de Ruído ===\n")
    
    # Define noise levels to test
    noise_levels = [0, 0.0005, 0.001, 0.0015, 0.0020, 0.0025]
    
    # Dictionary to store results for each noise level, model, and temporal correlation setting
    noise_results = {
        'no_smoothing': {model: {'RMSE': []} for model in 
                         ['Outer-Product', 'Edge-Reduced', 
                          'Squares-Only', 
                          'Squares-Reduced']},
        'smoothing': {model: {'RMSE': []} for model in 
                      ['Outer-Product', 'Edge-Reduced', 
                       'Squares-Only', 
                       'Squares-Reduced']}
    }
    
    # Store the number of parameters (they don't change with noise or temporal correlation)
    model_params = {model: 0 for model in ['Outer-Product', 'Edge-Reduced', 
                                          'Squares-Only', 
                                          'Squares-Reduced']}
    
    # Setup for network without temporal correlation
    print("Configurando rede sem correlação temporal...")
    SlackBus, Net_Info, P, Ptest = load_data(apply_smoothing=False)
    
    # Build network matrices
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Build model matrices (these don't change with noise)
    # Original model
    X_orig_train = np.column_stack((
        P[:, 0]**2, 2*P[:, 0]*P[:, 1], 2*P[:, 0]*P[:, 2], 2*P[:, 0]*P[:, 3],
        P[:, 1]**2, 2*P[:, 1]*P[:, 2], 2*P[:, 1]*P[:, 3],
        P[:, 2]**2, 2*P[:, 2]*P[:, 3], P[:, 3]**2
    ))
    X_orig_test = np.column_stack((
        Ptest[:, 0]**2, 2*Ptest[:, 0]*Ptest[:, 1], 2*Ptest[:, 0]*Ptest[:, 2], 2*Ptest[:, 0]*Ptest[:, 3],
        Ptest[:, 1]**2, 2*Ptest[:, 1]*Ptest[:, 2], 2*Ptest[:, 1]*Ptest[:, 3],
        Ptest[:, 2]**2, 2*Ptest[:, 2]*Ptest[:, 3], Ptest[:, 3]**2
    ))
    
    # Edge-Reduced model
    X_topo_train = edge_reduced_X(P, linha_indices, SlackBus)
    X_topo_test = edge_reduced_X(Ptest, linha_indices, SlackBus)
    
    # Squares-Only model
    X_sq_train = squares_only_X(P)
    X_sq_test = squares_only_X(Ptest)
    
    # Squares-Reduced model
    X_nearby_train = squares_reduced_X(P, linha_indices, SlackBus)
    X_nearby_test = squares_reduced_X(Ptest, linha_indices, SlackBus)
    
    # Setup for network with temporal correlation
    print("Configurando rede com correlação temporal...")
    _, _, P_smooth, Ptest_smooth = load_data(apply_smoothing=True)
    
    # Build model matrices with smoothed data
    # Original model
    X_orig_train_smooth = np.column_stack((
        P_smooth[:, 0]**2, 2*P_smooth[:, 0]*P_smooth[:, 1], 2*P_smooth[:, 0]*P_smooth[:, 2], 2*P_smooth[:, 0]*P_smooth[:, 3],
        P_smooth[:, 1]**2, 2*P_smooth[:, 1]*P_smooth[:, 2], 2*P_smooth[:, 1]*P_smooth[:, 3],
        P_smooth[:, 2]**2, 2*P_smooth[:, 2]*P_smooth[:, 3], P_smooth[:, 3]**2
    ))
    X_orig_test_smooth = np.column_stack((
        Ptest_smooth[:, 0]**2, 2*Ptest_smooth[:, 0]*Ptest_smooth[:, 1], 2*Ptest_smooth[:, 0]*Ptest_smooth[:, 2], 2*Ptest_smooth[:, 0]*Ptest_smooth[:, 3],
        Ptest_smooth[:, 1]**2, 2*Ptest_smooth[:, 1]*Ptest_smooth[:, 2], 2*Ptest_smooth[:, 1]*Ptest_smooth[:, 3],
        Ptest_smooth[:, 2]**2, 2*Ptest_smooth[:, 2]*Ptest_smooth[:, 3], Ptest_smooth[:, 3]**2
    ))
    
    # Edge-Reduced model
    X_topo_train_smooth = edge_reduced_X(P_smooth, linha_indices, SlackBus)
    X_topo_test_smooth = edge_reduced_X(Ptest_smooth, linha_indices, SlackBus)
    
    # Squares-Only model
    X_sq_train_smooth = squares_only_X(P_smooth)
    X_sq_test_smooth = squares_only_X(Ptest_smooth)
    
    # Squares-Reduced model
    X_nearby_train_smooth = squares_reduced_X(P_smooth, linha_indices, SlackBus)
    X_nearby_test_smooth = squares_reduced_X(Ptest_smooth, linha_indices, SlackBus)
    
    # Store the number of parameters for each model
    model_params['Outer-Product'] = X_orig_train.shape[1]
    model_params['Edge-Reduced'] = X_topo_train.shape[1]
    model_params['Squares-Only)'] = X_sq_train.shape[1]
    model_params['Squares-Reduced'] = X_nearby_train.shape[1]
    
    # Main loop for different noise levels
    for noise_level in noise_levels:
        print(f"\nAvaliando modelos com nível de ruído: {noise_level}")
        
        # ===== Without temporal correlation =====
        print("Processando rede sem correlação temporal...")
        
        # Calculate true losses with current noise level
        PL2_train_true = compute_PL2(P, B, Cl, Gv, noise_level)
        PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noise_level)
        
        # 1. Original model
        beta_original = inv(X_orig_train.T @ X_orig_train) @ (X_orig_train.T @ PL2_train_true)
        pred_test_original = X_orig_test @ beta_original
        
        # Calculate metrics
        rmse_original = np.sqrt(mean_squared_error(PL2_test_true, pred_test_original))
        
        noise_results['no_smoothing']['Outer-Product']['RMSE'].append(rmse_original)
        
        # 2. Edge-Reduced model
        beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
        pred_test_topo = X_topo_test @ beta_topo
        
        # Calculate metrics
        rmse_topo = np.sqrt(mean_squared_error(PL2_test_true, pred_test_topo))
        
        noise_results['no_smoothing']['Edge-Reduced']['RMSE'].append(rmse_topo)
        
        # 3. Squares-Only model
        beta_sq = inv(X_sq_train.T @ X_sq_train) @ (X_sq_train.T @ PL2_train_true)
        pred_test_sq = X_sq_test @ beta_sq
        
        # Calculate metrics
        rmse_sq = np.sqrt(mean_squared_error(PL2_test_true, pred_test_sq))
        
        noise_results['no_smoothing']['Squares-Only']['RMSE'].append(rmse_sq)
        
        # 4. Squares-Reduced model
        beta_nearby = inv(X_nearby_train.T @ X_nearby_train) @ (X_nearby_train.T @ PL2_train_true)
        pred_test_nearby = X_nearby_test @ beta_nearby
        
        # Calculate metrics
        rmse_nearby = np.sqrt(mean_squared_error(PL2_test_true, pred_test_nearby))
        
        noise_results['no_smoothing']['Squares-Reduced']['RMSE'].append(rmse_nearby)
        
        # ===== With temporal correlation =====
        print("  Processando rede com correlação temporal...")
        
        # Calculate true losses with current noise level
        PL2_train_true_smooth = compute_PL2(P_smooth, B, Cl, Gv, noise_level)
        PL2_test_true_smooth = compute_PL2(Ptest_smooth, B, Cl, Gv, noise_level)
        
        # 1. Outer-Product model
        beta_original_smooth = inv(X_orig_train_smooth.T @ X_orig_train_smooth) @ (X_orig_train_smooth.T @ PL2_train_true_smooth)
        pred_test_original_smooth = X_orig_test_smooth @ beta_original_smooth
        
        # Calculate metrics
        rmse_original_smooth = np.sqrt(mean_squared_error(PL2_test_true_smooth, pred_test_original_smooth))
        
        noise_results['smoothing']['Outer-Product']['RMSE'].append(rmse_original_smooth)
        
        # 2. Edge-Reduced model
        beta_topo_smooth = inv(X_topo_train_smooth.T @ X_topo_train_smooth) @ (X_topo_train_smooth.T @ PL2_train_true_smooth)
        pred_test_topo_smooth = X_topo_test_smooth @ beta_topo_smooth
        
        # Calculate metrics
        rmse_topo_smooth = np.sqrt(mean_squared_error(PL2_test_true_smooth, pred_test_topo_smooth))
        
        noise_results['smoothing']['Edge-Reduced']['RMSE'].append(rmse_topo_smooth)
        
        # 3. Squares-Only model
        beta_sq_smooth = inv(X_sq_train_smooth.T @ X_sq_train_smooth) @ (X_sq_train_smooth.T @ PL2_train_true_smooth)
        pred_test_sq_smooth = X_sq_test_smooth @ beta_sq_smooth
        
        # Calculate metrics
        rmse_sq_smooth = np.sqrt(mean_squared_error(PL2_test_true_smooth, pred_test_sq_smooth))
        
        noise_results['smoothing']['Squares-Only']['RMSE'].append(rmse_sq_smooth)
        
        # 4. Squares-Reduced model
        beta_nearby_smooth = inv(X_nearby_train_smooth.T @ X_nearby_train_smooth) @ (X_nearby_train_smooth.T @ PL2_train_true_smooth)
        pred_test_nearby_smooth = X_nearby_test_smooth @ beta_nearby_smooth
        
        # Calculate metrics
        rmse_nearby_smooth = np.sqrt(mean_squared_error(PL2_test_true_smooth, pred_test_nearby_smooth))
        
        noise_results['smoothing']['Squares-Reduced']['RMSE'].append(rmse_nearby_smooth)
    
    # Visualization
    
    # Plot RMSE vs Noise Level for all models (with and without temporal correlation)
    plt.figure(figsize=(20, 10))
    
    # Setup subplots for each model
    models = list(noise_results['no_smoothing'].keys())
    for i, model in enumerate(models):
        plt.subplot(2, 2, i+1)
        
        plt.plot(noise_levels, noise_results['no_smoothing'][model]['RMSE'], 
                 marker='o', linewidth=2, label='Sem Correlação Temporal')
        plt.plot(noise_levels, noise_results['smoothing'][model]['RMSE'], 
                 marker='s', linewidth=2, label='Com Correlação Temporal')
        
        plt.title(f'RMSE vs Nível de Ruído - {model}')
        plt.xlabel('Nível de Ruído')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('Plots/Noise/comparacao_ruido_correlacao_temporal.png', dpi=300)
    
    # Combined comparison showing percentage improvement with temporal correlation
    plt.figure(figsize=(16, 8))
    models = list(noise_results['no_smoothing'].keys())
    colors = ['blue', 'green', 'red', 'purple']
    
    # Plot lines for both with and without temporal correlation
    for i, model in enumerate(models):
        # Compute the percentage improvement
        improvement_pct = []
        for j in range(len(noise_levels)):
            no_smooth = noise_results['no_smoothing'][model]['RMSE'][j]
            smooth = noise_results['smoothing'][model]['RMSE'][j]
            pct = 100 * (no_smooth - smooth) / no_smooth if no_smooth != 0 else 0
            improvement_pct.append(pct)
        
        plt.plot(noise_levels, improvement_pct, 
                 marker='o', color=colors[i], linestyle='-', linewidth=2, 
                 label=f'{model} - Melhoria %')
    
    plt.title('Melhoria Percentual com Correlação Temporal vs Nível de Ruído')
    plt.xlabel('Nível de Ruído')
    plt.ylabel('Melhoria Percentual (%)')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('Plots/Noise/melhoria_percentual_correlacao_temporal.png', dpi=300)
    
    # Print summary table
    print("\nResumo dos Resultados - RMSE para Diferentes Níveis de Ruído e Correlação Temporal:")
    print("-" * 110)
    header = "Modelo".ljust(30) + " | " + "Correlação".ljust(15) + " | "
    for noise in noise_levels:
        header += f"Ruído {noise:.4f}".ljust(12) + " | "
    print(header)
    print("-" * 110)
    
    for model in models:
        # Without temporal correlation
        row_no_smooth = model.ljust(30) + " | " + "Sem".ljust(15) + " | "
        for i, noise in enumerate(noise_levels):
            row_no_smooth += f"{noise_results['no_smoothing'][model]['RMSE'][i]:.6f}".ljust(12) + " | "
        print(row_no_smooth)
        
        # With temporal correlation
        row_smooth = model.ljust(30) + " | " + "Com".ljust(15) + " | "
        for i, noise in enumerate(noise_levels):
            row_smooth += f"{noise_results['smoothing'][model]['RMSE'][i]:.6f}".ljust(12) + " | "
        print(row_smooth)
        
        # Improvement percentage
        row_improvement = model.ljust(30) + " | " + "Melhoria (%)".ljust(15) + " | "
        for i, noise in enumerate(noise_levels):
            no_smooth = noise_results['no_smoothing'][model]['RMSE'][i]
            smooth = noise_results['smoothing'][model]['RMSE'][i]
            pct = 100 * (no_smooth - smooth) / no_smooth if no_smooth != 0 else 0
            row_improvement += f"{pct:.2f}%".ljust(12) + " | "
        print(row_improvement)
        print("-" * 110)
    
    plt.show()
    
    return noise_results, model_params

def compare_temporal_correlation():
    """Compares model performance with and without temporal correlation in the data."""
    print("\n=== Comparando Desempenho dos Modelos Com e Sem Correlação Temporal ===\n")
    
    # Store results
    results = {
        'no_smoothing': {model: {'RMSE': 0} for model in 
                       ['Outer-Product', 'Edge-Reduced', 
                        'Squares-Only', 
                        'Squares-Reduced']},
        'smoothing': {model: {'RMSE': 0} for model in 
                     ['Outer-Product', 'Edge-Reduced', 
                      'Squares-Only', 
                      'Squares-Reduced']}
    }
    
    # Run models without temporal smoothing
    print("\nExecutando modelos SEM correlação temporal...")
    SlackBus, Net_Info, P, Ptest = load_data(apply_smoothing=False)
    
    # Build network matrices
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Compute true losses
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # Outer-Product model
    X_orig = np.column_stack((
        P[:, 0]**2, 2*P[:, 0]*P[:, 1], 2*P[:, 0]*P[:, 2], 2*P[:, 0]*P[:, 3],
        P[:, 1]**2, 2*P[:, 1]*P[:, 2], 2*P[:, 1]*P[:, 3],
        P[:, 2]**2, 2*P[:, 2]*P[:, 3], P[:, 3]**2
    ))
    X_orig_test = np.column_stack((
        Ptest[:, 0]**2, 2*Ptest[:, 0]*Ptest[:, 1], 2*Ptest[:, 0]*Ptest[:, 2], 2*Ptest[:, 0]*Ptest[:, 3],
        Ptest[:, 1]**2, 2*Ptest[:, 1]*Ptest[:, 2], 2*Ptest[:, 1]*Ptest[:, 3],
        Ptest[:, 2]**2, 2*Ptest[:, 2]*Ptest[:, 3], Ptest[:, 3]**2
    ))
    beta_orig = inv(X_orig.T @ X_orig) @ (X_orig.T @ PL2_train_true)
    pred_test_orig = X_orig_test @ beta_orig
    
    # Compute metrics
    rmse_orig = np.sqrt(mean_squared_error(PL2_test_true, pred_test_orig))
    results['no_smoothing']['Outer-Product']['RMSE'] = rmse_orig
    
    # Edge-Reduced model
    X_topo = edge_reduced_X(P, linha_indices, SlackBus)
    X_topo_test = edge_reduced_X(Ptest, linha_indices, SlackBus)
    beta_topo = inv(X_topo.T @ X_topo) @ (X_topo.T @ PL2_train_true)
    pred_test_topo = X_topo_test @ beta_topo
    rmse_topo = np.sqrt(mean_squared_error(PL2_test_true, pred_test_topo))
    results['no_smoothing']['Edge-Reduced']['RMSE'] = rmse_topo
    
    # Squares-Only model
    X_sq = squares_only_X(P)
    X_sq_test = squares_only_X(Ptest)
    beta_sq = inv(X_sq.T @ X_sq) @ (X_sq.T @ PL2_train_true)
    pred_test_sq = X_sq_test @ beta_sq
    rmse_sq = np.sqrt(mean_squared_error(PL2_test_true, pred_test_sq))
    results['no_smoothing']['Squares-Only']['RMSE'] = rmse_sq
    
    # Squares-Reduced model
    X_nearby = squares_reduced_X(P, linha_indices, SlackBus)
    X_nearby_test = squares_reduced_X(Ptest, linha_indices, SlackBus)
    beta_nearby = inv(X_nearby.T @ X_nearby) @ (X_nearby.T @ PL2_train_true)
    pred_test_nearby = X_nearby_test @ beta_nearby
    rmse_nearby = np.sqrt(mean_squared_error(PL2_test_true, pred_test_nearby))
    results['no_smoothing']['Squares-Reduced']['RMSE'] = rmse_nearby
    
    # Run models WITH temporal smoothing
    print("\nExecutando modelos COM correlação temporal...")
    SlackBus, Net_Info, P_smooth, Ptest_smooth = load_data(apply_smoothing=True)
    
    # Compute true losses
    PL2_train_true_smooth = compute_PL2(P_smooth, B, Cl, Gv, noiseFactor)
    PL2_test_true_smooth = compute_PL2(Ptest_smooth, B, Cl, Gv, noiseFactor)
    
    # Outer-Product model
    X_orig_smooth = np.column_stack((
        P_smooth[:, 0]**2, 2*P_smooth[:, 0]*P_smooth[:, 1], 2*P_smooth[:, 0]*P_smooth[:, 2], 2*P_smooth[:, 0]*P_smooth[:, 3],
        P_smooth[:, 1]**2, 2*P_smooth[:, 1]*P_smooth[:, 2], 2*P_smooth[:, 1]*P_smooth[:, 3],
        P_smooth[:, 2]**2, 2*P_smooth[:, 2]*P_smooth[:, 3], P_smooth[:, 3]**2
    ))
    X_orig_test_smooth = np.column_stack((
        Ptest_smooth[:, 0]**2, 2*Ptest_smooth[:, 0]*Ptest_smooth[:, 1], 2*Ptest_smooth[:, 0]*Ptest_smooth[:, 2], 2*Ptest_smooth[:, 0]*Ptest_smooth[:, 3],
        Ptest_smooth[:, 1]**2, 2*Ptest_smooth[:, 1]*Ptest_smooth[:, 2], 2*Ptest_smooth[:, 1]*Ptest_smooth[:, 3],
        Ptest_smooth[:, 2]**2, 2*Ptest_smooth[:, 2]*Ptest_smooth[:, 3], Ptest_smooth[:, 3]**2
    ))
    beta_orig_smooth = inv(X_orig_smooth.T @ X_orig_smooth) @ (X_orig_smooth.T @ PL2_train_true_smooth)
    pred_test_orig_smooth = X_orig_test_smooth @ beta_orig_smooth
    rmse_orig_smooth = np.sqrt(mean_squared_error(PL2_test_true_smooth, pred_test_orig_smooth))
    results['smoothing']['Outer-Product']['RMSE'] = rmse_orig_smooth
    
    # Edge-Reduced model
    X_topo_smooth = edge_reduced_X(P_smooth, linha_indices, SlackBus)
    X_topo_test_smooth = edge_reduced_X(Ptest_smooth, linha_indices, SlackBus)
    beta_topo_smooth = inv(X_topo_smooth.T @ X_topo_smooth) @ (X_topo_smooth.T @ PL2_train_true_smooth)
    pred_test_topo_smooth = X_topo_test_smooth @ beta_topo_smooth
    rmse_topo_smooth = np.sqrt(mean_squared_error(PL2_test_true_smooth, pred_test_topo_smooth))
    results['smoothing']['Edge-Reduced']['RMSE'] = rmse_topo_smooth
    
    # Squares-Only model
    X_sq_smooth = squares_only_X(P_smooth)
    X_sq_test_smooth = squares_only_X(Ptest_smooth)
    beta_sq_smooth = inv(X_sq_smooth.T @ X_sq_smooth) @ (X_sq_smooth.T @ PL2_train_true_smooth)
    pred_test_sq_smooth = X_sq_test_smooth @ beta_sq_smooth
    rmse_sq_smooth = np.sqrt(mean_squared_error(PL2_test_true_smooth, pred_test_sq_smooth))
    results['smoothing']['Squares-Only']['RMSE'] = rmse_sq_smooth
    
    # Squares-Reduced model
    X_nearby_smooth = squares_reduced_X(P_smooth, linha_indices, SlackBus)
    X_nearby_test_smooth = squares_reduced_X(Ptest_smooth, linha_indices, SlackBus)
    beta_nearby_smooth = inv(X_nearby_smooth.T @ X_nearby_smooth) @ (X_nearby_smooth.T @ PL2_train_true_smooth)
    pred_test_nearby_smooth = X_nearby_test_smooth @ beta_nearby_smooth
    rmse_nearby_smooth = np.sqrt(mean_squared_error(PL2_test_true_smooth, pred_test_nearby_smooth))
    results['smoothing']['Squares-Reduced']['RMSE'] = rmse_nearby_smooth
    
    # Visualize results - bar chart comparing RMSE
    models = list(results['no_smoothing'].keys())
    x = np.arange(len(models))
    width = 0.35
    
    
    # RMSE plot
    rmse_no_smoothing = [results['no_smoothing'][model]['RMSE'] for model in models]
    rmse_smoothing = [results['smoothing'][model]['RMSE'] for model in models]
    
    fig, ax = plt.subplots(figsize=(10, 7))  # Single plot
    rects1 = ax.bar(x - width/2, rmse_no_smoothing, width, label='Sem Correlação Temporal')
    rects2 = ax.bar(x + width/2, rmse_smoothing, width, label='Com Correlação Temporal')
    ax.set_ylabel('RMSE')
    ax.set_title('Erro (RMSE) Com e Sem Correlação Temporal')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig('Plots/Comp/comparacao_correlacao_temporal.png', dpi=300)
    plt.show()
    
    # Plot power signals comparison
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # Make sure time_range doesn't exceed the length of available data
    time_range = min(100, P.shape[0], P_smooth.shape[0])  # Just plot a portion for clarity
    time_indices = np.arange(time_range)  # Create x values

    for i in range(4):
        row, col = i // 2, i % 2
        ax[row, col].step(time_indices, P[:time_range, i], where='post', label='Sem Correlação Temporal')
        ax[row, col].step(time_indices, P_smooth[:time_range, i], where='post', label='Com Correlação Temporal')
        ax[row, col].set_title(f'Bus {i+1} - Injeção de Potência')
        ax[row, col].set_xlabel('Tempo [15 min]')
        ax[row, col].set_ylabel('Potência [p.u.]')
        ax[row, col].legend()
        ax[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig('Plots/Comp/sinais_potencia_comparacao.png', dpi=300)
    plt.show()
    
    # Calculate and display the improvement percentag

    print("\nResultados Detalhados:")
    print("-" * 80)
    print("| Modelo                  | Métrica | Sem Correlação | Com Correlação | Melhoria (%) |")
    print("-" * 80)

    for model in models:
        rmse_no_smooth = results['no_smoothing'][model]['RMSE']
        rmse_smooth = results['smoothing'][model]['RMSE']
        improvement = 100 * (rmse_no_smooth - rmse_smooth) / rmse_no_smooth
        
        print(f"| {model.ljust(24)} | RMSE    | {rmse_no_smooth:.6f}    | {rmse_smooth:.6f}    | {improvement:.2f}%      |")

    print("-" * 80)
    print("\nConclusão: A correlação temporal " + 
        ("melhora" if all(results['no_smoothing'][m]['RMSE'] > results['smoothing'][m]['RMSE'] for m in models) 
        else "nem sempre melhora") + 
        " o desempenho dos modelos na estimação de perdas.")
    
    return results

def compare_models_by_parameters_and_noise():
    """Compares the 4 models showing their parameter counts and RMSE across noise levels."""
    print("\n=== Comparando Modelos por Número de Parâmetros e Ruído ===\n")
    
    # Define noise levels (0 to 0.001)
    noise_levels = [0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001]
    
    # Dictionary to store results for each model
    results = {
        model: {'RMSE': [], 'params': 0} for model in 
        ['Outer-Product', 'Edge-Reduced', 'Squares-Only', 'Squares-Reduced']
    }
    
    # Setup network
    SlackBus, Net_Info, P, Ptest = load_data(apply_smoothing=False)
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Build model matrices (these don't change with noise)
    # Original model (Outer-Product)
    X_orig_train = np.column_stack((
        P[:, 0]**2, 2*P[:, 0]*P[:, 1], 2*P[:, 0]*P[:, 2], 2*P[:, 0]*P[:, 3],
        P[:, 1]**2, 2*P[:, 1]*P[:, 2], 2*P[:, 1]*P[:, 3],
        P[:, 2]**2, 2*P[:, 2]*P[:, 3], P[:, 3]**2
    ))
    X_orig_test = np.column_stack((
        Ptest[:, 0]**2, 2*Ptest[:, 0]*Ptest[:, 1], 2*Ptest[:, 0]*Ptest[:, 2], 2*Ptest[:, 0]*Ptest[:, 3],
        Ptest[:, 1]**2, 2*Ptest[:, 1]*Ptest[:, 2], 2*Ptest[:, 1]*Ptest[:, 3],
        Ptest[:, 2]**2, 2*Ptest[:, 2]*Ptest[:, 3], Ptest[:, 3]**2
    ))
    
    # Edge-Reduced model
    X_topo_train = edge_reduced_X(P, linha_indices, SlackBus)
    X_topo_test = edge_reduced_X(Ptest, linha_indices, SlackBus)
    
    # Squares-Only model
    X_sq_train = squares_only_X(P)
    X_sq_test = squares_only_X(Ptest)
    
    # Squares-Reduced model
    X_nearby_train = squares_reduced_X(P, linha_indices, SlackBus)
    X_nearby_test = squares_reduced_X(Ptest, linha_indices, SlackBus)
    
    # Store the number of parameters for each model
    results['Outer-Product']['params'] = X_orig_train.shape[1]
    results['Edge-Reduced']['params'] = X_topo_train.shape[1]
    results['Squares-Only']['params'] = X_sq_train.shape[1]
    results['Squares-Reduced']['params'] = X_nearby_train.shape[1]
    
    # Loop through different noise levels
    for noise_level in noise_levels:
        print(f"  Processando com nível de ruído: {noise_level}")
        
        # Calculate true losses with current noise level
        PL2_train_true = compute_PL2(P, B, Cl, Gv, noise_level)
        PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noise_level)
        
        # 1. Outer-Product model
        beta_original = inv(X_orig_train.T @ X_orig_train) @ (X_orig_train.T @ PL2_train_true)
        pred_test_original = X_orig_test @ beta_original
        rmse_original = np.sqrt(mean_squared_error(PL2_test_true, pred_test_original))
        results['Outer-Product']['RMSE'].append(rmse_original)
        
        # 2. Edge-Reduced model
        beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
        pred_test_topo = X_topo_test @ beta_topo
        rmse_topo = np.sqrt(mean_squared_error(PL2_test_true, pred_test_topo))
        results['Edge-Reduced']['RMSE'].append(rmse_topo)
        
        # 3. Squares-Only model
        beta_sq = inv(X_sq_train.T @ X_sq_train) @ (X_sq_train.T @ PL2_train_true)
        pred_test_sq = X_sq_test @ beta_sq
        rmse_sq = np.sqrt(mean_squared_error(PL2_test_true, pred_test_sq))
        results['Squares-Only']['RMSE'].append(rmse_sq)
        
        # 4. Squares-Reduced model
        beta_nearby = inv(X_nearby_train.T @ X_nearby_train) @ (X_nearby_train.T @ PL2_train_true)
        pred_test_nearby = X_nearby_test @ beta_nearby
        rmse_nearby = np.sqrt(mean_squared_error(PL2_test_true, pred_test_nearby))
        results['Squares-Reduced']['RMSE'].append(rmse_nearby)
    
    # Visualization - Line plot of RMSE vs Noise Level
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    markers = ['o', 's', '^', 'd']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, model in enumerate(models):
        plt.plot(noise_levels, results[model]['RMSE'], 
                 marker=markers[i], color=colors[i], linewidth=2, 
                 label=f"{model} ({results[model]['params']} params)")
    
    plt.title('RMSE vs Nível de Ruído para Diferentes Modelos')
    plt.xlabel('Nível de Ruído (σ²)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Plots/Comp/comparacao_modelos_parametros_ruido.png', dpi=300)
    
    # Create a bar plot comparing the number of parameters for each model
    plt.figure(figsize=(10, 6))
    param_counts = [results[model]['params'] for model in models]
    
    # Calculate the average RMSE across noise levels for each model
    avg_rmse = [np.mean(results[model]['RMSE']) for model in models]
    
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar plot for parameter count (left y-axis)
    bars = ax1.bar(models, param_counts, color=colors, alpha=0.7)
    ax1.set_xlabel('Modelo')
    ax1.set_ylabel('Número de Parâmetros', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Line plot for average RMSE (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(models, avg_rmse, 'r-', marker='o', linewidth=2)
    ax2.set_ylabel('RMSE Médio', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add parameter count labels on bars
    for bar, count in zip(bars, param_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', color='blue', fontweight='bold')
    
    plt.title('Comparação de Modelos: Número de Parâmetros vs RMSE Médio')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('Plots/Comp/parametros_vs_rmse_medio.png', dpi=300)
    
    # Print summary table
    print("\nResumo dos Resultados - RMSE para Diferentes Níveis de Ruído por Modelo:")
    print("-" * 90)
    header = "Modelo".ljust(20) + " | " + "Parâmetros".ljust(10) + " | "
    for noise in noise_levels:
        header += f"σ²={noise:.4f}".ljust(10) + " | "
    print(header)
    print("-" * 90)
    
    for model in models:
        row = model.ljust(20) + " | " + str(results[model]['params']).ljust(10) + " | "
        for i, noise in enumerate(noise_levels):
            row += f"{results[model]['RMSE'][i]:.6f}".ljust(10) + " | "
        print(row)
    
    print("-" * 90)
    
    plt.show()
    
    return results
# ============================================================================================================================================
# Funções para o menu principal e execução de comparações
# ============================================================================================================================================
def show_menu():
    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
    print("=" * 80)
    print("                     Projeto 1 - Correlação de Perdas de Potência                 ")
    print("=" * 80)
    print("\nEscolha uma opção:")
    print("  1. Visualizar dados iniciais")
    print("  2. Executar o modelo com maior dimensionalidade de X (Outer Product)")
    print("  3. Executar o modelo com X reduzida (edge-reduced)")
    print("  4. Executar o modelo com X reduzida (squares only)")
    print("  5. Executar o modelo com X reduzida (squares-reduced)")
    print("  6. Comparar desempenho dos com diferentes níveis de ruído")
    print("  7. Comparar modelos com e sem correlação temporal")
    print("  8. Comparar modelos por parâmetros e ruído (0 a 0.001)\n")
    print("  \nSair do programa: '0'")
    
    try:
        opcao = int(input("\nDigite a opção desejada: "))
        return opcao
    except ValueError:
        return -1  # Retorna -1 se o usuário inserir um valor não numérico

# Atualizar o bloco de execução principal
if __name__ == "__main__":
    # Criar diretórios para gráficos se não existirem
    ensure_plot_directories()
    
    while True:
        opcao = show_menu()
        
        if opcao == 0:
            print("\nPrograma finalizado. Obrigado!")
            break
        elif opcao == 1:
            plot_initial_data()
            input("\nPressione Enter para continuar...")
        elif opcao == 2:
            apply_smoothing = input("\nAplicar correlação temporal aos dados? (s/n): ").lower() == 's'
            run_outer_product_model(apply_smoothing=apply_smoothing)
            input("\nPressione Enter para continuar...")
        elif opcao == 3:
            apply_smoothing = input("\nAplicar correlação temporal aos dados? (s/n): ").lower() == 's'
            run_edge_reduced_model(apply_smoothing=apply_smoothing)
            input("\nPressione Enter para continuar...")
        elif opcao == 4:
            apply_smoothing = input("\nAplicar correlação temporal aos dados? (s/n): ").lower() == 's'
            run_squared_model(apply_smoothing=apply_smoothing)
            input("\nPressione Enter para continuar...")
        elif opcao == 5:
            apply_smoothing = input("\nAplicar correlação temporal aos dados? (s/n): ").lower() == 's'
            run_nearby_model(apply_smoothing=apply_smoothing)
            input("\nPressione Enter para continuar...")
        elif opcao == 6:
            compare_temporal_correlation_different_noise()
            input("\nPressione Enter para continuar...")
        elif opcao == 7:
            compare_temporal_correlation()
            input("\nPressione Enter para continuar...")
        elif opcao == 8:
            compare_models_by_parameters_and_noise()
            input("\nPressione Enter para continuar...")
        else:
            print("\nOpção inválida! Por favor, escolha uma opção válida.")
            input("\nPressione Enter para continuar...")