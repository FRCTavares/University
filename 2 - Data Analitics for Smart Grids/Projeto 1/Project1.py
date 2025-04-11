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
def load_data(file_path='DATA.xlsx'):
    """
    Carrega os dados do arquivo Excel e retorna as informações necessárias.
    
    Parameters:
        file_path (str): Path to the Excel file containing the data
        
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
    for dir_path in ['Plots/OG', 'Plots/LV', 'Plots/Top', 'Plots/Quad', 
                     'Plots/Inje', 'Plots/Large', 'Plots/Noise', 'Plots/Comp']:
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


# ============================================================================================================================================
# Função para o modelo original (Outer-Product)
# ============================================================================================================================================
def run_outer_product_model(noise_factor=0.0025, network_factor=100, ptest_factor=3):
    """
    Executa o modelo original para o cálculo de perdas da rede.
    
    Parameters:
        noise_factor (float): Factor for adding noise to the data (default: 0.0025)
        network_factor (float): Factor to scale network characteristics (default: 100)
        ptest_factor (float): Factor to scale test power values (default: 3)
        
    Returns:
        dict: Dictionary containing model results and metrics
    """
    print("\n=== Executando o Modelo Original ===\n")
    
    # Ensure plot directories exist
    ensure_plot_directories()
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
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
    train_mae = mean_absolute_error(PL2_train_true, PL2_train_pred)
    test_rmse = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred))
    test_mae = mean_absolute_error(PL2_test_true, PL2_test_pred)
    
    print("Modelo Original:")
    print("Erro de treino - RMSE:", train_rmse)
    print("Erro de treino - MAE: ", train_mae)
    print("Erro de teste - RMSE:", test_rmse)
    print("Erro de teste - MAE: ", test_mae)
    
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
    ax[0, 0].step(time_intervals_train, PL2_train_pred, where='post', label='Perdas Preditas')
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
    ax[0, 1].step(time_intervals_test, PL2_test_pred, where='post', label='Perdas Preditas')
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
    plt.savefig('Plots/OG/resultados_modelo_original.png', dpi=300)
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
      X_topo: Matriz de features considerando a topologia (edge-reduced) da rede
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

def run_edge_reduced_model():
    """Executa o modelo que considera a topologia (edge-reduced) da rede para o cálculo de perdas."""
    print("\n=== Executando o Modelo com Topologia (edge-reduced) da Rede ===\n")
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    
    # Construir matriz X' baseada na topologia (edge-reduced) para os dados de treino
    X_topo_train = edge_reduced_X(P, linha_indices, SlackBus)
    
    # Cálculo do beta utilizando OLS
    beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_topo = X_topo_train @ beta_topo
    
    # Construir matriz X' baseada na topologia (edge-reduced) para os dados de teste
    X_topo_test = edge_reduced_X(Ptest, linha_indices, SlackBus)
    
    # Predição para os dados de teste
    PL2_test_pred_topo = X_topo_test @ beta_topo
    
    # Cálculo das perdas físicas para os dados de teste
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # Cálculo das métricas de erro
    train_rmse_topo = np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_topo))
    train_mae_topo = mean_absolute_error(PL2_train_true, PL2_train_pred_topo)
    test_rmse_topo = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_topo))
    test_mae_topo = mean_absolute_error(PL2_test_true, PL2_test_pred_topo)
    
    print("Modelo com Topologia (edge-reduced) da Rede:")
    print("Erro de treino - RMSE:", train_rmse_topo)
    print("Erro de treino - MAE: ", train_mae_topo)
    print("Erro de teste - RMSE:", test_rmse_topo)
    print("Erro de teste - MAE: ", test_mae_topo)
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
    ax[0, 0].step(time_intervals_train, PL2_train_pred_topo, where='post', label='Perdas Preditas')
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
    ax[1, 0].step(time_intervals_test, PL2_test_pred_topo, where='post', label='Perdas Preditas')
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
    plt.savefig('Plots/Top/resultados_modelo_topologia.png', dpi=300)
    plt.show()
    
    # Mostrar os coeficientes beta e sua relação com a estrutura da rede
    print("\nCOEFICIENTES DO MODELO COM TOPOLOGIA (edge-reduced)")
    
    # Termos quadráticos (squares only)
    print("\nTermos Quadráticos (squares only):")
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

def run_squared_model():
    """Executa o modelo simplificado que considera apenas termos quadráticos (squares only) P_i²."""
    print("\n=== Executando o Modelo Simplificado (Apenas Termos Quadráticos (squares only)) ===\n")
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
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
    train_mae_sq = mean_absolute_error(PL2_train_true, PL2_train_pred_sq)
    test_rmse_sq = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_sq))
    test_mae_sq = mean_absolute_error(PL2_test_true, PL2_test_pred_sq)
    
    print("Modelo Simplificado (Apenas Termos Quadráticos (squares only)):")
    print("Erro de treino - RMSE:", train_rmse_sq)
    print("Erro de treino - MAE: ", train_mae_sq)
    print("Erro de teste - RMSE:", test_rmse_sq)
    print("Erro de teste - MAE: ", test_mae_sq)
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
    ax[0, 0].step(time_intervals_train, PL2_train_pred_sq, where='post', label='Treino: Perdas Preditas (Quadrático (squares-only))')
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
    ax[1, 0].step(time_intervals_test, PL2_test_pred_sq, where='post', label='Teste: Perdas Preditas (Quadrático (squares-only))', color='orange')
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
    plt.savefig('Plots/Quad/resultados_modelo_quadratico.png', dpi=300)
    plt.show()
    
    # Mostrar os coeficientes beta
    print("\nCOEFICIENTES DO MODELO SIMPLIFICADO")
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

def run_nearby_model():
    """Executa o modelo simplificado que considera a soma de injeções eletricamente próximas."""
    print("\n=== Executando o Modelo Simplificado (Soma de Injeções Eletricamente Próximas) ===\n")
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
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
    train_mae_nearby = mean_absolute_error(PL2_train_true, PL2_train_pred_nearby)
    test_rmse_nearby = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_nearby))
    test_mae_nearby = mean_absolute_error(PL2_test_true, PL2_test_pred_nearby)
    
    print("Modelo Simplificado (Soma de Injeções Eletricamente Próximas):")
    print("Erro de treino - RMSE:", train_rmse_nearby)
    print("Erro de treino - MAE: ", train_mae_nearby)
    print("Erro de teste - RMSE:", test_rmse_nearby)
    print("Erro de teste - MAE: ", test_mae_nearby)

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
    ax[0, 0].step(time_intervals_train, PL2_train_pred_nearby, where='post', label='Perdas Preditas')
    ax[0, 0].set_title('Comparação de Perdas - Treino')
    ax[0, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência [p.u.]')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Test data comparison
    ax[0, 1].step(time_intervals_test, PL2_test_true, where='post', label='Perdas Reais')
    ax[0, 1].step(time_intervals_test, PL2_test_pred_nearby, where='post', label='Perdas Preditas')
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
    plt.savefig('Plots/Inje/resultados_modelo_injeções_proximas.png', dpi=300)
    plt.show()


# ============================================================================================================================================
# Funções para o modelo de teste de diferentes valores de ruído
# ============================================================================================================================================
def compare_network_sizes_different_noise():
    """Compares model performance with different noise levels."""
    print("\n=== Comparando Desempenho dos Modelos com Diferentes Níveis de Ruído ===\n")
    
    # Define noise levels to test
    noise_levels = [0, 0.001, 0.003, 0.005, 0.01]
    
    # Dictionary to store results for each noise level and model
    noise_results = {
        'small': {model: {'RMSE': [], 'MAE': []} for model in ['Original', 'Topologia (edge-reduced)', 
                                                              'Quadrático (squares-only)', 
                                                              'Injeções Próximas (squares-reduced)']},
        'large': {model: {'RMSE': [], 'MAE': []} for model in ['Original', 'Topologia (edge-reduced)', 
                                                              'Quadrático (squares-only)', 
                                                              'Injeções Próximas (squares-reduced)']}
    }
    
    # We'll also keep track of parameters (they don't change with noise)
    model_params = {
        'small': {model: 0 for model in ['Original', 'Topologia (edge-reduced)', 
                                        'Quadrático (squares-only)', 
                                        'Injeções Próximas (squares-reduced)']},
        'large': {model: 0 for model in ['Original', 'Topologia (edge-reduced)', 
                                        'Quadrático (squares-only)', 
                                        'Injeções Próximas (squares-reduced)']}
    }
    
    # ===== Setup for small network =====
    print("Configurando rede original (5 barramentos)...")
    
    # Load data for small network
    SlackBus, Net_Info, P, Ptest = load_data()
    
    # Build network matrices
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Build model matrices for small network (these don't change with noise)
    X_orig_train_small = np.column_stack((
        P[:, 0]**2, 2*P[:, 0]*P[:, 1], 2*P[:, 0]*P[:, 2], 2*P[:, 0]*P[:, 3],
        P[:, 1]**2, 2*P[:, 1]*P[:, 2], 2*P[:, 1]*P[:, 3],
        P[:, 2]**2, 2*P[:, 2]*P[:, 3], P[:, 3]**2
    ))
    X_orig_test_small = np.column_stack((
        Ptest[:, 0]**2, 2*Ptest[:, 0]*Ptest[:, 1], 2*Ptest[:, 0]*Ptest[:, 2], 2*Ptest[:, 0]*Ptest[:, 3],
        Ptest[:, 1]**2, 2*Ptest[:, 1]*Ptest[:, 2], 2*Ptest[:, 1]*Ptest[:, 3],
        Ptest[:, 2]**2, 2*Ptest[:, 2]*Ptest[:, 3], Ptest[:, 3]**2
    ))
    
    X_topo_train_small = edge_reduced_X(P, linha_indices, SlackBus)
    X_topo_test_small = edge_reduced_X(Ptest, linha_indices, SlackBus)
    
    X_sq_train_small = squares_only_X(P)
    X_sq_test_small = squares_only_X(Ptest)
    
    X_nearby_train_small = squares_reduced_X(P, linha_indices, SlackBus)
    X_nearby_test_small = squares_reduced_X(Ptest, linha_indices, SlackBus)
    
    # Store the number of parameters for each model (small network)
    model_params['small']['Original'] = X_orig_train_small.shape[1]
    model_params['small']['Topologia (edge-reduced)'] = X_topo_train_small.shape[1]
    model_params['small']['Quadrático (squares-only)'] = X_sq_train_small.shape[1]
    model_params['small']['Injeções Próximas (squares-reduced)'] = X_nearby_train_small.shape[1]
    
    # ===== Setup for large network =====
    print("Configurando rede maior (25 barramentos)...")
    
    # Setup large network (similar to compare_network_sizes function)
    n_buses = 25
    
    # Generate network topology - mix of radial structure with additional connections
    edge_list = [(i, i+1) for i in range(n_buses-1)]  # Base radial
    extra_edges = [(0, 5), (5, 10), (10, 15), (15, 20), 
                  (2, 7), (7, 12), (12, 17), (17, 22),
                  (3, 8), (8, 13), (13, 18), (18, 23)]
    edge_list.extend(extra_edges)
    
    # Build Y matrix for synthetic network
    Y_large = np.zeros((n_buses, n_buses), dtype=complex)
    np.random.seed(42)
    linha_indices_large = []
    
    for i, j in edge_list:
        # Typical values for transmission lines
        r = 0.01 + 0.05 * np.random.random()
        x = 0.05 + 0.15 * np.random.random()
        
        y = 1 / complex(r, x)
        Y_large[i, j] = Y_large[j, i] = -y
        Y_large[i, i] += y
        Y_large[j, j] += y
        linha_indices_large.append((i, j))
    
    # Reference bus (slack)
    slack_bus = 0
    
    # Generate load/injection data
    n_instantes_train = 500
    n_instantes_test = 100
    
    # Power matrices (excluding slack)
    P_large = np.random.randn(n_instantes_train, n_buses-1) * 0.5
    Ptest_large = np.random.randn(n_instantes_test, n_buses-1) * 0.7
    
    # Prepare matrices for loss calculation
    Y_large_no_slack = np.delete(np.delete(Y_large, slack_bus, axis=0), slack_bus, axis=1)
    G_large = Y_large_no_slack.real
    B_large = Y_large_no_slack.imag
    
    # Incidence matrix
    C_large = np.zeros((n_buses, len(edge_list)))
    for idx, (i, j) in enumerate(edge_list):
        C_large[i, idx] = 1
        C_large[j, idx] = -1
    
    Cl_large = np.delete(C_large, slack_bus, axis=0)
    
    # Line conductance vector
    Gv_large = np.zeros(len(edge_list))
    for idx, (i, j) in enumerate(edge_list):
        Gv_large[idx] = -Y_large[i, j].real
    
    # Build model matrices for large network
    # Original model
    X_orig_large = []
    # Quadratic terms
    for i in range(n_buses-1):
        X_orig_large.append(P_large[:, i]**2)
    # Cross terms
    for i in range(n_buses-1):
        for j in range(i+1, n_buses-1):
            X_orig_large.append(2 * P_large[:, i] * P_large[:, j])
    X_orig_train_large = np.column_stack(X_orig_large)
    
    X_orig_large_test = []
    for i in range(n_buses-1):
        X_orig_large_test.append(Ptest_large[:, i]**2)
    for i in range(n_buses-1):
        for j in range(i+1, n_buses-1):
            X_orig_large_test.append(2 * Ptest_large[:, i] * Ptest_large[:, j])
    X_orig_test_large = np.column_stack(X_orig_large_test)
    
    # Topology model
    X_topo_train_large = edge_reduced_X(P_large, linha_indices_large, slack_bus+1)
    X_topo_test_large = edge_reduced_X(Ptest_large, linha_indices_large, slack_bus+1)
    
    # Squared model
    X_sq_train_large = squares_only_X(P_large)
    X_sq_test_large = squares_only_X(Ptest_large)
    
    # Nearby injections model (clusters)
    num_clusters = 5
    buses_per_cluster = (n_buses - 1) // num_clusters
    remainder = (n_buses - 1) % num_clusters
    
    clusters = []
    start_idx = 0
    
    for i in range(num_clusters):
        size = buses_per_cluster + (1 if i < remainder else 0)
        end_idx = start_idx + size
        clusters.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    
    # Build nearby injections matrices
    X_nearby_large = []
    # Quadratic terms for each cluster
    for cluster in clusters:
        if cluster:
            cluster_sum = np.zeros(n_instantes_train)
            for bus in cluster:
                if bus < P_large.shape[1]:
                    cluster_sum += P_large[:, bus]
            X_nearby_large.append(cluster_sum**2)
    
    # Cross terms between clusters
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if clusters[i] and clusters[j]:
                sum_i = np.zeros(n_instantes_train)
                sum_j = np.zeros(n_instantes_train)
                
                for bus in clusters[i]:
                    if bus < P_large.shape[1]:
                        sum_i += P_large[:, bus]
                        
                for bus in clusters[j]:
                    if bus < P_large.shape[1]:
                        sum_j += P_large[:, bus]
                        
                X_nearby_large.append(2 * sum_i * sum_j)
    
    X_nearby_train_large = np.column_stack(X_nearby_large)
    
    X_nearby_large_test = []
    # Quadratic terms for test data
    for cluster in clusters:
        if cluster:
            cluster_sum = np.zeros(n_instantes_test)
            for bus in cluster:
                if bus < Ptest_large.shape[1]:
                    cluster_sum += Ptest_large[:, bus]
            X_nearby_large_test.append(cluster_sum**2)
    
    # Cross terms between clusters for test data
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if clusters[i] and clusters[j]:
                sum_i = np.zeros(n_instantes_test)
                sum_j = np.zeros(n_instantes_test)
                
                for bus in clusters[i]:
                    if bus < Ptest_large.shape[1]:
                        sum_i += Ptest_large[:, bus]
                        
                for bus in clusters[j]:
                    if bus < Ptest_large.shape[1]:
                        sum_j += Ptest_large[:, bus]
                        
                X_nearby_large_test.append(2 * sum_i * sum_j)
    
    X_nearby_test_large = np.column_stack(X_nearby_large_test)
    
    # Store the number of parameters for each model (large network)
    model_params['large']['Original'] = X_orig_train_large.shape[1]
    model_params['large']['Topologia (edge-reduced)'] = X_topo_train_large.shape[1]
    model_params['large']['Quadrático (squares-only)'] = X_sq_train_large.shape[1]
    model_params['large']['Injeções Próximas (squares-reduced)'] = X_nearby_train_large.shape[1]

    # ===== Main loop for different noise levels =====
    for noise_level in noise_levels:
        print(f"\nAvaliando modelos com nível de ruído: {noise_level}")
        
        # ==== Small network ====
        print("  Processando rede original (5 barramentos)...")
        
        # Calculate true losses with current noise level
        PL2_train_true_small = compute_PL2(P, B, Cl, Gv, noise_level)
        PL2_test_true_small = compute_PL2(Ptest, B, Cl, Gv, noise_level)
        
        # 1. Original model
        beta_original = inv(X_orig_train_small.T @ X_orig_train_small) @ (X_orig_train_small.T @ PL2_train_true_small)
        pred_test_original = X_orig_test_small @ beta_original
        
        # Calculate metrics
        rmse_original = np.sqrt(mean_squared_error(PL2_test_true_small, pred_test_original))
        mae_original = mean_absolute_error(PL2_test_true_small, pred_test_original)
        
        noise_results['small']['Original']['RMSE'].append(rmse_original)
        noise_results['small']['Original']['MAE'].append(mae_original)
        
        # 2. Topology model
        beta_topo = inv(X_topo_train_small.T @ X_topo_train_small) @ (X_topo_train_small.T @ PL2_train_true_small)
        pred_test_topo = X_topo_test_small @ beta_topo
        
        # Calculate metrics
        rmse_topo = np.sqrt(mean_squared_error(PL2_test_true_small, pred_test_topo))
        mae_topo = mean_absolute_error(PL2_test_true_small, pred_test_topo)
        
        noise_results['small']['Topologia (edge-reduced)']['RMSE'].append(rmse_topo)
        noise_results['small']['Topologia (edge-reduced)']['MAE'].append(mae_topo)
        
        # 3. Squared model
        beta_sq = inv(X_sq_train_small.T @ X_sq_train_small) @ (X_sq_train_small.T @ PL2_train_true_small)
        pred_test_sq = X_sq_test_small @ beta_sq
        
        # Calculate metrics
        rmse_sq = np.sqrt(mean_squared_error(PL2_test_true_small, pred_test_sq))
        mae_sq = mean_absolute_error(PL2_test_true_small, pred_test_sq)
        
        noise_results['small']['Quadrático (squares-only)']['RMSE'].append(rmse_sq)
        noise_results['small']['Quadrático (squares-only)']['MAE'].append(mae_sq)
        
        # 4. Nearby injections model
        beta_nearby = inv(X_nearby_train_small.T @ X_nearby_train_small) @ (X_nearby_train_small.T @ PL2_train_true_small)
        pred_test_nearby = X_nearby_test_small @ beta_nearby
        
        # Calculate metrics
        rmse_nearby = np.sqrt(mean_squared_error(PL2_test_true_small, pred_test_nearby))
        mae_nearby = mean_absolute_error(PL2_test_true_small, pred_test_nearby)
        
        noise_results['small']['Injeções Próximas (squares-reduced)']['RMSE'].append(rmse_nearby)
        noise_results['small']['Injeções Próximas (squares-reduced)']['MAE'].append(mae_nearby)
        
        # ==== Large network ====
        print("  Processando rede maior (25 barramentos)...")
        
        # Calculate true losses for current noise level
        invB_large = inv(B_large)
        PL2_train_true_large = np.zeros(n_instantes_train)
        PL2_test_true_large = np.zeros(n_instantes_test)
        
        for m in range(n_instantes_train):
            teta = invB_large @ P_large[m, :].T
            grau = Cl_large.T @ teta
            PL2_m = (2 * Gv_large) @ (1 - np.cos(grau))
            PL2_train_true_large[m] = PL2_m * (1 + np.random.normal(0, noise_level))
        
        for m in range(n_instantes_test):
            teta = invB_large @ Ptest_large[m, :].T
            grau = Cl_large.T @ teta
            PL2_m = (2 * Gv_large) @ (1 - np.cos(grau))
            PL2_test_true_large[m] = PL2_m * (1 + np.random.normal(0, noise_level))
        
        # 1. Original model (large)
        beta_original_large = inv(X_orig_train_large.T @ X_orig_train_large) @ (X_orig_train_large.T @ PL2_train_true_large)
        pred_test_original_large = X_orig_test_large @ beta_original_large
        
        # Calculate metrics
        rmse_original_large = np.sqrt(mean_squared_error(PL2_test_true_large, pred_test_original_large))
        mae_original_large = mean_absolute_error(PL2_test_true_large, pred_test_original_large)
        
        noise_results['large']['Original']['RMSE'].append(rmse_original_large)
        noise_results['large']['Original']['MAE'].append(mae_original_large)
        
        # 2. Topology model (large)
        beta_topo_large = inv(X_topo_train_large.T @ X_topo_train_large) @ (X_topo_train_large.T @ PL2_train_true_large)
        pred_test_topo_large = X_topo_test_large @ beta_topo_large
        
        # Calculate metrics
        rmse_topo_large = np.sqrt(mean_squared_error(PL2_test_true_large, pred_test_topo_large))
        mae_topo_large = mean_absolute_error(PL2_test_true_large, pred_test_topo_large)
        
        noise_results['large']['Topologia (edge-reduced)']['RMSE'].append(rmse_topo_large)
        noise_results['large']['Topologia (edge-reduced)']['MAE'].append(mae_topo_large)
        
        # 3. Squared model (large)
        beta_sq_large = inv(X_sq_train_large.T @ X_sq_train_large) @ (X_sq_train_large.T @ PL2_train_true_large)
        pred_test_sq_large = X_sq_test_large @ beta_sq_large
        
        # Calculate metrics
        rmse_sq_large = np.sqrt(mean_squared_error(PL2_test_true_large, pred_test_sq_large))
        mae_sq_large = mean_absolute_error(PL2_test_true_large, pred_test_sq_large)
        
        noise_results['large']['Quadrático (squares-only)']['RMSE'].append(rmse_sq_large)
        noise_results['large']['Quadrático (squares-only)']['MAE'].append(mae_sq_large)
        
        # 4. Nearby injections model (large)
        beta_nearby_large = inv(X_nearby_train_large.T @ X_nearby_train_large) @ (X_nearby_train_large.T @ PL2_train_true_large)
        pred_test_nearby_large = X_nearby_test_large @ beta_nearby_large
        
        # Calculate metrics
        rmse_nearby_large = np.sqrt(mean_squared_error(PL2_test_true_large, pred_test_nearby_large))
        mae_nearby_large = mean_absolute_error(PL2_test_true_large, pred_test_nearby_large)
        
        noise_results['large']['Injeções Próximas (squares-reduced)']['RMSE'].append(rmse_nearby_large)
        noise_results['large']['Injeções Próximas (squares-reduced)']['MAE'].append(mae_nearby_large)
    
    # ===== Visualization =====
    
    # Plot 1: Small network RMSE vs Noise Level for all models
    plt.figure(figsize=(14, 10))
    
    plt.subplot(1, 2, 1)
    for model in noise_results['small'].keys():
        plt.plot(noise_levels, noise_results['small'][model]['RMSE'], 
                 marker='o', linewidth=2, label=model)
    
    plt.title('RMSE vs Nível de Ruído - Rede Original (5 barramentos)')
    plt.xlabel('Nível de Ruído')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    
    
    # Plot 3: Large network RMSE vs Noise Level for all models
    plt.subplot(1, 2, 2)
    for model in noise_results['large'].keys():
        plt.plot(noise_levels, noise_results['large'][model]['RMSE'], 
                 marker='o', linewidth=2, label=model)
    
    plt.title('RMSE vs Nível de Ruído - Rede Maior (25 barramentos)')
    plt.xlabel('Nível de Ruído')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    
    
    plt.tight_layout()
    plt.savefig('Plots/Noise/comparacao_ruido_todas_metricas.png', dpi=300)
    
    # Plot 5: Combined comparison across networks for RMSE
    plt.figure(figsize=(16, 7))
    models = list(noise_results['small'].keys())
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, model in enumerate(models):
        plt.plot(noise_levels, noise_results['small'][model]['RMSE'], 
                 marker='o', color=colors[i], linestyle='-', linewidth=2, 
                 label=f'{model} (5 barramentos)')
        plt.plot(noise_levels, noise_results['large'][model]['RMSE'], 
                 marker='s', color=colors[i], linestyle='--', linewidth=2, 
                 label=f'{model} (25 barramentos)')
    
    plt.title('RMSE vs Nível de Ruído - Comparação entre Redes')
    plt.xlabel('Nível de Ruído')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    
    plt.tight_layout()
    plt.savefig('Plots/Noise/comparacao_ruido_parametros.png', dpi=300)
    
    # Print summary table
    print("\nResumo dos Resultados - RMSE para Diferentes Níveis de Ruído:")
    print("-" * 95)
    header = "Modelo".ljust(30) + " | "
    for noise in noise_levels:
        header += f"Ruído {noise:.4f}".ljust(12) + " | "
    print(header)
    print("-" * 95)
    
    for net_size in ['small', 'large']:
        print(f"Rede {'Original (5 barramentos)' if net_size=='small' else 'Maior (25 barramentos)'}")
        for model in models:
            row = model.ljust(30) + " | "
            for i, noise in enumerate(noise_levels):
                row += f"{noise_results[net_size][model]['RMSE'][i]:.6f}".ljust(12) + " | "
            print(row)
        print("-" * 95)

    plt.show()
    
    return noise_results, model_params

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
    print("  1. Executar o modelo com maior dimensionalidade de X (Outer Product)")
    print("  2. Executar o modelo com X reduzida (edge-reduced)")
    print("  3. Executar o modelo com X reduzida (squares only)")
    print("  4. Executar o modelo com X reduzida (squares-reduced)")
    print("  5. Comparar desempenho dos com diferentes níveis de ruído\n")
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
            run_outer_product_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 2:
            run_edge_reduced_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 3:
            run_squared_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 4:
            run_nearby_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 5:
            compare_network_sizes_different_noise()
            input("\nPressione Enter para continuar...")
        else:
            print("\nOpção inválida! Por favor, escolha uma opção válida.")
            input("\nPressione Enter para continuar...")