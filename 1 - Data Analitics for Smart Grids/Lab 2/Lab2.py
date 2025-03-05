###############################################################################################
# Laboratorio 2 - Perdas de Potência                                                          #
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
# Funções comuns
# ============================================================================================================================================
def load_data():
    """Carrega os dados do arquivo Excel e retorna as informações necessárias."""
    Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
    SlackBus = Info[0, 1]  # Informação acerca do bus slack

    Net_Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
    Power_Info = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
    Power_Info = np.delete(Power_Info, [0], 1)
    Power_Test = np.array(pd.read_excel(r'DASG_Prob2_new.xlsx', sheet_name='Test_Load(t,Bus)'))
    Power_Test = np.delete(Power_Test, [0], 1)

    time = Power_Info.shape[0]
    P = Power_Info
    Ptest = Power_Test * PtestFactor

    return SlackBus, Net_Info, P, Ptest

def build_network_matrices(SlackBus, Net_Info):
    """Constrói as matrizes da rede (Y, G, B, C, Cl, Gv, Gd) e retorna os valores."""
    # Construção da matriz de admitância Y
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

def compute_PL2(P_mat, B, Cl, Gv, noiseFactor):
    """
    Calcula as perdas físicas (PL2) para cada instante de tempo utilizando a Eq (13).

    Parâmetros:
      P_mat: Matriz de potência, com dimensão (n_instantes, nBus)  (ou nBus-1 se o slack for removido)
      B: Matriz de susceptância, dimensão (nBus-1, nBus-1)
      Cl: Matriz de incidência (com slack removido), dimensão (nBus-1, nLines)
      Gv: Vetor de condutância das linhas, dimensão (1, nLines)
      noiseFactor: Fator de ruído multiplicativo

    Retorna:
      PL2_vec: Vetor das perdas para cada instante (valores escalares)
    """
    n_instantes = P_mat.shape[0]
    PL2_vec = np.zeros(n_instantes)
    invB = inv(B)  # Pré-computa a inversa de B
    for m in range(n_instantes):
        # Calcular os ângulos de tensão: teta = inv(B) @ P_mat[m, :].T
        teta = invB @ P_mat[m, :].T
        # Calcular as diferenças dos ângulos: grau = Cl.T @ teta
        grau = Cl.T @ teta
        # Calcular as perdas físicas: PL2 = 2 * Gv @ (1 - cos(grau))
        PL2_m = (2 * Gv) @ (1 - np.cos(grau))
        # Extrair o valor escalar a partir do array 1x1
        PL2_scalar = PL2_m.item()
        # Adicionar ruído multiplicativo
        PL2_vec[m] = PL2_scalar * (1 + np.random.normal(0, noiseFactor))
    return PL2_vec


# ============================================================================================================================================
# Função para o modelo original
# ============================================================================================================================================
def run_original_model():
    """Executa o modelo original para o cálculo de perdas da rede."""
    print("\n=== Executando o Modelo Original ===\n")
    
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
# Funções para o modelo de baixa tensão (LV) Desafio 1
# ============================================================================================================================================
def compute_losses_LV(P_mat, Y, linha_indices, SlackBus, noiseFactor, v0=1.0):
    """
    Calcula as perdas físicas em uma rede de baixa tensão, assumindo diferenças de ângulo 
    negligíveis e focando nas diferenças de magnitude de tensão.
    """
    n_instantes = P_mat.shape[0]
    nBus = Y.shape[0]
    losses_vec = np.zeros(n_instantes)
    
    # Matriz de condutância completa (parte real da matriz Y)
    G_full = Y.real
    
    for m in range(n_instantes):
        # Em redes LV, estimamos as tensões com base nos injetados de potência
        V_magnitudes = np.ones(nBus)  # Inicialmente todos os barramentostêm tensão nominal
        
        # Ajustar magnitudes baseando-se nos injetados (valores positivos reduzem tensão)
        # Esta é uma aproximação linear simplificada adequada para redes LV
        for i in range(nBus):
            if i != (SlackBus-1):  # O slack mantém tensão fixa
                idx = i if i < (SlackBus-1) else i-1
                V_magnitudes[i] = v0 - 0.01 * P_mat[m, idx]  # Aproximação linear simples
        
        # Calcular perdas usando diferenças de tensão
        total_loss = 0
        for idx, (i, j) in enumerate(linha_indices):
            delta_V_squared = (V_magnitudes[i] - V_magnitudes[j])**2
            G_ij = -G_full[i, j]  # Condutância da linha i-j
            loss_ij = G_ij * delta_V_squared
            total_loss += loss_ij
        
        # Adicionar ruído
        losses_vec[m] = total_loss * (1 + np.random.normal(0, noiseFactor))
        
    return losses_vec

def build_X_prime(P_mat, Y, linha_indices, slack_bus, v0=1.0):
    """
    Constrói a matriz X' para o modelo LV baseada nas diferenças de tensão ao quadrado
    entre os barramentosconectados por linhas.
    """
    n_instantes = P_mat.shape[0]
    nBus = Y.shape[0]
    nLines = len(linha_indices)
    X_prime = np.zeros((n_instantes, nLines))
    
    for m in range(n_instantes):
        # Estimar magnitudes de tensão
        V_magnitudes = np.ones(nBus)
        for i in range(nBus):
            if i != (slack_bus-1):
                idx = i if i < (slack_bus-1) else i-1
                V_magnitudes[i] = v0 - 0.01 * P_mat[m, idx]
        
        # Calcular (V_i - V_j)^2 para cada linha
        for l, (i, j) in enumerate(linha_indices):
            X_prime[m, l] = (V_magnitudes[i] - V_magnitudes[j])**2
    
    return X_prime

def run_LV_model():
    """Executa o modelo de baixa tensão (LV) para o cálculo de perdas da rede."""
    print("\n=== Executando o Modelo de Baixa Tensão (LV) ===\n")
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino usando o modelo de baixa tensão
    PL2_train_true_LV = compute_losses_LV(P, Y, linha_indices, SlackBus, noiseFactor)
    
    # Construir matriz X' para os dados de treino
    X_prime_train = build_X_prime(P, Y, linha_indices, SlackBus)
    
    # Cálculo do beta para o modelo LV usando OLS
    beta_LV = inv(X_prime_train.T @ X_prime_train) @ (X_prime_train.T @ PL2_train_true_LV)
    
    # Predição para os dados de treino usando o modelo LV
    PL2_train_pred_LV = X_prime_train @ beta_LV
    
    # Cálculo das perdas físicas para os dados de teste
    PL2_test_true_LV = compute_losses_LV(Ptest, Y, linha_indices, SlackBus, noiseFactor)
    
    # Construir matriz X' para os dados de teste
    X_prime_test = build_X_prime(Ptest, Y, linha_indices, SlackBus)
    
    # Predição para os dados de teste usando o modelo LV
    PL2_test_pred_LV = X_prime_test @ beta_LV
    
    # Cálculo das métricas de erro
    train_rmse_LV = np.sqrt(mean_squared_error(PL2_train_true_LV, PL2_train_pred_LV))
    train_mae_LV = mean_absolute_error(PL2_train_true_LV, PL2_train_pred_LV)
    test_rmse_LV = np.sqrt(mean_squared_error(PL2_test_true_LV, PL2_test_pred_LV))
    test_mae_LV = mean_absolute_error(PL2_test_true_LV, PL2_test_pred_LV)
    
    print("Modelo de Baixa Tensão (LV):")
    print("Erro de treino - RMSE (LV):", train_rmse_LV)
    print("Erro de treino - MAE (LV) :", train_mae_LV)
    print("Erro de teste - RMSE (LV):", test_rmse_LV)
    print("Erro de teste - MAE (LV) :", test_mae_LV)
    
    # Visualização dos resultados
    time_intervals_train = np.arange(len(PL2_train_true_LV))
    time_intervals_test = np.arange(len(PL2_test_true_LV))
    
    # Calculate errors
    train_errors = 100 * np.abs(PL2_train_true_LV - PL2_train_pred_LV) / np.abs(PL2_train_true_LV)
    test_errors = 100 * np.abs(PL2_test_true_LV - PL2_test_pred_LV) / np.abs(PL2_test_true_LV)
    
    # Create a 2x2 subplot grid
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Top-left: Training data comparison
    ax[0, 0].step(time_intervals_train, PL2_train_true_LV, where='post', label='Perdas Reais (PL2 com ruído)')
    ax[0, 0].step(time_intervals_train, PL2_train_pred_LV, where='post', label='Perdas Preditas')
    ax[0, 0].set_title('Comparação de Perdas - Treino')
    ax[0, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Test data comparison
    ax[0, 1].step(time_intervals_test, PL2_test_true_LV, where='post', label='Perdas Reais')
    ax[0, 1].step(time_intervals_test, PL2_test_pred_LV, where='post', label='Perdas Preditas')
    ax[0, 1].set_title('Comparação de Perdas - Teste')
    ax[0, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax[0, 1].set_ylabel('Perdas de Potência')
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
    
    # Update y-axis limits for percentage error plots
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[1, 0].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    
    plt.tight_layout()
    plt.savefig('Plots/LV/resultados_modelo_LV.png', dpi=300)
    plt.show()
    
    # Comparar os coeficientes beta com as condutâncias das linhas
    print("\nCOMPARAÇÃO DOS COEFICIENTES E CONDUTÂNCIAS")
    print("Coeficientes beta (modelo LV):")
    for i, (bus1, bus2) in enumerate(linha_indices):
        g_ij = -np.real(Y[bus1, bus2])
        print(f"Linha {i+1} (Bus {bus1+1}-{bus2+1}): Beta = {beta_LV[i]:.6f}, G_ij = {g_ij:.6f}")


# ============================================================================================================================================
# Funções para o modelo baseado na topologia (edge-reduced) da rede Desafio 2.1
# ============================================================================================================================================
def build_X_topology(P_mat, linha_indices, SlackBus):
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

def run_topology_model():
    """Executa o modelo que considera a topologia (edge-reduced) da rede para o cálculo de perdas."""
    print("\n=== Executando o Modelo com Topologia (edge-reduced) da Rede ===\n")
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    
    # Construir matriz X' baseada na topologia (edge-reduced) para os dados de treino
    X_topo_train = build_X_topology(P, linha_indices, SlackBus)
    
    # Cálculo do beta utilizando OLS
    beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_topo = X_topo_train @ beta_topo
    
    # Construir matriz X' baseada na topologia (edge-reduced) para os dados de teste
    X_topo_test = build_X_topology(Ptest, linha_indices, SlackBus)
    
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
# Funções para o modelo simplificado (apenas termos quadráticos (squares only)) Desafio 2.2
# ============================================================================================================================================
def build_X_squared(P_mat):
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
    X_sq_train = build_X_squared(P)
    
    # Cálculo do beta utilizando OLS
    beta_sq = inv(X_sq_train.T @ X_sq_train) @ (X_sq_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_sq = X_sq_train @ beta_sq
    
    # Construir matriz X_sq para os dados de teste
    X_sq_test = build_X_squared(Ptest)
    
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
# Funções para o modelo simplificado (soma de injeções eletricamente proximas (squares-reduced)) - Desafio 2.3 
# ============================================================================================================================================
def build_X_nearby(P_mat, linha_indices, SlackBus):
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
    X_nearby_train = build_X_nearby(P, linha_indices, SlackBus)
    
    # Cálculo do beta utilizando OLS
    beta_nearby = inv(X_nearby_train.T @ X_nearby_train) @ (X_nearby_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_nearby = X_nearby_train @ beta_nearby
    
    # Construir matriz X_nearby para os dados de teste
    X_nearby_test = build_X_nearby(Ptest, linha_indices, SlackBus)
    
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

    # Calculate errors as percentage (consistente com run_original_model)
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

    # Update y-axis limits for percentage error plots (consistente com run_original_model)
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[1, 0].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)

    plt.tight_layout()
    plt.savefig('Plots/Inje/resultados_modelo_injeções_proximas.png', dpi=300)
    plt.show()


# ============================================================================================================================================
# Funções para comparação de modelos em rede de maior dimensão (20-30 barramentos ) - Desafio 2.4
# ============================================================================================================================================
def run_large_network_comparison():
    """Compara os diferentes modelos em uma rede de maior dimensão (20-30 barramentos )."""
    print("\n=== Comparação de Modelos em Rede de Maior Dimensão ===\n")
    
    # === 1. Gerar/carregar dados para rede maior ===
    n_buses = 25  # Podemos testar com diferentes tamanhos (20-30 barramentos )
    
    # Gerar topologia (edge-reduced) - uma mistura de estrutura radial com algumas conexões adicionais
    edge_list = [(i, i+1) for i in range(n_buses-1)]  # Base radial
    # Adicionar algumas ligações para criar malhas
    extra_edges = [(0, 5), (5, 10), (10, 15), (15, 20), 
                  (2, 7), (7, 12), (12, 17), (17, 22),
                  (3, 8), (8, 13), (13, 18), (18, 23)]
    edge_list.extend(extra_edges)
    
    # Construir matriz Y para a rede sintética
    Y_large = np.zeros((n_buses, n_buses), dtype=complex)
    np.random.seed(42)
    linha_indices_large = []
    
    for i, j in edge_list:
        # Valores típicos para linhas de transmissão
        r = 0.01 + 0.05 * np.random.random()  # Resistência [p.u.]
        x = 0.05 + 0.15 * np.random.random()  # Reatância [p.u.]
        
        y = 1 / complex(r, x)
        Y_large[i, j] = Y_large[j, i] = -y
        Y_large[i, i] += y
        Y_large[j, j] += y
        linha_indices_large.append((i, j))
    
    # Barramento de referência (slack)
    slack_bus = 0
    
    # === 2. Gerar dados de carga/injeção ===
    n_instantes_train = 500
    n_instantes_test = 100
    
    # Matrizes de potência (sem o slack)
    P_large = np.random.randn(n_instantes_train, n_buses-1) * 0.5
    Ptest_large = np.random.randn(n_instantes_test, n_buses-1) * 0.7
    
    # === 3. Preparar matrizes para cálculo de perdas ===
    Y_large_no_slack = np.delete(np.delete(Y_large, slack_bus, axis=0), slack_bus, axis=1)
    G_large = Y_large_no_slack.real
    B_large = Y_large_no_slack.imag
    
    # Matriz de incidência
    C_large = np.zeros((n_buses, len(edge_list)))
    for idx, (i, j) in enumerate(edge_list):
        C_large[i, idx] = 1
        C_large[j, idx] = -1
    
    Cl_large = np.delete(C_large, slack_bus, axis=0)
    
    # Vetor de condutâncias das linhas
    Gv_large = np.zeros(len(edge_list))
    for idx, (i, j) in enumerate(edge_list):
        Gv_large[idx] = -Y_large[i, j].real
    
    # === 4. Calcular perdas "reais" ===
    PL2_train_true = np.zeros(n_instantes_train)
    PL2_test_true = np.zeros(n_instantes_test)
    invB_large = inv(B_large)
    
    for m in range(n_instantes_train):
        teta = invB_large @ P_large[m, :].T
        grau = Cl_large.T @ teta
        PL2_m = (2 * Gv_large) @ (1 - np.cos(grau))
        PL2_train_true[m] = PL2_m * (1 + np.random.normal(0, noiseFactor))
    
    for m in range(n_instantes_test):
        teta = invB_large @ Ptest_large[m, :].T
        grau = Cl_large.T @ teta
        PL2_m = (2 * Gv_large) @ (1 - np.cos(grau))
        PL2_test_true[m] = PL2_m * (1 + np.random.normal(0, noiseFactor))
    
    # === 5. Implementar e avaliar cada modelo ===
    results = {
        'Modelo': [],
        'RMSE Treino': [],
        'MAE Treino': [],
        'RMSE Teste': [],
        'MAE Teste': [],
        'Parâmetros': []
    }
    
    # --- 5.1 Modelo Original ---
    print("Construindo modelo original (completo)...")
    
    # Construir matriz X para treino (matriz completa com todos os termos)
    feature_count = 0
    X_original = []
    
    # Termos quadráticos (squares only)
    for i in range(n_buses-1):
        X_original.append(P_large[:, i]**2)
        feature_count += 1
    
    # Termos cruzados
    for i in range(n_buses-1):
        for j in range(i+1, n_buses-1):
            X_original.append(2 * P_large[:, i] * P_large[:, j])
            feature_count += 1
    
    X_original = np.column_stack(X_original)
    
    # Fazer o mesmo para teste
    X_original_test = []
    for i in range(n_buses-1):
        X_original_test.append(Ptest_large[:, i]**2)
    
    for i in range(n_buses-1):
        for j in range(i+1, n_buses-1):
            X_original_test.append(2 * Ptest_large[:, i] * Ptest_large[:, j])
    
    X_original_test = np.column_stack(X_original_test)
    
    # Treino e predição
    beta_original = inv(X_original.T @ X_original) @ (X_original.T @ PL2_train_true)
    PL2_train_pred_original = X_original @ beta_original
    PL2_test_pred_original = X_original_test @ beta_original
    
    results['Modelo'].append('Original')
    results['RMSE Treino'].append(np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_original)))
    results['MAE Treino'].append(mean_absolute_error(PL2_train_true, PL2_train_pred_original))
    results['RMSE Teste'].append(np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_original)))
    results['MAE Teste'].append(mean_absolute_error(PL2_test_true, PL2_test_pred_original))
    results['Parâmetros'].append(len(beta_original))
    
    # --- 5.2 Modelo de Topologia (edge-reduced) ---
    print("Construindo modelo de topologia (edge-reduced)...")
    
    X_topo_train = build_X_topology(P_large, linha_indices_large, slack_bus)
    X_topo_test = build_X_topology(Ptest_large, linha_indices_large, slack_bus)
    beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
    PL2_train_pred_topo = X_topo_train @ beta_topo
    PL2_test_pred_topo = X_topo_test @ beta_topo
    
    results['Modelo'].append('Topologia (edge-reduced)')
    results['RMSE Treino'].append(np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_topo)))
    results['MAE Treino'].append(mean_absolute_error(PL2_train_true, PL2_train_pred_topo))
    results['RMSE Teste'].append(np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_topo)))
    results['MAE Teste'].append(mean_absolute_error(PL2_test_true, PL2_test_pred_topo))
    results['Parâmetros'].append(len(beta_topo))
    
    # --- 5.3 Modelo Quadrático ---
    print("Construindo modelo quadrático (squares-only)...")
    
    X_sq_train = build_X_squared(P_large)
    X_sq_test = build_X_squared(Ptest_large)
    beta_sq = inv(X_sq_train.T @ X_sq_train) @ (X_sq_train.T @ PL2_train_true)
    PL2_train_pred_sq = X_sq_train @ beta_sq
    PL2_test_pred_sq = X_sq_test @ beta_sq
    
    results['Modelo'].append('Quadrático (squares-only)')
    results['RMSE Treino'].append(np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_sq)))
    results['MAE Treino'].append(mean_absolute_error(PL2_train_true, PL2_train_pred_sq))
    results['RMSE Teste'].append(np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_sq)))
    results['MAE Teste'].append(mean_absolute_error(PL2_test_true, PL2_test_pred_sq))
    results['Parâmetros'].append(len(beta_sq))
    
    # --- 5.4 Modelo de Injeções Próximas (squares-reduced)---
    print("Construindo modelo de injeções próximas (squares-reduced)...")
    
    # Para redes maiores, precisamos agrupar os barramentos em clusters
    # Vamos criar grupos baseados na topologia (edge-reduced) da rede
    
    # Para simplificar, vamos dividir os barramentos em 5 clusters de tamanhos aproximadamente iguais
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
    
    print(f"Divisão em clusters: {clusters}")
    
    # Construir matriz X_nearby para os dados de treino
    X_nearby = []
    
    # Termos quadráticos (squares only) para cada cluster
    for cluster in clusters:
        if cluster:  # Se o cluster não está vazio
            cluster_sum = np.zeros(n_instantes_train)
            for bus in cluster:
                if bus < P_large.shape[1]:  # Verificar se o índice está dentro do limite
                    cluster_sum += P_large[:, bus]
            X_nearby.append(cluster_sum**2)
    
    # Termos cruzados entre clusters
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if clusters[i] and clusters[j]:  # Se ambos clusters não estão vazios
                sum_i = np.zeros(n_instantes_train)
                sum_j = np.zeros(n_instantes_train)
                
                for bus in clusters[i]:
                    if bus < P_large.shape[1]:
                        sum_i += P_large[:, bus]
                        
                for bus in clusters[j]:
                    if bus < P_large.shape[1]:
                        sum_j += P_large[:, bus]
                        
                X_nearby.append(2 * sum_i * sum_j)
    
    X_nearby_train = np.column_stack(X_nearby)
    
    # Fazer o mesmo para o conjunto de teste
    X_nearby_test = []
    
    # Termos quadráticos (squares only) para cada cluster
    for cluster in clusters:
        if cluster:
            cluster_sum = np.zeros(n_instantes_test)
            for bus in cluster:
                if bus < Ptest_large.shape[1]:
                    cluster_sum += Ptest_large[:, bus]
            X_nearby_test.append(cluster_sum**2)
    
    # Termos cruzados entre clusters
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
                        
                X_nearby_test.append(2 * sum_i * sum_j)
    
    X_nearby_test = np.column_stack(X_nearby_test)
    
    # Treino e predição
    beta_nearby = inv(X_nearby_train.T @ X_nearby_train) @ (X_nearby_train.T @ PL2_train_true)
    PL2_train_pred_nearby = X_nearby_train @ beta_nearby
    PL2_test_pred_nearby = X_nearby_test @ beta_nearby
    
    results['Modelo'].append('Injeções Próximas (squares-reduced)')
    results['RMSE Treino'].append(np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_nearby)))
    results['MAE Treino'].append(mean_absolute_error(PL2_train_true, PL2_train_pred_nearby))
    results['RMSE Teste'].append(np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_nearby)))
    results['MAE Teste'].append(mean_absolute_error(PL2_test_true, PL2_test_pred_nearby))
    results['Parâmetros'].append(len(beta_nearby))
    
    # Visualizar resultados para o modelo de injeções próximas (squares-reduced)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(PL2_train_true, PL2_train_pred_nearby, alpha=0.5)
    min_val = min(np.min(PL2_train_true), np.min(PL2_train_pred_nearby))
    max_val = max(np.max(PL2_train_true), np.max(PL2_train_pred_nearby))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Modelo de Injeções Próximas (squares-reduced) - Treino')
    plt.xlabel('Perdas Reais')
    plt.ylabel('Perdas Preditas')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(PL2_test_true, PL2_test_pred_nearby, alpha=0.5, color='orange')
    min_val = min(np.min(PL2_test_true), np.min(PL2_test_pred_nearby))
    max_val = max(np.max(PL2_test_true), np.max(PL2_test_pred_nearby))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Modelo de Injeções Próximas (squares-reduced) - Teste')
    plt.xlabel('Perdas Reais')
    plt.ylabel('Perdas Preditas')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Plots/Large/injecoes_proximas_large_network.png', dpi=300)
    
    # Criar tabela de comparação
    print("\nComparação de Modelos:")
    print("-" * 80)
    print(f"{'Modelo':<20} {'RMSE Treino':<15} {'MAE Treino':<15} {'RMSE Teste':<15} {'MAE Teste':<15} {'# Parâmetros'}")
    print("-" * 80)
    
    for i in range(len(results['Modelo'])):
        print(f"{results['Modelo'][i]:<20} {results['RMSE Treino'][i]:<15.6f} {results['MAE Treino'][i]:<15.6f} "
              f"{results['RMSE Teste'][i]:<15.6f} {results['MAE Teste'][i]:<15.6f} {results['Parâmetros'][i]}")
              
    # Visualizar comparação gráfica de erros
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(results['Modelo']))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    plt.bar(x - width/2, results['RMSE Teste'], width, label='RMSE Teste')
    plt.bar(x + width/2, results['MAE Teste'], width, label='MAE Teste')
    plt.xlabel('Modelo')
    plt.ylabel('Erro')
    plt.title('Comparação de Métricas de Erro entre Modelos (Dados de Teste)')
    plt.xticks(x, results['Modelo'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Visualizar comparação de número de parâmetros
    plt.subplot(2, 1, 2)
    plt.bar(x, results['Parâmetros'], color='green')
    plt.xlabel('Modelo')
    plt.ylabel('Número de Parâmetros')
    plt.title('Comparação do Número de Parâmetros por Modelo')
    plt.xticks(x, results['Modelo'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores sobre as barras
    for i, v in enumerate(results['Parâmetros']):
        plt.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('Plots/Large/comparacao_modelos_grande.png', dpi=300)
    plt.show()
    
    # Destacar o trade-off entre complexidade e precisão
    trade_off = pd.DataFrame({
        'Modelo': results['Modelo'],
        'RMSE Teste': results['RMSE Teste'],
        'Parâmetros': results['Parâmetros']
    })
    
    plt.figure(figsize=(10, 6))
    plt.scatter(trade_off['Parâmetros'], trade_off['RMSE Teste'], s=100)
    
    # Adicionar rótulos aos pontos
    for i, modelo in enumerate(trade_off['Modelo']):
        plt.annotate(modelo, 
                    (trade_off['Parâmetros'][i], trade_off['RMSE Teste'][i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Número de Parâmetros (Complexidade)')
    plt.ylabel('RMSE Teste (Erro)')
    plt.title('Trade-off entre Complexidade e Erro de Predição')
    plt.grid(True)
    plt.savefig('Plots/Large/trade_off_complexidade_erro.png', dpi=300)
    plt.show()


# ============================================================================================================================================
# Funções para o modelo baseado em sensibilidade - Desafio 2.5
# ============================================================================================================================================
def build_X_sensitivity(P_mat, Y, slack_bus, sensitivity_threshold=0.1):
    """
    Builds feature matrix X based on sensitivity analysis of each bus's contribution to losses.
    
    Parameters:
      P_mat: Matrix of power injections, with dimension (n_instances, n_buses-1)
      Y: Admittance matrix
      slack_bus: Index of slack bus
      sensitivity_threshold: Threshold to include buses in the full model
      
    Returns:
      X_sens: Feature matrix based on sensitivity analysis
    """
    n_instances, n_buses_no_slack = P_mat.shape
    
    # Step 1: Calculate loss sensitivity factors (LSF) for each bus
    # This is a simplified approach - in practice you'd need a full power flow calculation
    G = Y.real  # Conductance matrix
    
    # Calculate sensitivity factors (approximate)
    sensitivity = np.zeros(n_buses_no_slack)
    for i in range(n_buses_no_slack):
        # For simplicity, we'll use row sum of G as a proxy for sensitivity
        # In a real implementation, we'd calculate ∂PL/∂P_i for each bus
        bus_idx = i if i < slack_bus else i + 1  # Adjust for slack bus index
        sensit_i = np.sum(np.abs(G[bus_idx, :]))
        sensitivity[i] = sensit_i
    
    # Step 2: Normalize sensitivity factors
    sensitivity = sensitivity / np.max(sensitivity)
    
    # Step 3: Identify high and low sensitivity buses
    high_sens_buses = [i for i, s in enumerate(sensitivity) if s >= sensitivity_threshold]
    low_sens_buses = [i for i, s in enumerate(sensitivity) if s < sensitivity_threshold]
    
    print(f"High sensitivity buses: {len(high_sens_buses)} out of {n_buses_no_slack}")
    
    # Step 4: Build feature matrix X_sens
    X_features = []
    
    # Include squared terms for all buses
    for i in range(n_buses_no_slack):
        X_features.append(P_mat[:, i]**2)
        
    # Include cross-terms only for high sensitivity buses
    for i in high_sens_buses:
        for j in high_sens_buses:
            if i < j:  # Avoid duplicates
                X_features.append(2 * P_mat[:, i] * P_mat[:, j])
    
    # Additionally, include grouped terms for low sensitivity buses
    if low_sens_buses:
        low_sens_sum = np.zeros(n_instances)
        for i in low_sens_buses:
            low_sens_sum += P_mat[:, i]
        X_features.append(low_sens_sum**2)
        
        # Cross terms between the low-sensitivity group and each high-sensitivity bus
        for i in high_sens_buses:
            X_features.append(2 * P_mat[:, i] * low_sens_sum)
    
    return np.column_stack(X_features)

def run_sensitivity_model():
    """Executes the sensitivity-based model for power loss prediction."""
    print("\n=== Executando o Modelo Baseado em Sensibilidade ===\n")
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino e teste
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # ==== MODELO BASEADO EM SENSIBILIDADE ====
    # Construir matriz X_sensitivity para os dados de treino
    X_sens_train = build_X_sensitivity(P, Y, SlackBus-1, sensitivity_threshold=0.3)
    
    # Cálculo do beta utilizando OLS
    beta_sens = inv(X_sens_train.T @ X_sens_train) @ (X_sens_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_sens = X_sens_train @ beta_sens
    
    # Construir matriz X_sensitivity para os dados de teste
    X_sens_test = build_X_sensitivity(Ptest, Y, SlackBus-1, sensitivity_threshold=0.3)
    
    # Predição para os dados de teste
    PL2_test_pred_sens = X_sens_test @ beta_sens
    
    # ==== MODELO ORIGINAL PARA COMPARAÇÃO ====
    # Construir matriz X para os dados de treino (modelo original)
    X_orig_train = np.column_stack((
        P[:, 0]**2, 2*P[:, 0]*P[:, 1], 2*P[:, 0]*P[:, 2], 2*P[:, 0]*P[:, 3],
        P[:, 1]**2, 2*P[:, 1]*P[:, 2], 2*P[:, 1]*P[:, 3],
        P[:, 2]**2, 2*P[:, 2]*P[:, 3],
        P[:, 3]**2
    ))
    
    # Cálculo do beta utilizando OLS para o modelo original
    beta_orig = inv(X_orig_train.T @ X_orig_train) @ (X_orig_train.T @ PL2_train_true)
    
    # Predição para os dados de treino com o modelo original
    PL2_train_pred_orig = X_orig_train @ beta_orig
    
    # Construir matriz X para os dados de teste (modelo original)
    X_orig_test = np.column_stack((
        Ptest[:, 0]**2, 2*Ptest[:, 0]*Ptest[:, 1], 2*Ptest[:, 0]*Ptest[:, 2], 2*Ptest[:, 0]*Ptest[:, 3],
        Ptest[:, 1]**2, 2*Ptest[:, 1]*Ptest[:, 2], 2*Ptest[:, 1]*Ptest[:, 3],
        Ptest[:, 2]**2, 2*Ptest[:, 2]*Ptest[:, 3],
        Ptest[:, 3]**2
    ))
    
    # Predição para os dados de teste com o modelo original
    PL2_test_pred_orig = X_orig_test @ beta_orig
    
    # ==== CALCULATE METRICS FOR BOTH MODELS ====
    # Métricas para o modelo baseado em sensibilidade
    train_rmse_sens = np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_sens))
    train_mae_sens = mean_absolute_error(PL2_train_true, PL2_train_pred_sens)
    test_rmse_sens = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_sens))
    test_mae_sens = mean_absolute_error(PL2_test_true, PL2_test_pred_sens)
    
    # Métricas para o modelo original
    train_rmse_orig = np.sqrt(mean_squared_error(PL2_train_true, PL2_train_pred_orig))
    train_mae_orig = mean_absolute_error(PL2_train_true, PL2_train_pred_orig)
    test_rmse_orig = np.sqrt(mean_squared_error(PL2_test_true, PL2_test_pred_orig))
    test_mae_orig = mean_absolute_error(PL2_test_true, PL2_test_pred_orig)
    
    # ==== PRINT METRICS ====
    print("==== Comparação entre Modelo Baseado em Sensibilidade e Modelo Original ====")
    print("\nModelo Baseado em Sensibilidade:")
    print(f"  Erro de treino - RMSE: {train_rmse_sens:.6f}")
    print(f"  Erro de treino - MAE:  {train_mae_sens:.6f}")
    print(f"  Erro de teste - RMSE:  {test_rmse_sens:.6f}")
    print(f"  Erro de teste - MAE:   {test_mae_sens:.6f}")
    print(f"  Número de parâmetros:  {len(beta_sens)}")
    
    print("\nModelo Original:")
    print(f"  Erro de treino - RMSE: {train_rmse_orig:.6f}")
    print(f"  Erro de treino - MAE:  {train_mae_orig:.6f}")
    print(f"  Erro de teste - RMSE:  {test_rmse_orig:.6f}")
    print(f"  Erro de teste - MAE:   {test_mae_orig:.6f}")
    print(f"  Número de parâmetros:  {len(beta_orig)}")
    
    # ==== CALCULATE ERRORS FOR VISUALIZATION ====
    time_intervals_train = np.arange(len(PL2_train_true))
    time_intervals_test = np.arange(len(PL2_test_true))

    # Calculate percentage errors for both models
    train_errors_sens = 100 * np.abs(PL2_train_true - PL2_train_pred_sens) / np.abs(PL2_train_true)
    test_errors_sens = 100 * np.abs(PL2_test_true - PL2_test_pred_sens) / np.abs(PL2_test_true)
    
    train_errors_orig = 100 * np.abs(PL2_train_true - PL2_train_pred_orig) / np.abs(PL2_train_true)
    test_errors_orig = 100 * np.abs(PL2_test_true - PL2_test_pred_orig) / np.abs(PL2_test_true)

    # ==== VISUALIZATION OF THE SENSITIVITY MODEL ALONE ====
    # (Original plots of the sensitivity model)
    fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Top-left: Training data comparison
    ax1[0, 0].step(time_intervals_train, PL2_train_true, where='post', label='Perdas Reais')
    ax1[0, 0].step(time_intervals_train, PL2_train_pred_sens, where='post', label='Perdas Preditas')
    ax1[0, 0].set_title('Comparação de Perdas - Treino')
    ax1[0, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax1[0, 0].set_ylabel('Perdas de Potência [p.u.]')
    ax1[0, 0].legend(loc='upper right')
    ax1[0, 0].grid(True)

    # Top-right: Test data comparison
    ax1[0, 1].step(time_intervals_test, PL2_test_true, where='post', label='Perdas Reais')
    ax1[0, 1].step(time_intervals_test, PL2_test_pred_sens, where='post', label='Perdas Preditas')
    ax1[0, 1].set_title('Comparação de Perdas - Teste')
    ax1[0, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax1[0, 1].set_ylabel('Perdas de Potência [p.u.]')
    ax1[0, 1].legend(loc='upper right')
    ax1[0, 1].grid(True)

    # Bottom-left: Training error
    ax1[1, 0].step(time_intervals_train, train_errors_sens, where='post', label='Erro Percentual', color='blue')
    ax1[1, 0].set_title('Erro Percentual - Treino')
    ax1[1, 0].set_xlabel('Intervalo Temporal [15 min]')
    ax1[1, 0].set_ylabel('Erro Percentual (%)')
    ax1[1, 0].legend(loc='upper right')
    ax1[1, 0].grid(True)

    # Bottom-right: Test error
    ax1[1, 1].step(time_intervals_test, test_errors_sens, where='post', label='Erro Percentual', color='blue')
    ax1[1, 1].set_title('Erro Percentual - Teste')
    ax1[1, 1].set_xlabel('Intervalo Temporal [15 min]')
    ax1[1, 1].set_ylabel('Erro Percentual (%)')
    ax1[1, 1].legend(loc='upper right')
    ax1[1, 1].grid(True)

    # Update y-axis limits for percentage error plots
    y_max_error_pct = max(np.max(train_errors_sens), np.max(test_errors_sens))
    y_margin_error_pct = 0.1 * y_max_error_pct

    ax1[1, 0].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax1[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)

    plt.tight_layout()
    plt.savefig('Plots/Sensitivity/resultados_modelo_sensibilidade.png', dpi=300)
    
    # ==== NEW VISUALIZATION COMPARING THE TWO MODELS ====
    # Create additional plots for comparison
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))
    
    # Subplot 1: Bar chart comparing RMSE and MAE
    model_names = ['Sensibilidade', 'Original']
    x = np.arange(len(model_names))
    width = 0.2
    
    # Training metrics
    ax2[0].bar(x - width*1.5, [train_rmse_sens, train_rmse_orig], width, label='RMSE Treino', color='skyblue')
    ax2[0].bar(x - width/2, [train_mae_sens, train_mae_orig], width, label='MAE Treino', color='lightgreen')
    
    # Test metrics
    ax2[0].bar(x + width/2, [test_rmse_sens, test_rmse_orig], width, label='RMSE Teste', color='orange')
    ax2[0].bar(x + width*1.5, [test_mae_sens, test_mae_orig], width, label='MAE Teste', color='salmon')
    
    ax2[0].set_ylabel('Valor do Erro')
    ax2[0].set_title('Comparação de Métricas entre Modelos')
    ax2[0].set_xticks(x)
    ax2[0].set_xticklabels(model_names)
    ax2[0].legend()
    ax2[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores nas barras
    bar_values = [train_rmse_sens, train_rmse_orig, train_mae_sens, train_mae_orig, 
                 test_rmse_sens, test_rmse_orig, test_mae_sens, test_mae_orig]
    bar_positions = [x[0]-width*1.5, x[1]-width*1.5, x[0]-width/2, x[1]-width/2, 
                    x[0]+width/2, x[1]+width/2, x[0]+width*1.5, x[1]+width*1.5]
    
    for i, v in enumerate(bar_values):
        ax2[0].text(bar_positions[i], v + 0.0005, f'{v:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Subplot 2: Side-by-side comparison of test error percentages over time
    ax2[1].step(time_intervals_test, test_errors_sens, where='post', label='Modelo Sensibilidade', color='blue')
    ax2[1].step(time_intervals_test, test_errors_orig, where='post', label='Modelo Original', color='red')
    ax2[1].set_title('Comparação de Erro Percentual no Teste')
    ax2[1].set_xlabel('Intervalo Temporal [15 min]')
    ax2[1].set_ylabel('Erro Percentual (%)')
    ax2[1].legend(loc='upper right')
    ax2[1].grid(True)
    
    # Ajustar limite do eixo y para comportar ambos os erros
    y_max_error_compare = max(np.max(test_errors_sens), np.max(test_errors_orig))
    y_margin_error_compare = 0.1 * y_max_error_compare
    ax2[1].set_ylim(0, y_max_error_compare + y_margin_error_compare)
    
    # Information table as text
    params_reduction = (len(beta_orig) - len(beta_sens)) / len(beta_orig) * 100
    error_improvement = (test_rmse_orig - test_rmse_sens) / test_rmse_orig * 100
    
    info_text = (
        f"Comparação de Modelos:\n"
        f"- Modelo Sensibilidade: {len(beta_sens)} parâmetros, RMSE Teste = {test_rmse_sens:.6f}\n"
        f"- Modelo Original: {len(beta_orig)} parâmetros, RMSE Teste = {test_rmse_orig:.6f}\n"
        f"- Redução de parâmetros: {params_reduction:.1f}%\n"
        f"- {'Melhoria' if error_improvement > 0 else 'Aumento'} do erro: {abs(error_improvement):.1f}%"
    )
    
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig('Plots/Sensitivity/comparacao_sensibilidade_vs_original.png', dpi=300)
    plt.show()

# ============================================================================================================================================
# Funções para o menu principal e execução de comparações
# ============================================================================================================================================
def show_menu():
    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
    print("=" * 80)
    print("                     LABORATÓRIO 2 - ANÁLISE DE PERDAS DE POTÊNCIA                     ")
    print("=" * 80)
    print("\nEscolha uma opção:")
    print("  1. Executar o modelo original (baseado em ângulos)")
    print("  2. Executar o modelo de baixa tensão (LV) (baseado em tensões)")
    print("  3. Executar o modelo baseado na topologia (edge-reduced) da rede")
    print("  4. Executar o modelo simplificado (apenas termos quadráticos (squares only))")
    print("  5. Executar o modelo simplificado (com soma das injeções eletricamente próximas)")
    print("  6. Comparar todos os modelos\n")
    print("  7. Comparar modelos em rede de maior dimensão (20-30 barramentos )\n")
    print("  8. Comparar desempenho dos modelos em diferentes tamanhos de rede\n")
    print("  9. Executar o modelo baseado em sensibilidade\n")
    print("  \nSair do programa: '0'")
    
    try:
        opcao = int(input("\nDigite a opção desejada: "))
        return opcao
    except ValueError:
        return -1  # Retorna -1 se o usuário inserir um valor não numérico
    
def run_comparison():
    """Compara todos os modelos implementados em termos de métricas de erro."""
    print("\n=== Comparando Todos os Modelos ===\n")
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino e teste
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # Resultados para armazenar as métricas
    results = {
        'Modelo': [],
        'RMSE Treino': [],
        'MAE Treino': [],
        'RMSE Teste': [],
        'MAE Teste': [],
        'Parâmetros': []
    }
    
    # 1. Modelo Original
    X_train = np.column_stack((
        P[:, 0]**2, 2*P[:, 0]*P[:, 1], 2*P[:, 0]*P[:, 2], 2*P[:, 0]*P[:, 3],
        P[:, 1]**2, 2*P[:, 1]*P[:, 2], 2*P[:, 1]*P[:, 3],
        P[:, 2]**2, 2*P[:, 2]*P[:, 3],
        P[:, 3]**2
    ))
    X_test = np.column_stack((
        Ptest[:, 0]**2, 2*Ptest[:, 0]*Ptest[:, 1], 2*Ptest[:, 0]*Ptest[:, 2], 2*Ptest[:, 0]*Ptest[:, 3],
        Ptest[:, 1]**2, 2*Ptest[:, 1]*Ptest[:, 2], 2*Ptest[:, 1]*Ptest[:, 3],
        Ptest[:, 2]**2, 2*Ptest[:, 2]*Ptest[:, 3],
        Ptest[:, 3]**2
    ))
    beta_original = inv(X_train.T @ X_train) @ (X_train.T @ PL2_train_true)
    pred_train_original = X_train @ beta_original
    pred_test_original = X_test @ beta_original
    
    results['Modelo'].append('Original')
    results['RMSE Treino'].append(np.sqrt(mean_squared_error(PL2_train_true, pred_train_original)))
    results['MAE Treino'].append(mean_absolute_error(PL2_train_true, pred_train_original))
    results['RMSE Teste'].append(np.sqrt(mean_squared_error(PL2_test_true, pred_test_original)))
    results['MAE Teste'].append(mean_absolute_error(PL2_test_true, pred_test_original))
    results['Parâmetros'].append(len(beta_original))
    
    # 2. Modelo de Topologia (edge-reduced)
    X_topo_train = build_X_topology(P, linha_indices, SlackBus)
    X_topo_test = build_X_topology(Ptest, linha_indices, SlackBus)
    beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
    pred_train_topo = X_topo_train @ beta_topo
    pred_test_topo = X_topo_test @ beta_topo
    
    results['Modelo'].append('Topologia (edge-reduced)')
    results['RMSE Treino'].append(np.sqrt(mean_squared_error(PL2_train_true, pred_train_topo)))
    results['MAE Treino'].append(mean_absolute_error(PL2_train_true, pred_train_topo))
    results['RMSE Teste'].append(np.sqrt(mean_squared_error(PL2_test_true, pred_test_topo)))
    results['MAE Teste'].append(mean_absolute_error(PL2_test_true, pred_test_topo))
    results['Parâmetros'].append(len(beta_topo))
    
    # 3. Modelo Quadrático (squares-only)
    X_sq_train = build_X_squared(P)
    X_sq_test = build_X_squared(Ptest)
    beta_sq = inv(X_sq_train.T @ X_sq_train) @ (X_sq_train.T @ PL2_train_true)
    pred_train_sq = X_sq_train @ beta_sq
    pred_test_sq = X_sq_test @ beta_sq
    
    results['Modelo'].append('Quadrático (squares-only)')
    results['RMSE Treino'].append(np.sqrt(mean_squared_error(PL2_train_true, pred_train_sq)))
    results['MAE Treino'].append(mean_absolute_error(PL2_train_true, pred_train_sq))
    results['RMSE Teste'].append(np.sqrt(mean_squared_error(PL2_test_true, pred_test_sq)))
    results['MAE Teste'].append(mean_absolute_error(PL2_test_true, pred_test_sq))
    results['Parâmetros'].append(len(beta_sq))
    
    # 4. Modelo de Injeções Próximas (squares-reduced)
    X_nearby_train = build_X_nearby(P, linha_indices, SlackBus)
    X_nearby_test = build_X_nearby(Ptest, linha_indices, SlackBus)
    beta_nearby = inv(X_nearby_train.T @ X_nearby_train) @ (X_nearby_train.T @ PL2_train_true)
    pred_train_nearby = X_nearby_train @ beta_nearby
    pred_test_nearby = X_nearby_test @ beta_nearby
    
    results['Modelo'].append('Injeções Próximas (squares-reduced)')
    results['RMSE Treino'].append(np.sqrt(mean_squared_error(PL2_train_true, pred_train_nearby)))
    results['MAE Treino'].append(mean_absolute_error(PL2_train_true, pred_train_nearby))
    results['RMSE Teste'].append(np.sqrt(mean_squared_error(PL2_test_true, pred_test_nearby)))
    results['MAE Teste'].append(mean_absolute_error(PL2_test_true, pred_test_nearby))
    results['Parâmetros'].append(len(beta_nearby))
    
    # Criar tabela de comparação
    print("\nComparação de Modelos:")
    print("-" * 80)
    print(f"{'Modelo':<20} {'RMSE Treino':<15} {'MAE Treino':<15} {'RMSE Teste':<15} {'MAE Teste':<15} {'# Parâmetros'}")
    print("-" * 80)
    
    for i in range(len(results['Modelo'])):
        print(f"{results['Modelo'][i]:<20} {results['RMSE Treino'][i]:<15.6f} {results['MAE Treino'][i]:<15.6f} "
              f"{results['RMSE Teste'][i]:<15.6f} {results['MAE Teste'][i]:<15.6f} {results['Parâmetros'][i]}")
              
    # Visualizar comparação gráfica
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(results['Modelo']))
    width = 0.35
    
    plt.bar(x - width/2, results['RMSE Teste'], width, label='RMSE Teste')
    plt.bar(x + width/2, results['MAE Teste'], width, label='MAE Teste')
    
    plt.xlabel('Modelo')
    plt.ylabel('Erro')
    plt.title('Comparação de Métricas de Erro entre Modelos (Dados de Teste)')
    plt.xticks(x, results['Modelo'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('Plots/Comp/comparacao_modelos.png', dpi=300)
    plt.show()
    
def compare_network_sizes():
    """Compares model performance between original network size and larger network size."""
    print("\n=== Comparando Desempenho dos Modelos em Diferentes Tamanhos de Rede ===\n")
    
    # Dictionary to store results from both network sizes
    all_results = {
        'small': {
            'Modelo': [],
            'RMSE': [],
            'MAE': [],
            'Parâmetros': []
        },
        'large': {
            'Modelo': [],
            'RMSE': [],
            'MAE': [],
            'Parâmetros': []
        }
    }
    
    # ===== Get results for small network (original size) =====
    print("Executando modelos na rede original (5 barramentos )...")
    
    # Carregar dados da rede original
    SlackBus, Net_Info, P, Ptest = load_data()
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de teste
    PL2_test_true = compute_PL2(Ptest, B, Cl, Gv, noiseFactor)
    
    # 1. Modelo Original
    X_test = np.column_stack((
        Ptest[:, 0]**2, 2*Ptest[:, 0]*Ptest[:, 1], 2*Ptest[:, 0]*Ptest[:, 2], 2*Ptest[:, 0]*Ptest[:, 3],
        Ptest[:, 1]**2, 2*Ptest[:, 1]*Ptest[:, 2], 2*Ptest[:, 1]*Ptest[:, 3],
        Ptest[:, 2]**2, 2*Ptest[:, 2]*Ptest[:, 3],
        Ptest[:, 3]**2
    ))
    
    # Treinamento rápido para obter beta
    X_train = np.column_stack((
        P[:, 0]**2, 2*P[:, 0]*P[:, 1], 2*P[:, 0]*P[:, 2], 2*P[:, 0]*P[:, 3],
        P[:, 1]**2, 2*P[:, 1]*P[:, 2], 2*P[:, 1]*P[:, 3],
        P[:, 2]**2, 2*P[:, 2]*P[:, 3],
        P[:, 3]**2
    ))
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    beta_original = inv(X_train.T @ X_train) @ (X_train.T @ PL2_train_true)
    
    # Predição para teste
    pred_test_original = X_test @ beta_original
    
    all_results['small']['Modelo'].append('Original')
    all_results['small']['RMSE'].append(np.sqrt(mean_squared_error(PL2_test_true, pred_test_original)))
    all_results['small']['MAE'].append(mean_absolute_error(PL2_test_true, pred_test_original))
    all_results['small']['Parâmetros'].append(len(beta_original))
    
    # 2. Modelo de Topologia (edge-reduced)
    X_topo_train = build_X_topology(P, linha_indices, SlackBus)
    X_topo_test = build_X_topology(Ptest, linha_indices, SlackBus)
    beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
    pred_test_topo = X_topo_test @ beta_topo
    
    all_results['small']['Modelo'].append('Topologia (edge-reduced)')
    all_results['small']['RMSE'].append(np.sqrt(mean_squared_error(PL2_test_true, pred_test_topo)))
    all_results['small']['MAE'].append(mean_absolute_error(PL2_test_true, pred_test_topo))
    all_results['small']['Parâmetros'].append(len(beta_topo))
    
    # 3. Modelo Quadrático (squares-only)
    X_sq_train = build_X_squared(P)
    X_sq_test = build_X_squared(Ptest)
    beta_sq = inv(X_sq_train.T @ X_sq_train) @ (X_sq_train.T @ PL2_train_true)
    pred_test_sq = X_sq_test @ beta_sq
    
    all_results['small']['Modelo'].append('Quadrático (squares-only)')
    all_results['small']['RMSE'].append(np.sqrt(mean_squared_error(PL2_test_true, pred_test_sq)))
    all_results['small']['MAE'].append(mean_absolute_error(PL2_test_true, pred_test_sq))
    all_results['small']['Parâmetros'].append(len(beta_sq))
    
    # 4. Modelo de Injeções Próximas (squares-reduced)
    X_nearby_train = build_X_nearby(P, linha_indices, SlackBus)
    X_nearby_test = build_X_nearby(Ptest, linha_indices, SlackBus)
    beta_nearby = inv(X_nearby_train.T @ X_nearby_train) @ (X_nearby_train.T @ PL2_train_true)
    pred_test_nearby = X_nearby_test @ beta_nearby
    
    all_results['small']['Modelo'].append('Injeções Próximas')
    all_results['small']['RMSE'].append(np.sqrt(mean_squared_error(PL2_test_true, pred_test_nearby)))
    all_results['small']['MAE'].append(mean_absolute_error(PL2_test_true, pred_test_nearby))
    all_results['small']['Parâmetros'].append(len(beta_nearby))
    
    # ===== Get results for large network =====
    print("Executando modelos na rede maior (25 barramentos )...")
    
    # Configurar rede maior (código adaptado de run_large_network_comparison())
    n_buses = 25
    
    # Gerar topologia (edge-reduced)- uma mistura de estrutura radial com algumas conexões adicionais
    edge_list = [(i, i+1) for i in range(n_buses-1)]  # Base radial
    extra_edges = [(0, 5), (5, 10), (10, 15), (15, 20), 
                  (2, 7), (7, 12), (12, 17), (17, 22),
                  (3, 8), (8, 13), (13, 18), (18, 23)]
    edge_list.extend(extra_edges)
    
    # Construir matriz Y para a rede sintética
    Y_large = np.zeros((n_buses, n_buses), dtype=complex)
    np.random.seed(42)
    linha_indices_large = []
    
    for i, j in edge_list:
        # Valores típicos para linhas de transmissão
        r = 0.01 + 0.05 * np.random.random()
        x = 0.05 + 0.15 * np.random.random()
        
        y = 1 / complex(r, x)
        Y_large[i, j] = Y_large[j, i] = -y
        Y_large[i, i] += y
        Y_large[j, j] += y
        linha_indices_large.append((i, j))
    
    # Barramento de referência (slack)
    slack_bus = 0
    
    # Gerar dados de carga/injeção
    n_instantes_train = 500
    n_instantes_test = 100
    
    # Matrizes de potência (sem o slack)
    P_large = np.random.randn(n_instantes_train, n_buses-1) * 0.5
    Ptest_large = np.random.randn(n_instantes_test, n_buses-1) * 0.7
    
    # Preparar matrizes para cálculo de perdas
    Y_large_no_slack = np.delete(np.delete(Y_large, slack_bus, axis=0), slack_bus, axis=1)
    G_large = Y_large_no_slack.real
    B_large = Y_large_no_slack.imag
    
    # Matriz de incidência
    C_large = np.zeros((n_buses, len(edge_list)))
    for idx, (i, j) in enumerate(edge_list):
        C_large[i, idx] = 1
        C_large[j, idx] = -1
    
    Cl_large = np.delete(C_large, slack_bus, axis=0)
    
    # Vetor de condutâncias das linhas
    Gv_large = np.zeros(len(edge_list))
    for idx, (i, j) in enumerate(edge_list):
        Gv_large[idx] = -Y_large[i, j].real
    
    # Calcular perdas "reais" para treino e teste
    PL2_train_true_large = np.zeros(n_instantes_train)
    PL2_test_true_large = np.zeros(n_instantes_test)
    invB_large = inv(B_large)
    
    for m in range(n_instantes_train):
        teta = invB_large @ P_large[m, :].T
        grau = Cl_large.T @ teta
        PL2_m = (2 * Gv_large) @ (1 - np.cos(grau))
        PL2_train_true_large[m] = PL2_m * (1 + np.random.normal(0, noiseFactor))
    
    for m in range(n_instantes_test):
        teta = invB_large @ Ptest_large[m, :].T
        grau = Cl_large.T @ teta
        PL2_m = (2 * Gv_large) @ (1 - np.cos(grau))
        PL2_test_true_large[m] = PL2_m * (1 + np.random.normal(0, noiseFactor))
    
    # 1. Modelo Original (Large)
    # Construir matriz X para treino (matriz completa com todos os termos)
    X_original_large = []
    
    # Termos quadráticos (squares only)
    for i in range(n_buses-1):
        X_original_large.append(P_large[:, i]**2)
    
    # Termos cruzados
    for i in range(n_buses-1):
        for j in range(i+1, n_buses-1):
            X_original_large.append(2 * P_large[:, i] * P_large[:, j])
    
    X_original_large = np.column_stack(X_original_large)
    
    # Fazer o mesmo para teste
    X_original_large_test = []
    for i in range(n_buses-1):
        X_original_large_test.append(Ptest_large[:, i]**2)
    
    for i in range(n_buses-1):
        for j in range(i+1, n_buses-1):
            X_original_large_test.append(2 * Ptest_large[:, i] * Ptest_large[:, j])
    
    X_original_large_test = np.column_stack(X_original_large_test)
    
    # Treino e predição
    beta_original_large = inv(X_original_large.T @ X_original_large) @ (X_original_large.T @ PL2_train_true_large)
    PL2_test_pred_original_large = X_original_large_test @ beta_original_large
    
    all_results['large']['Modelo'].append('Original')
    all_results['large']['RMSE'].append(np.sqrt(mean_squared_error(PL2_test_true_large, PL2_test_pred_original_large)))
    all_results['large']['MAE'].append(mean_absolute_error(PL2_test_true_large, PL2_test_pred_original_large))
    all_results['large']['Parâmetros'].append(len(beta_original_large))
    
    # 2. Modelo de Topologia (edge-reduced) (Large)
    X_topo_large_train = build_X_topology(P_large, linha_indices_large, slack_bus+1)
    X_topo_large_test = build_X_topology(Ptest_large, linha_indices_large, slack_bus+1)
    beta_topo_large = inv(X_topo_large_train.T @ X_topo_large_train) @ (X_topo_large_train.T @ PL2_train_true_large)
    PL2_test_pred_topo_large = X_topo_large_test @ beta_topo_large
    
    all_results['large']['Modelo'].append('Topologia (edge-reduced)')
    all_results['large']['RMSE'].append(np.sqrt(mean_squared_error(PL2_test_true_large, PL2_test_pred_topo_large)))
    all_results['large']['MAE'].append(mean_absolute_error(PL2_test_true_large, PL2_test_pred_topo_large))
    all_results['large']['Parâmetros'].append(len(beta_topo_large))
    
    # 3. Modelo Quadrático (squares-only) (Large)
    X_sq_large_train = build_X_squared(P_large)
    X_sq_large_test = build_X_squared(Ptest_large)
    beta_sq_large = inv(X_sq_large_train.T @ X_sq_large_train) @ (X_sq_large_train.T @ PL2_train_true_large)
    PL2_test_pred_sq_large = X_sq_large_test @ beta_sq_large
    
    all_results['large']['Modelo'].append('Quadrático (squares-only)')
    all_results['large']['RMSE'].append(np.sqrt(mean_squared_error(PL2_test_true_large, PL2_test_pred_sq_large)))
    all_results['large']['MAE'].append(mean_absolute_error(PL2_test_true_large, PL2_test_pred_sq_large))
    all_results['large']['Parâmetros'].append(len(beta_sq_large))
    
    # 4. Modelo de Injeções Próximas (squares-reduced) (Large)
    # Para simplificar, vamos dividir os barramentosem 5 clusters de tamanhos aproximadamente iguais
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
    
    # Construir matriz X_nearby para os dados de treino
    X_nearby_large = []
    
    # Termos quadráticos  (squares only) para cada cluster
    for cluster in clusters:
        if cluster:  # Se o cluster não está vazio
            cluster_sum = np.zeros(n_instantes_train)
            for bus in cluster:
                if bus < P_large.shape[1]:  # Verificar se o índice está dentro do limite
                    cluster_sum += P_large[:, bus]
            X_nearby_large.append(cluster_sum**2)
    
    # Termos cruzados entre clusters
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if clusters[i] and clusters[j]:  # Se ambos clusters não estão vazios
                sum_i = np.zeros(n_instantes_train)
                sum_j = np.zeros(n_instantes_train)
                
                for bus in clusters[i]:
                    if bus < P_large.shape[1]:
                        sum_i += P_large[:, bus]
                        
                for bus in clusters[j]:
                    if bus < P_large.shape[1]:
                        sum_j += P_large[:, bus]
                        
                X_nearby_large.append(2 * sum_i * sum_j)
    
    X_nearby_large_train = np.column_stack(X_nearby_large)
    
    # Fazer o mesmo para o conjunto de teste
    X_nearby_large_test = []
    
    # Termos quadráticos (squares only) para cada cluster
    for cluster in clusters:
        if cluster:
            cluster_sum = np.zeros(n_instantes_test)
            for bus in cluster:
                if bus < Ptest_large.shape[1]:
                    cluster_sum += Ptest_large[:, bus]
            X_nearby_large_test.append(cluster_sum**2)
    
    # Termos cruzados entre clusters
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
    
    X_nearby_large_test = np.column_stack(X_nearby_large_test)
    
    # Treino e predição
    beta_nearby_large = inv(X_nearby_large_train.T @ X_nearby_large_train) @ (X_nearby_large_train.T @ PL2_train_true_large)
    PL2_test_pred_nearby_large = X_nearby_large_test @ beta_nearby_large
    
    all_results['large']['Modelo'].append('Injeções Próximas (squares-reduced)')
    all_results['large']['RMSE'].append(np.sqrt(mean_squared_error(PL2_test_true_large, PL2_test_pred_nearby_large)))
    all_results['large']['MAE'].append(mean_absolute_error(PL2_test_true_large, PL2_test_pred_nearby_large))
    all_results['large']['Parâmetros'].append(len(beta_nearby_large))
    
    # ===== Create comparison plots =====
    
    # Plot 1: RMSE comparison between small and large networks
    plt.figure(figsize=(12, 10))
    
    # Plot RMSE comparison
    plt.subplot(2, 1, 1)
    x = np.arange(len(all_results['small']['Modelo']))
    width = 0.35
    
    plt.bar(x - width/2, all_results['small']['RMSE'], width, label='Rede Original (5 barramentos)')
    plt.bar(x + width/2, all_results['large']['RMSE'], width, label='Rede Maior (25 barramentos)')
    
    plt.xlabel('Modelo')
    plt.ylabel('RMSE no Conjunto de Teste')
    plt.title('Comparação de RMSE entre Redes de Diferentes Tamanhos')
    plt.xticks(x, all_results['small']['Modelo'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar labels para valores
    for i, v in enumerate(all_results['small']['RMSE']):
        plt.text(i - width/2, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(all_results['large']['RMSE']):
        plt.text(i + width/2, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=9)
    
    # Plot parameters comparison (log scale for better visualization)
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, all_results['small']['Parâmetros'], width, label='Rede Original (5 barramentos)')
    plt.bar(x + width/2, all_results['large']['Parâmetros'], width, label='Rede Maior (25 barramentos)')
    
    plt.xlabel('Modelo')
    plt.ylabel('Número de Parâmetros (escala log)')
    plt.title('Comparação do Número de Parâmetros entre Redes de Diferentes Tamanhos')
    plt.xticks(x, all_results['small']['Modelo'])
    plt.yscale('log')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar labels para valores
    for i, v in enumerate(all_results['small']['Parâmetros']):
        plt.text(i - width/2, v * 1.1, str(v), ha='center', va='bottom')
    
    for i, v in enumerate(all_results['large']['Parâmetros']):
        plt.text(i + width/2, v * 1.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('Plots/Large/comparacao_redes_diferentes_tamanhos.png', dpi=300)
    
    # Plot 2: Efficiency plot (RMSE vs Parameter count) for each network size
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for i, model_name in enumerate(all_results['small']['Modelo']):
        plt.scatter(all_results['small']['Parâmetros'][i], all_results['small']['RMSE'][i], s=100, label=model_name)
    
    plt.xlabel('Número de Parâmetros')
    plt.ylabel('RMSE Teste')
    plt.title('Trade-off em Rede Original (5 barramentos)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i, model_name in enumerate(all_results['large']['Modelo']):
        plt.scatter(all_results['large']['Parâmetros'][i], all_results['large']['RMSE'][i], s=100, label=model_name)
    
    plt.xlabel('Número de Parâmetros')
    plt.ylabel('RMSE Teste')
    plt.title('Trade-off em Rede Maior (25 barramentos)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Plots/Large/trade_off_redes_diferentes_tamanhos.png', dpi=300)
    plt.show()
    
    # Print tabela de resumo
    print("\nComparação entre Redes de Diferentes Tamanhos:")
    print("-" * 100)
    print(f"{'Modelo':<20} {'RMSE (5 barramentos)':<15} {'RMSE (25 barramentos)':<15} {'Parâmetros (5 barramentos)':<20} {'Parâmetros (25 barramentos)':<20}")
    print("-" * 100)
    
    for i in range(len(all_results['small']['Modelo'])):
        print(f"{all_results['small']['Modelo'][i]:<20} "
              f"{all_results['small']['RMSE'][i]:<15.6f} "
              f"{all_results['large']['RMSE'][i]:<15.6f} "
              f"{all_results['small']['Parâmetros'][i]:<20} "
              f"{all_results['large']['Parâmetros'][i]:<20}")


# Atualizar o bloco de execução principal
if __name__ == "__main__":
    while True:
        opcao = show_menu()
        
        if opcao == 0:
            print("\nPrograma finalizado. Obrigado!")
            break
        elif opcao == 1:
            run_original_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 2:
            run_LV_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 3:
            run_topology_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 4:
            run_squared_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 5:
            run_nearby_model()
            input("\nPressione Enter para continuar...")
        elif opcao == 6:
            run_comparison()
            input("\nPressione Enter para continuar...")
        elif opcao == 7:
            run_large_network_comparison()
            input("\nPressione Enter para continuar...")
        elif opcao == 8:
            compare_network_sizes()
            input("\nPressione Enter para continuar...")
        elif opcao == 9:
            run_sensitivity_model()
            input("\nPressione Enter para continuar...")
        else:
            print("\nOpção inválida! Por favor, escolha uma opção válida.")
            input("\nPressione Enter para continuar...")