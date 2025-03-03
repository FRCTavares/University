###############################################################################################
# Laboratorio 2 - Power Losses                                                                #
#                                                                                             #
# Grupo X                                                                                     #
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
    linha_indices = []  # Armazenar os índices dos buses conectados por cada linha
    nLine_Aux = 0
    for i in range(nBus):
        for j in range(i+1, nBus):
            if abs(Y[i, j]) != 0:
                C[i, nLine_Aux] = 1
                C[j, nLine_Aux] = -1
                linha_indices.append((i, j))  # Guarda os buses conectados por esta linha
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
    ax[0, 0].set_xlabel('Carimbo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência [p.u.]')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Training error
    ax[1, 0].step(time_intervals_train, train_errors , where='post', label='Erro Percentual', color='blue')
    ax[1, 0].set_title('Erro Percentual - Treino')
    ax[1, 0].set_xlabel('Carimbo Temporal [15 min]')
    ax[1, 0].set_ylabel('Erro Percentual (%)')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].grid(True)
    
    # Bottom-left: Test data comparison
    ax[0, 1].step(time_intervals_test, PL2_test_true, where='post', label='Perdas Reais')
    ax[0, 1].step(time_intervals_test, PL2_test_pred, where='post', label='Perdas Preditas')
    ax[0, 1].set_title('Comparação de Perdas - Teste')
    ax[0, 1].set_xlabel('Carimbo Temporal [15 min]')
    ax[0, 1].set_ylabel('Perdas de Potência [p.u.]')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].grid(True)
    
    # Bottom-right: Test error
    ax[1, 1].step(time_intervals_test, test_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 1].set_title('Erro Percentual - Teste')
    ax[1, 1].set_xlabel('Carimbo Temporal [15 min]')
    ax[1, 1].set_ylabel('Erro Percentual (%)')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].grid(True)

    # Update y-axis limits for error plots
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[1, 0].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    
    plt.tight_layout()
    plt.savefig('resultados_modelo_original.png', dpi=300)
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
        V_magnitudes = np.ones(nBus)  # Inicialmente todos os buses têm tensão nominal
        
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
    entre os buses conectados por linhas.
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
    ax[0, 0].set_xlabel('Carimbo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Training error
    ax[0, 1].step(time_intervals_train, train_errors, where='post', label='Erro Percentual', color='blue')
    ax[0, 1].set_title('Erro Percentual - Treino')
    ax[0, 1].set_xlabel('Carimbo Temporal [15 min]')
    ax[0, 1].set_ylabel('Erro Percentual (%)')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].grid(True)
    
    # Bottom-left: Test data comparison
    ax[1, 0].step(time_intervals_test, PL2_test_true_LV, where='post', label='Perdas Reais')
    ax[1, 0].step(time_intervals_test, PL2_test_pred_LV, where='post', label='Perdas Preditas')
    ax[1, 0].set_title('Comparação de Perdas - Teste')
    ax[1, 0].set_xlabel('Carimbo Temporal [15 min]')
    ax[1, 0].set_ylabel('Perdas de Potência')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].grid(True)
    
    # Bottom-right: Test error
    ax[1, 1].step(time_intervals_test, test_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 1].set_title('Erro Percentual - Teste')
    ax[1, 1].set_xlabel('Carimbo Temporal [15 min]')
    ax[1, 1].set_ylabel('Erro Percentual (%)')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].grid(True)
    
    # Update y-axis limits for percentage error plots
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[0, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    
    plt.tight_layout()
    plt.savefig('resultados_modelo_LV.png', dpi=300)
    plt.show()
    
    # Comparar os coeficientes beta com as condutâncias das linhas
    print("\nCOMPARAÇÃO DOS COEFICIENTES E CONDUTÂNCIAS")
    print("Coeficientes beta (modelo LV):")
    for i, (bus1, bus2) in enumerate(linha_indices):
        g_ij = -np.real(Y[bus1, bus2])
        print(f"Linha {i+1} (Bus {bus1+1}-{bus2+1}): Beta = {beta_LV[i]:.6f}, G_ij = {g_ij:.6f}")

# ============================================================================================================================================
# Funções para o modelo baseado na topologia da rede Desafio 2.1
# ============================================================================================================================================

def build_X_topology(P_mat, linha_indices, SlackBus):
    """
    Constrói a matriz X' considerando apenas termos P_i^2 para todos os buses
    e termos cruzados 2*P_i*P_j apenas quando existe uma linha ligando os buses i e j.
    
    Parâmetros:
      P_mat: Matriz de potência, com dimensão (n_instantes, nBus_sem_slack)
      linha_indices: Lista de tuplas (i,j) com os índices dos buses ligados por linhas
      SlackBus: Índice do bus slack (1-based)
      
    Retorna:
      X_topo: Matriz de features considerando a topologia da rede
    """
    n_instantes = P_mat.shape[0]
    nBus_sem_slack = P_mat.shape[1]
    
    # Lista para armazenar todas as features (quadráticas e cruzadas)
    features = []
    
    # 1. Adicionar termos quadráticos P_i^2 para cada bus
    for i in range(nBus_sem_slack):
        features.append(('quad', i))
    
    # 2. Adicionar termos cruzados 2*P_i*P_j apenas para linhas existentes
    slack_0based = SlackBus - 1  # Converter para 0-based
    
    for i, j in linha_indices:
        # Ignorar se algum dos buses for o slack
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
    """Executa o modelo que considera a topologia da rede para o cálculo de perdas."""
    print("\n=== Executando o Modelo com Topologia da Rede ===\n")
    
    # Carregar dados
    SlackBus, Net_Info, P, Ptest = load_data()
    
    # Construir matrizes da rede
    nBus, nLines, Y, Yl, G, B, C, Cl, Gv, Gd, linha_indices = build_network_matrices(SlackBus, Net_Info)
    
    # Cálculo das perdas físicas para os dados de treino
    PL2_train_true = compute_PL2(P, B, Cl, Gv, noiseFactor)
    
    # Construir matriz X' baseada na topologia para os dados de treino
    X_topo_train = build_X_topology(P, linha_indices, SlackBus)
    
    # Cálculo do beta utilizando OLS
    beta_topo = inv(X_topo_train.T @ X_topo_train) @ (X_topo_train.T @ PL2_train_true)
    
    # Predição para os dados de treino
    PL2_train_pred_topo = X_topo_train @ beta_topo
    
    # Construir matriz X' baseada na topologia para os dados de teste
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
    
    print("Modelo com Topologia da Rede:")
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
    ax[0, 0].set_xlabel('Carimbo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Training error
    ax[0, 1].step(time_intervals_train, train_errors, where='post', label='Erro Percentual', color='blue')
    ax[0, 1].set_title('Erro Percentual - Treino')
    ax[0, 1].set_xlabel('Carimbo Temporal [15 min]')
    ax[0, 1].set_ylabel('Erro Percentual (%)')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].grid(True)
    
    # Bottom-left: Test data comparison
    ax[1, 0].step(time_intervals_test, PL2_test_true, where='post', label='Perdas Reais')
    ax[1, 0].step(time_intervals_test, PL2_test_pred_topo, where='post', label='Perdas Preditas')
    ax[1, 0].set_title('Comparação de Perdas - Teste')
    ax[1, 0].set_xlabel('Carimbo Temporal [15 min]')
    ax[1, 0].set_ylabel('Perdas de Potência')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].grid(True)
    
    # Bottom-right: Test error
    ax[1, 1].step(time_intervals_test, test_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 1].set_title('Erro Percentual - Teste')
    ax[1, 1].set_xlabel('Carimbo Temporal [15 min]')
    ax[1, 1].set_ylabel('Erro Percentual (%)')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].grid(True)
    
    # Update y-axis limits for percentage error plots
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[0, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    
    plt.tight_layout()
    plt.savefig('resultados_modelo_topologia.png', dpi=300)
    plt.show()
    
    # Mostrar os coeficientes beta e sua relação com a estrutura da rede
    print("\nCOEFICIENTES DO MODELO COM TOPOLOGIA")
    
    # Termos quadráticos
    print("\nTermos Quadráticos:")
    for i in range(P.shape[1]):  # Número de buses sem o slack
        print(f"P{i+1}²: beta = {beta_topo[i]:.6f}")
    
    # Termos cruzados
    print("\nTermos Cruzados:")
    for idx, (bus1, bus2) in enumerate(linha_indices):
        if bus1 != (SlackBus-1) and bus2 != (SlackBus-1):  # Verifica se não envolve o slack
            coef_idx = P.shape[1] + idx  # termos quadráticos + índice atual
            if coef_idx < len(beta_topo):
                print(f"2*P{bus1+1}*P{bus2+1}: beta = {beta_topo[coef_idx]:.6f}")

# ============================================================================================================================================
# Funções para o modelo simplificado (apenas termos quadráticos) Desafio 2.2
# ============================================================================================================================================

def build_X_squared(P_mat):
    """
    Constrói a matriz X_sq considerando apenas os termos quadráticos P_i² para cada bus.
    
    Parâmetros:
      P_mat: Matriz de potência, com dimensão (n_instantes, nBus_sem_slack)
      
    Retorna:
      X_sq: Matriz de features contendo apenas os termos quadráticos
    """
    n_instantes, nBus_sem_slack = P_mat.shape
    X_sq = np.zeros((n_instantes, nBus_sem_slack))
    
    for i in range(nBus_sem_slack):
        X_sq[:, i] = P_mat[:, i]**2
        
    return X_sq

def run_squared_model():
    """Executa o modelo simplificado que considera apenas termos quadráticos P_i²."""
    print("\n=== Executando o Modelo Simplificado (Apenas Termos Quadráticos) ===\n")
    
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
    
    print("Modelo Simplificado (Apenas Termos Quadráticos):")
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
    ax[0, 0].step(time_intervals_train, PL2_train_pred_sq, where='post', label='Treino: Perdas Preditas (Quadrático)')
    ax[0, 0].set_title('Comparação de Perdas - Treino (Modelo Simplificado)')
    ax[0, 0].set_xlabel('Carimbo Temporal [15 min]')
    ax[0, 0].set_ylabel('Perdas de Potência')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].grid(True)
    
    # Top-right: Training error
    ax[0, 1].step(time_intervals_train, train_errors, where='post', label='Erro Percentual', color='blue')
    ax[0, 1].set_title('Erro Percentual - Treino')
    ax[0, 1].set_xlabel('Carimbo Temporal [15 min]')
    ax[0, 1].set_ylabel('Erro Percentual (%)')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].grid(True)
    
    # Bottom-left: Test data comparison
    ax[1, 0].step(time_intervals_test, PL2_test_true, where='post', label='Teste: Perdas Reais')
    ax[1, 0].step(time_intervals_test, PL2_test_pred_sq, where='post', label='Teste: Perdas Preditas (Quadrático)', color='orange')
    ax[1, 0].set_title('Comparação de Perdas - Teste (Modelo Simplificado)')
    ax[1, 0].set_xlabel('Carimbo Temporal [15 min]')
    ax[1, 0].set_ylabel('Perdas de Potência')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].grid(True)
    
    # Bottom-right: Test error
    ax[1, 1].step(time_intervals_test, test_errors, where='post', label='Erro Percentual', color='blue')
    ax[1, 1].set_title('Erro Percentual - Teste')
    ax[1, 1].set_xlabel('Carimbo Temporal [15 min]')
    ax[1, 1].set_ylabel('Erro Percentual (%)')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].grid(True)
    
    # Update y-axis limits for percentage error plots
    y_max_error_pct = max(np.max(train_errors), np.max(test_errors))
    y_margin_error_pct = 0.1 * y_max_error_pct
    
    ax[0, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    ax[1, 1].set_ylim(0, y_max_error_pct + y_margin_error_pct)
    
    plt.tight_layout()
    plt.savefig('resultados_modelo_quadratico.png', dpi=300)
    plt.show()
    
    # Mostrar os coeficientes beta
    print("\nCOEFICIENTES DO MODELO SIMPLIFICADO")
    for i in range(P.shape[1]):  # Número de buses sem o slack
        print(f"P{i+1}²: beta = {beta_sq[i]:.6f}")

def show_squared_challenge_info():
    """Mostra informações sobre o desafio de modelagem usando apenas termos quadráticos."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
    print("=" * 80)
    print("                DESAFIO: MODELO SIMPLIFICADO (APENAS TERMOS QUADRÁTICOS)                ")
    print("=" * 80)
    print("""
Neste desafio, a matriz X do modelo de regressão y = X β + ε é simplificada para incluir apenas 
os termos quadráticos P_i² para cada bus, excluindo completamente os termos de produtos cruzados 
2*P_i*P_j.

Na abordagem original, a matriz X inclui:
- Termos quadráticos P_i² para todos os buses
- Termos cruzados 2*P_i*P_j para todas as combinações possíveis de buses i e j

No modelo simplificado:
- Mantemos APENAS os termos quadráticos P_i² para cada bus
- Eliminamos completamente todos os termos cruzados 2*P_i*P_j

Benefícios desta abordagem:
- Redução dramática da dimensionalidade (de n(n+1)/2 para apenas n parâmetros)
- Modelo extremamente simples e fácil de interpretar
- Menor risco de sobreajuste (overfitting)
- Computacionalmente mais eficiente

A principal questão deste desafio é verificar o quanto se perde em termos de precisão 
ao adotar uma representação tão simplificada do modelo.
""")
    input("\nPressione Enter para voltar ao menu principal...")

# Atualizar o menu para incluir a nova opção
def show_menu():
    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
    print("=" * 80)
    print("                     LABORATÓRIO 2 - ANÁLISE DE PERDAS DE POTÊNCIA                    ")
    print("=" * 80)
    print("\nEscolha uma opção:")
    print("  1. Executar o modelo original (baseado em ângulos)")
    print("  2. Executar o modelo de baixa tensão (LV) (baseado em tensões)")
    print("  3. Executar o modelo baseado na topologia da rede")
    print("  4. Executar o modelo simplificado (apenas termos quadráticos)")
    print("  \nSair do programa: '0'")
    
    try:
        opcao = int(input("\nDigite a opção desejada: "))
        return opcao
    except ValueError:
        return -1  # Retorna -1 se o usuário inserir um valor não numérico

 
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
        else:
            print("\nOpção inválida! Por favor, escolha uma opção válida.")
            input("\nPressione Enter para continuar...")

