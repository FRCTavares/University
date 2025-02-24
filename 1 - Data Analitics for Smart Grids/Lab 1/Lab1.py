###############################################################################################
# Laboratorio 1 - Phase Identification - Single_Bus structure                                 #
#                                                                                             #
# Grupo X                                                                                     #
#                                                                                             #
# Membros:                                                                                    #
#   Diogo Sampaio (103068)                                                                    #
#   Francisco Tavares (103402)                                                                #
#   Marta Valente (103574)                                                                    #
###############################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import randint
from sklearn.metrics import confusion_matrix

'''---------------------------------------------------------------------------------------------------------------------------------------------------------#
# Modo 1: Modo Principal (Single_Bus) - Identificação de Fases
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    Quando M (número de observações) é apenas ligeiramente superior a N (número de variáveis), o sistema encontra-se praticamente “determinístico”,
    o que significa que uma pequena perturbação ou ruído nos dados pode levar a oscilações significativas nas estimativas das fases. Em termos práticos, 
    há poucas medições excedentes para compensar erros ou redundâncias no modelo, tornando a solução mais sensível a alterações mínimas nos valores observados.
    Já quando M é substancialmente maior do que N, o sistema fica mais “sobredeterminado”, e essa redundância adicional de medições ajuda a estabilizar o 
    processo de estimação, resultando em menos erros globais mesmo na presença de ruído ou variações no consumo.
'''
def modo_principal():
    
    nc = 4                         # Número de consumidores (1 a nc)
    ts = 40                        # Período inicial de análise (pode ser de 0 a 13, mas aqui vai de 40 a 63)
    te = 53                        # Período final de análise

    # Gera valores aleatórios para as fases
    phase = randint(1, 4, nc)
    
    # Importa os dados do arquivo Excel
    file_path = r"Prob1_Conso_Data.xlsx"
    raw_data = np.array(pd.read_excel(file_path, header=None))
    
    # Remove zeros e organiza por consumidores
    checks = 0
    nr = 1
    data = np.zeros((1, 96))
    h = raw_data[0:96, 0]
    for i in range(1, raw_data.shape[0] + 1):
        if raw_data[i - 1, 0] == h[checks]:
            checks += 1
        else:
            checks = 0
        if checks == 96:
            if np.sum(raw_data[i - 96:i, 1]) != 0:
                data[nr - 1, 0:96] = raw_data[i - 96:i, 1]
                data.resize((nr + 1, 96))
                nr += 1
            checks = 0
    data.resize((nr - 1, 96))
    
    # Cria a matriz X
    data_Aux1 = data[0:nc, :]
    pw = data_Aux1[:, ts:te + 1]
    X = np.transpose(4 * pw)

    
    # Cria a matriz Y com base nas fases (fase aleatória)
    Y = np.zeros((X.shape[0], 3))
    for i in range(nc):
        phase_index = phase[i] - 1  # Ajusta para índice base zero
        Y[:, phase_index] += X[:, i]
    
    # Adiciona ruído aos dados de Y
    Y = Y + np.random.normal(0, 0.01, Y.shape)
    
    
    # Estima a matriz beta via mínimos quadrados
    lambda_ = 0.1  # Parâmetro de regularização, ajustar conforme necessário
    beta = np.linalg.inv(X.T @ X + lambda_ * np.eye(X.shape[1])) @ X.T @ Y

    beta = np.where(beta > 0.5, 1, 0)               # Converte para binário
    phase_estimated = np.argmax(beta, axis=1) + 1   # Ajusta para índice 1-based
    
    print("\nO vetor beta estimado:\n", phase_estimated)
    print("\nO vetor de fase inicial:\n", phase)
    
    # Cria os intervalos de tempo para os gráficos (de ts a te, inclusive)
    time_intervals = np.arange(ts, te + 1)
    
    # Cria a figura com dois subplots lado a lado
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    # Gráfico: Leituras dos Consumidores (matriz X)
    for i in range(X.shape[1]):
        ax[0].step(time_intervals, X[:, i], where='post', label=f'Consumer {i + 1}')
    ax[0].set_title('Customer Readings')
    ax[0].set_xlabel('Time Stamp [15min]')
    ax[0].set_ylabel('Power [kW]')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    
    # Gráfico: Totais por Fase (matriz Y)
    for i in range(Y.shape[1]):
        ax[1].step(time_intervals, Y[:, i], where='post', label=f'Fase {i + 1}')
    ax[1].set_title('Per-Phase Totals')
    ax[1].set_xlabel('Time Stamp [15min]')
    ax[1].set_ylabel('Power [kW]')
    ax[1].legend(loc='upper right')  
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Análise dos Resultados Finais, erro de classificação
    labels = [1, 2, 3]
    cm = confusion_matrix(phase, phase_estimated, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Fase Estimada')
    plt.ylabel('Fase Real')
    plt.title('Matriz de Confusão: Fase Real vs. Fase Estimada')
    plt.show()

    # Soma da diagonal da matriz de confusão para obter o número de classificações corretas
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total
    print(f'\nA percentagem de classificações corretas é {accuracy:.2f}')

def precisao_mono():
    """
    Esta função varia o número de observações (M) desde M = N (ou seja, M = nc) 
    até um valor máximo (definido em função do número total de períodos disponíveis),
    e para cada M calcula a precisão do modelo da identificação de fases.
    
    O gráfico resultante tem no eixo x a diferença (M - N) e no eixo y a precisão do modelo.
    """
    nc = 10                       # Número de consumidores (N)
    ts = 0                        # Período inicial fixo
    M_max = 96                     # Valor máximo de M (por exemplo, até te = 96, pois 96 - 40 + 1 = 57)
    
    # Gerar uma única distribuição aleatória de fases para todos os experimentos
    true_phase = randint(1, 4, nc)
    print("Distribuição dos consumidores em cada fase (verdadeira):", true_phase)
    
    # Importa os dados do ficheiro Excel
    file_path = r"Prob1_Conso_Data.xlsx"
    raw_data = np.array(pd.read_excel(file_path, header=None))
    
    # Remove zeros e organiza os dados por consumidores (assume-se que os dados têm 96 períodos)
    checks = 0
    nr = 1
    data = np.zeros((1, 96))
    h = raw_data[0:96, 0]
    for i in range(1, raw_data.shape[0] + 1):
        if raw_data[i - 1, 0] == h[checks]:
            checks += 1
        else:
            checks = 0
        if checks == 96:
            if np.sum(raw_data[i - 96:i, 1]) != 0:
                data[nr - 1, 0:96] = raw_data[i - 96:i, 1]
                data.resize((nr + 1, 96))
                nr += 1
            checks = 0
    data.resize((nr - 1, 96))
    
    differences_list = []
    accuracy_list = []
    
    # Variação de M: de M = nc (ou seja, M - nc = 0) até M_max
    for M in range(nc+1, M_max + 1):
        aux = []
        for i in range(0,30):

            te = ts + M - 1  # Calcula te com base em M
            
            # Seleciona os dados dos primeiros 'nc' consumidores e o intervalo [ts, te]
            data_Aux1 = data[0:nc, :]
            pw = data_Aux1[:, ts:te + 1]
            X = np.transpose(4 * pw)  # X terá dimensão (M, nc)
            
            # Cria a matriz Y com base na fase verdadeira (true_phase)
            Y = np.zeros((X.shape[0], 3))
            for i in range(nc):
                phase_index = true_phase[i] - 1  # Ajusta para índice base zero
                Y[:, phase_index] += X[:, i]
            
            # Adiciona ruído aos dados de Y
            Y = Y + np.random.normal(0, 0.03, Y.shape)
            
            # Estima a matriz beta utilizando mínimos quadrados
            lambda_ = 0.1  # Parâmetro de regularização, ajustar conforme necessário
            beta = np.linalg.inv(X.T @ X + lambda_ * np.eye(X.shape[1])) @ X.T @ Y

            beta = np.where(beta > 0.5, 1, 0)  # Converte para binário
            phase_estimated = np.argmax(beta, axis=1) + 1  # Resultado: vector de dimensão (nc,)
            
            # Calcula a Precisão do Modelo: proporção de consumidores corretamente classificados
            acc = np.mean(phase_estimated == true_phase)
            aux.append(acc)

        differences_list.append(M - nc)  # Diferença entre número de observações e consumidores
        accuracy_list.append(np.mean(aux))
        #print(f"M = {M}, M - N = {M - nc}, Precisão do Modelo = {np.mean(aux):.2f}")
    
    # Plot do gráfico: eixo x -> (M - N), eixo y -> Precisão do Modelo
    plt.figure(figsize=(8, 5))
    plt.plot(differences_list, accuracy_list, linestyle='-')
    plt.xlabel('Diferença (M - N)')
    plt.ylabel('Média da Precisão do Modelo')
    plt.title('Média da Precisão do Modelo vs Diferença entre M e N')
    plt.grid(True)
    plt.show()


'''---------------------------------------------------------------------------------------------------------------------------------------------------------#
# Modo 2: Análise de Sensibilidade
#
# Consumidores com consumo idêntico ou com pequenas diferenças
# Aqui é para adicionar dois clientes novos à matriz X e arranjar valores de energia
# para os dois começando com eles muito parecidos e alterando um pouco até que ele identifica corretamente a fase, 
# porque quase de certeza que quando eles são muito muito parecidos o modelo não os vai consegui diferenciar.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
    Quando dois consumidores apresentam padrões de consumo idênticos ou quase idênticos, as colunas da matriz X tornam-se linearmente dependentes
    ou quase linearmente dependentes, o que implica que a matriz X.T @ X se torna singular ou mal condicionada e, portanto, não pode ser invertida
    de forma estável. Essa situação dificulta a aplicação direta da fórmula dos mínimos quadrados para estimar os coeficientes β. Para contornar
    esse problema, é recomendada a utilização da pseudo-inversa, que permite obter uma solução de mínimos quadrados mesmo quando a matriz 
    original não é invertível, como foi sugerido pelo professor. Essa abordagem garante uma estimação robusta dos coeficientes mesmo em casos 
    de alta correlação entre as variáveis.
'''
def analise_sensabilidade():
    nc = 10  # Número de consumidores (1 a nc)
    ts = 40  # Período inicial de análise (pode ser de 0 a 13, mas aqui vai de 40 a 63)
    te = 52  # Período final de análise (M = 12)
    
    # Importa os dados do arquivo Excel
    file_path = r"Prob1_Conso_Data.xlsx"
    raw_data = np.array(pd.read_excel(file_path, header=None))
    
    # Remove zeros e organiza por consumidores
    checks = 0
    nr = 1
    data = np.zeros((1, 96))
    h = raw_data[0:96, 0]
    for i in range(1, raw_data.shape[0] + 1):
        if raw_data[i - 1, 0] == h[checks]:
            checks += 1
        else:
            checks = 0
        if checks == 96:
            if np.sum(raw_data[i - 96:i, 1]) != 0:
                data[nr - 1, 0:96] = raw_data[i - 96:i, 1]
                data.resize((nr + 1, 96))
                nr += 1
            checks = 0
    data.resize((nr - 1, 96))
    
    # Inicializa variáveis para o ciclo
    confusion_matrices_no_L2 = []
    confusion_matrices_with_L2 = []
    accuracies_no_L2 = []
    accuracies_with_L2 = []
    
    # Ciclo de 100 iterações
    for _ in range(100):
        # Aleatoriza a fase em cada iteração (vetor sempre de comprimento nc+1)
        phase = np.random.randint(1, 4, nc+1)  # Fase aleatória em cada iteração
        
        # Modifica ligeiramente os dados adicionando ruído aleatório ou alterando fases
        data_Aux1 = data[0:nc, :]
        pw = data_Aux1[:, ts:te + 1]
        X = np.transpose(4 * pw)

        # Introduz um novo consumidor com uma ligeira variação
        new_consumer_data = X[:, -1] + np.random.normal(0, 0.02, X.shape[0])
        X = np.column_stack([X, new_consumer_data])

        # Cria a matriz Y com base nas fases aleatórias
        Y = np.zeros((X.shape[0], 3))
        for i in range(nc+1):
            phase_index = phase[i] - 1  # Ajusta para índice base zero
            Y[:, phase_index] += X[:, i]

        # Adiciona ruído a Y
        Y = Y + np.random.normal(0, 0.03, Y.shape)

        # Estima a matriz beta via mínimos quadrados (sem regularização L2)
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        phase_estimated = np.argmax(beta, axis=1) + 1   # Ajusta para índice base 1

        # Regularização L2
        lambda_reg = 0.01  
        I = np.eye(X.shape[1])
        beta_L2 = np.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ Y
        phase_estimated_L2 = np.argmax(beta_L2, axis=1) + 1

        # Calcula matrizes de confusão com rótulos fixos para manter sempre a dimensão (3,3)
        cm_no_L2 = confusion_matrix(phase, phase_estimated, labels=[1, 2, 3])
        cm_with_L2 = confusion_matrix(phase, phase_estimated_L2, labels=[1, 2, 3])
        
        confusion_matrices_no_L2.append(cm_no_L2)
        confusion_matrices_with_L2.append(cm_with_L2)
        
        # Calcula a precisão para esta iteração
        correct_no_L2 = np.trace(cm_no_L2)
        total_no_L2 = np.sum(cm_no_L2)
        accuracy_no_L2 = correct_no_L2 / total_no_L2
        accuracies_no_L2.append(accuracy_no_L2)
        
        correct_with_L2 = np.trace(cm_with_L2)
        total_with_L2 = np.sum(cm_with_L2)
        accuracy_with_L2 = correct_with_L2 / total_with_L2
        accuracies_with_L2.append(accuracy_with_L2)
    
    # Calcula a precisão média para cada caso
    avg_accuracy_no_L2 = np.mean(accuracies_no_L2)
    avg_accuracy_with_L2 = np.mean(accuracies_with_L2)
    
    print(f'Precisão do modelo sem regularização: {avg_accuracy_no_L2: .2f}')
    print(f'Precisão do modelo com regularização: {avg_accuracy_with_L2: .2f}')
    
    # Matrizes de confusão médias ao longo de 100 iterações
    avg_cm_no_L2 = np.mean(confusion_matrices_no_L2, axis=0)
    avg_cm_with_L2 = np.mean(confusion_matrices_with_L2, axis=0)
    
    # Desenha as matrizes de confusão para ambos os casos
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  

    sns.heatmap(avg_cm_no_L2.astype(int), annot=True, fmt='d', cmap='Blues', cbar=False, square=True, ax=ax[0],
                annot_kws={'size': 12})  
    ax[0].set_title('Matriz de Confusão: Sem Regularização', fontsize=14)
    ax[0].set_xlabel('Fase estimada', fontsize=12)
    ax[0].set_ylabel('Fase real', fontsize=12)

    sns.heatmap(avg_cm_with_L2.astype(int), annot=True, fmt='d', cmap='Blues', cbar=False, square=True, ax=ax[1],
                annot_kws={'size': 12})  
    ax[1].set_title('Matriz de Confusão: Com Regularização', fontsize=14)
    ax[1].set_xlabel('Fase estimada', fontsize=12)
    ax[1].set_ylabel('Fase real', fontsize=12)

    plt.tight_layout()
    plt.show()


'''---------------------------------------------------------------------------------------------------------------------------------------------------------#
# Modo 3: Clientes Trifásicos 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
'''
def trifasico():
    # Parâmetros
    nc = 4      # Número de clientes monofásicos (dados reais do Excel)
    n_tri = 2   # Número de clientes trifásicos (dados simulados)
    ts = 60     # Período inicial de análise
    te = 83     # Período final de análise
    M = te - ts + 1  # Número de períodos de análise

    # Gera os rótulos verdadeiros para os clientes monofásicos (ligados a uma única fase)
    true_phase_single = randint(1, 4, nc)  # Valores aleatórios entre 1 e 3
    print("Fases verdadeiras dos clientes monofásicos:", true_phase_single)

    # -------------------------------
    # Dados dos clientes monofásicos (importados do Excel)
    # -------------------------------
    file_path = r"Prob1_Conso_Data.xlsx"
    raw_data = np.array(pd.read_excel(file_path, header=None))

    # Remove zeros e organiza os dados em blocos de 96 períodos por cliente
    checks = 0
    nr = 1
    data = np.zeros((1, 96))
    h = raw_data[0:96, 0]
    for i in range(1, raw_data.shape[0] + 1):
        if raw_data[i-1, 0] == h[checks]:
            checks += 1
        else:
            checks = 0
        if checks == 96:
            if np.sum(raw_data[i-96:i, 1]) != 0:
                data[nr-1, 0:96] = raw_data[i-96:i, 1]
                data.resize((nr+1, 96))
                nr += 1
            checks = 0
    data.resize((nr-1, 96))

    # Seleciona os dados dos primeiros nc clientes (monofásicos) para o intervalo [ts, te]
    data_mono = data[0:nc, :]
    pw_mono = data_mono[:, ts:te+1]
    X = np.transpose(4 * pw_mono)   # Matriz X de dimensão (M, nc)

    # --- Agora simula o consumo global para 2 clientes trifásicos ---
    # Para cada cliente trifásico, simulamos o seu consumo total (global) por período.
    # Amostramos valores de uma distribuição normal (média 1, desvio padrão 0.25).
    X_tri = np.random.normal(loc=1.0, scale=0.25, size=(M, n_tri)) 

    print("\nMatriz X dos clientes monofásicos:\n", X)
    print("\nMatriz X dos clientes trifásicos:\n", X_tri)

    # Concatena as colunas simuladas dos clientes trifásicos à matriz X dos clientes monofásicos.
    X_final = np.hstack([X, X_tri])

    # -------------------------------
    # Construção da Matriz Y (agregada por fase) para os clientes monofásicos
    # -------------------------------
    # Para cada cliente monofásico, adiciona o consumo à fase definida por true_phase_single.
    Y = np.zeros((M, 3))
    for i in range(nc):
        phase_index = true_phase_single[i] - 1  # Ajusta para índice base zero
        Y[:, phase_index] += X_final[:, i]

    # -------------------------------
    # Adiciona as contribuições dos clientes trifásicos
    # -------------------------------
    # Para cada cliente trifásico, amostramos percentagens (em %) de uma distribuição normal
    # com média 33 e desvio padrão 4, aplicamos um clip para valores negativos e normalizamos para que a soma seja 1.
    Y_tri = np.zeros((M, 3))
    for j in range(n_tri):
        # Amostra valores percentuais (em pontos percentuais)
        perc = np.random.normal(loc=33, scale=4, size=3)
        # Aplica um clip a valores negativos (mínimo 0.01)
        perc = np.clip(perc, 0.01, None)
        # Normaliza para que a soma seja 100% (ou 1 quando usados como frações)
        perc = perc / np.sum(perc)
        # Adiciona a contribuição do j-ésimo cliente trifásico (consumo global por período)
        # distribuído entre as três fases segundo as percentagens amostradas.
        Y_tri[:, 0] += X_tri[:, j] * perc[0]
        Y_tri[:, 1] += X_tri[:, j] * perc[1]
        Y_tri[:, 2] += X_tri[:, j] * perc[2]

    # -------------------------------
    # Remove a contribuição dos clientes trifásicos de Y
    # -------------------------------
    Y_adjusted = Y - Y_tri  # Subtrai o consumo estimado dos clientes trifásicos
    # Adiciona um pouco de ruído a Y_adjusted (para simular imperfeições na medição)
    Y_adjusted = Y_adjusted + np.random.normal(0, 0.01, Y_adjusted.shape)
    # -------------------------------
    # Estima a matriz beta via mínimos quadrados com regularização L2 (usando X e Y_adjusted)
    # -------------------------------
    lambda_ = 0.01
    beta = np.linalg.inv(X.T @ X + lambda_ * np.eye(X.shape[1])) @ (X.T @ Y_adjusted)
    beta = np.where(beta > 0.5, 1, 0)  # Binariza a matriz beta
    phase_estimated = np.argmax(beta, axis=1) + 1  # A fase estimada para cada cliente monofásico

    print("\nO vetor beta estimado:\n", phase_estimated)
    print("\nO vetor de fase inicial (clientes monofásicos):\n", true_phase_single)

    # -------------------------------
    # Visualizações
    # -------------------------------
    time_intervals = np.arange(ts, te + 1)

    # Plot das leituras dos consumidores (X)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    for i in range(X.shape[1]):
        ax[0].step(time_intervals, X[:, i], where='post', label=f'Cliente {i+1}')
    ax[0].set_title('Leituras dos Clientes Monofásicos')
    ax[0].set_xlabel('Carimbo Temporal [15min]')
    ax[0].set_ylabel('Potência [kW]')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)

    # Plot dos totais por fase (Y_adjusted)
    for i in range(Y_adjusted.shape[1]):
        ax[1].step(time_intervals, Y_adjusted[:, i], where='post', label=f'Fase {i+1}')
    ax[1].set_title('Totais por Fase (Após Remoção dos Trifásicos)')
    ax[1].set_xlabel('Carimbo Temporal [15min]')
    ax[1].set_ylabel('Potência [kW]')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Avaliação: Matriz de Confusão para os Clientes Monofásicos
    # -------------------------------
    from sklearn.metrics import confusion_matrix
    labels = [1, 2, 3]
    cm = confusion_matrix(true_phase_single, phase_estimated, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Fase Estimada')
    plt.ylabel('Fase Real')
    plt.title('Matriz de Confusão: Fase Real vs. Fase Estimada (Clientes Monofásicos)')
    plt.show()

    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total if total != 0 else 0
    print(f'\nA percentagem de classificações corretas é {accuracy:.2f}')

def precisao_tri():
    """
    Esta função varia o número de observações (M) desde M = N (ou seja, M = nc) 
    até um valor máximo (definido em função do número total de períodos disponíveis),
    e para cada M calcula a precisão do modelo da identificação de fases.
    
    O gráfico resultante tem no eixo x a diferença (M - N) e no eixo y a precisão do modelo.
    """
    nc = 12                     # Número de clientes monofásicos (N)
    n_tri = 2                   # Número de clientes trifásicos (dados simulados)
    ts = 0                      # Período inicial fixo
    M_max = 96                  # Valor máximo de M 
    
    # Gera as fases verdadeiras para os clientes monofásicos
    true_phase_single = randint(1, 4, nc)  
    print("Fases verdadeiras dos clientes monofásicos:", true_phase_single)

    file_path = r"Prob1_Conso_Data.xlsx"
    raw_data = np.array(pd.read_excel(file_path, header=None))
    
    # Remove zeros e organiza os dados por consumidores 
    checks = 0
    nr = 1
    data = np.zeros((1, 96))
    h = raw_data[0:96, 0]
    for i in range(1, raw_data.shape[0] + 1):
        if raw_data[i - 1, 0] == h[checks]:
            checks += 1
        else:
            checks = 0
        if checks == 96:
            if np.sum(raw_data[i - 96:i, 1]) != 0:
                data[nr - 1, 0:96] = raw_data[i - 96:i, 1]
                data.resize((nr + 1, 96))
                nr += 1
            checks = 0
    data.resize((nr - 1, 96))
    
    differences_list = []
    accuracy_list = []
    
    # Variação de M: de M = nc + n_tri (ou seja, M - nc = 0) até M_max
    for M in range(nc+n_tri+1, M_max + 1):
        aux = []
        for i in range(0,30):

            te = ts + M - 1  # Calcula te com base em M
            
            # Seleciona os dados dos primeiros nc clientes (monofásicos) para o intervalo [ts, te]
            data_mono = data[0:nc, :]
            pw_mono = data_mono[:, ts:te+1]
            X = np.transpose(4 * pw_mono)   # Matriz X de dimensão (M, nc)

            # --- Agora simula o consumo global para 2 clientes trifásicos ---
            # Para cada cliente trifásico, simulamos o seu consumo total (global) por período.
            # Amostramos valores de uma distribuição normal (média 1, desvio padrão 0.25).
            X_tri = np.random.normal(loc=1.0, scale=0.25, size=(M, n_tri)) 

            # Concatena as colunas simuladas dos clientes trifásicos à matriz X dos clientes monofásicos.
            X_final = np.hstack([X, X_tri])

            # -------------------------------
            # Construção da Matriz Y (agregada por fase) para os clientes monofásicos
            # -------------------------------
            # Para cada cliente monofásico, adiciona o consumo à fase definida por true_phase_single.
            Y = np.zeros((M, 3))
            for i in range(nc):
                phase_index = true_phase_single[i] - 1  # Ajusta para índice base zero
                Y[:, phase_index] += X_final[:, i]

            # -------------------------------
            # Adiciona as contribuições dos clientes trifásicos
            # -------------------------------
            # Para cada cliente trifásico, amostramos percentagens (em %) de uma distribuição normal
            # com média 33 e desvio padrão 4, aplicamos um clip para valores negativos e normalizamos para que a soma seja 1.
            Y_tri = np.zeros((M, 3))
            for j in range(n_tri):
                # Amostra valores percentuais (em pontos percentuais)
                perc = np.random.normal(loc=33, scale=4, size=3)
                # Aplica um clip a valores negativos (mínimo 0.01)
                perc = np.clip(perc, 0.01, None)
                # Normaliza para que a soma seja 100% (ou 1 quando usados como frações)
                perc = perc / np.sum(perc)
                # Adiciona a contribuição do j-ésimo cliente trifásico (consumo global por período)
                # distribuído entre as três fases segundo as percentagens amostradas.
                Y_tri[:, 0] += X_tri[:, j] * perc[0]
                Y_tri[:, 1] += X_tri[:, j] * perc[1]
                Y_tri[:, 2] += X_tri[:, j] * perc[2]

            # -------------------------------
            # Remove a contribuição dos clientes trifásicos de Y
            # -------------------------------
            Y_adjusted = Y - Y_tri  # Subtrai o consumo estimado dos clientes trifásicos
            # Adiciona um pouco de ruído a Y_adjusted (para simular imperfeições na medição)
            Y_adjusted = Y_adjusted + np.random.normal(0, 0.01, Y_adjusted.shape)
            # -------------------------------
            # Estima a matriz beta via mínimos quadrados com regularização L2 (usando X e Y_adjusted)
            # -------------------------------
            lambda_ = 0.01
            beta = np.linalg.inv(X.T @ X + lambda_ * np.eye(X.shape[1])) @ (X.T @ Y_adjusted)
            beta = np.where(beta > 0.5, 1, 0)  # Binariza a matriz beta
            phase_estimated = np.argmax(beta, axis=1) + 1  # A fase estimada para cada cliente monofásico

            # Calcula a Precisão do Modelo: proporção de consumidores corretamente classificados
            acc = np.mean(phase_estimated == true_phase_single)
            aux.append(acc)

        differences_list.append(M - nc)  # Diferença entre número de observações e consumidores
        accuracy_list.append(np.mean(aux))
        #print(f"M = {M}, M - N = {M - nc}, Precisão do Modelo = {np.mean(aux):.2f}")
    
    # Plot do gráfico: eixo x -> (M - N), eixo y -> Precisão do Modelo
    plt.figure(figsize=(8, 5))
    plt.plot(differences_list, accuracy_list, linestyle='-')
    plt.xlabel('Diferença (M - N)')
    plt.ylabel('Média da Precisão do Modelo')
    plt.title('Média da Precisão do Modelo vs Diferença entre M e N')
    plt.grid(True)
    plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
# Menu de Execução
#-----------------------------------------------------------------------------------------------------------------------------------------------------------#
def main_menu():
    while True:
        print("\nMenu de Modos:")
        print("1 - Modo Principal: Identificação de Fases")
        print("2 - Análise de Precisão com Clientes Monofásico apenas")
        print("3 - Análise de Sensibilidade (Consumidores com Consumo Idêntico ou Quase Idêntico)")
        print("4 - Identificação de Fases para Clientes Monofásicos + Trifásicos")
        print("5 - Análise de Precisão para Clientes Monofásicos + Trifásicos")

        print("E - Sair")
        choice = input("\nEscolha uma opção: ")
        if choice == '1':
            modo_principal()
        elif choice == '2':
            precisao_mono()
        elif choice == '3':
            analise_sensabilidade()
        elif choice == '4':
            trifasico()
        elif choice == '5':
            precisao_tri()
        elif choice == 'E' or choice == 'e':
            print("\nA terminar...\n")
            break
        else:
            print("\nOpção inválida. Por favor, escolha novamente.\n")

if __name__ == "__main__":
    main_menu()