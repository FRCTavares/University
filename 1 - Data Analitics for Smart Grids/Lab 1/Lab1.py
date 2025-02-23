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
from numpy.random import randint, random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#------------------------------#
# Modo 1: Modo Principal (Single_Bus) - Identificação de Fases
#------------------------------#

'''
    Quando M (número de observações) é apenas ligeiramente superior a N (número de variáveis), o sistema encontra-se praticamente “determinístico”,
    o que significa que uma pequena perturbação ou ruído nos dados pode levar a oscilações significativas nas estimativas das fases. Em termos práticos, 
    há poucas medições excedentes para compensar erros ou redundâncias no modelo, tornando a solução mais sensível a alterações mínimas nos valores observados.
    Já quando M é substancialmente maior do que N, o sistema fica mais “sobredeterminado”, e essa redundância adicional de medições ajuda a estabilizar o 
    processo de estimação, resultando em menos erros globais mesmo na presença de ruído ou variações no consumo.
'''

def main_lab_mode():
    print("\nExecutando o Modo Principal: Phase Identification - Single_Bus structure\n")
    
    nc = 10                        # Número de consumidores (1 a nc)
    ts = 40                        # Período inicial de análise (pode ser de 0 a 13, mas aqui vai de 40 a 63)
    te = 52                        # Período final de análise M = 12

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
    Y = Y + np.random.normal(0, 0.10, Y.shape)
    
    # Estima a matriz beta via mínimos quadrados
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    beta = np.where(beta > 0.5, 1, 0)  # Converte para binário
    phase_estimated = np.argmax(beta, axis=1) + 1  # Ajusta para índice 1-based
    
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
    ax[0].legend()
    ax[0].grid(True)
    
    # Gráfico: Totais por Fase (matriz Y)
    for i in range(Y.shape[1]):
        ax[1].step(time_intervals, Y[:, i], where='post', label=f'Fase {i + 1}')
    ax[1].set_title('Per-Phase Totals')
    ax[1].set_xlabel('Time Stamp [15min]')
    ax[1].set_ylabel('Power [kW]')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Análise dos Resultados Finais, erro de classificação
    cm = confusion_matrix(phase, phase_estimated)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Fase Estimada')
    plt.ylabel('Fase Real')
    plt.title('Matriz de Confusão: Fase Real vs. Fase Estimada')
    plt.show()

    # some the diagonal elements of the confusion matrix
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total
    print(f'A percentagem de classificações corretas é {accuracy:.2f}')


def analyze_accuracy_vs_MminusN():
    """
    Esta função varia o número de observações (M) desde M = N (ou seja, M = nc) 
    até um valor máximo (definido em função do número total de períodos disponíveis),
    e para cada M calcula a acurácia da identificação de fases.
    
    O gráfico resultante tem no eixo x a diferença (M - N) e no eixo y a acurácia.
    """
    nc = 4                       # Número de consumidores (N)
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
        Y = Y + np.random.normal(0, 0.10, Y.shape)
        
        # Estima a matriz beta utilizando mínimos quadrados
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        beta = np.where(beta > 0.5, 1, 0)  # Converte para binário
        phase_estimated = np.argmax(beta, axis=1) + 1  # Resultado: vector de dimensão (nc,)
        
        # Calcula a acurácia: proporção de consumidores corretamente classificados
        acc = np.mean(phase_estimated == true_phase)
        differences_list.append(M - nc)  # Diferença entre número de observações e consumidores
        accuracy_list.append(acc)
        
        print(f"M = {M}, M - N = {M - nc}, Acurácia = {acc:.2f}")
    
    # Plot do gráfico: eixo x -> (M - N), eixo y -> acurácia
    plt.figure(figsize=(8, 5))
    plt.plot(differences_list, accuracy_list, linestyle='-')
    plt.xlabel('Diferença (M - N)')
    plt.ylabel('Acurácia')
    plt.title('Acurácia vs Diferença entre M e N')
    plt.grid(True)
    plt.show()

#------------------------------#
# Modo 2: Análise de Sensibilidade
# Consumidores com consumo idêntico ou com pequenas diferenças
#------------------------------#

'''
Quando dois consumidores apresentam padrões de consumo idênticos ou quase idênticos, as colunas da matriz X tornam-se linearmente dependentes
    ou quase linearmente dependentes, o que implica que a matriz X.T @ X se torna singular ou mal condicionada e, portanto, não pode ser invertida
    de forma estável. Essa situação dificulta a aplicação direta da fórmula dos mínimos quadrados para estimar os coeficientes β. Para contornar
    esse problema, é recomendada a utilização da pseudo-inversa, que permite obter uma solução de mínimos quadrados mesmo quando a matriz 
    original não é invertível, como foi sugerido pelo professor. Essa abordagem garante uma estimação robusta dos coeficientes mesmo em casos 
    de alta correlação entre as variáveis.
'''

def sensitivity_analysis_mode():
    print("\nExecutando o Modo: Análise de Sensibilidade\n")
    
    n_periods = 14                    # Períodos de 0 a 13
    true_phase = 1                    # Supomos que ambos os consumidores pertencem à fase 1
    n_consumers = 2                   # Dois consumidores para o teste
    
    # Gera um padrão base de consumo
    np.random.seed(0)
    base_pattern = np.random.normal(loc=10, scale=1, size=n_periods)
    
    phase_estimates = []              # Armazenará a acurácia em cada cenário
    differences = np.linspace(0, 0.5, 20)  # Diferenças de 0 a 0.5 (20 passos)
    
    for diff in differences:
        # Cria os padrões de consumo para os dois consumidores
        consumer1 = base_pattern.copy()
        consumer2 = base_pattern.copy() + diff  # Adiciona uma pequena diferença ao segundo consumidor
        
        # Monta a matriz X: cada coluna corresponde a um consumidor (escalado por 4)
        X = np.transpose(np.array([4 * consumer1, 4 * consumer2]))
        
        # Monta a matriz Y: ambos os consumidores estão na mesma fase (fase 1 → índice 0)
        Y = np.zeros((n_periods, 3))
        for i in range(n_consumers):
            phase_index = true_phase - 1
            Y[:, phase_index] += X[:, i]
        Y = Y + np.random.normal(0, 0.01, Y.shape)
        
        # Estima os coeficientes beta via mínimos quadrados
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        beta_binary = np.where(beta > 0.5, 1, 0)
        estimated_phase = np.argmax(beta_binary, axis=1) + 1
        
        # Calcula a acurácia (proporção de períodos com fase correta)
        accuracy = np.mean(estimated_phase == true_phase)
        phase_estimates.append(accuracy)
    
    # Plota a acurácia em função da diferença adicionada
    plt.figure(figsize=(8, 5))
    plt.plot(differences, phase_estimates, marker='o')
    plt.xlabel('Diferença adicionada ao consumo do Consumer 2')
    plt.ylabel('Proporção de períodos com fase estimada correta')
    plt.title('Análise de Sensibilidade: Pequenas Diferenças de Consumo')
    plt.grid(True)
    plt.show()

#------------------------------#
# Modo 3: Clientes Trifásicos - Clustering e PCA
#------------------------------#
def three_phase_customers_mode():
    print("\nExecutando o Modo: Clientes Trifásicos (Clustering e PCA)\n")
    
    np.random.seed(42)
    customers = 100
    three_phase_data = np.random.normal(loc=1500, scale=200, size=(customers, 3))
    
    # Cria DataFrame com os dados simulados
    df_three = pd.DataFrame(three_phase_data, columns=['Phase1', 'Phase2', 'Phase3'])
    df_three['Total'] = df_three.sum(axis=1)
    
    # Normaliza os dados para melhorar o clustering
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_three[['Phase1', 'Phase2', 'Phase3']])
    
    # Aplica o K-Means para identificar 3 clusters (potencialmente correspondentes às fases)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)
    df_three['Cluster'] = clusters
    
    # Reduz a dimensionalidade para 2 componentes usando PCA para visualização
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(normalized_data)
    
    # Plota os resultados do clustering via PCA
    plt.figure(figsize=(8, 5))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', edgecolor='k')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Clustering de Consumidores Trifásicos (PCA)')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

#------------------------------#
# Menu de Execução
#------------------------------#
def main_menu():
    while True:
        print("\nMenu de Modos:")
        print("1 - Modo Principal: Phase Identification - Single_Bus")
        print("2 - Análise de Sensibilidade (Consumidores com Consumo Idêntico ou Quase Idêntico)")
        print("3 - Clientes Trifásicos (Clustering e PCA)")
        print("4 - Análise de Acuraccy vs. Diferença entre M e N")
        print("E - Sair")
        choice = input("Escolha uma opção: ")
        if choice == '1':
            main_lab_mode()
        elif choice == '2':
            sensitivity_analysis_mode()
        elif choice == '3':
            three_phase_customers_mode()
        elif choice == '4':
            analyze_accuracy_vs_MminusN()
        elif choice == 'E' or choice == 'e':
            print("A terminar...")
            break
        else:
            print("Opção inválida. Por favor, escolha novamente.")

if __name__ == "__main__":
    main_menu()
