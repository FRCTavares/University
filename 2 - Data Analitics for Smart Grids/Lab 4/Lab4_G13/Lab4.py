###############################################################################################
# Laboratório 4 - Regressão sobre Previsão Autoregressiva                                     #
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
networkFactor = 100    # Para alterar as características da rede (Y)
cosPhi=0.95            # Valor de teta
time=24                # Período de Treino    
timeForecast=12        # Período de Teste

# ============================================================================================================================================
# Funções para carregar e processar dados
# ============================================================================================================================================
def load_data(file_path):
    Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Info', header=None))
    # Informação sobre o barramento de referênciae referência
    SlackBus=Info[0,1]
    print ("Barramento de referência: ", SlackBus)

    # Informação da Rede
    Net_Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Y_Data'))
    print ("Informação das linhas (Admitâncias)\n", Net_Info, "\n")

    #Informação de Potência (Treino)reino)
    Power_Info = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Load(t,Bus)'))
    Power_Info = np.delete(Power_Info,[0],1)
    print ("Informação do consumo de potência (tempo, Barramento)\n", Power_Info, "\n")

    #Informação de Potência (Teste)Teste)
    Power_Test = np.array(pd.read_excel (r'DASG_Prob2_new.xlsx', sheet_name='Test_Load(t,Bus)'))
    Power_Test = np.delete(Power_Test,[0],1)
    print ("Informação do consumo de potência (tempo, Barramento)\n", Power_Test, "\n")

    # Cria diretorio para guardar os gráficos
    if not os.path.exists('plots'):
        os.makedirs('plots')
    

    return SlackBus, Net_Info, Power_Info, Power_Test

def prepare_network(Net_Info, Power_Info, SlackBus):
    P=np.dot(-Power_Info,np.exp(complex(0,1)*np.arccos(cosPhi)))
    I=np.conj(P[2,:])


    P=np.dot(-Power_Info,np.exp(complex(0,1)*np.arccos(cosPhi)))
    print(P)
    I=np.conj(P[2,:])


    # Determinar o número de Barramentosamentos
    nBus=max(np.max(Net_Info[:,0]),np.max(Net_Info[:,1]))

    # Criar a variável número de linhas e a matriz de admitâncias (Y)
    nLines=Net_Info.shape[0]

    Y=np.zeros((nBus,nBus), dtype=complex)

    #Completar a matriz Y e atualizar o número de linhas
    for i in range (Net_Info.shape[0]):
        y_aux=Net_Info[i,2].replace(",",".")
        y_aux=y_aux.replace("i","j")
        Y[Net_Info[i,0]-1,Net_Info[i,0]-1]=Y[Net_Info[i,0]-1,Net_Info[i,0]-1]+complex(y_aux)*networkFactor
        Y[Net_Info[i,1]-1,Net_Info[i,1]-1]=Y[Net_Info[i,1]-1,Net_Info[i,1]-1]+complex(y_aux)*networkFactor
        Y[Net_Info[i,0]-1,Net_Info[i,1]-1]=Y[Net_Info[i,0]-1,Net_Info[i,1]-1]-complex(y_aux)*networkFactor
        Y[Net_Info[i,1]-1,Net_Info[i,0]-1]=Y[Net_Info[i,1]-1,Net_Info[i,0]-1]-complex(y_aux)*networkFactor

                
    # Remover o barramento de referência da matriz de admitâncias            s            
    Yl=np.delete(Y, SlackBus-1, axis=0)
    Yl=np.delete(Yl, SlackBus-1, axis=1)

    # Matriz de Condutânciascias
    G=Yl.real

    # Matriz de Susceptânciasncias
    B=Yl.imag 

    print("A matriz de admitâncias Y é:\n", Y, "\n")
    print("A matriz de condutâncias G é\n", G, "\n")
    print("A matriz de susceptâncias B é\n",B, "\n")

    return Y, Yl, G, B, I, nBus

# ============================================================================================================================================
# Funções para definir a matriz de erros
# ============================================================================================================================================
def error_matrix_1(Y, Yl, I, nBus, time, timeForecast):
    np.random.seed(50)
   # e1=np.random.randn(time+timeForecast)*0.5  #Erros associados à Geração Eólica
   # e=np.random.randn(time+timeForecast)*0.25  #Erros associados à Injeção de Potência (Consumo) 


    e1 = [0.2878,   0.0145,   0.5846,  -0.0029,  -0.2718,  -0.1411,
        -0.2058,  -0.1793,  -0.9878,  -0.4926,  -0.1480,   0.7222,
        -0.3123,   0.4541,   0.9474,  -0.1584,   0.4692,   1.0173,
        -0.0503,   0.4684,  -0.3604,   0.4678,   0.3047,  -1.5098,
        -0.5515,  -0.5159,   0.3657,   0.7160,   0.1407,   0.5424,
        0.0409,   0.0450,   0.2365,  -0.3875,   1.4783,  -0.8487]

    e =  [ -0.0106,   0.0133,   0.2226,   0.2332,   0.1600,  -0.0578,
        -0.2293,  -0.2843,  -0.2732,  -0.1203,  -0.1757,  -0.1891,
        0.1541,  -0.0093,  -0.1691,   0.2211,  -0.4515,  -0.1786,
        -0.2031,  -0.3634,  -0.1105,  -0.1413,  -0.5900,  -0.1729,
        -0.0810,  -0.0023,  -0.0556,   0.1858,  -0.0324,  -0.1071,
        -0.0845,  -0.0743,  -0.0479,  -0.0870,  -0.1834,  -0.1432]


    #Criação da Matriz
    II=np.zeros((nBus-1,time+timeForecast), dtype=complex)
    i12=np.zeros(time+timeForecast)
    i1w=np.zeros(time+timeForecast)
 
    # Inicialização do processo de geração de dados
    II[:,0]=I                                      #Injeções de Potência
    v=1+np.dot(np.linalg.inv(Yl),I)
    i12[0]=np.absolute(np.dot(Y[0,1],v[0]-v[1]))   #Corrente I12 no período t=0
    i1w[0]=np.real(I[0])                           #Injeção no barramento 1 (Eólica) no período t=0

    # Processo de geração de dadosos
    for t in range(time+timeForecast-1):             
        II[:,t+1]=0.95*II[:,t]+e[t]                                 # Injeção de potência baseada em períodos anteriores e nos erros..
                                                                    # Os valores estão mais ou menos relacionados considerando
                                                                    # o valor de 0,95. Este valor pode mudar entre 0 e 1.  
        i1w[t+1]=0.75*i1w[t]+e1[t]                                  # Potência eólica baseada nos períodos anteriores  
        II[0,t+1]=i1w[t+1]+np.complex128(0,np.imag(II[0,t+1]))      # Adicionar a geração eólica
        v=1+np.dot(np.linalg.inv(Yl),II[:,t+1])                     # Calcular as tensões
        I12=np.dot(-Y[0,1],v[0]-v[1])                               # Calcular o fluxo de carga na linha 1-2 (Complexo)
        i12[t+1]=np.absolute(I12)*np.sign(np.real(I12))             # Calcular o fluxo de carga na linha 1-2 (RMS com sinal)

        
    print ('A injeção de potência no Barramento 1 é:\n',II[0,:])
    print ('\nO fluxo de potência na Linha 1-2 é:\n',i12)

    return II, i12, i1w

def error_matrix_2(Y, Yl, I, nBus, time, timeForecast):
    ee1 = [ 0.2878,   0.0145,   0.5846,  -0.0029,  -0.2718,  -0.1411,
        -0.2058,  -0.1793,  -0.9878,  -0.4926,  -0.1480,   0.7222,
        -0.3123,   0.4541,   0.9474,  -0.1584,   0.4692,   1.0173,
        -0.0503,   0.4684,  -0.3604,   0.4678,   0.3047,  -1.5098,
        -0.5515,  -0.5159,   0.3657,   0.7160,   0.1407,   0.5424,
            0.0409,   0.0450,   0.2365,  -0.3875,   1.4783,  -0.8487]

    ee =  [ 0.2226,  -0.2293,  -0.1757,  -0.1691,  -0.2031,  -0.5900,
        -0.0556,  -0.0845,  -0.1834,   0.2798,   0.1534,   0.0751,
        -0.1089,   0.3545,   0.0228,  -0.2139,   0.4409,   0.6044,
        -0.2187,  -0.1233,   0.0026,   0.4980,   0.3703,   0.0812,
            0.1183,   0.2486,  -0.0686,  -0.0727,  -0.0009,  -0.1180,
            0.2443,   0.6224,  -0.4600,  -0.3878,   0.4734,  -0.4050]

    #Criação da Matriz
    II=np.zeros((nBus-1,time+timeForecast), dtype=complex)
    i12=np.zeros(time+timeForecast)
    i1w=np.zeros(time+timeForecast)
 
    # Inicialização do processo de geração de dados
    II[:,0]=I                                      #Injeções de Potência
    v=1+np.dot(np.linalg.inv(Yl),I)
    i12[0]=np.absolute(np.dot(Y[0,1],v[0]-v[1]))   #Corrente I12 no período t=0
    i1w[0]=np.real(I[0])                           #Injeção no barramento 1 (Eólica) no período t=0

    # Processo de geração de dadosos
    for t in range(time+timeForecast-1):             
        II[:,t+1]=0.95*II[:,t]+ee[t]                                 # Injeção de potência baseada em períodos anteriores e nos erros..
                                                                    # Os valores estão mais ou menos relacionados considerando
                                                                    # o valor de 0,95. Este valor pode mudar entre 0 e 1.  
        i1w[t+1]=0.75*i1w[t]+ee1[t]                                  # Potência eólica baseada nos períodos anteriores  
        II[0,t+1]=i1w[t+1]+np.complex128(0,np.imag(II[0,t+1]))      # Adicionar a geração eólica
        v=1+np.dot(np.linalg.inv(Yl),II[:,t+1])                     # Calcular as tensões
        I12=np.dot(-Y[0,1],v[0]-v[1])                               # Calcular o fluxo de carga na linha 1-2 (Complexo)
        i12[t+1]=np.absolute(I12)*np.sign(np.real(I12))             # Calcular o fluxo de carga na linha 1-2 (RMS com sinal)

        
    print ('A injeção de potência no Barramento 1 é:\n',II[0,:])
    print ('\nO fluxo de potência na Linha 1-2 é:\n',i12)

    return II, i12, i1w

# ============================================================================================================================================
# Funções para OLS 
# ============================================================================================================================================
def OLS(i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False):

    # Define the OLS regression relating the Current I12 with the Pinjection I1. See Equation (30) in the lecture notes
    AA=np.ones((time,2))        #Vector Xt with ones
    AA[:,1]=i1w[0:time]         #Vector Xt with ones in first column and wind injection in column 2 
    AATransp=np.transpose(AA)
    beta=np.dot(np.dot(np.linalg.inv(np.dot(AATransp,AA)),AATransp),i12[0:time])  # Beta values
    print ("\nO valor dos Betas, usando OLS, são:")
    print("β₀ = {:.4f}, β₁ = {:.4f}".format(beta[0], beta[1]))

    # Definir os gráficos
    x = range(time)
    yy1 = i12[0:time]
    yy2 = i1w[0:time]
    rss_1=beta[0]+np.dot(beta[1],i1w[0:time])   #OLS regresion line
    yy3 = rss_1
    yy4 = i12[0:time]-beta[0]-np.dot(beta[1],i1w[0:time])

    #Primeiro Gráfico (Injeção de potência no barramento 1 e Corrente I12)to 1 e Corrente I12)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, yy2, 'g-')
    ax2.plot(x, yy1, 'b-')
    ax1.set_xlabel('Instante de Tempo [h]')
    ax1.set_ylabel('P_1 Injetada', color='g')
    ax2.set_ylabel('Corrente I_12', color='b')
    plt.xlabel("Instante de Tempo [h]") 
    plt.grid(True)
    plt.savefig('plots/OLS1.png')
    if show_plots:
        plt.show()
    else:
        plt.close()


    #Segundo Gráfico (Relação I1 vs I12 e regressão OLS)
    plt.plot(yy2, yy1, 'C1o', label='P1')
    plt.plot(yy2, yy3, label='Linha de regressão')
    plt.legend()
    plt.xlabel("Injeção P1")
    plt.ylabel("Corrente I12")
    plt.grid(True)
    plt.savefig('plots/OLS2.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    

    #Terceiro Gráfico (Resíduos - Diferença entre a corrente real I12 e a obtida pela regressão OLS)
    plt.plot(x, yy4, 'C0o', label='Resíduos')
    plt.legend()
    plt.xlabel("Instante de Tempo [h]")
    plt.ylabel("Resíduos")
    plt.grid(True)
    plt.savefig('plots/OLS3.png') 
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Calcular previsões para o período de teste
    I12f1 = beta[0] + np.dot(beta[1], i1w[time:time+timeForecast])
    print("Previsão da Corrente I12 considerando OLS:", I12f1)

    # Gráfico da Projeção para o fluxo de corrente I12
    forecast_range = range(time, time + timeForecast)
    plt.plot(forecast_range, i12[time:time+timeForecast], color='red', marker='o', label='I12 medido')
    plt.plot(forecast_range, I12f1, color='black', linestyle='--', marker='s', label='Previsão OLS')
    plt.xlabel("Intervalo de tempo [h]")
    plt.ylabel("Corrente I12")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/OLS4.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    while True:
        print("Deseja voltar ao menu principal? (s/n)")
        choice = input()
        if choice == 's':
            return beta, rss_1
        elif choice == 'n':
            exit()
        else:
            print("Escolha inválida. Tente novamente.")


# ============================================================================================================================================
# Funções para Cochrane-Orcutt (CO)
# ============================================================================================================================================
def CO(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False):
    # Definir a regressão Cochrane-Orcutt (CO)
    # Definir os resíduos

    # Cálculo do valor de Durdin-Watson (DW) e do valor de rho
    dw1 = np.sum((rss_1[1:time-1]-rss_1[0:time-2])**2)  # numerador 
    dw2 = np.sum((rss_1[0:time-2])**2)                  # denominador
    D = np.divide(dw1,dw2) 
    ro = 1-D/2
    print("O valor de Durdin-Watson (DW) é:", D)
    print("O valor de rho é: ", ro)

    res_1 = i12[0:time] - rss_1
    for k in range(3):                  #Iterações para estimar o valor de Beta e Rho
        r2 = res_1[0:time-1]
        r1 = res_1[1:time]
        ro = np.dot(np.dot((np.dot(np.transpose(r2),r2))**(-1),np.transpose(r2)),r1) #Estima Rho através de (28) 
        i1w_s = i1w[1:time] - np.dot(ro, i1w[0:time-1])   #Transforma yt*=yt
        i12_s = i12[1:time] - np.dot(ro, i12[0:time-1])   #Transforma xt*=Xt
        B = np.ones((time-1,2))
        B[:,1] = i1w_s
        b_s = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B),B)),np.transpose(B)),np.transpose(i12_s)) #Regressão de yt* por xt*
        
        b_s[0] = np.divide(b_s[0], 1-ro)               # Transforma Beta_0
        rss_s = b_s[0] + np.dot(b_s[1], i1w_s[0:time-1]) # Faz o update da regressão 
        rss_2 = b_s[0] + np.dot(b_s[1], i1w[0:time])  
        res_2 = i12[0:time] - rss_2
        res_1 = res_2[:]                 
    b_ss = b_s

    # Adicionar esta linha para mostrar os betas de CO
    print("O valor dos Betas, usando Cochrane-Orcutt, são:\n", b_ss)
    print("Comparação dos Betas:")
    print("OLS: β₀ = {:.4f}, β₁ = {:.4f}".format(beta[0], beta[1]))
    print("CO:  β₀ = {:.4f}, β₁ = {:.4f}".format(b_ss[0], b_ss[1]))

    I12f1=beta[0]+np.dot(beta[1],i1w[time:time+timeForecast]) #using Ordinary least Squares (OLS)
    I12f2=b_ss[0]+np.dot(b_ss[1],i1w[time:time+timeForecast]) #using Cochrane-Orcutt (CO)
    print ("Forecast Corrent I12 considering OLS:",I12f1,"\n")
    print ("Forecast Corrent I12 considering CO:",I12f2)

    x = range(time)
    xx = range(time-1)
    xxx = range(time, time+timeForecast)

    yy1 = i12[0:time]
    yy2 = i1w[0:time]
    yy3 = rss_1
    yy4 = rss_2
    yy5 = i12[0:time] - rss_1
    yy6 = i12[0:time-1] - rss_2[0:time-1]
    yy7 = i12[time:time+timeForecast]
    yy8 = I12f1
    yy9 = I12f2

    # Gráfico da Corrente I12
    plt.plot(xxx, yy7, color='red', label='Medido')
    plt.plot(xxx, yy8, color='black', linestyle='dashed', marker='o', label='OLS')
    plt.plot(xxx, yy9, color='red', linestyle='-.', marker='*', label='CO')
    plt.legend()
    plt.xlabel('Instante de tempo [h]')
    plt.ylabel('Corrente I12')
    plt.grid(True)
    plt.savefig('plots/CO1.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Gráfico de dispersão e reta de regressão
    plt.plot(yy2, yy1, 'C1o', label='i12')
    plt.plot(yy2, yy3, label='Regressão OLS')
    plt.plot(yy2, yy4, label='Regressão CO')
    plt.legend()
    plt.xlabel("Injeção P1")
    plt.ylabel("Corrente I12")
    plt.grid(True) 
    plt.savefig('plots/CO2.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Gráfico de resíduos
    plt.plot(x, yy5, 'C1o', label='Resíduos OLS')
    plt.plot(xx, yy6, 'C0*', label='Resíduos CO')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/CO3.png') 
    if show_plots:
        plt.show()
    else:
        plt.close()

    while True:
        print("Deseja voltar ao menu principal? (s/n)")
        choice = input()
        if choice == 's':
            return I12f1, I12f2
        elif choice == 'n':
            exit()
        else:
            print("Escolha inválida. Tente novamente.")

   
# ============================================================================================================================================
# Funções para Autorregressão AR
# ============================================================================================================================================
def AR(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False):
    # Para calcular I12: usar P1 + o valor anterior da corrente
    beta, rss_1 = OLS(i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False)
    I12f1, I12f2 = CO(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False)

    II, i12, i1w = error_matrix_2(Y, Yl, I, nBus, time, timeForecast)

    # Recalcular os resíduos e valores ajustados do CO para comparação
    dw1 = np.sum((rss_1[1:time-1]-rss_1[0:time-2])**2)  # numerador 
    dw2 = np.sum((rss_1[0:time-2])**2)                  # denominador
    D = np.divide(dw1,dw2) 
    ro = 1-D/2
    
    res_1 = i12[0:time] - rss_1
    for k in range(3):
        r2 = res_1[0:time-1]
        r1 = res_1[1:time]
        ro = np.dot(np.dot((np.dot(np.transpose(r2),r2))**(-1),np.transpose(r2)),r1)
        i1w_s = i1w[1:time] - np.dot(ro, i1w[0:time-1])
        i12_s = i12[1:time] - np.dot(ro, i12[0:time-1])
        B = np.ones((time-1,2))
        B[:,1] = i1w_s
        b_s = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B),B)),np.transpose(B)),np.transpose(i12_s))
        
        b_s[0] = np.divide(b_s[0], 1-ro)
        rss_s = b_s[0] + np.dot(b_s[1], i1w_s[0:time-1])
        rss_2 = b_s[0] + np.dot(b_s[1], i1w[0:time])
        res_2 = i12[0:time] - rss_2
        res_1 = res_2[:]
    b_ss = b_s
    
    # Valores ajustados e resíduos do CO
    co_fitted = rss_2
    co_residuals = i12[0:time] - co_fitted

    X = np.ones((time-1, 3))    # [1, i1w(t), i12(t-1)]
    X[:,1] = i1w[1:time]        # valores atuais de i1w
    X[:,2] = i12[0:time-1]      # valores anteriores de i12
    y = i12[1:time]             # valores atuais de i12
    
    # Estimar os parâmetros do modelo AR
    X_transp = X.T
    ar_params = np.linalg.inv(X_transp @ X) @ (X_transp @ y)
    print("\nO valor dos Betas, usando AR, são:")
    print("β₀ = {:.4f}, β₁ = {:.4f}, β₂ = {:.4f}".format(ar_params[0], ar_params[1], ar_params[2]))
    
    # Calcular os valores ajustados do modelo AR
    fitted = np.zeros(time)
    fitted[0] = i12[0]  # usar o valor inicial real
    for t in range(time-1):
        fitted[t+1] = ar_params[0] + ar_params[1]*i1w[t+1] + ar_params[2]*i12[t]
    
    # Calcular os resíduos
    ols_residuals = i12[0:time]-rss_1
    residuals = i12[1:time] - fitted[1:time]
    
    # Fazer previsões para o período de teste
    predictions = np.zeros(timeForecast)
    predictions[0] = ar_params[0] + ar_params[1]*i1w[time] + ar_params[2]*i12[time-1]
    for t in range(1, timeForecast):
        predictions[t] = ar_params[0] + ar_params[1]*i1w[time+t] + ar_params[2]*predictions[t-1]
    
    # Previsão OLS e CO para comparação
    I12f1 = beta[0] + np.dot(beta[1], i1w[time:time+timeForecast])
    I12f2 = b_ss[0] + np.dot(b_ss[1], i1w[time:time+timeForecast])

    # Preparar dados para gráficos
    x = range(time)
    xx = range(time-1)
    xxx = range(time, time+timeForecast)
    
    # Gráfico 1: Projeção de 12h para o fluxo de corrente I12 (período de teste)
    plt.figure(figsize=(10, 6))
    plt.plot(xxx, i12[time:time+timeForecast], color='red', marker='o', label='I12 medido')
    plt.plot(xxx, I12f1, color='black', linestyle='--', marker='s', label='Previsão OLS')
    plt.plot(xxx, I12f2, color='green', linestyle=':', marker='d', label='Previsão CO')
    plt.plot(xxx, predictions, color='blue', linestyle='-.', marker='*', label='Previsão AR')
    plt.xlabel("Intervalo de tempo [h]")
    plt.ylabel("Corrente I12")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/AR1.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Gráfico 3: Resíduos
    plt.figure(figsize=(10, 6))
    plt.scatter(x, ols_residuals, color='black', marker='o', label='Resíduos OLS', alpha=0.7)
    plt.scatter(x, co_residuals, color='green', marker='s', label='Resíduos CO', alpha=0.7)
    plt.scatter(xx, residuals, color='red', marker='x', label='Resíduos AR', alpha=0.7)
    plt.xlabel("Intervalo de tempo [h]")
    plt.ylabel("Resíduos")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/AR3.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Análise estatística dos modelos
    ols_mse = np.mean(ols_residuals**2)
    co_mse = np.mean(co_residuals**2)
    ar_mse = np.mean(residuals**2)
    
    print("\nAnálise estatística (treino):")
    print(f"MSE OLS: {ols_mse:.6f}")
    print(f"MSE CO:  {co_mse:.6f}")
    print(f"MSE AR:  {ar_mse:.6f}")
    
    # MSE para o período de teste
    test_mse_ols = np.mean((i12[time:time+timeForecast] - I12f1)**2)
    test_mse_co = np.mean((i12[time:time+timeForecast] - I12f2)**2)
    test_mse_ar = np.mean((i12[time:time+timeForecast] - predictions)**2)
    
    print("\nAnálise estatística (teste):")
    print(f"MSE OLS: {test_mse_ols:.6f}")
    print(f"MSE CO:  {test_mse_co:.6f}")
    print(f"MSE AR:  {test_mse_ar:.6f}")
        
    while True:
        print("Deseja voltar ao menu principal? (s/n)")
        choice = input()
        if choice == 's':
            return I12f1, predictions
        elif choice == 'n':
            exit()
        else:
            print("Escolha inválida. Tente novamente.")
  
    
# ============================================================================================================================================
# Funções para Autorregressão AR + Soma de Cargas
# ============================================================================================================================================
def AR_Soma(i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False):
    """
    Modelo autoregressivo que incorpora a soma das cargas como variável explicativa adicional.
    Compara os resultados com OLS, CO e AR standard.
    """
    # Obter resultados dos modelos OLS e CO para comparação
    beta, rss_1 = OLS(i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False)
    I12f1, I12f2 = CO(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False)

    II, i12, i1w = error_matrix_2(Y, Yl, I, nBus, time, timeForecast)
    
    # Calcular o modelo AR standard para comparação
    X_ar = np.ones((time-1, 3))    # [1, i1w(t), i12(t-1)]
    X_ar[:,1] = i1w[1:time]        # valores atuais de i1w
    X_ar[:,2] = i12[0:time-1]      # valores anteriores de i12
    y = i12[1:time]                # valores atuais de i12
    
    X_ar_transp = X_ar.T
    ar_params = np.linalg.inv(X_ar_transp @ X_ar) @ (X_ar_transp @ y)
    
    # Calcular valores ajustados do modelo AR standard
    fitted_ar = np.zeros(time)
    fitted_ar[0] = i12[0]
    for t in range(time-1):
        fitted_ar[t+1] = ar_params[0] + ar_params[1]*i1w[t+1] + ar_params[2]*i12[t]
    
    # Calcular previsões do modelo AR standard
    predictions_ar = np.zeros(timeForecast)
    predictions_ar[0] = ar_params[0] + ar_params[1]*i1w[time] + ar_params[2]*i12[time-1]
    for t in range(1, timeForecast):
        predictions_ar[t] = ar_params[0] + ar_params[1]*i1w[time+t] + ar_params[2]*predictions_ar[t-1]

    # Preparar os dados para o modelo AR + Soma das Cargas
    X = np.ones((time-1, 4))    # [1, i1w(t), i12(t-1), soma_cargas(t)]
    X[:,1] = i1w[1:time]        # valores atuais de i1w (potência eólica)
    X[:,2] = i12[0:time-1]      # valores anteriores de i12 (corrente)
    X[:,3] = np.sum(np.real(II[0:nBus-1,1:time]), axis=0)  # soma das cargas atuais
    
    sum_loads = np.sum(np.real(II[0:nBus-1,:]), axis=0)  # soma das cargas para todo o período
    
    # Estimar os parâmetros do modelo AR + Soma das Cargas
    X_transp = X.T
    ar_soma_params = np.linalg.inv(X_transp @ X) @ (X_transp @ y)
    
    print("\nParâmetros do modelo AR standard:")
    print("β₀ = {:.4f}, β₁ = {:.4f}, β₂ = {:.4f}".format(ar_params[0], ar_params[1], ar_params[2]))
    
    print("\nParâmetros do modelo AR(1) + Soma das Cargas:")
    print("β₀ = {:.4f}, β₁ = {:.4f}, β₂ = {:.4f}, β₃ = {:.4f}".format(
        ar_soma_params[0], ar_soma_params[1], ar_soma_params[2], ar_soma_params[3]))
    
    # Calcular os valores ajustados do modelo AR+Soma
    fitted = np.zeros(time)
    fitted[0] = i12[0]  # usar o valor inicial real
    for t in range(time-1):
        fitted[t+1] = ar_soma_params[0] + ar_soma_params[1]*i1w[t+1] + \
                      ar_soma_params[2]*i12[t] + ar_soma_params[3]*sum_loads[t+1]
    
    # Calcular os resíduos
    ols_residuals = i12[0:time] - rss_1
    ar_residuals = i12[1:time] - fitted_ar[1:time]
    ar_soma_residuals = i12[1:time] - fitted[1:time]
    
    # Fazer previsões para o período de teste
    predictions = np.zeros(timeForecast)
    predictions[0] = ar_soma_params[0] + ar_soma_params[1]*i1w[time] + \
                     ar_soma_params[2]*i12[time-1] + ar_soma_params[3]*sum_loads[time]
    
    for t in range(1, timeForecast):
        predictions[t] = ar_soma_params[0] + ar_soma_params[1]*i1w[time+t] + \
                         ar_soma_params[2]*predictions[t-1] + ar_soma_params[3]*sum_loads[time+t]
    
    # Preparar dados para gráficos
    x = range(time)
    xx = range(time-1)
    xxx = range(time, time+timeForecast)

    # Gráfico 1: Período de treino - valores medidos vs ajustados
    plt.figure(figsize=(10, 6))
    plt.plot(x, i12[0:time], 'bo', label='I12 medido', alpha=0.7)
    plt.plot(x, rss_1, 'g-', label='OLS ajustado', linewidth=2)
    plt.plot(x[1:], fitted_ar[1:], 'y--', label='AR ajustado', linewidth=1.5)
    plt.plot(x[1:], fitted[1:], 'r-', label='AR+Soma ajustado', linewidth=2)
    plt.xlabel("Intervalo de tempo [h]")
    plt.ylabel("Corrente I12")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/AR_Soma1.png')   
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Gráfico 2: Período de teste - previsões
    plt.figure(figsize=(10, 6))
    plt.plot(xxx, i12[time:time+timeForecast], 'ro', label='I12 medido', markersize=8)
    plt.plot(xxx, I12f1, 'g--', marker='s', label='Previsão OLS')
    plt.plot(xxx, I12f2, 'c-.', marker='d', label='Previsão CO')
    plt.plot(xxx, predictions_ar, 'y-', marker='^', label='Previsão AR')
    plt.plot(xxx, predictions, 'b-', marker='*', label='Previsão AR+Soma', linewidth=2)
    plt.xlabel("Intervalo de tempo [h]")
    plt.ylabel("Corrente I12")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/AR_Soma2.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Gráfico 3: Resíduos
    plt.figure(figsize=(10, 6))
    plt.scatter(x, ols_residuals, color='green', marker='o', label='Resíduos OLS', alpha=0.7)
    plt.scatter(xx, ar_residuals, color='orange', marker='^', label='Resíduos AR', alpha=0.7)
    plt.scatter(xx, ar_soma_residuals, color='blue', marker='x', label='Resíduos AR+Soma', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1)
    plt.xlabel("Intervalo de tempo [h]")
    plt.ylabel("Resíduos")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/AR_Soma3.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Gráfico 4: Cargas e injeção de potência eólica (período de treino)
    plt.figure(figsize=(12, 6))
    # Plotar a injeção de potência eólica (P1)
    plt.plot(x, i1w[:time], 'b-', linewidth=2, label='P1 (Eólica)')
    
    # Cores para os diferentes barramentos
    colors = ['g', 'r', 'm', 'c', 'y']
    
    # Plotar as cargas dos outros barramentos
    for i in range(1, nBus-1):
        if i < len(colors):
            color = colors[i]
        else:
            color = colors[i % len(colors)]
        plt.plot(x, np.real(II[i, :time]), f'{color}-', linewidth=1.5, 
                 label=f'P{i+1} (Carga)')
    
    # Plotar a soma das cargas
    plt.plot(x, sum_loads[:time], 'k--', linewidth=2, 
             label='Soma das cargas')
    
    plt.xlabel("Intervalo de tempo [h]")
    plt.ylabel("Potência injetada")
    plt.title("Cargas e injeção de potência eólica no período de treino")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/AR_Soma4.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Análise estatística dos modelos
    ols_mse = np.mean(ols_residuals**2)
    ar_mse = np.mean(ar_residuals**2)
    ar_soma_mse = np.mean(ar_soma_residuals**2)
    
    print("\nAnálise estatística (treino):")
    print(f"MSE OLS:      {ols_mse:.6f}")
    print(f"MSE AR:       {ar_mse:.6f}")
    print(f"MSE AR+Soma:  {ar_soma_mse:.6f}")
    
    # MSE para o período de teste
    test_mse_ols = np.mean((i12[time:time+timeForecast] - I12f1)**2)
    test_mse_co = np.mean((i12[time:time+timeForecast] - I12f2)**2)
    test_mse_ar = np.mean((i12[time:time+timeForecast] - predictions_ar)**2)
    test_mse_ar_soma = np.mean((i12[time:time+timeForecast] - predictions)**2)
    
    print("\nAnálise estatística (teste):")
    print(f"MSE OLS:      {test_mse_ols:.6f}")
    print(f"MSE CO:       {test_mse_co:.6f}")
    print(f"MSE AR:       {test_mse_ar:.6f}")
    print(f"MSE AR+Soma:  {test_mse_ar_soma:.6f}")
    print(f"Melhoria vs OLS:     {(1 - test_mse_ar_soma/test_mse_ols)*100:.2f}%")
    print(f"Melhoria vs AR:      {(1 - test_mse_ar_soma/test_mse_ar)*100:.2f}%")

    while True:
        print("Deseja voltar ao menu principal? (s/n)")
        choice = input()
        if choice == 's':
            return I12f1, predictions
        elif choice == 'n':
            exit()
        else:
            print("Escolha inválida. Tente novamente.")
    

# ============================================================================================================================================
# Função para Prais-Winsten (PW) - Desafio 2
# ============================================================================================================================================
def PW(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False):
    """
    Implementação do método Prais-Winsten para corrigir a autocorrelação dos resíduos.
    A diferença principal em relação ao Cochrane-Orcutt é que o Prais-Winsten preserva 
    a primeira observação, transformando-a de forma especial.
    """
    # Executar OLS independentemente para garantir dados limpos
    X = np.ones((time, 2))
    X[:, 1] = i1w[:time]
    y = i12[:time]
    
    # Ajuste OLS inicial
    X_t = X.T
    beta_ols = np.linalg.inv(X_t @ X) @ (X_t @ y)
    y_hat_ols = X @ beta_ols
    residuals_ols = y - y_hat_ols
    
    # Calcular estatística DW e estimar rho inicial
    dw_num = np.sum((residuals_ols[1:] - residuals_ols[:-1])**2)
    dw_den = np.sum(residuals_ols**2)
    dw = dw_num / dw_den
    rho = 1 - dw/2
    
    print("\nO valor de Durbin-Watson (DW) é:", dw)
    print("O valor inicial de rho é:", rho)
    
    # Garantir que rho esteja no intervalo válido
    rho = max(min(rho, 0.99), -0.99)
    
    # Inicializar parâmetros para iteração
    beta_pw = beta_ols.copy()
    max_iter = 3
    
    # Armazenar histórico de parâmetros
    betas_history = [beta_pw]
    rho_history = [rho]
    
    # Iterações do método Prais-Winsten
    for k in range(max_iter):
        print(f"\nIteração {k+1}:")
        print(f"rho = {rho:.4f}")
        
        # Criar matrizes para dados transformados
        X_pw = np.zeros_like(X)
        y_pw = np.zeros_like(y)
        
        # Transformação para t=1 (primeira observação - específico de Prais-Winsten)
        factor = np.sqrt(1 - rho**2)
        X_pw[0, :] = factor * X[0, :]
        y_pw[0] = factor * y[0]
        
        # Transformação para t=2,...,T (igual a Cochrane-Orcutt)
        for t in range(1, time):
            X_pw[t, :] = X[t, :] - rho * X[t-1, :]
            y_pw[t] = y[t] - rho * y[t-1]
        
        # Regressão com dados transformados
        X_pw_t = X_pw.T
        beta_pw = np.linalg.inv(X_pw_t @ X_pw) @ (X_pw_t @ y_pw)
        
        print(f"β₀ = {beta_pw[0]:.4f}, β₁ = {beta_pw[1]:.4f}")
        
        # Calcular valores ajustados para os dados originais
        y_hat_pw = X @ beta_pw
        
        # Calcular novos resíduos
        residuals_pw = y - y_hat_pw
        
        # Reestimar rho para próxima iteração usando os novos resíduos
        r_lagged = residuals_pw[:-1]
        r_current = residuals_pw[1:]
        
        # Calcular novo rho de maneira mais estável
        if np.sum(r_lagged**2) > 0:
            rho_new = np.sum(r_lagged * r_current) / np.sum(r_lagged**2)
            # Garantir que rho está no intervalo válido
            rho_new = max(min(rho_new, 0.99), -0.99)
        else:
            rho_new = 0
            
        # Verificar convergência
        if abs(rho_new - rho) < 0.01:
            print(f"Convergência atingida na iteração {k+1}")
            rho = rho_new
            break
            
        rho = rho_new
        betas_history.append(beta_pw)
        rho_history.append(rho)
    
    # Valores ajustados finais
    y_hat_pw = X @ beta_pw
    residuals_pw = y - y_hat_pw
    
    # Mostrar resultados da regressão
    print("\nResultados finais:")
    print("O valor dos Betas, usando Prais-Winsten, são:")
    print("β₀ = {:.4f}, β₁ = {:.4f}".format(beta_pw[0], beta_pw[1]))
    
    print("\nComparação dos Betas:")
    print("OLS: β₀ = {:.4f}, β₁ = {:.4f}".format(beta_ols[0], beta_ols[1]))
    print("PW:  β₀ = {:.4f}, β₁ = {:.4f}".format(beta_pw[0], beta_pw[1]))
    
    # Obter a previsão de OLS e CO para comparar (usando as funções existentes)
    I12f1 = beta_ols[0] + beta_ols[1] * i1w[time:time+timeForecast]  # OLS
    
    # Obter resultados do Cochrane-Orcutt (sem modificar os dados da função atual)
    # Chamar CO para obter os coeficientes e previsões
    _, I12f2 = CO(y_hat_ols, beta_ols, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False)
    
    # Vamos também calcular resíduos CO para comparação
    # Primeiro precisamos calcular os betas CO
    res_1 = i12[0:time] - y_hat_ols
    for k in range(3):
        r2 = res_1[0:time-1]
        r1 = res_1[1:time]
        ro_co = np.dot(np.dot((np.dot(np.transpose(r2),r2))**(-1),np.transpose(r2)),r1)
        i1w_s = i1w[1:time] - np.dot(ro_co, i1w[0:time-1])
        i12_s = i12[1:time] - np.dot(ro_co, i12[0:time-1])
        B = np.ones((time-1,2))
        B[:,1] = i1w_s
        b_co = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(B),B)),np.transpose(B)),np.transpose(i12_s))
        b_co[0] = np.divide(b_co[0], 1-ro_co)
        y_hat_co = b_co[0] + np.dot(b_co[1], i1w[0:time])
        res_1 = i12[0:time] - y_hat_co
    
    residuals_co = i12[0:time] - y_hat_co
    
    # Previsão usando Prais-Winsten
    I12f3 = beta_pw[0] + beta_pw[1] * i1w[time:time+timeForecast]  # PW
    
    print("\nPrevisão da Corrente I12 usando OLS:", I12f1)
    print("Previsão da Corrente I12 usando PW:", I12f3)
    
    # Preparar dados para gráficos
    x = range(time)
    xxx = range(time, time+timeForecast)
    
    # Valores reais e ajustados
    yy1 = i12[0:time]                      # valores reais de i12 no treino
    yy3 = y_hat_ols                        # valores ajustados OLS
    yy4 = y_hat_pw                         # valores ajustados PW
    yy5 = residuals_ols                    # resíduos OLS
    yy6 = residuals_pw                     # resíduos PW
    yy7 = i12[time:time+timeForecast]      # valores reais no teste
    
    # Gráfico 1: Previsões no período de teste
    plt.figure(figsize=(10, 6))
    plt.plot(xxx, yy7, 'ro-', label='I12 medido', linewidth=2)
    plt.plot(xxx, I12f1, 'k--', marker='o', label='OLS')
    plt.plot(xxx, I12f2, 'g-.', marker='^', label='CO')
    plt.plot(xxx, I12f3, 'b-', marker='s', label='PW')
    plt.legend()
    plt.xlabel('Instante de tempo [h]')
    plt.ylabel('Corrente I12')
    plt.title('Comparação das previsões: OLS, CO e PW')
    plt.grid(True)
    plt.savefig('plots/PW1.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    # Gráfico 2: Valores ajustados no período de treino
    plt.figure(figsize=(10, 6))
    plt.plot(x, yy1, 'ro', label='I12 medido')
    plt.plot(x, yy3, 'k--', label='OLS ajustado')
    plt.plot(x, y_hat_co, 'g-.', label='CO ajustado')
    plt.plot(x, yy4, 'b-', label='PW ajustado')
    plt.legend()
    plt.xlabel("Instante de tempo [h]")
    plt.ylabel("Corrente I12")
    plt.title('Valores ajustados: OLS, CO e PW')
    plt.grid(True)
    plt.savefig('plots/PW2.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Gráfico 3: Resíduos de todos os modelos
    plt.figure(figsize=(10, 6))
    plt.scatter(x, residuals_ols, color='black', marker='o', label='Resíduos OLS', alpha=0.7)
    plt.scatter(x, residuals_co, color='green', marker='^', label='Resíduos CO', alpha=0.7)
    plt.scatter(x, residuals_pw, color='blue', marker='s', label='Resíduos PW', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    plt.xlabel("Instante de tempo [h]")
    plt.ylabel("Resíduos")
    plt.title('Comparação dos resíduos: OLS, CO e PW')
    plt.grid(True)
    plt.savefig('plots/PW3.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # NOVO GRÁFICO: Função de Autocorrelação (ACF) dos resíduos para cada modelo
    max_lag = 10
    acf_vals_ols = []
    acf_vals_co = []
    acf_vals_pw = []
    
    # Calcular ACF para cada modelo
    for lag in range(max_lag+1):
        if lag == 0:
            acf_vals_ols.append(1.0)
            acf_vals_co.append(1.0)
            acf_vals_pw.append(1.0)
        else:
            # ACF para OLS
            if lag < len(residuals_ols):
                r_ols = np.corrcoef(residuals_ols[lag:], residuals_ols[:-lag])[0,1]
                acf_vals_ols.append(r_ols)
            
            # ACF para CO
            if lag < len(residuals_co):
                r_co = np.corrcoef(residuals_co[lag:], residuals_co[:-lag])[0,1]
                acf_vals_co.append(r_co)
            
            # ACF para PW
            if lag < len(residuals_pw):
                r_pw = np.corrcoef(residuals_pw[lag:], residuals_pw[:-lag])[0,1]
                acf_vals_pw.append(r_pw)
    
    # Plotar ACF
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 3, 1)
    plt.stem(range(len(acf_vals_ols)), acf_vals_ols, linefmt='k-', markerfmt='ko')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=1.96/np.sqrt(time), color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=-1.96/np.sqrt(time), color='b', linestyle='--', alpha=0.5)
    plt.title("ACF - Resíduos OLS")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelação")
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.stem(range(len(acf_vals_co)), acf_vals_co, linefmt='g-', markerfmt='go')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=1.96/np.sqrt(time), color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=-1.96/np.sqrt(time), color='b', linestyle='--', alpha=0.5)
    plt.title("ACF - Resíduos CO")
    plt.xlabel("Lag")
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.stem(range(len(acf_vals_pw)), acf_vals_pw, linefmt='b-', markerfmt='bo')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=1.96/np.sqrt(time), color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=-1.96/np.sqrt(time), color='b', linestyle='--', alpha=0.5)
    plt.title("ACF - Resíduos PW")
    plt.xlabel("Lag")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/PW4_ACF.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    # NOVO GRÁFICO: Estatísticas de Durbin-Watson para cada modelo
    # Calcular estatística DW para os outros modelos
    dw_co_num = np.sum((residuals_co[1:] - residuals_co[:-1])**2)
    dw_co_den = np.sum(residuals_co**2)
    dw_co = dw_co_num / dw_co_den
    
    dw_pw_num = np.sum((residuals_pw[1:] - residuals_pw[:-1])**2)
    dw_pw_den = np.sum(residuals_pw**2)
    dw_pw = dw_pw_num / dw_pw_den
    
    rho_ols = 1 - dw/2
    rho_co = 1 - dw_co/2
    rho_pw = 1 - dw_pw/2
    
    print(f"\nComparação da estatística Durbin-Watson (mais próximo de 2 = melhor):")
    print(f"DW OLS: {dw:.4f} (rho = {rho_ols:.4f})")
    print(f"DW CO:  {dw_co:.4f} (rho = {rho_co:.4f})")
    print(f"DW PW:  {dw_pw:.4f} (rho = {rho_pw:.4f})")
    
    plt.figure(figsize=(10, 6))
    models = ['OLS', 'Cochrane-Orcutt', 'Prais-Winsten']
    dw_values = [dw, dw_co, dw_pw]
    colors = ['black', 'green', 'blue']
    
    bars = plt.bar(models, dw_values, color=colors, alpha=0.7)
    plt.axhline(y=2.0, color='r', linestyle='--', label='Valor ideal (2.0)', alpha=0.8)
    
    # Adicionar valores
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('Comparação das Estatísticas de Durbin-Watson')
    plt.ylabel('Valor de DW')
    plt.ylim(0, max(dw_values) * 1.2)  # Ajustar escala
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('plots/PW5_DW.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Análise da qualidade das previsões
    mse_ols = np.mean((yy7 - I12f1)**2)
    mse_co = np.mean((yy7 - I12f2)**2)
    mse_pw = np.mean((yy7 - I12f3)**2)
    
    print("\nAnálise de erro quadrático médio (MSE) no teste:")
    print(f"MSE OLS: {mse_ols:.6f}")
    print(f"MSE CO:  {mse_co:.6f}")
    print(f"MSE PW:  {mse_pw:.6f}")
    
    # NOVO GRÁFICO: Comparação do MSE
    plt.figure(figsize=(10, 6))
    models = ['OLS', 'Cochrane-Orcutt', 'Prais-Winsten']
    mse_values = [mse_ols, mse_co, mse_pw]
    colors = ['black', 'green', 'blue']
    
    bars = plt.bar(models, mse_values, color=colors, alpha=0.7)
    
    # Adicionar valores
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{height:.6f}', ha='center', va='bottom')
    
    plt.title('Comparação do MSE na previsão')
    plt.ylabel('MSE')
    plt.ylim(0, max(mse_values) * 1.2)  # Ajustar escala
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/PW6_MSE.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    if mse_pw < mse_co:
        print(f"\nPrais-Winsten melhorou a previsão em {(1-mse_pw/mse_co)*100:.2f}% comparado a Cochrane-Orcutt")
    else:
        print(f"\nCochrane-Orcutt teve melhor desempenho que Prais-Winsten neste caso")
        
    print(f"Melhoria do PW em relação ao OLS: {(1-mse_pw/mse_ols)*100:.2f}%")
    
    # Gráfico de evolução do rho e betas ao longo das iterações
    if len(rho_history) > 1:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(range(len(rho_history)), rho_history, 'bo-', linewidth=2)
        plt.title('Convergência do rho no método Prais-Winsten')
        plt.xlabel('Iteração')
        plt.ylabel('Valor de rho')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        beta0_hist = [b[0] for b in betas_history]
        beta1_hist = [b[1] for b in betas_history]
        plt.plot(range(len(beta0_hist)), beta0_hist, 'ro-', label='β₀', linewidth=2)
        plt.plot(range(len(beta1_hist)), beta1_hist, 'go-', label='β₁', linewidth=2)
        plt.title('Convergência dos coeficientes')
        plt.xlabel('Iteração')
        plt.ylabel('Valor do coeficiente')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/PW7_convergence.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    while True:
        print("\nDeseja voltar ao menu principal? (s/n)")
        choice = input()
        if choice == 's':
            return I12f1, I12f3
        elif choice == 'n':
            exit()
        else:
            print("Escolha inválida. Tente novamente.")


# ============================================================================================================================================
# Função para Hildreth-Lu - Desafio 3
# ============================================================================================================================================
def Hildreth_Lu(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False):
    """
    Implementação do método Hildreth–Lu para corrigir a autocorrelação dos resíduos
    através de uma busca em grade para encontrar o valor ótimo de ρ que minimiza a SSR.
    
    Parâmetros:
      rss_1      - Valores ajustados pelo modelo OLS (usado apenas para comparação)
      beta       - Parâmetros do modelo OLS
      i12        - Série de corrente I12 (dados originais)
      i1w        - Série de injeção de potência eólica (dados originais)
      nBus       - Número total de barramentos da rede
      I, Y, Yl   - Variáveis relativas à rede (matriz de admitâncias, etc.)
      time       - Número de períodos de treino
      timeForecast - Número de períodos de teste
      show_plots - Flag para exibição dos gráficos
      
    Retorna:
      I12f1, I12f4 - Previsões para I12 usando OLS e Hildreth–Lu, respectivamente.
    """
    print("\n" + "="*80)
    print(" " * 30 + "DESAFIO 3 - HILDRETH-LU")
    print("="*80 + "\n")
    
    # 1. Ajuste OLS inicial (dados de treino)
    X = np.ones((time, 2))
    X[:, 1] = i1w[:time]
    y = i12[:time]
    
    beta_ols = np.linalg.inv(X.T @ X) @ (X.T @ y)
    y_hat_ols = X @ beta_ols
    residuals_ols = y - y_hat_ols
    
    # 2. Estimar ρ inicial via Durbin-Watson
    dw_num = np.sum((residuals_ols[1:] - residuals_ols[:-1])**2)
    dw_den = np.sum(residuals_ols**2)
    dw = dw_num / dw_den
    rho_initial = 1 - dw/2
    print("Valor de Durbin-Watson (DW):", dw)
    print("Valor inicial de ρ:", rho_initial)
    
    # Garantir que ρ esteja no intervalo [-0.95, 0.95]
    rho_initial = max(min(rho_initial, 0.95), -0.95)
    
    # 3. Definir intervalo de busca para ρ centralizado no valor inicial
    rho_min = max(-0.95, rho_initial - 0.5)
    rho_max = min(0.95, rho_initial + 0.5)
    rho_grid = np.linspace(rho_min, rho_max, 100)
    
    # Variáveis para armazenar o melhor ajuste
    min_ssr = float('inf')
    best_rho = 0
    best_beta = np.zeros(2)
    all_ssr = []  # Para plotar a curva da SSR
    
    print("\nIniciando busca em grade para o ρ ótimo...")
    for rho in rho_grid:
        # Transformação dos dados: construir X* e y*
        X_hl = np.zeros_like(X)
        y_hl = np.zeros_like(y)
        
        # Para t=0 (primeira observação) – transformação especial
        factor = np.sqrt(1 - rho**2)
        X_hl[0, :] = factor * X[0, :]
        y_hl[0] = factor * y[0]
        
        # Para t >= 1
        for t in range(1, time):
            X_hl[t, :] = X[t, :] - rho * X[t-1, :]
            y_hl[t] = y[t] - rho * y[t-1]
        
        # Ajustar regressão OLS nos dados transformados
        beta_hl = np.linalg.inv(X_hl.T @ X_hl) @ (X_hl.T @ y_hl)
        y_hat_hl = X @ beta_hl  # Voltar à escala original
        residuals_hl = y - y_hat_hl
        ssr = np.sum(residuals_hl**2)
        all_ssr.append(ssr)
        
        if ssr < min_ssr:
            min_ssr = ssr
            best_rho = rho
            best_beta = beta_hl.copy()
    
    print(f"Busca concluída: ρ ótimo = {best_rho:.4f}, SSR minimizado = {min_ssr:.6f}")
    
    # 4. Recalcular os valores ajustados e resíduos com os parâmetros ótimos
    X_hl = np.zeros_like(X)
    y_hl = np.zeros_like(y)
    factor = np.sqrt(1 - best_rho**2)
    X_hl[0, :] = factor * X[0, :]
    y_hl[0] = factor * y[0]
    for t in range(1, time):
        X_hl[t, :] = X[t, :] - best_rho * X[t-1, :]
        y_hl[t] = y[t] - best_rho * y[t-1]
    best_beta = np.linalg.inv(X_hl.T @ X_hl) @ (X_hl.T @ y_hl)
    y_hat_hl = X @ best_beta
    residuals_hl = y - y_hat_hl
    
    print("\nParâmetros finais obtidos com Hildreth-Lu:")
    print("β₀ = {:.4f}, β₁ = {:.4f}".format(best_beta[0], best_beta[1]))
    print("\nComparação com OLS:")
    print("OLS: β₀ = {:.4f}, β₁ = {:.4f}".format(beta_ols[0], beta_ols[1]))
    
    # 5. Obter previsões para o período de teste
    # Para efeito de comparação, usamos a mesma abordagem para previsão:
    I12f_OLS = beta_ols[0] + beta_ols[1] * i1w[time:time+timeForecast]
    I12f_HL  = best_beta[0] + best_beta[1] * i1w[time:time+timeForecast]
    
    print("\nPrevisão da Corrente I12:")
    print("OLS:", I12f_OLS)
    print("Hildreth-Lu:", I12f_HL)
    
    # 6. Calcular a estatística Durbin-Watson para os resíduos transformados (HL)
    dw_hl = np.sum((residuals_hl[1:] - residuals_hl[:-1])**2) / np.sum(residuals_hl**2)
    print(f"\nEstatística Durbin-Watson para os resíduos HL: {dw_hl:.4f} (valor ideal: 2.0)")
    print(f"Valor implícito de ρ nos resíduos HL: {1 - dw_hl/2:.4f}")
    
    # 7. Comparação com os outros métodos (chamadas para CO e PW, se existirem)
    # Aqui assumimos que as funções CO e PW já estão implementadas e não interferem
    # na execução deste método.
    # Por exemplo:
    _, I12f_CO = CO(y_hat_ols, beta_ols, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False)
    _, I12f_PW = PW(y_hat_ols, beta_ols, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots=False)
    
    # Calcular MSE para cada método no período de teste
    actual_test = i12[time:time+timeForecast]
    mse_OLS = np.mean((actual_test - I12f_OLS)**2)
    mse_CO  = np.mean((actual_test - I12f_CO)**2)
    mse_PW  = np.mean((actual_test - I12f_PW)**2)
    mse_HL  = np.mean((actual_test - I12f_HL)**2)
    
    print("\nMSE no teste:")
    print(f"OLS: {mse_OLS:.6f}")
    print(f"CO:  {mse_CO:.6f}")
    print(f"PW:  {mse_PW:.6f}")
    print(f"HL:  {mse_HL:.6f}")
    
    # Comparar os métodos
    best_method = min([(mse_OLS, "OLS"), (mse_CO, "CO"), (mse_PW, "PW"), (mse_HL, "HL")], key=lambda x: x[0])
    print(f"\nO melhor método é: {best_method[1]} com MSE = {best_method[0]:.6f}")
    
    # 8. Visualizações
    x_train = range(time)
    x_test  = range(time, time+timeForecast)
    
    # Gráfico: Previsões no período de teste
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, actual_test, 'ro-', label='I12 medido', linewidth=2)
    plt.plot(x_test, I12f_OLS, 'k--', marker='o', label='OLS')
    plt.plot(x_test, I12f_CO, 'g-.', marker='^', label='CO')
    plt.plot(x_test, I12f_PW, 'b-', marker='s', label='PW')
    plt.plot(x_test, I12f_HL, 'm:', marker='*', label='HL', linewidth=2)
    plt.xlabel('Tempo [h]')
    plt.ylabel('Corrente I12')
    plt.title('Comparação das Previsões: OLS, CO, PW e HL')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/HL1.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Gráfico: Evolução da SSR em função de ρ
    plt.figure(figsize=(10, 6))
    plt.plot(rho_grid, all_ssr, 'b-', linewidth=2)
    plt.axvline(x=best_rho, color='r', linestyle='--')
    plt.text(best_rho+0.02, min(all_ssr)+0.1, f'ρ ótimo = {best_rho:.4f}', fontsize=12)
    plt.title('SSR vs. Valor de ρ')
    plt.xlabel('Valor de ρ')
    plt.ylabel('SSR')
    plt.grid(True)
    plt.savefig('plots/HL3_SSR.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Gráfico: Comparação do MSE entre métodos
    plt.figure(figsize=(10, 6))
    methods = ['OLS', 'CO', 'PW', 'HL']
    mse_values = [mse_OLS, mse_CO, mse_PW, mse_HL]
    colors = ['black', 'green', 'blue', 'magenta']
    bars = plt.bar(methods, mse_values, color=colors, alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                 f'{height:.6f}', ha='center', va='bottom')
    plt.title('Comparação do MSE na Previsão')
    plt.ylabel('MSE')
    plt.ylim(0, max(mse_values)*1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/HL4_MSE.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    while True:
        print("\nDeseja voltar ao menu principal? (s/n)")
        choice = input().lower().strip()
        if choice == 's':
            return I12f_OLS, I12f_HL
        elif choice == 'n':
            exit()
        else:
            print("Opção inválida. Tente novamente.")


# ============================================================================================================================================
# Função Principal
# ============================================================================================================================================
def show_menu():
    # Pretende exebir o menu principal?
    print("Preciona ENTER para exibir o menu principal.")
    input()

    """Exibe o menu principal e retorna a opção selecionada pelo usuário."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa a tela
    print("=" * 80)
    print("                 LABORATÓRIO 4 - Regressão sobre Previsão Autoregressiva                 ")
    print("=" * 80)
    print("\nEscolha uma opção (essencial que seja por ordem):")
    print("0 - Sair")
    print("1 - Regressão OLS")
    print("2 - Cochrane-Orcutt (CO)")
    print("3 - Autorregressão (AR)")
    print("4 - Autorregressão + Soma de Cargas")
    print("5 - Desafio 1: Prais-Winsten (PW)")
    print("6 - Desafio 2: Hildreth-Lu (HL)")
    print("\nAdicione 'p' após o número para mostrar os gráficos (ex: 1p)")
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
    # Carregar e processar os dados
    SlackBus, Net_Info, Power_Info, Power_Test = load_data('DASG_Prob2_new.xlsx')

    # Preparar a rede
    Y, Yl, G, B, I, nBus = prepare_network(Net_Info, Power_Info, SlackBus)

    # Definir a matriz de erros
    II, i12, i1w = error_matrix_1(Y, Yl, I, nBus, time, timeForecast)   

    rss_1 = 0
    beta = np.zeros(2)

    while True:
        option, show_plots = show_menu()
        if option == 0:
            break

        elif option == 1:    
            # OLS
            beta, rss_1 = OLS(i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots)  

        elif option == 2:
            # Cochrane-Orcutt (CO)
            I12f1, I12f2 = CO(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots)

        elif option == 3:
            # AR
            I12f1, predictions = AR(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots)
        
        elif option == 4:
            # AR + Soma de Cargas
            I12f1, I12_pred = AR_Soma(i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots)

        elif option == 5:
            # Desafio 2
            I12f1, I12f3 = PW(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots)

        elif option == 6:
            # Desafio 3
            I12f1, I12f4 = Hildreth_Lu(rss_1, beta, i12, i1w, nBus, I, Y, Yl, time, timeForecast, show_plots)
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()