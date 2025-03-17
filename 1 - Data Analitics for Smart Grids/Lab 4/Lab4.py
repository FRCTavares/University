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
    ax1.set_ylim([-2, 2])
    ax2.set_ylim([-1, 1])
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
# Funções para Processo Estacionário - Desafio 1
# ============================================================================================================================================
def processo_estacionario(II, Y, Yl, I, nBus, time, timeForecast, show_plots=False):
    """
    Desafio Extra:
      - Usar P2 para definir o mesmo processo estacionário para todas as cargas (barramentos 2..nBus-1).
      - Usar os dados de x_t (eólica) como no enunciado para W_t (barramento 1).
      - Calcular I12 com Y^-1.
      - Ajustar AR(1) nos primeiros 12 períodos e prever os próximos 12.
      - Analisar a autocorrelação dos resíduos e aplicar Cochrane-Orcutt.
      - Comparar as previsões (AR vs. CO). Verificar se houve melhora.
    """

    print("\n" + "="*80)
    print(" " * 30 + "DESAFIO - PROCESSO ESTACIONÁRIO")
    print("="*80 + "\n")

    # -------------------------------------------------------------------------
    # (1) Criar um processo estacionário para as cargas
    # -------------------------------------------------------------------------
    # Copiar a matriz de injeções original (II)
    II_stationary = II.copy()

    # Pegar o perfil de carga do barramento 2 (índice 1 na matriz)
    p2_profile = np.real(II[1, :])  # parte real do barramento 2
    np.random.seed(42)  # fixar semente para reprodutibilidade

    print("Criando processo estacionário em que todas as cargas seguem o perfil de P2.")
    for i in range(1, nBus-1):
        if i != 1:  # se não for o próprio barramento 2
            # Definir fator aleatório [0.8..1.2] (por ex.)
            factor = 0.8 + 0.4 * np.random.rand()
            # Aplicar o perfil do barramento 2
            load_real = p2_profile * factor
            load_imag = np.imag(II_stationary[i, :])  # mantém parte imaginária original (caso queira)
            II_stationary[i, :] = load_real + 1j*load_imag
            print(f"Barramento {i+1} -> fator={factor:.2f}")

    # Barramento 1 (eólico) permanece inalterado (i.e. i1w).
    i1w = np.real(II_stationary[0, :])

    # -------------------------------------------------------------------------
    # (2) Calcular I12(t) usando Y^-1 com as novas cargas
    # -------------------------------------------------------------------------
    i12_stationary = np.zeros(time + timeForecast)
    for t in range(time + timeForecast):
        # Tensões
        v = 1 + np.dot(np.linalg.inv(Yl), II_stationary[:, t])
        # Corrente I12
        I12_complex = np.dot(-Y[0, 1], v[0] - v[1])
        i12_stationary[t] = np.abs(I12_complex)*np.sign(np.real(I12_complex))

    # -------------------------------------------------------------------------
    # (3) Ajustar AR(1) nos primeiros 12 períodos e (4) prever +12
    # -------------------------------------------------------------------------
    train_periods = 12
    test_periods = 12

    i12_train = i12_stationary[:train_periods]
    i12_test  = i12_stationary[train_periods:train_periods+test_periods]

    print(f"\nTreinando com os primeiros {train_periods} períodos, prevendo {test_periods} seguintes...")

    # Montar matriz para AR(1): i12(t) = alpha0 + alpha1*i12(t-1)
    X_ar = np.ones((train_periods-1, 2))
    X_ar[:, 1] = i12_train[:-1]
    y_ar = i12_train[1:]

    # OLS
    ar_params = np.linalg.inv(X_ar.T @ X_ar) @ (X_ar.T @ y_ar)
    alpha0, alpha1 = ar_params
    print(f"\nModelo AR(1): i12(t) = {alpha0:.4f} + {alpha1:.4f} * i12(t-1)")

    # Previsão AR(1) no treino
    ar_fitted = np.zeros(train_periods)
    ar_fitted[0] = i12_train[0]
    for t in range(1, train_periods):
        ar_fitted[t] = alpha0 + alpha1*i12_train[t-1]

    # Resíduos no treino
    ar_residuals = i12_train - ar_fitted

    # Previsão AR(1) no teste
    ar_preds = np.zeros(test_periods)
    # usar o último valor do treino como inicial
    ar_preds[0] = alpha0 + alpha1*i12_train[-1]
    for t in range(1, test_periods):
        ar_preds[t] = alpha0 + alpha1*ar_preds[t-1]

    # -------------------------------------------------------------------------
    # (5) Analisar autocorrelação dos resíduos e aplicar Cochrane-Orcutt
    # -------------------------------------------------------------------------
    # Estatística Durbin-Watson
    dw_num = np.sum((ar_residuals[1:] - ar_residuals[:-1])**2)
    dw_den = np.sum(ar_residuals**2)
    dw = dw_num / dw_den
    rho = 1 - dw/2

    print(f"\nAnálise de Autocorrelação (Durbin-Watson): DW={dw:.4f}, rho~={rho:.4f}")

    # Se a autocorrelação for significativa, aplicamos Cochrane-Orcutt
    co_alpha0, co_alpha1 = alpha0, alpha1  # default (caso não seja necessário)
    if abs(dw - 2) > 0.5:  # limiar arbitrário para "significativo"
        print("Autocorrelação significativa detectada. Aplicando Cochrane-Orcutt...")
        # Transformar as séries
        # i12^*(t) = i12(t) - rho*i12(t-1)
        # Para t=1..(train_periods-1)
        i12_star = i12_train[1:] - rho*i12_train[:-1]

        # Montar B
        B = np.ones((train_periods-1, 2))
        B[:,1] = i12_star[:-1]  # i12^*(t-1)
        y_co = i12_star[1:]     # i12^*(t)

        # Reajustar
        co_params = np.linalg.inv(B.T @ B) @ (B.T @ y_co)
        co_alpha0, co_alpha1 = co_params
        # Ajustar intercepto
        co_alpha0 = co_alpha0/(1-rho)

        print(f"Parâmetros Cochrane-Orcutt: α0={co_alpha0:.4f}, α1={co_alpha1:.4f}")

    # Previsão CO no teste
    co_preds = np.zeros(test_periods)
    co_preds[0] = co_alpha0 + co_alpha1*i12_train[-1]
    for t in range(1, test_periods):
        co_preds[t] = co_alpha0 + co_alpha1*co_preds[t-1]

    # -------------------------------------------------------------------------
    # Comparar previsões AR vs. CO
    # -------------------------------------------------------------------------
    mse_ar = np.mean((i12_test - ar_preds)**2)
    mse_co = np.mean((i12_test - co_preds)**2)

    print(f"\nMSE no teste:")
    print(f"AR(1) = {mse_ar:.6f}")
    print(f"CO    = {mse_co:.6f}")
    if mse_co < mse_ar:
        print(f"Cochrane-Orcutt melhorou a previsão em ~{(1 - mse_co/mse_ar)*100:.2f}%.")
    else:
        print("Cochrane-Orcutt não melhorou a previsão (ou piorou).")

    # -------------------------------------------------------------------------
    # (6) Visualizações
    # -------------------------------------------------------------------------
    x_train = np.arange(train_periods)
    x_test  = np.arange(train_periods, train_periods+test_periods)

    # Plot 1: Sinais de carga
    plt.figure(figsize=(12,5))
    for i in range(nBus-1):
        plt.plot(np.real(II_stationary[i,:]), label=f'Barramento {i+1}')
    plt.axvline(train_periods-0.5, color='k', linestyle='--', alpha=0.7)
    plt.title("Perfil de Cargas Estacionárias (baseadas em P2)")
    plt.xlabel("Tempo [h]")
    plt.ylabel("Potência Real")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Estacionario_Loads.png')
    if show_plots: plt.show()
    else: plt.close()

    # Plot 2: Sinal i12 e previsões
    plt.figure(figsize=(12,5))
    plt.plot(x_train, i12_train, 'bo-', label='i12 (treino)')
    plt.plot(x_test, i12_test, 'ro-', label='i12 (teste)')
    plt.plot(x_test, ar_preds, 'g--s', label='AR(1)')
    plt.plot(x_test, co_preds, 'm--d', label='Cochrane-Orcutt')
    plt.title("Previsão de i12 (Estacionário)")
    plt.xlabel("Tempo [h]")
    plt.ylabel("Corrente i12")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Estacionario_Preds.png')
    if show_plots: plt.show()
    else: plt.close()

    # Plot 3: Autocorrelação Resíduos AR no treino
    plt.figure(figsize=(12,4))
    ar_res = ar_residuals[1:]  # ignorar a 1ª amostra, pois foi seed
    max_lag = min(10, len(ar_res)-1)
    acf_vals = []
    for lag in range(max_lag+1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            r = np.corrcoef(ar_res[lag:], ar_res[:-lag])[0,1]
            acf_vals.append(r)
    plt.stem(range(max_lag+1), acf_vals, linefmt='k-', markerfmt='ko')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title("Função de Autocorrelação dos Resíduos (AR) - Treino")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.grid(True)
    plt.savefig('plots/Estacionario_ACF.png')
    if show_plots: plt.show()
    else: plt.close()

    while True:
        print("\nDeseja voltar ao menu principal? (s/n)")
        choice = input().lower().strip()
        if choice == 's':
            # Retorna as previsões AR e CO e a série de teste
            return ar_preds, co_preds, i12_test
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
    print("5 - Desafio 1: Processo Estacionário")
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
            # Desafio 1
            ar_preds, co_preds, actual = processo_estacionario(II, Y, Yl, I, nBus, time, timeForecast, show_plots)

        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()