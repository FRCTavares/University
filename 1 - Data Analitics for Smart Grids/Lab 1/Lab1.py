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

# NOTAS: Temos de fazer um report - entrega na 2a/3a feira

import pandas as pd
import numpy as np
from numpy.random import randint  # To random values in the phases
from numpy.random import random   # To random values in the phases
import matplotlib.pyplot as plt

nc=4                        # Number of consumers (1 to nc)                   %%Data Notes: nc=4
ts=60                       # start period of analysis (Can be from 1 to 96)  %%Data Notes: ts=60
te=71                       # Last period of analysis (Can be from 1 to 96)   %%Data Notes: te=71
#phase = [3,2,1,3]           # To obtain the same values of lecture notes
noise = 0

phase = randint(1, 4, nc)  # To obtain random values
print ("The distribution of consumers in each phase is:\n", phase)


# Import data (from Excel file)
raw_data = np.array(pd.read_excel ('Prob1_Conso_Data.xlsx', header=None))


# Delete zeros and Organize by consumers
checks=0
nr=1
data=np.zeros((1,96))
#h=np.arange(1/96, 1, 1/96).tolist()
h=raw_data[0:96,0]
for i in range(1,raw_data.shape[0]+1):
    if i==0:
        print(i)
    if raw_data[i-1,0]==h[checks]:
        checks=checks+1
    else:
        checks=0
    if checks==96:
        if np.sum(raw_data[i-96:i,1])!=0:
            data[nr-1,0:96]=raw_data[i-96:i,1]
            data.resize((nr+1,96))
            nr=nr+1
        checks=0
data.resize((nr-1,96))


# Create Matriz X
data_Aux1=data[0:nc,:]
pw=data_Aux1[:,ts-1:te]

##############################################################################################################
# Create Y Matrix - phase = [3,2,1,3]
# Y = [ X[t,3]  X[t,2]  X[t,1]+X[t,4] ]
#X = np.transpose(4*pw)
#Y = X[:, 2].reshape(-1, 1)                                              # Primeira coluna de Y
#Y = np.concatenate((Y, X[:, 1].reshape(-1, 1)), axis=1)                 # Segunda coluna de Y
#Y = np.concatenate((Y, (X[:, 0] + X[:, 3]).reshape(-1, 1)), axis=1)     # Terceira coluna de Y
##############################################################################################################

# Create Y Matrix - phase = random
X = np.transpose(4*pw)
# Y matrix will be of size (X.shape[0], 3)
Y = np.zeros((X.shape[0], 3))

for i in range(nc):
    # Subtract 1 to convert phase values (1..3) to zero-based column indexes
    phase_index = phase[i] - 1
    Y[:, phase_index] += X[:, i]


# Noise
Y = Y + np.random.normal(0, 0.01, Y.shape)


# Beta Matrix
beta = np.linalg.inv(X.T @ X) @ X.T @ Y

# Redefine beta as a binary matrix 
beta = np.where(beta > 0.5, 1, 0)

# Write beta as a vector
phase_estimated = np.argmax(beta, axis=1) + 1  # Adding 1 to get 1-based index

# Compare beta with the phase vector
print("\nO vetor beta estimado:\n", phase_estimated)
print("\nO vetor de fase inicial:\n", phase)

################################################################################################################
#
#                                              MUDAR O FORMATO GRAFICOS!
#
################################################################################################################

# Plots of the graphs
time_intervals = np.arange(ts, te + 1)

# Example data (replace with your actual data)
# X = np.random.rand(12, 4)  # Example: 12 periods for 4 customers (replace with actual data)
# Y = np.random.rand(12, 3)  # Example: 12 periods for 3 phases (replace with actual data)

# Create a single figure with two subplots side by side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Left subplot for X
for i in range(X.shape[1]):
    ax[0].step(time_intervals, X[:, i], where='post', label=f'Customer {i+1}')
ax[0].set_title('Customer Readings')
ax[0].set_xlabel('Time Stamp [15min]')
ax[0].set_ylabel('Power [kW]')
ax[0].legend()
ax[0].grid(True)

# Right subplot for Y
for i in range(Y.shape[1]):
    ax[1].step(time_intervals, Y[:, i], where='post', label=f'Phase {i+1}')
ax[1].set_title('Per-Phase Totals')
ax[1].set_xlabel('Time Stamp [15min]')
ax[1].set_ylabel('Power [kW]')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show() 
