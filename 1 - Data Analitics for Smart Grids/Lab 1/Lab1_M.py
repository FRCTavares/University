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
import seaborn as sns
import numpy as np
from numpy.random import randint  # To random values in the phases
from numpy.random import random   # To random values in the phases
import matplotlib.pyplot as plt

nc=4                        # Number of consumers (1 to nc)                   %%Data Notes: nc=4
ts=60                       # start period of analysis (Can be from 1 to 96)  %%Data Notes: ts=60
te=71                       # Last period of analysis (Can be from 1 to 96)   %%Data Notes: te=71
phase = [3,2,1,3]           # To obtain the same values of lecture notes
noise = 0

#phase = randint(1, 4, nc)  # To obtain random values
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
X = np.transpose(4*pw)

# Create Y Matrix - phase = [3,2,1,3]
# Y = [ X[t,3]  X[t,2]  X[t,1]+X[t,4] ]
Y = X[:, 2].reshape(-1, 1)                                              # Primeira coluna de Y
Y = np.concatenate((Y, X[:, 1].reshape(-1, 1)), axis=1)                 # Segunda coluna de Y
Y = np.concatenate((Y, (X[:, 0] + X[:, 3]).reshape(-1, 1)), axis=1)     # Terceira coluna de Y

# Create Y Matrix - phase = random
# X = np.transpose(4*pw)


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

###############################################################################################
#                                                                                             #
#                                  MUDAR O FORMATO GRAFICOS!                                  #
#                                                                                             #
###############################################################################################

# Plots of the graphs
# Define time intervals (12 periods, assuming data is for 12 time periods)
time_intervals = np.arange(1, 13)  # 12 time periods from 1 to 12

# Example data (replace with your actual data)
# X = np.random.rand(12, 4)  # Example: 12 periods for 4 customers (replace with actual data)
# Y = np.random.rand(12, 3)  # Example: 12 periods for 3 phases (replace with actual data)

# Plot the customer readings
plt.figure(figsize=(10, 6))
width = 0.2  # Bar width (narrower to avoid overlap)

# Loop through each customer (column in X) and plot their data
for i in range(X.shape[1]):  # Loop through each customer (column in X)
    # Plot each customer's data without shifting bar positions
    plt.bar(time_intervals + (i - 1.5) * width, X[:, i], width=width, label=f'Customer {i+1}', align='center')

plt.title('Customer Readings')
plt.xlabel('Time Stamp [15min]')
plt.ylabel('Power [kW]')
plt.legend()
plt.grid(True)
plt.show()

# Plot the per-phase totals
plt.figure(figsize=(10, 6))

# Loop through each phase total (column in Y) and plot their data
for i in range(Y.shape[1]):  # Loop through each phase total (column in Y)
    # Plot each phase's total using the same time intervals, ensuring no negative bars
    plt.bar(time_intervals + (i - 1.5) * width, Y[:, i], width=width, label=f'Phase {i+1}', align='center')

plt.title('Per-Phase Totals')
plt.xlabel('Time Stamp [15min]')
plt.ylabel('Power [kW]')
plt.legend()
plt.grid(True)
plt.show()



###############################################################################################
#                                                                                             #
#                                       EXTRA Challenges                                      #
#                                                                                             #
###############################################################################################

def mean_absolute_difference(X, threshold=0.1):
    num_customers = X.shape[1]
    for i in range(num_customers):
        for j in range(i+1, num_customers):
            mad = np.mean(np.abs(X[:, i] - X[:, j]))
            if mad < threshold:
                print(f"Customers {i+1} and {j+1} have very similar consumption (MAD = {mad:.4f})")

# Call the function with your X data
mean_absolute_difference(X, threshold=0.05)

# Assuming X is structured so that each customer has three consecutive columns (one per phase)
num_customers = X.shape[1] // 3    # Since each customer has 3 phases

# Compute per-phase totals
Y_new = np.zeros((X.shape[0], 3))  # 3 phases

for i in range(num_customers):
    Y_new[:, 0] += X[:, i*3]       # Phase A
    Y_new[:, 1] += X[:, i*3 + 1]   # Phase B
    Y_new[:, 2] += X[:, i*3 + 2]   # Phase C

# Now Y_new contains the per-phase total power considering three-phase clients
time_intervals = [f"T{i}" for i in range(Y_new.shape[0])]
plt.figure(figsize=(10, 6))
sns.heatmap(Y_new.T, cmap="coolwarm", xticklabels=time_intervals, yticklabels=["Phase A", "Phase B", "Phase C"])
plt.title("Per-Phase Power Distribution")
plt.xlabel("Time Stamp [15min]")
plt.ylabel("Phase")
plt.show()