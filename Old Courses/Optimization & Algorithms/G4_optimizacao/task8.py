import cvxpy as cp
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Open the data file .mat
data = scipy.io.loadmat('classifier_dataset.mat')

# Extract the variables from the data
traindataset = data['traindataset']
testdataset = data['testdataset']

trainlabels = data['trainlabels'].flatten()  # Flatten to convert into 1D array
testlabels = data['testlabels'].flatten()    # Flatten to convert into 1D array

# Save the number of rows in N (number of samples) and columns in D (number of features)
N, D = traindataset.shape

# Regularization parameter
ro = 0.1

# Define the optimization variables
w0 = cp.Variable()        # Bias term (scalar)
w = cp.Variable(D)        # Weights vector (D-dimensional)

# Define the hinge loss function h(u) = max(0, 1 - u)
hinge_losses = cp.pos(1 - cp.multiply(trainlabels, traindataset @ w + w0))

# Define the objective function: hinge loss + regularization term
objective = cp.Minimize((1/N) * cp.sum(hinge_losses) + ro * cp.norm(w, 2)**2)

# Define the problem and solve it
problem = cp.Problem(objective)
problem.solve()

# Retrieve the optimal parameters
w_optimal = w.value
w0_optimal = w0.value

# Function to evaluate the classifier error rate fD on a given dataset
def evaluate_error_rate(dataset, labels, w0_opt, w_opt):
    predictions = np.sign(dataset @ w_opt + w0_opt)  # Classify
    misclassifications = np.sum(predictions != labels)  # Count errors
    error_rate = misclassifications / len(labels)  # Calculate error rate
    return error_rate

P=0.18

#####################################################
#make the x~ vector (atacker vector)
#####################################################
N2, D2 = testdataset.shape

x_attack_final=np.empty(shape=(N2, D2))

#For each sample, calculate x~
for k in range(0,N2):
    #calculate y*w
    yw=testlabels[k]*w_optimal
    i=0
    
    #aply sign function to all elements in the vector yw
    for num in yw:
        if num>=0:
            yw[i]= 1
        else:
            yw[i]= -1
        i+=1
    
    x=testdataset[k]
    #calculate x~ for this sample
    x_attack= x - P*yw
    #add to matrix with all x~
    x_attack_final[k]=x_attack

#####################################################
#calculate error with attacked input
#####################################################

test_error_rate = evaluate_error_rate(x_attack_final, testlabels, w0_optimal, w_optimal)

print(f"Test dataset error rate with attack vector: {test_error_rate * 100:.2f}%")

