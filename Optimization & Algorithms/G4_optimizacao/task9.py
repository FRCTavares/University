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



w0 = cp.Variable()        
w = cp.Variable(D)

ro = 0.1


#Vector manipulation
train_labels_col = cp.reshape(trainlabels, (400,1))
w_T = cp.reshape(w, (1,784))

#Compute yw
product_matrix = train_labels_col @ w_T

#Equivalent to calculating the l1-norm for every row
l1_norm = cp.sum(cp.abs(product_matrix), axis=1)

#Define the hinge loss function
hinge_losses_9 = cp.pos(1 - (cp.multiply(trainlabels, traindataset @ w + w0) - P * l1_norm))

#Problem definition
objective_9 = cp.Minimize((1/N) * cp.sum(hinge_losses_9) + ro * cp.norm(w, 2)**2)
problem_9 = cp.Problem(objective_9)
problem_9.solve()

w_optimal = w.value
w0_optimal = w0.value


#Result Evaluation
train_error_rate = evaluate_error_rate(traindataset, trainlabels, w0_optimal, w_optimal)
print(f"Training dataset error rate: {train_error_rate * 100:.2f}%")

test_error_rate = evaluate_error_rate(testdataset, testlabels, w0_optimal, w_optimal)
print(f"Test dataset error rate: {test_error_rate * 100:.2f}%")

attack_test_error_rate = evaluate_error_rate(x_attack_final, testlabels, w0_optimal, w_optimal)
print(f"Test dataset error rate with attack vector: {attack_test_error_rate * 100:.2f}%")