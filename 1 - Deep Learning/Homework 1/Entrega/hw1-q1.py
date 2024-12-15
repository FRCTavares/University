# Deep Learning Homework 1

# Import necessary libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import utils

class LinearModel(object):
    """
    A base class for linear models.
    """
    def __init__(self, n_classes, n_features, **kwargs):
        """
        Initialize the model with weights set to zero.
        
        Args:
            n_classes (int): Number of classes.
            n_features (int): Number of features.
            **kwargs: Additional arguments (unused).
        """
        self.W = np.zeros((n_classes, n_features))  # Weight matrix initialized to zeros

    def update_weight(self, x_i, y_i, **kwargs):
        """
        Update the model's weights based on a single training example.
        To be implemented by subclasses.
        
        Args:
            x_i (np.ndarray): Feature vector of a single training example.
            y_i (int): True label of the training example.
            **kwargs: Additional arguments (unused).
        """
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        """
        Train the model for one epoch over the dataset.
        
        Args:
            X (np.ndarray): Feature matrix (n_examples x n_features).
            y (np.ndarray): Labels vector (n_examples).
            **kwargs: Additional arguments passed to the update_weight method.
        """
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)  # Update weights for each example

    def predict(self, X):
        """
        Predict the labels for given examples.
        
        Args:
            X (np.ndarray): Feature matrix (n_examples x n_features).
        
        Returns:
            np.ndarray: Predicted labels (n_examples).
        """
        scores = np.dot(self.W, X.T)  # Compute scores for each class
        predicted_labels = scores.argmax(axis=0)  # Choose the class with the highest score
        return predicted_labels

    def evaluate(self, X, y):
        """
        Evaluate the model's accuracy on a dataset.
        
        Args:
            X (np.ndarray): Feature matrix (n_examples x n_features).
            y (np.ndarray): True labels (n_examples).
        
        Returns:
            float: Accuracy of the model.
        """
        y_hat = self.predict(X)  # Get predictions
        n_correct = (y == y_hat).sum()  # Count correct predictions
        n_possible = y.shape[0]  # Total number of examples
        return n_correct / n_possible  # Calculate accuracy

class Perceptron(LinearModel):
    """
    Perceptron classifier implementing the update rule.
    """
    def __init__(self, n_classes, n_features, learning_rate=0.01, **kwargs):
        """
        Initialize the Perceptron model with a learning rate.
        
        Args:
            n_classes (int): Number of output classes.
            n_features (int): Number of input features.
            learning_rate (float): Learning rate for weight updates.
            **kwargs: Additional arguments (unused).
        """
        super(Perceptron, self).__init__(n_classes, n_features)
        self.learning_rate = learning_rate  # Learning rate for weight updates


    def update_weight(self, x_i, y_i, **kwargs):
        """
        Update the weights based on a single training example using the Perceptron rule.
        
        Args:
            x_i (np.ndarray): Feature vector of a single training example.
            y_i (int): True label of the training example.
            **kwargs: Additional arguments (unused).
        
        Returns:
            np.ndarray: Updated weights for the true class.
        """
        y_pred = np.argmax(np.dot(self.W, x_i))  # Predict the class
        
        if y_pred != y_i:
            # Update weights for the true class
            self.W[y_i] += self.learning_rate * x_i
            # Penalize the weights for the incorrectly predicted class
            self.W[y_pred] -= self.learning_rate * x_i

        return self.W[y_i]  # Return updated weights for the true class

class LogisticRegression(LinearModel):
    """
    Logistic Regression classifier with optional L2 regularization.
    """
    def __init__(self, n_classes, n_features, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        Initialize the Logistic Regression model with a learning rate and L2 penalty.

        Args:
            n_classes (int): Number of output classes.
            n_features (int): Number of input features.
            learning_rate (float): Learning rate for weight updates.
            l2_penalty (float): L2 regularization penalty.
            **kwargs: Additional arguments (unused).
        """
        super().__init__(n_classes, n_features, **kwargs)
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty

    def update_weight(self, x_i, y_i, **kwargs):
        """
        Update the weights based on a single training example using gradient descent.

        Args:
            x_i (np.ndarray): Feature vector of a single training example.
            y_i (int): True label of the training example.
            **kwargs: Additional arguments (unused).
        """
        # Compute the scores for each class
        scores = np.dot(self.W, x_i)
        
        # Compute the probabilities using softmax function
        exp_scores = np.exp(scores - np.max(scores))  # For numerical stability
        probs = exp_scores / np.sum(exp_scores)
        
        # Create one-hot encoding of the true label
        y_one_hot = np.zeros_like(probs)
        y_one_hot[y_i] = 1

        # Compute the gradient of the loss with respect to the weights
        grad_W = np.outer((probs - y_one_hot), x_i)
        
        # Apply L2 regularization if l2_penalty > 0
        if self.l2_penalty > 0:
            grad_W += self.l2_penalty * self.W
            #print(f"applied l2 penalty: {self.l2_penalty}")

        # Update the weights using gradient descent
        self.W -= self.learning_rate * grad_W

class MLP(object):
    """
    Multi-Layer Perceptron (MLP) classifier.
    """
    def __init__(self, n_classes, n_features, hidden_size):
        """
        Initialize the MLP with a single hidden layer.
        
        Args:
            n_classes (int): Number of output classes.
            n_features (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
        
        """
        self.hidden_size = hidden_size # The number of neurons in the hieden layer
        self.learning_rate = 0.001 

        # Initialize the weights using np.random.normal with μ = 0.1 and σ² = 0.1² and biases
        self.W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))
        self.b2 = np.zeros(n_classes)

    def predict(self, X):
        """
        Compute the forward pass of the network to make predictions.
        
        Args:
            X (np.ndarray): Feature matrix (n_examples x n_features).
        
        Returns:
            np.ndarray: Predicted labels (n_examples).
        """
        # Forward pass 
        z1 = X.dot(self.W1.T) + self.b1 # Compute the hidden layer
        a1 = np.maximum(0,z1) # Apply ReLU
        z2 = a1.dot(self.W2.T) + self.b2 # Compute the output layer

        # Softmax Activation
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True)) # For numerical stability
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # Compute probabilities
        predictions = np.argmax(probs, axis=1)

        return predictions


    def evaluate(self, X, y):
        """
        Evaluate the model's accuracy on a dataset.
        
        Args:
            X (np.ndarray): Feature matrix (n_examples x n_features).
            y (np.ndarray): True labels (n_examples).
        
        Returns:
            float: Accuracy of the model.
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)  # Get predictions
        n_correct = (y == y_hat).sum()  # Count correct predictions
        n_possible = y.shape[0]  # Total number of examples
        return n_correct / n_possible  # Calculate accuracy

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Train the MLP for one epoch.
        
        Args:
            X (np.ndarray): Feature matrix (n_examples x n_features).
            y (np.ndarray): Labels vector (n_examples).
            learning_rate (float): Learning rate for weight updates.
            **kwargs: Additional arguments (unused).

        Returns:
            float: Loss of the epoch.
        """

        if learning_rate is None:
            learning_rate = self.learning_rate

        n_examples = X.shape[0]
        total_loss = 0

        for i in range(n_examples):
            x_i = X[i:i + 1]  # Shape (1, n_features)
            y_i = y[i]

            # Forward pass
            z1 = x_i.dot(self.W1.T) + self.b1  # Shape (1, hidden_size)
            a1 = np.maximum(0, z1)  # ReLU activation
            z2 = a1.dot(self.W2.T) + self.b2  # Shape (1, n_classes)

            # Softmax activation
            exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Compute the cross-entropy loss
            loss = -np.log(probs[0, y_i])
            total_loss += loss

            # Backward pass
            delta3 = probs.copy()
            delta3[0, y_i] -= 1  # Gradient w.r.t softmax output

            dW2 = delta3.T.dot(a1)  # Shape (n_classes, hidden_size)
            db2 = delta3.sum(axis=0)  # Shape (n_classes,)

            da1 = delta3.dot(self.W2)  # Shape (1, hidden_size)
            dz1 = da1 * (z1 > 0)  # ReLU derivative

            dW1 = dz1.T.dot(x_i)  # Shape (hidden_size, n_features)
            db1 = dz1.sum(axis=0)  # Shape (hidden_size,)

            # Update weights and biases
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        avg_loss = total_loss / n_examples
        return avg_loss


def plot(epochs, train_accs, val_accs, filename=None):
    """
    Plot training and validation accuracy over epochs.
    
    Args:
        epochs (np.ndarray): Array of epoch numbers.
        train_accs (list): List of training accuracies.
        val_accs (list): List of validation accuracies.
        filename (str, optional): Filename to save the plot. Defaults to None.
    """
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')  # Plot training accuracy
    plt.plot(epochs, val_accs, label='validation')  # Plot validation accuracy
    plt.legend()  # Show legend
    if filename:
        plt.savefig(filename, bbox_inches='tight')  # Save plot if filename is provided
    plt.show()  # Display the plot

def plot_loss(epochs, loss, filename=None):
    """
    Plot training loss over epochs.
    
    Args:
        epochs (np.ndarray): Array of epoch numbers.
        loss (list): List of loss values.
        filename (str, optional): Filename to save the plot. Defaults to None.
    """
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')  # Plot training loss
    plt.legend()  # Show legend
    if filename:
        plt.savefig(filename, bbox_inches='tight')  # Save plot if filename is provided
    plt.show()  # Display the plot

def plot_w_norm(epochs, w_norms, filename=None):
    """
    Plot the norm of the weights over epochs.
    
    Args:
        epochs (np.ndarray): Array of epoch numbers.
        w_norms (list): List of weight norms.
        filename (str, optional): Filename to save the plot. Defaults to None.
    """
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')  # Plot weight norms
    plt.legend()  # Show legend
    if filename:
        plt.savefig(filename, bbox_inches='tight')  # Save plot if filename is provided
    plt.show()  # Display the plot

def main():
    """
    The main function to run the training and evaluation pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()  # Parse command-line arguments

    utils.configure_seed(seed=42)  # Set random seed for reproducibility

    add_bias = opt.model != "mlp"  # Add bias term for non-MLP models
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)  # Load dataset
    train_X, train_y = data["train"]  # Training data
    dev_X, dev_y = data["dev"]  # Validation data
    test_X, test_y = data["test"]  # Test data
    n_classes = np.unique(train_y).size  # Number of unique classes
    n_feats = train_X.shape[1]  # Number of features

    # Initialize the model based on the chosen type
    if opt.model == 'perceptron':
        model = Perceptron(
            n_classes=n_classes, 
            n_features=n_feats,
            learning_rate=opt.learning_rate,
            )
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(
            n_classes=n_classes,
            n_features=n_feats,
            learning_rate=opt.learning_rate,
            l2_penalty=opt.l2_penalty
        )
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    
    epochs = np.arange(1, opt.epochs + 1)  # Array of epoch numbers
    train_loss = []  # List to store training loss
    weight_norms = []  # List to store weight norms
    valid_accs = []  # List to store validation accuracies
    train_accs = []  # List to store training accuracies

    start = time.time()  # Start timer

    # Print initial training and validation accuracy
    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    # Training loop over epochs
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])  # Shuffle training data
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        
        if opt.model == 'mlp':
            # Train MLP and record loss
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            # Train Perceptron or Logistic Regression
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        # Evaluate and record training and validation accuracy
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        
        if opt.model == 'mlp':
            # Print loss and accuracies for MLP
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            # Calculate and print weight norm for Logistic Regression
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            # Print accuracies for Perceptron
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    
    elapsed_time = time.time() - start  # Calculate elapsed time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    
    # Evaluate on the test set and print final accuracy
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
    ))
    print('Final train acc: {:.4f}'.format(train_accs[-1]))

    # Plot training and validation accuracy
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    
    if opt.model == 'mlp':
        # Plot training loss for MLP
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        # Plot weight norms for Logistic Regression
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    
    # Save final results to a text file
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Final train acc: {train_accs[-1]}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")

if __name__ == '__main__':
    main()  # Execute the main function when the script is run
