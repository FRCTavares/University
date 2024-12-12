import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit

# ----------------------------- Data Loading ----------------------------- #
def load_data(file_paths):
    """
    Load training and test data from specified file paths.

    Args:
    file_paths (dict): Paths for training outputs (y_train), training inputs (u_train), and test inputs (u_test).

    Returns:
    tuple: y_train, u_train, u_test arrays.
    """
    try:
        # Load data from .npy files
        y_train = np.load(file_paths['y_train'])
        u_train = np.load(file_paths['u_train'])
        u_test = np.load(file_paths['u_test'])

        # Check for dimension mismatches
        if y_train.shape[0] != u_train.shape[0]:
            raise ValueError("Mismatch between y_train and u_train dimensions.")
        if u_test.shape[0] != 510:
            raise ValueError("Unexpected shape for u_test.")
        
        print("Files loaded successfully!")
        return y_train, u_train, u_test

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except ValueError as e:
        print(f"Data shape error: {e}")
        raise

# ----------------------------- ARX Model Creation ----------------------------- #
def create_arx_model(y_train, u_train, n, m, d):
    """
    Build ARX model by constructing the phi vectors for given (n, m, d) orders.

    Args:
    y_train (array): Output training data.
    u_train (array): Input training data.
    n, m, d (int): ARX model parameters for output order, input order, and delay.

    Returns:
    X (array): Matrix of phi vectors (features).
    Y (array): Trimmed y_train corresponding to X.
    """
    N = len(u_train)
    p = max(n, m + d)  # Calculate number of past values to consider
    Y = np.zeros((N - p, 1))  # Output matrix
    X = np.zeros((N - p, n + m + 1))  # Design matrix

    # Fill Y and X matrices
    for k in range(p, N):
        Y[k - p] = y_train[k]  # Populate Y

        # Construct phi vector
        phi = np.zeros(n + m + 1)  # Initialize phi vector
        for i in range(n):
            phi[i] = y_train[k - i - 1]  # Previous outputs
        for i in range(m + 1):
            phi[n + i] = u_train[k - d - i]  # Delayed inputs

        X[k - p] = phi  # Assign phi to X

    return X, Y

# ----------------------------- Model Fitting ----------------------------- #
def fit_arx_model(X, Y, model_type):
    """
    Fit the ARX model using the specified regression model.

    Args:
    X (array): Feature matrix.
    Y (array): Output vector.
    model_type (str): Type of regression model to use.

    Returns:
    theta (array): Model coefficients.
    """
    # Initialize model based on type
    if model_type == "Linear":
        model = LinearRegression()
    elif model_type == "Ridge":
        model = Ridge(alpha=0.1)
    elif model_type == "Lasso":
        model = Lasso(alpha=0.1)
    else:
        raise ValueError("Invalid model type")

    # Fit the model to the data
    model.fit(X, Y)

    # Return coefficients (theta)
    theta = model.coef_.reshape(-1, 1)
    return theta

# ----------------------------- m, n & d Selection ----------------------------- #
def m_n_d_fitting(y_train, u_train, n_range, m_range, d_range, n_splits=10):
    """
    Perform cross-validation to find optimal (n, m, d) with lowest Sum of Squared Errors (SSE).

    Args:
    y_train (array): Output training data.
    u_train (array): Input training data.
    n_range, m_range, d_range (range): Ranges for ARX parameters n, m, and d.
    n_splits (int): Number of splits for cross-validation.

    Returns:
    tuple: Best (n, m, d) parameters and corresponding SSE.
    """
    best_sse = float('inf')
    best_params = (0, 0, 0)
    
    # Create TimeSeriesSplit object for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n in n_range:
        for m in m_range:
            for d in d_range:
                # Skip invalid combinations of parameters
                if d >= n or d >= m or d >= len(y_train) or d >= len(u_train):
                    continue
                
                # Prepare for cross-validation
                sse_list = []

                # Create ARX model design matrix and output
                X, Y = create_arx_model(y_train, u_train, n, m, d)

                # Cross-validation for SSE calculation
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    # Fit the model and predict on test set
                    theta = fit_arx_model(X_train, Y_train, "Linear")  # Choose the model you want to evaluate
                    y_pred_test = X_test @ theta
                    sse = np.sum((Y_test - y_pred_test) ** 2)
                    sse_list.append(sse)

                # Average SSE over all splits
                average_sse = np.mean(sse_list)

                # Update best parameters if current SSE is lower
                if average_sse < best_sse:
                    best_sse = average_sse
                    best_params = (n, m, d)

    return best_params, best_sse * n_splits  # Multiply by n_splits for total SSE

# ----------------------------- Test Set Prediction ----------------------------- #
def predict_arx_test(u_test, theta, n, m, d):
    """
    Predict test set output using the ARX model and learned parameters.

    Args:
    u_test (array): Input test data.
    theta (array): ARX model parameters.
    n, m, d (int): ARX model parameters for output order, input order, and delay.

    Returns:
    array: Predicted test set outputs.
    """
    N = len(u_test)
    p = max(n, m + d)  # Ensure enough lags are used
    output_test = np.zeros(N)  # Initialize output predictions

    # Loop through all time steps, predicting one step at a time
    for k in range(p, N):
        # Initialize phi vector for this time step
        phi = np.zeros(n + m + 1)
        
        # Collect previous output values (phi[0] to phi[n-1])
        for i in range(n):
            if k - i - 1 >= 0:
                phi[i] = output_test[k - i - 1]  # Use previous outputs
            else:
                phi[i] = 0  # No output available yet, so use 0
                
        # Collect delayed input values (phi[n] to phi[n+m])
        for i in range(m + 1):
            phi[n + i] = u_test[k - d - i]  # Delayed input values

        # Predict the output for time step k using the ARX model
        output_test[k] = (phi @ theta).item()  # Ensure output_test[k] is a scalar

    return output_test

# ----------------------------- Main Function ----------------------------- #
def main():
    """
    Orchestrate data loading, model fitting, cross-validation, and test set prediction.
    """
    # Define file paths for loading data
    file_paths = {
        'y_train': 'C:/Users/franc/OneDrive - Universidade de Lisboa/IST/Mestrado/AAut - Machine Learning/Projeto/Regression P2/output_train.npy',
        'u_train': 'C:/Users/franc/OneDrive - Universidade de Lisboa/IST/Mestrado/AAut - Machine Learning/Projeto/Regression P2/u_train.npy',
        'u_test': 'C:/Users/franc/OneDrive - Universidade de Lisboa/IST/Mestrado/AAut - Machine Learning/Projeto/Regression P2/u_test.npy'
    }
    
    # Load data
    y_train, u_train, u_test = load_data(file_paths)

    # Define ranges for ARX parameters to be optimized
    n_range = range(1, 10)
    m_range = range(1, 10)
    d_range = range(0, 10)

    # Cross-validation to find the best (n, m, d)
    best_params, best_sse = m_n_d_fitting(y_train, u_train, n_range, m_range, d_range, n_splits=5)

    print(f"Best parameters: n = {best_params[0]}, m = {best_params[1]}, d = {best_params[2]} with SSE = {best_sse:.5f}")

    # Fit the model with the best parameters and predict test set output
    X, Y = create_arx_model(y_train, u_train, best_params[0], best_params[1], best_params[2])

    # Define models to be tested
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso()
    }

    sse_results = {}
    mse_results = {}

    # Nested cross-validation for hyperparameter tuning and model evaluation
    tscv_outer = TimeSeriesSplit(n_splits=5)

    for model_name, model in models.items():
        sse_model_results = []
        mse_model_results = []

        # Hyperparameter tuning (if applicable)
        for train_index, test_index in tscv_outer.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            # Fit the model
            theta = fit_arx_model(X_train, Y_train, model_name.split()[0])  # Fit based on the model type

            # Predict on test set
            y_pred_test = X_test @ theta
            sse = np.sum((Y_test - y_pred_test) ** 2)
            sse_model_results.append(sse)

            mse = np.mean((Y_test - y_pred_test) ** 2)
            mse_model_results.append(mse)

        # Average SSE over all splits
        avg_sse = np.mean(sse_model_results)
        sse_results[model_name] = avg_sse

        # MSE for each model
        avg_mse = np.mean(mse_model_results)
        mse_results[model_name] = avg_mse

        print(f"Average MSE for {model_name}: {avg_mse:.5f}")
        print(f"Average SSE for {model_name}: {avg_sse:.5f}")

        # Save the theta coefficients for each model
        if model_name == "Linear Regression":
            thetaLinear = theta
        elif model_name == "Ridge Regression":
            thetaRidge = theta
        elif model_name == "Lasso Regression":
            thetaLasso = theta

    # Use the corresponding theta values for predictions on the test set
    for model_name, theta in zip(['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
                                 [thetaLinear, thetaRidge, thetaLasso]):
        # Predict the test set output using the respective theta
        y_test_pred = predict_arx_test(u_test, theta, best_params[0], best_params[1], best_params[2])

        # Save only the last 400 predictions
        y_test_pred = y_test_pred[-400:]

        # Save the predictions to a file
        if model_name == "Linear Regression":
            np.save(f"output_test.npy", y_test_pred)
            print("Predictions saved successfully for Linear model! in output_test.npy")

# Entry point of the program
if __name__ == "__main__":
    main()
