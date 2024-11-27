import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RANSACRegressor, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data():
    """
    Loads the training and test datasets from .npy files.
    Adds a column of ones (bias/intercept) to both training and test data.

    Returns:
    X_train (ndarray): Training dataset with bias column.
    Y_train (ndarray): Corresponding labels for the training data.
    X_test (ndarray): Test dataset with bias column.
    """
    X_train = np.load('/home/fct/Desktop/Machine Learning/Project/Regression/X_train.npy')
    Y_train = np.load('/home/fct/Desktop/Machine Learning/Project/Regression/y_train.npy')
    X_test = np.load('/home/fct/Desktop/Machine Learning/Project/Regression/X_test.npy')

    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    
    return X_train, Y_train, X_test

def remove_outliers(X, Y, num_outliers=48, regressor=LinearRegression()):
    """
    Removes outliers from the dataset based on the absolute error from the chosen regressor.

    Args:
    X (ndarray): Training dataset.
    Y (ndarray): Corresponding labels.
    num_outliers (int): Number of outliers to remove.
    regressor (sklearn regressor): The regressor used to identify outliers.

    Returns:
    X_cleaned (ndarray): Dataset without outliers.
    Y_cleaned (ndarray): Labels without outliers.
    """
    for _ in range(num_outliers):
        regressor.fit(X, Y)
        Y_pred = regressor.predict(X)
        errors = np.abs(Y_pred - Y)
        
        max_error_index = np.argmax(errors)
        
        X = np.delete(X, max_error_index, axis=0)
        Y = np.delete(Y, max_error_index, axis=0)
    
    return X, Y

def evaluate_model_cv(model, X, Y, features, model_name="Model", k=10):
    """
    Evaluates the model using K-Fold Cross Validation and prints performance metrics.

    Args:
    model (sklearn regressor): The model to be evaluated.
    X (ndarray): Training dataset.
    Y (ndarray): Corresponding labels.
    features (list): List of feature names to display coefficients.
    model_name (str): Name of the model.
    k (int): Number of folds for cross-validation.

    Returns:
    model (sklearn regressor): The trained model.
    total_sse (float): The total SSE (Sum of Squared Errors) for all folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    sse_list, mse_list, mad_list, r2_list = [], [], [], []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        sse = np.sum((Y_pred - Y_test) ** 2)
        mse = mean_squared_error(Y_test, Y_pred)
        mad = np.median(np.abs(Y_pred - Y_test))
        r2 = r2_score(Y_test, Y_pred)
        
        sse_list.append(sse)
        mse_list.append(mse)
        mad_list.append(mad)
        r2_list.append(r2)
    
    total_sse = np.sum(sse_list)
    
    print(f"\n---------------------------{model_name} (k-fold CV)---------------------------")
    print(f"Avg SSE: {total_sse:.6f} | Avg MSE: {np.mean(mse_list):.6f} | Avg MAD: {np.mean(mad_list):.6f} | Avg R²: {np.mean(r2_list):.6f}")
    
    # Print coefficients for linear models, including RANSAC's underlying model
    if hasattr(model, 'coef_'):
        print("\nBetas (coefficients):")
        for name, coef in zip(features, model.coef_):
            print(f"{name}: {coef:.6f}")
    
    # For RANSAC, get the coefficients from the underlying estimator
    if isinstance(model, RANSACRegressor) and hasattr(model.estimator_, 'coef_'):
        print("\nBetas (coefficients) from RANSAC's underlying estimator:")
        for name, coef in zip(features, model.estimator_.coef_):
            print(f"{name}: {coef:.6f}")
    
    model.fit(X, Y)
    
    return model, total_sse

def ridge_lasso_cv(model_name, X, Y, features, k=10):
    """
    Performs cross-validation for RidgeCV or LassoCV models to select the best alpha and evaluates the model using SSE, R², MAD, and MSE.

    Args:
    model_name (str): Name of the model ("Ridge" or "Lasso").
    X (ndarray): Training dataset.
    Y (ndarray): Corresponding labels.
    features (list): List of feature names to display coefficients.
    k (int): Number of folds for cross-validation.

    Returns:
    best_model (sklearn regressor): The best model after cross-validation.
    """
    alphas = np.logspace(-4, 4, 100)
    
    if model_name == 'Ridge':
        model = RidgeCV(alphas=alphas, store_cv_values=True)
    elif model_name == 'Lasso':
        model = LassoCV(alphas=alphas, cv=k)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    sse_list, mse_list, mad_list, r2_list = [], [], [], []
    
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        sse = np.sum((Y_pred - Y_test) ** 2)
        mse = mean_squared_error(Y_test, Y_pred)
        mad = np.median(np.abs(Y_pred - Y_test))
        r2 = r2_score(Y_test, Y_pred)
        
        sse_list.append(sse)
        mse_list.append(mse)
        mad_list.append(mad)
        r2_list.append(r2)
    
    total_sse = np.sum(sse_list)
    
    print(f"\n---------------------------{model_name} Regression (k-fold CV)---------------------------")
    print(f"Best alpha: {model.alpha_}")
    print(f"Avg SSE: {total_sse:.6f} | Avg MSE: {np.mean(mse_list):.6f} | Avg MAD: {np.mean(mad_list):.6f} | Avg R²: {np.mean(r2_list):.6f}")
    
    if hasattr(model, 'coef_'):
        print("\nBetas (coefficients):")
        for name, coef in zip(features, model.coef_):
            print(f"{name}: {coef:.6f}")
    
    model.fit(X_scaled, Y)
    
    return model, total_sse

def main():
    """
    Main function that loads data, removes outliers, trains, and evaluates models.
    """
    X_train, Y_train, X_test = load_data()
    features = ['Intercept', 'Air Temperature', 'Water Temperature', 'Wind Speed', 'Wind Direction', 'Illumination']

    # Remove outliers
    X_train_cleaned, Y_train_cleaned = remove_outliers(X_train, Y_train)

    best_model = None
    lowest_sse = np.inf
    
    models_to_test = {
        'Linear': LinearRegression(),
        'Ridge': 'Ridge',
        'Lasso': 'Lasso',
        'RANSAC': RANSACRegressor()
    }

    while True:
        print("\nChoose a method:\n 1 - Linear\n 2 - Ridge\n 3 - Lasso\n 4 - RANSAC\n 5 - Test all methods\n 6 - Exit")
        method = input("Your choice: ")

        if method == '6':  # Exit option
            print("Exiting the program.")
            break

        if method in ['1', '2', '3', '4']:
            if method == '1':
                model, sse = evaluate_model_cv(LinearRegression(), X_train_cleaned, Y_train_cleaned, features, model_name="Linear Regression")
            elif method == '2':
                model, sse = ridge_lasso_cv('Ridge', X_train_cleaned, Y_train_cleaned, features)
            elif method == '3':
                model, sse = ridge_lasso_cv('Lasso', X_train_cleaned, Y_train_cleaned, features)
            elif method == '4':
                model, sse = evaluate_model_cv(RANSACRegressor(), X_train_cleaned, Y_train_cleaned, features, model_name="RANSAC Regression")

            best_model = model

        elif method == '5':  # Test all methods
            for name, model in models_to_test.items():
                if isinstance(model, str):  # For Ridge and Lasso
                    current_model, sse = ridge_lasso_cv(model, X_train_cleaned, Y_train_cleaned, features)
                else:  # For Linear and RANSAC
                    current_model, sse = evaluate_model_cv(model, X_train_cleaned, Y_train_cleaned, features, model_name=name)

                if sse < lowest_sse:
                    lowest_sse = sse
                    best_model = current_model

            print(f"\nBest model selected with SSE: {lowest_sse}")
            break  # End the loop after testing all methods

        else:
            print("Invalid option. Please try again.")

    # Make predictions on the test set after exiting the loop
    if best_model:
        Y_test_pred = best_model.predict(X_test)
        
        # Ensure that Y_test_pred has the same shape as Y_train
        if Y_test_pred.shape != Y_train.shape:
            print(f"Reshaping Y_test_pred from {Y_test_pred.shape} to match Y_train shape {Y_train.shape}.")
            Y_test_pred = Y_test_pred.reshape(Y_train.shape)

        # Ensure that Y_test_pred is saved in the same format as Y_train
        np.save('/home/fct/Desktop/Machine Learning/Project/Regression/Y_test.npy', Y_test_pred)
        print(f"Predictions saved in the same format as Y_train to Y_test.npy.")

if __name__ == "__main__":
    main()
