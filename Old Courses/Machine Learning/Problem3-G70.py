import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score # type: ignore
from sklearn.metrics import f1_score, make_scorer # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import SVC # type: ignore


''' ----------------------------------------- Data Preprocessing ----------------------------------------- '''

def load_and_preprocess_data(x_train_path, y_train_path, x_train_extra_path, x_test_path):
    """
    Loads and preprocesses the training data. Balances the dataset by augmenting '0' class images 
    through flipping and rotating.

    Args:
    x_train_path (str): Path to the Xtrain .npy file.
    y_train_path (str): Path to the Ytrain .npy file.

    Returns:
    tuple: Preprocessed and reshaped X training data, one-hot encoded Y labels, extra data, and reshaped test data.
    """
    # Load data
    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    x_train_extra = np.load(x_train_extra_path)
    x_test = np.load(x_test_path)

    print(f"Ytrain shape: {y_train.shape}")

    x_test = x_test.reshape((x_test.shape[0], 48, 48, 1))/255.0
    x_train = x_train/255.0
    x_train_extra = x_train_extra/255.0

    return x_train, y_train, x_train_extra, x_test

''' ------------------------------------------- Data Balancing ------------------------------------------- '''

def balance_the_dataset(x_train, y_train):
    """
    Balances the dataset by augmenting images through flipping and rotating.
    """
    x_train = x_train.reshape((x_train.shape[0], 48 * 48))
    y_train = np.argmax(y_train, axis=1)

    indices_of_ones = np.where(y_train == 1)[0]
    indices_of_zeros = np.where(y_train == 0)[0]

    np.random.seed(42)
    n_to_augment = len(indices_of_ones) - len(indices_of_zeros)

    if n_to_augment > 0: 
        random_indices_of_zeros = np.random.choice(indices_of_zeros, n_to_augment, replace=False)
        x_remaining_zeros = x_train[indices_of_zeros].reshape(len(indices_of_zeros), 48, 48)
        y_remaining_zeros = y_train[indices_of_zeros]
        x_remaining_ones = x_train[indices_of_ones].reshape(len(indices_of_ones), 48, 48)
        y_remaining_ones = y_train[indices_of_ones]

        x_augment_zeros = x_train[random_indices_of_zeros].reshape(n_to_augment, 48, 48)
        y_augment_zeros = y_train[random_indices_of_zeros]
        x_augment_rotated_zeros = np.array([np.rot90(img, np.random.choice([1, 2, 3])) for img in x_augment_zeros])

        x_train_balanced = np.concatenate((x_remaining_ones, x_remaining_zeros, x_augment_rotated_zeros), axis=0)
        y_train_balanced = np.concatenate((y_remaining_ones, y_remaining_zeros, y_augment_zeros), axis=0)

    elif n_to_augment < 0:
        random_indices_of_ones = np.random.choice(indices_of_ones, -n_to_augment, replace=False)
        x_remaining_ones = x_train[indices_of_ones].reshape(len(indices_of_ones), 48, 48)
        y_remaining_ones = y_train[indices_of_ones]
        x_remaining_zeros = x_train[indices_of_zeros].reshape(len(indices_of_zeros), 48, 48)
        y_remaining_zeros = y_train[indices_of_zeros]

        x_augment_ones = x_train[random_indices_of_ones].reshape(-n_to_augment, 48, 48)
        y_augment_ones = y_train[random_indices_of_ones]
        x_augment_rotated_ones = np.array([np.rot90(img, np.random.choice([1, 2, 3])) for img in x_augment_ones])

        x_train_balanced = np.concatenate((x_remaining_ones, x_remaining_zeros, x_augment_rotated_ones), axis=0)
        y_train_balanced = np.concatenate((y_remaining_ones, y_remaining_zeros, y_augment_ones), axis=0)

    shuffled_indices = np.random.permutation(len(y_train_balanced))
    x_train_balanced = x_train_balanced[shuffled_indices]
    y_train_balanced = y_train_balanced[shuffled_indices]

    n_samples = x_train_balanced.shape[0]
    x_train_reshaped = x_train_balanced.reshape((n_samples, 48, 48, 1))

    y_train_one_hot = to_categorical(y_train_balanced, 2)
    return x_train_reshaped, y_train_one_hot

''' -------------------------------------------- Model Creation ------------------------------------------ '''

def create_cnn_model(learning_rate=0.001):
    """
    Creates and compiles a CNN model with a tunable learning rate.
    """
    model = Sequential()

    model.add(Input(shape=(48, 48, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
''' --------------------------------------- Train and Evaluate Model ------------------------------------- '''

def train_cnn_with_kfold(x_train, y_train, n_splits=10, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Trains the CNN using KFold cross-validation with tunable learning rate.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    fold = 1

    x_train = x_train.reshape((x_train.shape[0], 48, 48, 1))
    y_train = to_categorical(y_train, 2)

    best_f1 = -1
    best_model = None

    for train_index, val_index in kf.split(x_train):
        print(f"Training fold {fold}...")

        X_train_fold, X_val_fold = x_train[train_index], x_train[val_index]
        Y_train_fold, Y_val_fold = y_train[train_index], y_train[val_index]

        X_train_fold_balanced, Y_train_fold_balanced = balance_the_dataset(X_train_fold, Y_train_fold)

        model = create_cnn_model(learning_rate)
        model.fit(X_train_fold_balanced, Y_train_fold_balanced, epochs=epochs, batch_size=batch_size, validation_data=(X_val_fold, Y_val_fold))

        y_val_pred = model.predict(X_val_fold)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        y_val_true_classes = np.argmax(Y_val_fold, axis=1)

        f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='weighted')
        f1_scores.append(f1)
        print(f"F1 score for fold {fold}: {f1:.5f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        fold += 1

    avg_f1 = np.mean(f1_scores)
    print(f"\nAverage F1 score across all folds: {avg_f1:.5f}")

    return best_model

''' --------------------------------------- Semi-Supervised Learning ------------------------------------- '''

def semi_supervised_learning(x_train, y_train, x_train_extra, model, confidence_threshold=0.85):
    """
    This function performs semi-supervised learning by using a pre-trained CNN model to generate labels
    for additional unlabeled data (`x_train_extra`). The model predicts labels for the extra data twice,
    and only retains predictions where the model is confident (above the threshold) both times.

    Args:
    x_train (np.array): The balanced initial dataset (in flattened 2D form, with rows representing flattened images).
    y_train (np.array): Labels corresponding to the initial dataset (either one-hot encoded or class labels).
    x_train_extra (np.array): Unlabeled extra data in flattened 2D form (each row is a flattened 48x48 image).
    model (tf.keras.Model): A pre-trained CNN model used to predict labels for `x_train_extra`.
    confidence_threshold (float): Confidence threshold for accepting predictions.

    Returns:
    np.array, np.array: Updated training data and corresponding labels, after incorporating confidently 
                        predicted extra data.
    """

    # Reshape x_train_extra to match the shape expected by the model and normalize
    x_train_extra = x_train_extra.reshape((x_train_extra.shape[0], 48, 48, 1))

    # First prediction round
    y_train_extra_pred_1 = model.predict(x_train_extra)
    
    # Second prediction round
    y_train_extra_pred_2 = model.predict(x_train_extra)

    # Only keep predictions where the model is confident in both rounds
    confident_indices = np.where(
        (np.max(y_train_extra_pred_1, axis=1) > confidence_threshold) & 
        (np.max(y_train_extra_pred_2, axis=1) > confidence_threshold) & 
        (np.argmax(y_train_extra_pred_1, axis=1) == np.argmax(y_train_extra_pred_2, axis=1))
    )[0]

    # Filter out uncertain predictions
    x_train_filtered = x_train_extra[confident_indices]
    y_train_filtered = y_train_extra_pred_1[confident_indices]  # Use predictions from the first round

    # Decode one-hot encoding
    y_train_filtered = np.argmax(y_train_filtered, axis=1)

    # Reshape x_train to 2D
    x_train_filtered = x_train_filtered.reshape((x_train_filtered.shape[0], 48 * 48))

    # One-hot encode labels for training
    y_train = to_categorical(y_train, 2)
    y_train_filtered = to_categorical(y_train_filtered, 2)

    # Combine the confidently predicted extra data with the original balanced training set
    x_train = np.concatenate((x_train, x_train_filtered), axis=0)
    y_train = np.concatenate((y_train, y_train_filtered), axis=0)

    # Return the updated training data and labels
    return x_train, y_train

''' --------------------------------------- Support Vector Machine --------------------------------------- '''

def svm(x_train, y_train):
    """
    Train an SVM model with cross-validation and hyperparameter tuning using F1 score.

    Args:
        X_train (numpy.ndarray): 2D array of shape (n_samples, n_features) for training data.
        y_train (numpy.ndarray): 1D array of shape (n_samples,) for labels (0 or 1).

    Returns:
        mean_f1_score (float): Mean F1 score from cross-validation.
        best_model (SVC): The fitted SVM model.
    """
    # Reshape and normalize the training and test data
    x_train = x_train.reshape((x_train.shape[0], 48, 48, 1))  # Reshape to (samples, 48, 48, 1)
    # One-hot encode labels for training
    y_train = to_categorical(y_train, 2)

    x_train, y_train = balance_the_dataset(x_train, y_train)

    x_train = x_train.reshape((x_train.shape[0], 48 * 48))  # Reshape to 2D
    y_train = np.argmax(y_train, axis=1)
    
    # Step 1: Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)

    # Step 2: Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }

    # Step 3: Set up Grid Search with Cross-Validation using F1 score
    f1_scorer = make_scorer(f1_score)
    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring=f1_scorer, verbose=1, n_jobs=-1)

    # Step 4: Fit the grid search to the data
    grid.fit(X_train_scaled, y_train)

    # Step 5: Get the best model
    best_model = grid.best_estimator_

    # Step 6: Perform cross-validation on the best model using F1 score
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring=f1_scorer)

    # Step 7: Output the results
    mean_f1_score = np.mean(cv_scores)

    print(f"Best Hyperparameters: {grid.best_params_}")
    print(f"Mean F1 Score: {mean_f1_score:.4f}\n")

    return mean_f1_score, best_model

''' ---------------------------------------- Main Script Execution --------------------------------------- '''

def main():
    # Load and preprocess the dataset
    x_train, y_train, x_train_extra, x_test = load_and_preprocess_data(
        'C:/Users/franc/OneDrive - Universidade de Lisboa/IST/Mestrado/AAut - Machine Learning/Projeto/Classification P1/Xtrain1.npy', 
        'C:/Users/franc/OneDrive - Universidade de Lisboa/IST/Mestrado/AAut - Machine Learning/Projeto/Classification P1/Ytrain1.npy',
        'C:/Users/franc/OneDrive - Universidade de Lisboa/IST/Mestrado/AAut - Machine Learning/Projeto/Classification P1/Xtrain1_extra.npy',
        'C:/Users/franc/OneDrive - Universidade de Lisboa/IST/Mestrado/AAut - Machine Learning/Projeto/Classification P1/Xtest1.npy'
    )

    # Train and evaluate the model using KFold cross-validation
    best_model = train_cnn_with_kfold(x_train, y_train, n_splits=5, epochs=10, batch_size=32, learning_rate=0.0005)

    # Train an SVM model
    print("Training an SVM model...")
    mean_f1_score, best_svm_model = svm(x_train, y_train)

    # Semi-supervised learning
    print("Starting semi-supervised learning...")
    x_train, y_train = semi_supervised_learning(x_train, y_train, x_train_extra, best_model)

    # Balance the dataset again after adding pseudo-labeled data
    print("Balancing the dataset again...")
    x_train, y_train = balance_the_dataset(x_train, y_train)

    # Create a new model or retrain the best model
    print("Creating and training the final CNN model...")
    final_model = create_cnn_model()
    final_model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Predict on the test set
    y_test_pred = np.argmax(final_model.predict(x_test), axis=1)

    # Save predictions in folder Group70_P3
    if not os.path.exists('Group70_P3'):
        os.makedirs('Group70_P3')
    np.save('Group70_P3/Ytest1.npy', y_test_pred)

    print(f"\nFinal Y Matrix: {y_test_pred}\n")

    # Number of images evaluated
    print(f"Number of images evaluated: {len(y_test_pred)}")



if __name__ == "__main__":
    main()
