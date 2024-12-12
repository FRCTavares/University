import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

import numpy as np
import random 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers # type: ignore 
from tensorflow.keras import Model # type: ignore
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.optimizers import Adam  # type: ignore


from tqdm import tqdm  # Add tqdm for progress bars

# ------------- Load data --------------
def load_files():
    Xtrain2a = np.load('Xtrain2_a.npy')
    Ytrain2a = np.load('Ytrain2_a.npy')
    Xtest2a = np.load('Xtest2_a.npy')

    Xtrain2b = np.load('Xtrain2_b.npy')
    Ytrain2b = np.load('Ytrain2_b.npy')
    Xtest2b = np.load('Xtest2_b.npy')

    # Reshape to (num_images, 48, 48, 1)
    Xtrain2b = Xtrain2b.reshape((Xtrain2b.shape[0], 48, 48, 1))
    Ytrain2b = Ytrain2b.reshape((Ytrain2b.shape[0], 48, 48, 1))  # Segmentation maps
    Xtest2b = Xtest2b.reshape((Xtest2b.shape[0], 48, 48, 1))
    
    return Xtrain2a, Ytrain2a, Xtest2a, Xtrain2b, Ytrain2b, Xtest2b

# ------------- Data Augmentation --------------
def augment_images(X, Y):
    """Augment images by applying random rotations (90, 180, 270)"""
    augmented_X = []
    augmented_Y = []
    
    for img, mask in zip(X, Y):
        # Original image and mask
        augmented_X.append(img)
        augmented_Y.append(mask)

        # Apply random rotations (90, 180, 270 degrees)
        angle = np.random.choice([1, 2, 3])  # Choose a random angle (1=90°, 2=180°, 3=270°)
        rotated_img = np.rot90(img, angle)
        rotated_mask = np.rot90(mask, angle)

        augmented_X.append(rotated_img)
        augmented_Y.append(rotated_mask)
        
    return np.array(augmented_X), np.array(augmented_Y)

# ------------- U-Net Model --------------
def unet_model(input_size=(48, 48, 1), learning_rate=0.001):
    inputs = layers.Input(input_size)
    
    # Encoder (downsampling path)
    # Block 1
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4 (bottleneck)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Block 5 (bottom of the U)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    
    # Decoder (upsampling path)
    # Block 6
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    
    # Block 7
    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    
    # Block 8
    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    
    # Block 9
    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    
    # Output layer
    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    
    return model

# ------------- K-Folds CV ---------------
def cross_validate_model(X, Y, n_splits=5, epochs=15, batch_size=16, model_type='Model', learning_rate=0.001):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    balanced_accuracies = []
    best_model = None
    best_accuracy = 0.0


    for train_idx, val_idx in kf.split(X):
        print(f"Training fold {fold}...")
        
        # Split data into train and validation sets
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        Y_train_fold, Y_val_fold = Y[train_idx], Y[val_idx]

        if model_type == 'Random Forest':
            print("Training Random Forest model...")

            # Initialize RandomForestClassifier 
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

            # Display progress for training using tqdm
            for _ in tqdm(range(1), desc=f"Training RF on fold {fold}"):
                # Fit the model on the training data
                rf_model.fit(X_train_fold, Y_train_fold)
            
            # Predict on validation set
            Y_val_pred = rf_model.predict(X_val_fold)
            
        elif model_type == 'U-Net':
            # Augment training data
            X_train_fold, Y_train_fold = augment_images(X_train_fold, Y_train_fold) 
            Y_train_fold_flat = Y_train_fold.flatten()
            
            class_0_count = np.sum(Y_train_fold_flat == 0)
            class_1_count = np.sum(Y_train_fold_flat == 1)
            
            total_pixels = class_0_count + class_1_count
            
            weight_0 = total_pixels / (2 * class_0_count)
            weight_1 = total_pixels / (2 * class_1_count)
            
            class_weights = np.where(Y_train_fold == 0, weight_0, weight_1)
            
            # Create and compile the U-Net model
            model = unet_model(learning_rate=learning_rate)  # Pass the learning rate
            model.compile(optimizer=Adam(learning_rate=learning_rate),  # Use Adam optimizer with learning rate
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])

            model.fit(
                X_train_fold, Y_train_fold, 
                epochs=epochs, batch_size=batch_size, 
                verbose=1, 
                validation_data=(X_val_fold, Y_val_fold), 
                sample_weight=class_weights,
            )
        
            # Predict on validation set
            Y_val_pred = model.predict(X_val_fold)
            Y_val_pred = (Y_val_pred > 0.5).astype(int)  # Binary segmentation

        # Flatten for calculating Balanced Accuracy
        Y_val_true_flat = Y_val_fold.flatten()
        Y_val_pred_flat = Y_val_pred.flatten()

        # Calculate Balanced Accuracy
        balanced_accuracy = balanced_accuracy_score(Y_val_true_flat, Y_val_pred_flat)
        balanced_accuracies.append(balanced_accuracy)

        # Track the best model
        if balanced_accuracy > best_accuracy:
            best_accuracy = balanced_accuracy
            best_model = rf_model if model_type == 'Random Forest' else model

        print(f"Balanced Accuracy for fold {fold}: {balanced_accuracy:.5f}")
        fold += 1

    avg_balanced_accuracy = np.mean(balanced_accuracies)
    print(f"\nAverage Balanced Accuracy across all folds: {avg_balanced_accuracy:.5f}\n")

    return best_model  # Return the best model after all folds


# ------------- Main function to run the models --------------
def main():
    X_trainA, Y_trainA, X_testA, X_trainB, Y_trainB, X_testB = load_files()

    print("Choose the model type to run:")
    print("1: Random Forest (Dataset A)")
    print("2: U-Net (Dataset B)")
    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        X_train = X_trainA
        Y_train = Y_trainA
        X_test = X_testA

        model_type = "Random Forest"
    elif choice == '2':
        X_train = X_trainB
        Y_train = Y_trainB
        X_test = X_testB

        model_type = "U-Net"
    else:
        print("Invalid choice. Exiting.")
        return

    # Cross-validate and get the best model
    best_model = cross_validate_model(X_train, Y_train, n_splits=5, epochs=25, batch_size=16, model_type=model_type, learning_rate=0.001)

    # Once you have the best model, you can make predictions on the test set
    Y_test_pred = best_model.predict(X_test)


    print(f"Predictions shape: {Y_test_pred.shape}")
    # Shape here is (196, 48, 48, 1)


    # Reshape the predictions to binary masks (0 or 1)
    Y_test_pred = (Y_test_pred > 0.5).astype(np.uint8)  # Assuming the model outputs probabilities

    # Display 5 random images along with predictions
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows for original and predicted
    for i in range(5):
        idx = random.randint(0, X_test.shape[0] - 1)
        
        # Original image
        axes[0, i].imshow(X_test[idx].reshape((48, 48)), cmap='gray')
        axes[0, i].set_title("Original Image")
        axes[0, i].axis('off')

        # Predicted mask
        axes[1, i].imshow(Y_test_pred[idx].reshape((48, 48)), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title("Predicted Mask")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Predictions shape after reshaping: {Y_test_pred.shape}")

    # Save predictions to file
    np.save('Group70_P4/Y_test_predictions.npy', Y_test_pred)
    print("Predictions saved to 'Y_test_predictions.npy'")

if __name__ == "__main__":
    main()
