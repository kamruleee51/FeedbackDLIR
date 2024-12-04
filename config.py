double_GPU = True  # Set to True if using multiple GPUs for training
shuffle_ = True  # Whether to shuffle the training dataset
shuffle_val = True  # Whether to shuffle the validation dataset

# Batch Sizes
trainBatch = 4  # Batch size for training
testBatch = 4  # Batch size for testing
valBatch = 4  # Batch size for validation

# Training Configuration
Epochs = 250  # Number of training epochs
LR = 1e-4  # Learning rate for the optimizer

# Data Configuration
img_size = 64  # Input image size (e.g., 64x64x64 if working with 3D data)
num_classes = 3  # Number of output classes (background/MYO/LV)

# GPU and DataLoader Settings
num_workers = 0  # Number of parallel workers for data loading (0 for debugging/single-threaded)

# Miscellaneous Parameters
lower_bound = 0.999  # Lower bound for a specific threshold 
upper_bound = 1.99  # Upper bound for a specific threshold 
val_interval = 1  # Interval for validation during training 


# Set weights for the loss terms
# - These weights control the relative importance of different loss components in the total loss function.
# - Proper tuning of these weights is crucial to achieving a balanced optimization that aligns the model's focus
#   on different aspects of the problem, such as spatial alignment, feature similarity, and regularization.
w1 = 10.0
w2 = 10.0
w3 = 0.00001
w4 = 0.000001
w5 = 1.0
w6 = 1.0