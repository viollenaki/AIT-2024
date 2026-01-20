"""
Smile Recognition Model - From Scratch Implementation
=====================================================
This module implements a neural network for smile recognition using only NumPy.
No pre-built ML frameworks (TensorFlow, Keras, PyTorch) are used.

Features:
- Image loading and preprocessing (grayscale, resize, normalize)
- HOG (Histogram of Oriented Gradients) feature extraction
- Custom neural network with forward/backward propagation
- Training with backpropagation and gradient descent
- Model evaluation metrics
"""

import numpy as np
import os
from pathlib import Path


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

class ImagePreprocessor:
    """Handles image loading, preprocessing, and augmentation."""
    
    def __init__(self, target_size=(64, 64)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Tuple (height, width) for resizing images
        """
        self.target_size = target_size
    
    def load_image(self, image_path):
        """
        Load an image from file using basic file reading.
        Supports simple PPM/PGM formats or uses PIL if available.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of the image
        """
        try:
            # Try to use PIL for broader format support
            from PIL import Image
            img = Image.open(image_path)
            return np.array(img)
        except ImportError:
            # Fallback: read PPM/PGM format manually
            return self._read_ppm_pgm(image_path)
    
    def _read_ppm_pgm(self, filepath):
        """Read PPM or PGM image format manually."""
        with open(filepath, 'rb') as f:
            header = f.readline().decode().strip()
            if header == 'P5':  # PGM grayscale
                dims = f.readline().decode().split()
                width, height = int(dims[0]), int(dims[1])
                max_val = int(f.readline().decode().strip())
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data.reshape((height, width))
            elif header == 'P6':  # PPM color
                dims = f.readline().decode().split()
                width, height = int(dims[0]), int(dims[1])
                max_val = int(f.readline().decode().strip())
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data.reshape((height, width, 3))
        raise ValueError(f"Unsupported image format: {filepath}")
    
    def to_grayscale(self, image):
        """
        Convert image to grayscale.
        
        Args:
            image: numpy array (H, W, C) or (H, W)
            
        Returns:
            Grayscale image (H, W)
        """
        if len(image.shape) == 2:
            return image
        elif len(image.shape) == 3:
            # Weighted average for perceptual grayscale
            # Y = 0.299*R + 0.587*G + 0.114*B
            return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    
    def resize(self, image, size=None):
        """
        Resize image using bilinear interpolation.
        
        Args:
            image: numpy array (H, W)
            size: Tuple (new_height, new_width), defaults to self.target_size
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        
        old_h, old_w = image.shape[:2]
        new_h, new_w = size
        
        # Create coordinate grids for the new image
        row_ratio = old_h / new_h
        col_ratio = old_w / new_w
        
        # Generate new coordinates
        row_coords = np.arange(new_h) * row_ratio
        col_coords = np.arange(new_w) * col_ratio
        
        # Get integer and fractional parts
        row_floor = np.floor(row_coords).astype(int)
        col_floor = np.floor(col_coords).astype(int)
        row_ceil = np.minimum(row_floor + 1, old_h - 1)
        col_ceil = np.minimum(col_floor + 1, old_w - 1)
        
        row_frac = row_coords - row_floor
        col_frac = col_coords - col_floor
        
        # Bilinear interpolation
        resized = np.zeros((new_h, new_w), dtype=np.float32)
        
        for i in range(new_h):
            for j in range(new_w):
                top_left = image[row_floor[i], col_floor[j]]
                top_right = image[row_floor[i], col_ceil[j]]
                bottom_left = image[row_ceil[i], col_floor[j]]
                bottom_right = image[row_ceil[i], col_ceil[j]]
                
                top = top_left * (1 - col_frac[j]) + top_right * col_frac[j]
                bottom = bottom_left * (1 - col_frac[j]) + bottom_right * col_frac[j]
                resized[i, j] = top * (1 - row_frac[i]) + bottom * row_frac[i]
        
        return resized.astype(np.uint8)
    
    def normalize(self, image):
        """
        Normalize image to range [0, 1].
        
        Args:
            image: numpy array
            
        Returns:
            Normalized image as float32
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image):
        """
        Full preprocessing pipeline.
        
        Args:
            image: Raw image array
            
        Returns:
            Preprocessed image (grayscale, resized, normalized)
        """
        gray = self.to_grayscale(image)
        resized = self.resize(gray)
        normalized = self.normalize(resized)
        return normalized


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class HOGFeatureExtractor:
    """
    Histogram of Oriented Gradients (HOG) feature extractor.
    Implements HOG from scratch for extracting shape features from images.
    """
    
    def __init__(self, cell_size=8, block_size=2, num_bins=9):
        """
        Initialize HOG extractor.
        
        Args:
            cell_size: Size of each cell in pixels
            block_size: Number of cells per block (for normalization)
            num_bins: Number of orientation bins (typically 9 for 0-180°)
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.num_bins = num_bins
    
    def compute_gradients(self, image):
        """
        Compute image gradients using Sobel-like operators.
        
        Args:
            image: Grayscale image (H, W)
            
        Returns:
            magnitude: Gradient magnitude
            orientation: Gradient orientation (0-180°)
        """
        # Sobel kernels
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)
        
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]], dtype=np.float32)
        
        # Compute gradients using convolution
        gx = self._convolve2d(image, kernel_x)
        gy = self._convolve2d(image, kernel_y)
        
        # Magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180.0 / np.pi)
        
        # Convert to unsigned (0-180 range)
        orientation = np.mod(orientation, 180)
        
        return magnitude, orientation
    
    def _convolve2d(self, image, kernel):
        """
        2D convolution operation.
        
        Args:
            image: Input image (H, W)
            kernel: Convolution kernel
            
        Returns:
            Convolved image
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Output
        output = np.zeros_like(image, dtype=np.float32)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    def compute_cell_histograms(self, magnitude, orientation):
        """
        Compute histogram of gradients for each cell.
        
        Args:
            magnitude: Gradient magnitudes
            orientation: Gradient orientations
            
        Returns:
            Cell histograms array
        """
        h, w = magnitude.shape
        n_cells_y = h // self.cell_size
        n_cells_x = w // self.cell_size
        
        histograms = np.zeros((n_cells_y, n_cells_x, self.num_bins))
        bin_width = 180.0 / self.num_bins
        
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                # Extract cell region
                y_start = cy * self.cell_size
                y_end = y_start + self.cell_size
                x_start = cx * self.cell_size
                x_end = x_start + self.cell_size
                
                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ori = orientation[y_start:y_end, x_start:x_end]
                
                # Build histogram with interpolation
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        mag = cell_mag[i, j]
                        ori = cell_ori[i, j]
                        
                        # Find bin indices
                        bin_idx = ori / bin_width
                        lower_bin = int(bin_idx) % self.num_bins
                        upper_bin = (lower_bin + 1) % self.num_bins
                        
                        # Linear interpolation
                        upper_weight = bin_idx - int(bin_idx)
                        lower_weight = 1 - upper_weight
                        
                        histograms[cy, cx, lower_bin] += mag * lower_weight
                        histograms[cy, cx, upper_bin] += mag * upper_weight
        
        return histograms
    
    def normalize_blocks(self, histograms):
        """
        Normalize histograms over blocks for illumination invariance.
        
        Args:
            histograms: Cell histograms
            
        Returns:
            Normalized feature vector
        """
        n_cells_y, n_cells_x, _ = histograms.shape
        n_blocks_y = n_cells_y - self.block_size + 1
        n_blocks_x = n_cells_x - self.block_size + 1
        
        features = []
        eps = 1e-6  # Small constant to avoid division by zero
        
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                # Extract block
                block = histograms[by:by+self.block_size, 
                                   bx:bx+self.block_size, :].flatten()
                
                # L2 normalization
                norm = np.sqrt(np.sum(block**2) + eps)
                block = block / norm
                
                features.extend(block)
        
        return np.array(features)
    
    def extract(self, image):
        """
        Extract HOG features from an image.
        
        Args:
            image: Preprocessed grayscale image (normalized, 0-1 range)
            
        Returns:
            HOG feature vector
        """
        magnitude, orientation = self.compute_gradients(image)
        histograms = self.compute_cell_histograms(magnitude, orientation)
        features = self.normalize_blocks(histograms)
        return features


class SimpleFeatureExtractor:
    """
    Simple pixel-based feature extractor.
    Uses downsampled pixel intensities as features.
    """
    
    def __init__(self, feature_size=32):
        """
        Initialize feature extractor.
        
        Args:
            feature_size: Size to downsample to before flattening
        """
        self.feature_size = feature_size
    
    def extract(self, image):
        """
        Extract simple pixel intensity features.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Feature vector
        """
        # Downsample if needed
        if image.shape[0] != self.feature_size or image.shape[1] != self.feature_size:
            preprocessor = ImagePreprocessor((self.feature_size, self.feature_size))
            # Handle normalized images (scale up temporarily)
            temp_img = (image * 255).astype(np.uint8)
            image = preprocessor.resize(temp_img) / 255.0
        
        return image.flatten()


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

class ActivationFunctions:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation: 1 / (1 + exp(-x))"""
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(np.float32)
    
    @staticmethod
    def tanh(x):
        """Tanh activation"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh: 1 - tanh(x)^2"""
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU: x if x > 0, else alpha * x"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x):
        """Softmax activation for multi-class classification"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class LossFunctions:
    """Collection of loss functions and their derivatives."""
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """Binary cross-entropy loss."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        """Derivative of binary cross-entropy."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
    
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error loss."""
        return np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """Derivative of MSE."""
        return 2 * (y_pred - y_true) / y_true.size


# =============================================================================
# NEURAL NETWORK LAYERS
# =============================================================================

class DenseLayer:
    """Fully connected (dense) neural network layer."""
    
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Initialize dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of neurons in this layer
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'leaky_relu')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        
        # Xavier/He initialization
        if activation == 'relu' or activation == 'leaky_relu':
            # He initialization for ReLU
            std = np.sqrt(2.0 / input_size)
        else:
            # Xavier initialization for sigmoid/tanh
            std = np.sqrt(2.0 / (input_size + output_size))
        
        self.weights = np.random.randn(input_size, output_size) * std
        self.biases = np.zeros((1, output_size))
        
        # Gradient accumulators (for batch updates)
        self.d_weights = None
        self.d_biases = None
        
        # Cache for backpropagation
        self.input_cache = None
        self.z_cache = None  # Pre-activation values
        
        # Set activation function
        self._set_activation(activation)
    
    def _set_activation(self, activation):
        """Set activation function and its derivative."""
        activations = {
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'leaky_relu': (ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative),
            'linear': (lambda x: x, lambda x: np.ones_like(x))
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation, self.activation_derivative = activations[activation]
    
    def forward(self, X):
        """
        Forward pass through the layer.
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            Activated output (batch_size, output_size)
        """
        self.input_cache = X
        self.z_cache = np.dot(X, self.weights) + self.biases
        return self.activation(self.z_cache)
    
    def backward(self, d_output):
        """
        Backward pass through the layer.
        
        Args:
            d_output: Gradient from the next layer
            
        Returns:
            Gradient to pass to the previous layer
        """
        batch_size = d_output.shape[0]
        
        # Gradient through activation function
        d_z = d_output * self.activation_derivative(self.z_cache)
        
        # Compute gradients
        self.d_weights = np.dot(self.input_cache.T, d_z) / batch_size
        self.d_biases = np.mean(d_z, axis=0, keepdims=True)
        
        # Gradient for previous layer
        d_input = np.dot(d_z, self.weights.T)
        
        return d_input


# =============================================================================
# OPTIMIZERS
# =============================================================================

class SGDOptimizer:
    """Stochastic Gradient Descent optimizer with momentum."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient (0 = no momentum)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, layer, layer_id):
        """
        Update layer parameters.
        
        Args:
            layer: Layer to update
            layer_id: Unique identifier for the layer
        """
        # Initialize velocities if needed
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
        
        # Update velocities
        self.velocities[layer_id]['weights'] = (
            self.momentum * self.velocities[layer_id]['weights'] - 
            self.learning_rate * layer.d_weights
        )
        self.velocities[layer_id]['biases'] = (
            self.momentum * self.velocities[layer_id]['biases'] - 
            self.learning_rate * layer.d_biases
        )
        
        # Update parameters
        layer.weights += self.velocities[layer_id]['weights']
        layer.biases += self.velocities[layer_id]['biases']


class AdamOptimizer:
    """Adam optimizer."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, layer, layer_id):
        """
        Update layer parameters using Adam.
        
        Args:
            layer: Layer to update
            layer_id: Unique identifier for the layer
        """
        self.t += 1
        
        # Initialize moments if needed
        if layer_id not in self.m:
            self.m[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            self.v[layer_id] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
        
        # Update biased first moment estimate
        self.m[layer_id]['weights'] = (
            self.beta1 * self.m[layer_id]['weights'] + 
            (1 - self.beta1) * layer.d_weights
        )
        self.m[layer_id]['biases'] = (
            self.beta1 * self.m[layer_id]['biases'] + 
            (1 - self.beta1) * layer.d_biases
        )
        
        # Update biased second moment estimate
        self.v[layer_id]['weights'] = (
            self.beta2 * self.v[layer_id]['weights'] + 
            (1 - self.beta2) * layer.d_weights**2
        )
        self.v[layer_id]['biases'] = (
            self.beta2 * self.v[layer_id]['biases'] + 
            (1 - self.beta2) * layer.d_biases**2
        )
        
        # Bias correction
        m_hat_w = self.m[layer_id]['weights'] / (1 - self.beta1**self.t)
        m_hat_b = self.m[layer_id]['biases'] / (1 - self.beta1**self.t)
        v_hat_w = self.v[layer_id]['weights'] / (1 - self.beta2**self.t)
        v_hat_b = self.v[layer_id]['biases'] / (1 - self.beta2**self.t)
        
        # Update parameters
        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class NeuralNetwork:
    """
    Custom Neural Network for smile recognition.
    Implements forward propagation, backpropagation, and training.
    """
    
    def __init__(self, layer_sizes, activations=None, optimizer='adam', learning_rate=0.001):
        """
        Initialize neural network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activations: List of activation functions for each layer (excluding input)
            optimizer: 'sgd' or 'adam'
            learning_rate: Learning rate for optimization
        """
        self.layer_sizes = layer_sizes
        self.layers = []
        
        # Default activations: ReLU for hidden layers, sigmoid for output
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['sigmoid']
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                layer_sizes[i], 
                layer_sizes[i + 1], 
                activation=activations[i]
            )
            self.layers.append(layer)
        
        # Set optimizer
        if optimizer == 'sgd':
            self.optimizer = SGDOptimizer(learning_rate=learning_rate)
        elif optimizer == 'adam':
            self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Training history
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            Network output (batch_size, output_size)
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, y_pred):
        """
        Backward pass (backpropagation).
        
        Args:
            y_true: True labels
            y_pred: Predicted values
        """
        # Compute loss gradient
        d_output = LossFunctions.binary_cross_entropy_derivative(y_true, y_pred)
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)
    
    def update_weights(self):
        """Update weights using the optimizer."""
        for i, layer in enumerate(self.layers):
            self.optimizer.update(layer, i)
    
    def train_step(self, X_batch, y_batch):
        """
        Single training step.
        
        Args:
            X_batch: Batch of input data
            y_batch: Batch of labels
            
        Returns:
            loss: Batch loss
        """
        # Forward pass
        y_pred = self.forward(X_batch)
        
        # Compute loss
        loss = LossFunctions.binary_cross_entropy(y_batch, y_pred)
        
        # Backward pass
        self.backward(y_batch, y_pred)
        
        # Update weights
        self.update_weights()
        
        return loss
    
    def fit(self, X_train, y_train, epochs=100, batch_size=32, 
            validation_data=None, verbose=True, early_stopping_patience=10):
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            validation_data: Tuple (X_val, y_val) for validation
            verbose: Print progress
            early_stopping_patience: Stop if validation loss doesn't improve
            
        Returns:
            Training history
        """
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
            
            epoch_loss /= n_batches
            
            # Compute training accuracy
            train_pred = self.predict(X_train)
            train_acc = np.mean((train_pred >= 0.5) == y_train)
            
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(train_acc)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.forward(X_val)
                val_loss = LossFunctions.binary_cross_entropy(y_val, val_pred)
                val_acc = np.mean((val_pred >= 0.5) == y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f} - Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f} - Acc: {train_acc:.4f}")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.forward(X)
    
    def predict_classes(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Args:
            X: Input data
            threshold: Classification threshold
            
        Returns:
            Binary class labels
        """
        predictions = self.predict(X)
        return (predictions >= threshold).astype(int)
    
    def save_weights(self, filepath):
        """Save model weights to file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f'layer_{i}_weights'] = layer.weights
            weights[f'layer_{i}_biases'] = layer.biases
        np.savez(filepath, **weights)
    
    def load_weights(self, filepath):
        """Load model weights from file."""
        weights = np.load(filepath)
        for i, layer in enumerate(self.layers):
            layer.weights = weights[f'layer_{i}_weights']
            layer.biases = weights[f'layer_{i}_biases']


# =============================================================================
# EVALUATION METRICS
# =============================================================================

class Metrics:
    """Model evaluation metrics."""
    
    @staticmethod
    def accuracy(y_true, y_pred, threshold=0.5):
        """Compute accuracy."""
        y_pred_binary = (y_pred >= threshold).astype(int)
        return np.mean(y_pred_binary == y_true)
    
    @staticmethod
    def precision(y_true, y_pred, threshold=0.5):
        """Compute precision."""
        y_pred_binary = (y_pred >= threshold).astype(int)
        true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred_binary == 1)
        return true_positives / max(predicted_positives, 1)
    
    @staticmethod
    def recall(y_true, y_pred, threshold=0.5):
        """Compute recall (sensitivity)."""
        y_pred_binary = (y_pred >= threshold).astype(int)
        true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / max(actual_positives, 1)
    
    @staticmethod
    def f1_score(y_true, y_pred, threshold=0.5):
        """Compute F1 score."""
        prec = Metrics.precision(y_true, y_pred, threshold)
        rec = Metrics.recall(y_true, y_pred, threshold)
        return 2 * (prec * rec) / max(prec + rec, 1e-10)
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, threshold=0.5):
        """
        Compute confusion matrix.
        
        Returns:
            Dictionary with TP, TN, FP, FN counts
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true == 0))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        
        return {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    @staticmethod
    def classification_report(y_true, y_pred, threshold=0.5):
        """Generate a classification report."""
        acc = Metrics.accuracy(y_true, y_pred, threshold)
        prec = Metrics.precision(y_true, y_pred, threshold)
        rec = Metrics.recall(y_true, y_pred, threshold)
        f1 = Metrics.f1_score(y_true, y_pred, threshold)
        cm = Metrics.confusion_matrix(y_true, y_pred, threshold)
        
        report = f"""
Classification Report
=====================
Accuracy:  {acc:.4f}
Precision: {prec:.4f}
Recall:    {rec:.4f}
F1 Score:  {f1:.4f}

Confusion Matrix:
                Predicted
                Neg    Pos
Actual Neg      {cm['true_negatives']:<6} {cm['false_positives']:<6}
       Pos      {cm['false_negatives']:<6} {cm['true_positives']:<6}
"""
        return report


# =============================================================================
# SMILE RECOGNITION MODEL
# =============================================================================

class SmileRecognitionModel:
    """
    Complete smile recognition pipeline.
    Combines preprocessing, feature extraction, and neural network.
    """
    
    def __init__(self, image_size=(64, 64), feature_type='hog', 
                 hidden_layers=[128, 64], learning_rate=0.001):
        """
        Initialize smile recognition model.
        
        Args:
            image_size: Target image size
            feature_type: 'hog' or 'simple'
            hidden_layers: List of hidden layer sizes
            learning_rate: Learning rate for training
        """
        self.image_size = image_size
        self.feature_type = feature_type
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(target_size=image_size)
        
        if feature_type == 'hog':
            self.feature_extractor = HOGFeatureExtractor(cell_size=8, block_size=2)
        else:
            self.feature_extractor = SimpleFeatureExtractor(feature_size=32)
        
        self.model = None
        self.feature_size = None
    
    def _compute_feature_size(self, sample_image):
        """Compute feature size from a sample image."""
        preprocessed = self.preprocessor.preprocess(sample_image)
        features = self.feature_extractor.extract(preprocessed)
        return len(features)
    
    def prepare_data(self, images, labels):
        """
        Prepare data for training/prediction.
        
        Args:
            images: List of images (numpy arrays)
            labels: List of labels (0 = no smile, 1 = smile)
            
        Returns:
            X: Feature matrix
            y: Label array
        """
        features_list = []
        
        for img in images:
            preprocessed = self.preprocessor.preprocess(img)
            features = self.feature_extractor.extract(preprocessed)
            features_list.append(features)
        
        X = np.array(features_list)
        y = np.array(labels).reshape(-1, 1)
        
        return X, y
    
    def build_model(self, input_size):
        """
        Build the neural network model.
        
        Args:
            input_size: Size of input features
        """
        self.feature_size = input_size
        layer_sizes = [input_size] + self.hidden_layers + [1]
        
        # Use ReLU for hidden layers, sigmoid for output
        activations = ['relu'] * len(self.hidden_layers) + ['sigmoid']
        
        self.model = NeuralNetwork(
            layer_sizes=layer_sizes,
            activations=activations,
            optimizer='adam',
            learning_rate=self.learning_rate
        )
    
    def train(self, X_train, y_train, epochs=100, batch_size=32,
              validation_split=0.2, verbose=True):
        """
        Train the model.
        
        Args:
            X_train: Training features or images
            y_train: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            verbose: Print progress
            
        Returns:
            Training history
        """
        # Split data for validation
        n_samples = X_train.shape[0]
        n_val = int(n_samples * validation_split)
        
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Train
        history = self.model.fit(
            X_train_split, y_train_split,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_image(self, image):
        """
        Predict smile probability for a single image.
        
        Args:
            image: Input image
            
        Returns:
            Probability of smile
        """
        preprocessed = self.preprocessor.preprocess(image)
        features = self.feature_extractor.extract(preprocessed)
        prediction = self.model.predict(features.reshape(1, -1))
        return prediction[0, 0]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_test)
        
        return {
            'accuracy': Metrics.accuracy(y_test, predictions),
            'precision': Metrics.precision(y_test, predictions),
            'recall': Metrics.recall(y_test, predictions),
            'f1_score': Metrics.f1_score(y_test, predictions),
            'confusion_matrix': Metrics.confusion_matrix(y_test, predictions)
        }
    
    def save(self, filepath):
        """Save model weights."""
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        """Load model weights."""
        self.model.load_weights(filepath)


# =============================================================================
# DATA GENERATOR (FOR TESTING)
# =============================================================================

class SyntheticDataGenerator:
    """
    Generate synthetic smile/no-smile data for testing.
    Creates simple patterns that mimic facial expressions.
    """
    
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
    
    def generate_smile_face(self):
        """Generate a synthetic smiling face."""
        img = np.ones(self.image_size) * 200  # Light background
        h, w = self.image_size
        
        # Draw face circle
        y, x = np.ogrid[:h, :w]
        center = (h // 2, w // 2)
        radius = min(h, w) // 2 - 5
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        img[mask] = 255
        
        # Draw eyes
        eye_y = h // 3
        left_eye_x = w // 3
        right_eye_x = 2 * w // 3
        
        for eye_x in [left_eye_x, right_eye_x]:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy**2 + dx**2 <= 9:
                        img[eye_y + dy, eye_x + dx] = 50
        
        # Draw smile (curved line)
        mouth_y = 2 * h // 3
        for dx in range(-w // 4, w // 4):
            x_pos = w // 2 + dx
            y_offset = int((dx**2) / (w // 3))  # Parabola for smile
            y_pos = mouth_y + y_offset
            if 0 <= y_pos < h:
                img[y_pos, x_pos] = 50
                if y_pos + 1 < h:
                    img[y_pos + 1, x_pos] = 50
        
        return img.astype(np.uint8)
    
    def generate_no_smile_face(self):
        """Generate a synthetic non-smiling face."""
        img = np.ones(self.image_size) * 200
        h, w = self.image_size
        
        # Draw face circle
        y, x = np.ogrid[:h, :w]
        center = (h // 2, w // 2)
        radius = min(h, w) // 2 - 5
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        img[mask] = 255
        
        # Draw eyes
        eye_y = h // 3
        left_eye_x = w // 3
        right_eye_x = 2 * w // 3
        
        for eye_x in [left_eye_x, right_eye_x]:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy**2 + dx**2 <= 9:
                        img[eye_y + dy, eye_x + dx] = 50
        
        # Draw straight/frowning mouth
        mouth_y = 2 * h // 3
        for dx in range(-w // 4, w // 4):
            x_pos = w // 2 + dx
            y_offset = -int((dx**2) / (w // 2))  # Slight frown
            y_pos = mouth_y + y_offset
            if 0 <= y_pos < h:
                img[y_pos, x_pos] = 50
        
        return img.astype(np.uint8)
    
    def generate_dataset(self, n_samples=100, noise_level=0.1):
        """
        Generate a synthetic dataset.
        
        Args:
            n_samples: Total number of samples
            noise_level: Amount of noise to add
            
        Returns:
            images: List of images
            labels: List of labels (0 or 1)
        """
        images = []
        labels = []
        
        for i in range(n_samples):
            if i % 2 == 0:
                img = self.generate_smile_face()
                label = 1
            else:
                img = self.generate_no_smile_face()
                label = 0
            
            # Add noise
            noise = np.random.randn(*img.shape) * noise_level * 255
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Random transformations
            if np.random.random() > 0.5:
                img = np.fliplr(img)  # Horizontal flip
            
            images.append(img)
            labels.append(label)
        
        return images, labels


# =============================================================================
# DEMONSTRATION / MAIN
# =============================================================================

def demo():
    """
    Demonstration of the smile recognition model with synthetic data.
    """
    print("=" * 60)
    print("Smile Recognition Model - Demo")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic dataset...")
    generator = SyntheticDataGenerator(image_size=(64, 64))
    images, labels = generator.generate_dataset(n_samples=200, noise_level=0.1)
    print(f"   Generated {len(images)} images")
    print(f"   Smiles: {sum(labels)}, No smiles: {len(labels) - sum(labels)}")
    
    # Initialize model
    print("\n2. Initializing smile recognition model...")
    model = SmileRecognitionModel(
        image_size=(64, 64),
        feature_type='hog',  # Use HOG features
        hidden_layers=[128, 64],
        learning_rate=0.001
    )
    
    # Prepare data
    print("\n3. Preparing data (preprocessing & feature extraction)...")
    X, y = model.prepare_data(images, labels)
    print(f"   Feature shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Split data
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"   Training samples: {len(train_idx)}")
    print(f"   Test samples: {len(test_idx)}")
    
    # Train model
    print("\n4. Training neural network...")
    print("-" * 40)
    history = model.train(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=True
    )
    
    # Evaluate
    print("\n5. Evaluating model...")
    print("-" * 40)
    metrics = model.evaluate(X_test, y_test)
    
    print(f"\nTest Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"   True Positives:  {cm['true_positives']}")
    print(f"   True Negatives:  {cm['true_negatives']}")
    print(f"   False Positives: {cm['false_positives']}")
    print(f"   False Negatives: {cm['false_negatives']}")
    
    # Single prediction example
    print("\n6. Single image prediction example...")
    print("-" * 40)
    
    # Generate a new smile image
    smile_img = generator.generate_smile_face()
    no_smile_img = generator.generate_no_smile_face()
    
    smile_prob = model.predict_image(smile_img)
    no_smile_prob = model.predict_image(no_smile_img)
    
    print(f"   Smile image prediction: {smile_prob:.4f} (expected ~1.0)")
    print(f"   No-smile image prediction: {no_smile_prob:.4f} (expected ~0.0)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return model, history


def load_real_dataset(data_dir, smile_dir='smile', no_smile_dir='no_smile'):
    """
    Load a real dataset from directories.
    
    Args:
        data_dir: Root directory containing smile and no_smile subdirectories
        smile_dir: Name of directory containing smile images
        no_smile_dir: Name of directory containing no-smile images
        
    Returns:
        images: List of images
        labels: List of labels
    """
    images = []
    labels = []
    
    preprocessor = ImagePreprocessor()
    
    # Load smile images
    smile_path = os.path.join(data_dir, smile_dir)
    if os.path.exists(smile_path):
        for filename in os.listdir(smile_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.pgm')):
                img_path = os.path.join(smile_path, filename)
                try:
                    img = preprocessor.load_image(img_path)
                    images.append(img)
                    labels.append(1)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Load no-smile images
    no_smile_path = os.path.join(data_dir, no_smile_dir)
    if os.path.exists(no_smile_path):
        for filename in os.listdir(no_smile_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.pgm')):
                img_path = os.path.join(no_smile_path, filename)
                try:
                    img = preprocessor.load_image(img_path)
                    images.append(img)
                    labels.append(0)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(images)} images ({sum(labels)} smiles, {len(labels) - sum(labels)} no-smiles)")
    return images, labels


def train_on_real_data():
    """Train model on real dataset from the dataset folder."""
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "dataset")
    
    # Check if dataset exists and has images
    smile_path = os.path.join(data_dir, "smile")
    no_smile_path = os.path.join(data_dir, "no_smile")
    
    if not os.path.exists(smile_path) or not os.path.exists(no_smile_path):
        print("Error: dataset/smile and dataset/no_smile folders not found!")
        print(f"Expected location: {data_dir}")
        return None
    
    smile_count = len([f for f in os.listdir(smile_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.pgm'))])
    no_smile_count = len([f for f in os.listdir(no_smile_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.pgm'))])
    
    if smile_count == 0 or no_smile_count == 0:
        print("=" * 60)
        print("Dataset folders are empty!")
        print("=" * 60)
        print(f"\nPlease add images to:")
        print(f"  - Smile images:    {smile_path}")
        print(f"  - No-smile images: {no_smile_path}")
        print(f"\nSupported formats: .jpg, .jpeg, .png, .ppm, .pgm")
        print("\nRunning demo with synthetic data instead...")
        print("=" * 60 + "\n")
        return demo()
    
    print("=" * 60)
    print("Smile Recognition Model - Training on Real Data")
    print("=" * 60)
    
    # Load real dataset
    print("\n1. Loading dataset...")
    images, labels = load_real_dataset(data_dir)
    
    if len(images) == 0:
        print("No images found!")
        return None
    
    # Initialize model
    print("\n2. Initializing smile recognition model...")
    model = SmileRecognitionModel(
        image_size=(64, 64),
        feature_type='hog',
        hidden_layers=[128, 64],
        learning_rate=0.001
    )
    
    # Prepare data
    print("\n3. Preparing data (preprocessing & feature extraction)...")
    X, y = model.prepare_data(images, labels)
    print(f"   Feature shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Split data
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"   Training samples: {len(train_idx)}")
    print(f"   Test samples: {len(test_idx)}")
    
    # Train model
    print("\n4. Training neural network...")
    print("-" * 40)
    history = model.train(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=True
    )
    
    # Evaluate
    if len(test_idx) > 0:
        print("\n5. Evaluating model...")
        print("-" * 40)
        metrics = model.evaluate(X_test, y_test)
        
        print(f"\nTest Results:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"   True Positives:  {cm['true_positives']}")
        print(f"   True Negatives:  {cm['true_negatives']}")
        print(f"   False Positives: {cm['false_positives']}")
        print(f"   False Negatives: {cm['false_negatives']}")
    
    # Save model
    weights_path = os.path.join(script_dir, "smile_model_weights.npz")
    model.save(weights_path)
    print(f"\n6. Model weights saved to: {weights_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    # Try to train on real data first, fall back to demo if no data
    result = train_on_real_data()
