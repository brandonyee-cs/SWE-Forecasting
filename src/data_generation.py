import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SWEDataset(Dataset):
    """
    Dataset for Snow Water Equivalent (SWE) forecasting
    
    This dataset handles loading and preprocessing of the SWE data.
    Each sample consists of:
    - Input X: 12 months of data from 4 sensors (shape: [4, 12, 64, 64, 3])
    - Target y: 1 month of data from the first sensor (shape: [64, 64, 3])
    """
    def __init__(self, data_path, indices, transform=None):
        """
        Initialize the SWE dataset
        
        Args:
            data_path (str): Path to the numpy data file
            indices (list): List of indices to use
            transform (callable, optional): Optional transform to apply to samples
        """
        self.data = np.load(data_path, mmap_mode='r')
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (X, y) where X is the input data and y is the target
        """
        # Get the actual index from the indices list
        index = self.indices[idx]
        
        # Calculate month, row, and column
        month = index // 90 + 12  # Start from 12th month (0-based index)
        block = index % 90
        row = block // 9  # 0-9 for 10 rows
        col = block % 9   # 0-8 for 9 columns
        
        # Extract input and target data
        X = self.data[:, month-12:month, row, col]  # Shape: (4, 12, 64, 64, 3)
        y = self.data[0, month, row, col]           # Shape: (64, 64, 3)
        
        # Replace NaN values with 0
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            X, y = self.transform(X, y)
            
        return X, y

def prepare_data(data_path, batch_size=4, num_workers=4):
    """
    Prepare data loaders for training and testing
    
    Args:
        data_path (str): Path to the data file
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader, train_indices, test_indices)
    """
    # Calculate total number of samples
    total_samples = 100 * 90  # 9000 samples (100 months, 10x9 grid)
    
    # Create random indices for training and testing
    np.random.seed(42)  # for reproducibility
    all_indices = np.random.permutation(total_samples)
    train_indices = all_indices[:7200]  # 80% for training
    test_indices = all_indices[7200:]   # 20% for testing
    
    # Create datasets
    train_dataset = SWEDataset(data_path, train_indices)
    test_dataset = SWEDataset(data_path, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, train_indices, test_indices

# Extended functionality: Data augmentation
class SWETransform:
    """Base class for SWE data transformations"""
    def __call__(self, X, y):
        return X, y

class RandomNoise(SWETransform):
    """Add random noise to the input data"""
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level
        
    def __call__(self, X, y):
        # Add noise only to input data, not target
        noise = torch.randn_like(X) * self.noise_level
        X = X + noise
        return X, y

class RandomFlip(SWETransform):
    """Randomly flip the input and target data"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, X, y):
        if torch.rand(1) < self.p:
            # Flip horizontally
            X = torch.flip(X, dims=[-2])
            y = torch.flip(y, dims=[-2])
        if torch.rand(1) < self.p:
            # Flip vertically
            X = torch.flip(X, dims=[-3])
            y = torch.flip(y, dims=[-3])
        return X, y

class ComposeTransforms(SWETransform):
    """Compose multiple transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, X, y):
        for transform in self.transforms:
            X, y = transform(X, y)
        return X, y

# Example usage of data augmentation
def get_augmented_data(data_path, batch_size=4, num_workers=4, augment=True):
    """Get data loaders with optional augmentation"""
    # Calculate total number of samples
    total_samples = 100 * 90
    
    # Create random indices for training and testing
    np.random.seed(42)
    all_indices = np.random.permutation(total_samples)
    train_indices = all_indices[:7200]
    test_indices = all_indices[7200:]
    
    # Define transforms for training
    if augment:
        train_transform = ComposeTransforms([
            RandomNoise(noise_level=0.03),
            RandomFlip(p=0.3)
        ])
    else:
        train_transform = None
    
    # Create datasets
    train_dataset = SWEDataset(data_path, train_indices, transform=train_transform)
    test_dataset = SWEDataset(data_path, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, train_indices, test_indices

if __name__ == "__main__":
    # Example usage
    data_path = "path/to/your/data.npy"  # Replace with actual path
    
    # Test data loading without augmentation
    train_loader, test_loader, _, _ = prepare_data(data_path)
    
    # Print dataset sizes
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Test a single batch
    for X, y in train_loader:
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        break  # Just test the first batch