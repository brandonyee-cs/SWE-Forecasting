#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SWE Forecasting Model Training Script with Hardcoded Parameters

This script trains and evaluates the Snow Water Equivalent (SWE) Forecasting model
using hardcoded parameters and directly importing from models.py and data_generation.py.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import directly from the model and data files
from models import SWETransUNet
from data_generation import prepare_data


def main():
    # Hardcoded parameters
    data_path = "/path/to/your/data.npy"  # IMPORTANT: Replace with your actual data path
    output_dir = "./swe_model_results"
    
    # Model parameters
    model_params = {
        'emb_dim': 384,
        'num_heads': 4,
        'hidden_dim': 512,
        'num_transformer_layers': 2,
        'patch_size': (2, 4, 4),
        'dropout': 0.1
    }
    
    # Training parameters
    batch_size = 4
    num_epochs = 5
    learning_rate = 1e-5
    weight_decay = 1e-4
    early_stopping_patience = 3
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    # Prepare data
    print("Loading and preparing data...")
    train_loader, test_loader, _, _ = prepare_data(
        data_path,
        batch_size=batch_size,
        num_workers=4
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Initialize model
    print("Initializing model...")
    model = SWETransUNet(**model_params).to(device)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=2,
        verbose=True
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': []
    }
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_steps = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for X, y in train_loop:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            # Compute MAE
            with torch.no_grad():
                mae = F.l1_loss(y_pred, y).item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_mae += mae
            train_steps += 1
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item(), mae=mae)
        
        # Calculate average training metrics
        train_loss /= train_steps
        train_mae /= train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_steps = 0
        
        with torch.no_grad():
            val_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for X, y in val_loop:
                X, y = X.to(device), y.to(device)
                
                # Forward pass
                y_pred = model(X)
                loss = criterion(y_pred, y)
                
                # Compute MAE
                mae = F.l1_loss(y_pred, y).item()
                
                # Update metrics
                val_loss += loss.item()
                val_mae += mae
                val_steps += 1
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item(), mae=mae)
        
        # Calculate average validation metrics
        val_loss /= val_steps
        val_mae /= val_steps
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(f"  New best model saved with validation loss: {val_loss:.4f}")
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history
        }
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs without improvement")
            break
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Save history as pickle
    with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Load best model for visualization
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    
    # Visualize sample predictions
    print("Generating sample predictions...")
    plot_sample_predictions(model, test_loader, num_samples=5, 
                           save_path=os.path.join(output_dir, 'sample_predictions.png'), 
                           device=device)
    
    print(f"Training completed. Results saved in {output_dir}")


def plot_sample_predictions(model, test_loader, num_samples=5, save_path='./sample_predictions.png', device='cuda'):
    """Plot and save sample predictions."""
    model.eval()
    
    # Get sample batches
    samples = []
    for X, y in test_loader:
        if len(samples) < num_samples:
            samples.append((X, y))
        else:
            break
    
    plt.figure(figsize=(20, 4*num_samples))
    
    with torch.no_grad():
        for i, (X, y_true) in enumerate(samples):
            X, y_true = X.to(device), y_true.to(device)
            y_pred = model(X)
            
            # Move tensors to CPU for plotting
            X = X.cpu().numpy()
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            
            for j in range(3):  # Plot each channel
                plt.subplot(num_samples, 9, i*9 + j*3 + 1)
                plt.imshow(X[0, 0, -1, :, :, j], cmap='viridis')
                plt.title(f'Input Ch{j}')
                plt.axis('off')
                
                plt.subplot(num_samples, 9, i*9 + j*3 + 2)
                plt.imshow(y_true[0, :, :, j], cmap='viridis')
                plt.title(f'True Ch{j}')
                plt.axis('off')
                
                plt.subplot(num_samples, 9, i*9 + j*3 + 3)
                plt.imshow(y_pred[0, :, :, j], cmap='viridis')
                plt.title(f'Pred Ch{j}')
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    main()