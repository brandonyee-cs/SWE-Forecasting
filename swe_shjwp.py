# Snow Water Equivalent (SWE) Forecasting with XML Data - Chunked Processing
#
# This script implements a SWE forecasting system using the SWETransUNet model with XML data.
# It includes:
# - Memory-efficient chunked loading of XML data
# - Lazy loading and on-demand processing to minimize memory usage
# - Model training and evaluation with Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) metrics
# - Visualization of data and predictions, saved to a 'results' directory
#
# Setup Notes:
# - Requires Python 3.8+
# - For GPU support, install PyTorch with CUDA (adjust for your CUDA version if needed)
# - Ensure required dependencies are installed:
#   pip install numpy torch matplotlib tqdm scikit-learn h5py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import glob
import xml.etree.ElementTree as ET
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
import h5py
import gc
# import tempfile # Not explicitly used in the final script flow, but was in imports
from concurrent.futures import ThreadPoolExecutor
import time

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --- 1. Mount Google Drive and Set Up Data Access (if in Colab) ---
def mount_drive(drive_path='/content/drive'):
    if IN_COLAB:
        try:
            if not os.path.exists(drive_path):
                drive.mount(drive_path)
                print("Google Drive mounted successfully")
            else:
                print("Google Drive already mounted")
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")
    else:
        print("Not in Google Colab. Skipping Drive mount.")

# --- 2. XML Data Processing with Chunking ---
class XMLProcessor:
    def __init__(self, directory_path, cache_dir=None, chunk_size=5):
        """
        Initialize the XML processor with chunking capabilities.

        Args:
            directory_path: Path to the directory containing XML files
            cache_dir: Directory to store processed chunks (if None, no caching)
            chunk_size: Number of XML files to process at once
        """
        self.directory_path = directory_path
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.xml_files = None
        self.dimensions = None
        self._scan_files()

    def _scan_files(self):
        """Scan the directory for XML files and get their count"""
        if not os.path.exists(self.directory_path):
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")

        self.xml_files = sorted(glob.glob(os.path.join(self.directory_path, "*.xml")))
        print(f"Found {len(self.xml_files)} XML files in {self.directory_path}")

        if len(self.xml_files) == 0:
            raise FileNotFoundError(f"No XML files found in directory: {self.directory_path}")

        # Extract data dimensions from the first file
        sample_tree = ET.parse(self.xml_files[0])
        sample_root = sample_tree.getroot()
        self.dimensions = self._extract_dimensions(sample_root)
        print(f"Data dimensions: {self.dimensions}")

    def _extract_dimensions(self, root):
        """
        Extract data dimensions from XML root element.
        Adjust this function based on your XML structure.
        """
        sensors = int(root.find('.//sensors').text) if root.find('.//sensors') is not None else 4
        rows = int(root.find('.//grid_rows').text) if root.find('.//grid_rows') is not None else 10
        cols = int(root.find('.//grid_cols').text) if root.find('.//grid_cols') is not None else 9
        height = int(root.find('.//height').text) if root.find('.//height') is not None else 64
        width = int(root.find('.//width').text) if root.find('.//width') is not None else 64
        channels = int(root.find('.//channels').text) if root.find('.//channels') is not None else 3

        # If dimensions not found in XML, use defaults
        if sensors == 0 or rows == 0 or cols == 0 or height == 0 or width == 0 or channels == 0:
            print("Warning: Some dimensions not found in XML. Using defaults.")
            sensors = sensors or 4
            rows = rows or 10
            cols = cols or 9
            height = height or 64
            width = width or 64
            channels = channels or 3

        return {
            'sensors': sensors,
            'rows': rows,
            'cols': cols,
            'height': height,
            'width': width,
            'channels': channels,
            'total_files': len(self.xml_files)
        }

    def _extract_data_from_xml(self, root, sensors, rows, cols, height, width, channels):
        """
        Extract data from XML into numpy array.
        Adjust this function based on your XML structure.
        """
        # Initialize data array for this file
        data = np.zeros((sensors, rows, cols, height, width, channels), dtype=np.float32)

        # This is a placeholder - adjust based on your actual XML structure
        for sensor_idx, sensor_elem in enumerate(root.findall('.//sensor')):
            if sensor_idx >= sensors:
                break

            for row_idx, row_elem in enumerate(sensor_elem.findall('./row')):
                if row_idx >= rows:
                    break

                for col_idx, col_elem in enumerate(row_elem.findall('./col')):
                    if col_idx >= cols:
                        break

                    # Extract SWE, temperature, and precipitation data
                    swe_data = self._extract_channel_data(col_elem, 'swe', height, width)
                    temp_data = self._extract_channel_data(col_elem, 'temperature', height, width)
                    precip_data = self._extract_channel_data(col_elem, 'precipitation', height, width)

                    # Assign data to array
                    data[sensor_idx, row_idx, col_idx, :, :, 0] = swe_data
                    data[sensor_idx, row_idx, col_idx, :, :, 1] = temp_data
                    data[sensor_idx, row_idx, col_idx, :, :, 2] = precip_data

        # Handle missing values (NaN) - optional
        # Create a mask for NaN values (5% as in original code)
        mask = np.random.random(data.shape) < 0.05
        data[mask] = np.nan

        return data

    def _extract_channel_data(self, col_elem, channel_name, height, width):
        """
        Extract channel data (SWE, temperature, precipitation) from XML element.
        """
        channel_elem = col_elem.find(f'./{channel_name}')

        if channel_elem is not None:
            # Option 1: If data is stored as comma-separated values in text
            if channel_elem.text:
                try:
                    values = [float(x) for x in channel_elem.text.strip().split(',')]
                    if len(values) == height * width:
                        return np.array(values).reshape(height, width)
                except ValueError:
                    pass

            # Option 2: If data is stored as individual cell elements
            cells = channel_elem.findall('./cell')
            if len(cells) == height * width:
                data = np.zeros((height, width))
                for i, cell in enumerate(cells):
                    row_val = i // width
                    col_val = i % width
                    data[row_val, col_val] = float(cell.text)
                return data

        # If data not found or in unexpected format, generate placeholder data
        seasonal_factor = 0.5
        if channel_name == 'swe':
            return seasonal_factor * np.ones((height, width))
        elif channel_name == 'temperature':
            return -seasonal_factor * np.ones((height, width))
        else:  # precipitation
            return seasonal_factor * 0.5 * np.ones((height, width))

    def _get_cache_path(self, chunk_idx):
        """Generate a path for caching a specific chunk"""
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"chunk_{chunk_idx}.h5")

    def _process_chunk(self, chunk_idx):
        """
        Process a chunk of XML files and return the data or save to cache.

        Args:
            chunk_idx: Index of the chunk to process

        Returns:
            Numpy array with the processed data or path to cached file
        """
        cache_path = self._get_cache_path(chunk_idx)

        # If cache exists and we're using caching, return the cache path
        if cache_path and os.path.exists(cache_path):
            return cache_path

        # Calculate the indices for this chunk
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self.xml_files))
        chunk_files = self.xml_files[start_idx:end_idx]

        # Create a data array for this chunk
        sensors = self.dimensions['sensors']
        rows = self.dimensions['rows']
        cols = self.dimensions['cols']
        height = self.dimensions['height']
        width = self.dimensions['width']
        channels = self.dimensions['channels']

        chunk_data = np.zeros((sensors, len(chunk_files), rows, cols, height, width, channels), dtype=np.float32)

        # Process each file in the chunk
        for i, file_path in enumerate(chunk_files):
            tree = ET.parse(file_path)
            root = tree.getroot()
            file_data = self._extract_data_from_xml(root, sensors, rows, cols, height, width, channels)
            chunk_data[:, i, :, :, :, :, :] = file_data

        # If caching is enabled, save to cache
        if cache_path:
            with h5py.File(cache_path, 'w') as f:
                f.create_dataset('data', data=chunk_data)
            return cache_path

        return chunk_data

    def get_num_chunks(self):
        """Get the total number of chunks"""
        return (len(self.xml_files) + self.chunk_size - 1) // self.chunk_size

    def get_metadata(self):
        """Get metadata for the dataset"""
        return {
            'dimensions': self.dimensions,
            'num_files': len(self.xml_files),
            'num_chunks': self.get_num_chunks(),
            'chunk_size': self.chunk_size
        }

    def preprocess_all_chunks(self, num_workers=4):
        """
        Preprocess all chunks in parallel to create cache files.
        Only applicable if caching is enabled.

        Args:
            num_workers: Number of worker threads for parallel processing
        """
        if self.cache_dir is None:
            print("Caching is disabled. Set cache_dir to enable caching.")
            return

        num_chunks = self.get_num_chunks()
        print(f"Preprocessing {num_chunks} chunks in parallel with {num_workers} workers...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_chunk, i) for i in range(num_chunks)]

            # Show progress as chunks are processed
            for _ in tqdm(futures, desc="Preprocessing chunks"): # Iterate to ensure completion
                pass # result() is implicitly called by iterating

        print("All chunks preprocessed and cached successfully.")

    def get_chunk(self, chunk_idx):
        """
        Get a specific chunk of data, either from cache or by processing on-demand.

        Args:
            chunk_idx: Index of the chunk to retrieve

        Returns:
            Numpy array with the chunk data
        """
        if chunk_idx >= self.get_num_chunks():
            raise ValueError(f"Chunk index {chunk_idx} is out of range (max: {self.get_num_chunks()-1})")

        result = self._process_chunk(chunk_idx)

        # If the result is a file path (caching is enabled), load from cache
        if isinstance(result, str) and os.path.exists(result):
            with h5py.File(result, 'r') as f:
                return f['data'][:]  # Load the entire dataset into memory

        return result  # Return the data array directly if caching is disabled

    def get_file_indices_for_month(self, month_idx):
        """
        Map month index to file indices (XML files)

        Args:
            month_idx: Index of the month

        Returns:
            Tuple of (file_index, chunk_index, index_within_chunk)
        """
        if month_idx >= len(self.xml_files):
            raise ValueError(f"Month index {month_idx} is out of range (max: {len(self.xml_files)-1})")

        chunk_idx = month_idx // self.chunk_size
        idx_in_chunk = month_idx % self.chunk_size

        return month_idx, chunk_idx, idx_in_chunk

    def get_month_data(self, month_idx):
        """
        Get data for a specific month.

        Args:
            month_idx: Index of the month

        Returns:
            Numpy array with data for the specified month
        """
        _ , chunk_idx, idx_in_chunk = self.get_file_indices_for_month(month_idx) # file_idx not needed here
        chunk_data = self.get_chunk(chunk_idx)

        # Return the data for the specific month within the chunk
        return chunk_data[:, idx_in_chunk, :, :, :, :, :]

    def get_sequence_data(self, start_month, sequence_length):
        """
        Get data for a sequence of consecutive months.

        Args:
            start_month: Starting month index
            sequence_length: Number of consecutive months to retrieve

        Returns:
            Numpy array with data for the sequence
        """
        # Check if the sequence is valid
        if start_month + sequence_length > len(self.xml_files):
            raise ValueError(f"Sequence from month {start_month} with length {sequence_length} exceeds available data ({len(self.xml_files)} files)")

        # Initialize array for the sequence
        sensors = self.dimensions['sensors']
        rows = self.dimensions['rows']
        cols = self.dimensions['cols']
        height = self.dimensions['height']
        width = self.dimensions['width']
        channels = self.dimensions['channels']

        sequence_data = np.zeros((sensors, sequence_length, rows, cols, height, width, channels), dtype=np.float32)

        # Check if all months in the sequence are in the same chunk
        start_chunk = start_month // self.chunk_size
        end_chunk = (start_month + sequence_length - 1) // self.chunk_size

        if start_chunk == end_chunk:
            # All months are in the same chunk, so we can retrieve them at once
            chunk_data = self.get_chunk(start_chunk)
            start_idx_in_chunk = start_month % self.chunk_size
            sequence_data = chunk_data[:, start_idx_in_chunk : start_idx_in_chunk + sequence_length, :, :, :, :, :]
        else:
            # Sequence spans multiple chunks, retrieve month by month
            for i in range(sequence_length):
                month_data = self.get_month_data(start_month + i)
                sequence_data[:, i, :, :, :, :, :] = month_data

        return sequence_data

    def get_sample(self, sensor, start_month, row, col, sequence_length, target_month):
        """
        Get a specific sample for model training/testing.

        Args:
            sensor: Sensor index
            start_month: Starting month index
            row: Grid row index
            col: Grid column index
            sequence_length: Number of consecutive months for input sequence
            target_month: Month index for the target (prediction)

        Returns:
            Tuple of (X, y) - input sequence and target
        """
        # Get input sequence data
        sequence_data = self.get_sequence_data(start_month, sequence_length)
        X = sequence_data[sensor, :, row, col, :, :, :]

        # Get target data
        target_data = self.get_month_data(target_month)
        y = target_data[sensor, row, col, :, :, 0:1]  # Only the SWE channel for target

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        return X, y

    def visualize_sample(self, sensor=0, month=0, row=0, col=0, save_path=None):
        """
        Visualize a sample from the dataset.

        Args:
            sensor: Sensor index
            month: Month index
            row: Grid row index
            col: Grid column index
            save_path: Path to save the figure. If None, displays the plot (not suitable for script).
        """
        if save_path is None:
            print("Warning: save_path not provided for visualize_sample. Plot will not be saved.")
            return

        data = self.get_month_data(month)
        sample = data[sensor, row, col, :, :, :]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        channel_names = ['SWE', 'Temperature', 'Precipitation']
        for i in range(3):
            im = axes[i].imshow(sample[:, :, i], cmap='viridis')
            axes[i].set_title(channel_names[i])
            plt.colorbar(im, ax=axes[i])
        plt.suptitle(f"Sensor {sensor}, Month {month}, Row {row}, Col {col}")
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout
        
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Sample visualization saved to {save_path}")

# --- 5. Memory-Efficient Dataset Implementation ---
class ChunkedSWEDataset(Dataset):
    def __init__(self, xml_processor, sequence_length=12, split='train'):
        """
        Memory-efficient dataset that loads data on-demand from XML processor.

        Args:
            xml_processor: XMLProcessor instance
            sequence_length: Length of input sequence
            split: 'train', 'val', or 'test'
        """
        self.processor = xml_processor
        self.sequence_length = sequence_length
        self.split = split

        metadata = self.processor.get_metadata()
        self.dimensions = metadata['dimensions']
        self.num_months = metadata['num_files']

        # Define the indices for this dataset split
        self.indices = self._get_indices()

    def _get_indices(self):
        """Generate indices for this dataset split"""
        indices = []

        # Split data: 70% train, 15% validation, 15% test
        train_end = int(0.7 * self.num_months)
        val_end = int(0.85 * self.num_months)

        if self.split == 'train':
            start_split, end_split = 0, train_end
        elif self.split == 'val':
            start_split, end_split = train_end, val_end
        else:  # test
            start_split, end_split = val_end, self.num_months
        
        # Ensure that we can form a sequence of `sequence_length` and have a target month *after* it.
        # The last possible `start_month` for a sequence is `end_split - sequence_length -1`.
        # The target month will be `start_month + sequence_length`.
        # This target month must be `< end_split`.
        # Also, `target_month` must be `< self.num_months`.

        max_start_month_for_split = end_split - self.sequence_length -1

        for sensor in range(self.dimensions['sensors']):
            for month in range(start_split, max_start_month_for_split + 1):
                 # Ensure target_month is within total data available
                 if month + self.sequence_length < self.num_months:
                    for row in range(self.dimensions['rows']):
                        for col in range(self.dimensions['cols']):
                            indices.append((sensor, month, row, col))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (X, y) - input sequence and target
        """
        # Get the indices for this sample
        sensor, start_month, row, col = self.indices[idx]
        target_month = start_month + self.sequence_length

        # Get the sample from the processor
        X, y = self.processor.get_sample(
            sensor=sensor,
            start_month=start_month,
            row=row,
            col=col,
            sequence_length=self.sequence_length,
            target_month=target_month
        )

        # Reshape y if needed
        y = y.reshape(self.dimensions['height'], self.dimensions['width'], 1)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 6. Model Definition ---
class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer( # Renamed from self.encoder
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True 
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.norm(x)
        return x

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, return_sequences=True):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4) # (seq, batch, channel, height, width)

        batch_size, seq_len, _, h, w = x.size()

        if hidden_state is None:
            hidden_state = [self.cells[i].init_hidden(batch_size, (h, w)) for i in range(self.num_layers)]

        output_list = [] 
        h_last_layer = None # To store h from the last layer if not returning sequences
        for t in range(seq_len):
            cur_input = x[:, t, :, :, :] # Current time step: (batch, channel, height, width)
            for layer_idx in range(self.num_layers):
                h, c = self.cells[layer_idx](cur_input, hidden_state[layer_idx])
                hidden_state[layer_idx] = (h, c)
                cur_input = h # Output of current layer is input to next
            h_last_layer = h # Store h from the last layer at this time step
            if self.return_sequences:
                output_list.append(h)

        if self.return_sequences:
            final_output = torch.stack(output_list, dim=1) # (batch, seq_len, hidden_dim, H, W)
        else:
            final_output = h_last_layer # (batch, hidden_dim, H, W) - h from last time step, last layer

        return final_output, hidden_state


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class SWETransUNet(nn.Module):
    def __init__(self, in_channels=3, emb_dim=256, num_heads=8, hidden_dim=512, dropout=0.1, num_transformer_layers=2):
        super(SWETransUNet, self).__init__()
        self.patch_embed = nn.Conv2d(in_channels, emb_dim, kernel_size=4, stride=4) # Output H/4, W/4
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(emb_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_transformer_layers)
        ])
        self.conv_lstm = ConvLSTM(input_dim=emb_dim, hidden_dim=emb_dim, kernel_size=3,
                                  num_layers=1, batch_first=True, return_sequences=True)
        
        self.decoder_block1 = UNetDecoderBlock(emb_dim, 192) 
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        self.decoder_block2 = UNetDecoderBlock(192, 96) 
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        self.decoder_block3 = UNetDecoderBlock(96, 48) 
        self.decoder_block4 = UNetDecoderBlock(48, 32) 
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1) 

    def forward(self, x): # x: (batch, seq_len, height, width, channels)
        batch_size, seq_len, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(batch_size * seq_len, channels, height, width)
        
        x = self.patch_embed(x) # (bs*sq, emb_dim, h/4, w/4)
        _, c_emb, h_patch, w_patch = x.shape # c_emb is emb_dim
        
        x = x.permute(0, 2, 3, 1).reshape(batch_size * seq_len, h_patch * w_patch, c_emb) 
        for encoder in self.transformer_encoders:
            x = encoder(x)
        
        x = x.reshape(batch_size, seq_len, h_patch, w_patch, c_emb).permute(0, 1, 4, 2, 3) 
        x, _ = self.conv_lstm(x) 
        x = x[:, -1] # Take last_output: (bs, emb_dim, h_patch, w_patch)
        
        x = self.decoder_block1(x) 
        x = self.upsample1(x)      
        
        x = self.decoder_block2(x) 
        x = self.upsample2(x)      

        x = self.decoder_block3(x) 
        x = self.decoder_block4(x) 

        x = self.final_conv(x)     
        x = x.permute(0, 2, 3, 1)  # (bs, h, w, 1)
        return x

# --- 7. Training and Evaluation Functions ---
def compute_metrics(y_true, y_pred):
    y_true_np = y_true.cpu().detach().numpy().flatten()
    y_pred_np = y_pred.cpu().detach().numpy().flatten()
    mse = mean_squared_error(y_true_np, y_pred_np)
    mae = mean_absolute_error(y_true_np, y_pred_np)
    try: 
        r2 = r2_score(y_true_np, y_pred_np)
    except ValueError:
        r2 = 0.0 
    return mse, mae, r2

def train_model(model, train_loader, val_loader, num_epochs, lr, results_directory, cache_dir_model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    metrics_history = { # Renamed from 'metrics' to avoid conflict
        'train_mse': [], 'train_mae': [], 'train_r2': [],
        'val_mse': [], 'val_mae': [], 'val_r2': [],
        'learning_rates': []
    }

    best_val_mse = float('inf')
    best_model_path = os.path.join(cache_dir_model, 'best_model.pth') 

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training
        model.train()
        train_mse_epoch, train_mae_epoch, train_r2_epoch = 0.0, 0.0, 0.0
        train_samples = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            mse, mae, r2 = compute_metrics(y_batch, output)
            batch_size = X_batch.size(0)
            train_mse_epoch += mse * batch_size
            train_mae_epoch += mae * batch_size
            train_r2_epoch += r2 * batch_size
            train_samples += batch_size

            if train_samples > 0 and train_samples % 1000 == 0: 
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if train_samples > 0:
            metrics_history['train_mse'].append(train_mse_epoch / train_samples)
            metrics_history['train_mae'].append(train_mae_epoch / train_samples)
            metrics_history['train_r2'].append(train_r2_epoch / train_samples)
        else: 
            metrics_history['train_mse'].append(0)
            metrics_history['train_mae'].append(0)
            metrics_history['train_r2'].append(0)


        # Validation
        model.eval()
        val_mse_epoch, val_mae_epoch, val_r2_epoch = 0.0, 0.0, 0.0
        val_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                mse, mae, r2 = compute_metrics(y_batch, output)
                batch_size = X_batch.size(0)
                val_mse_epoch += mse * batch_size
                val_mae_epoch += mae * batch_size
                val_r2_epoch += r2 * batch_size
                val_samples += batch_size
        
        current_val_mse = float('inf')
        if val_samples > 0:
            current_val_mse = val_mse_epoch / val_samples
            metrics_history['val_mse'].append(current_val_mse)
            metrics_history['val_mae'].append(val_mae_epoch / val_samples)
            metrics_history['val_r2'].append(val_r2_epoch / val_samples)
        else: 
            metrics_history['val_mse'].append(0)
            metrics_history['val_mae'].append(0)
            metrics_history['val_r2'].append(0)

        metrics_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        if val_samples > 0:
            scheduler.step(current_val_mse)

        if val_samples > 0 and current_val_mse < best_val_mse:
            best_val_mse = current_val_mse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': current_val_mse,
                'metrics_history': metrics_history 
            }, best_model_path)
            print(f"New best model saved with validation MSE: {current_val_mse:.6f}")

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds')
        if train_samples > 0:
            print(f'Train - MSE: {metrics_history["train_mse"][-1]:.6f}, MAE: {metrics_history["train_mae"][-1]:.6f}, R²: {metrics_history["train_r2"][-1]:.6f}')
        if val_samples > 0:
            print(f'Val   - MSE: {metrics_history["val_mse"][-1]:.6f}, MAE: {metrics_history["val_mae"][-1]:.6f}, R²: {metrics_history["val_r2"][-1]:.6f}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(metrics_history['train_mse'], label='Train MSE')
    axes[0, 0].plot(metrics_history['val_mse'], label='Val MSE')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].legend()
    axes[0, 0].set_title('Mean Squared Error')

    axes[0, 1].plot(metrics_history['train_mae'], label='Train MAE')
    axes[0, 1].plot(metrics_history['val_mae'], label='Val MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].set_title('Mean Absolute Error')

    axes[1, 0].plot(metrics_history['train_r2'], label='Train R²')
    axes[1, 0].plot(metrics_history['val_r2'], label='Val R²')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].legend()
    axes[1, 0].set_title('R-squared')

    axes[1, 1].plot(metrics_history['learning_rates'], label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log') 

    plt.tight_layout()
    plot_save_path = os.path.join(results_directory, 'training_metrics.png')
    plt.savefig(plot_save_path)
    print(f"Training metrics plot saved to {plot_save_path}")
    plt.close(fig)

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation MSE: {checkpoint['val_mse']:.6f}")
    else:
        print("No best model was saved. Using current model state.")

    return model, metrics_history

def evaluate_model(model, test_loader):
    model.eval()
    test_mse_total, test_mae_total, test_r2_total = 0.0, 0.0, 0.0
    test_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating on test set"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            mse, mae, r2 = compute_metrics(y_batch, output)
            batch_size = X_batch.size(0)
            test_mse_total += mse * batch_size
            test_mae_total += mae * batch_size
            test_r2_total += r2 * batch_size
            test_samples += batch_size
    
    final_test_mse, final_test_mae, final_test_r2 = 0.0, 0.0, 0.0
    if test_samples > 0:
        final_test_mse = test_mse_total / test_samples
        final_test_mae = test_mae_total / test_samples
        final_test_r2 = test_r2_total / test_samples
    else:
        print("Warning: Test set is empty. Metrics will be zero.")

    print(f"Test Results:")
    print(f"  MSE: {final_test_mse:.6f}")
    print(f"  MAE: {final_test_mae:.6f}")
    print(f"  R²: {final_test_r2:.6f}")

    return final_test_mse, final_test_mae, final_test_r2

def visualize_predictions(model, test_loader, num_samples_to_plot, results_directory): # Renamed num_samples
    model.eval()
    plot_samples = []
    count = 0
    if len(test_loader) == 0:
        print("Test loader is empty, cannot visualize predictions.")
        return

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if count >= num_samples_to_plot:
                break
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred_batch = model(X_batch)
            
            for i in range(X_batch.size(0)):
                if count >= num_samples_to_plot:
                    break
                plot_samples.append((X_batch[i].cpu(), y_batch[i].cpu(), pred_batch[i].cpu()))
                count += 1

    for i, (X_single, y_single, pred_single) in enumerate(plot_samples):
        mse, mae, r2 = compute_metrics(y_single.unsqueeze(0), pred_single.unsqueeze(0))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(y_single[:, :, 0], cmap='viridis')
        axes[0].set_title('Ground Truth - SWE')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(pred_single[:, :, 0], cmap='viridis')
        axes[1].set_title('Prediction - SWE')
        plt.colorbar(im1, ax=axes[1])

        plt.suptitle(f'Sample {i+1}\nMSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        
        pred_plot_path = os.path.join(results_directory, f'prediction_sample_{i+1}.png')
        plt.savefig(pred_plot_path)
        print(f"Prediction visualization {i+1} saved to {pred_plot_path}")
        plt.close(fig)

        seq_len = X_single.shape[0] 
        
        num_input_plots = min(seq_len, 6)
        fig_seq, axes_seq = plt.subplots(1, num_input_plots, figsize=(3 * num_input_plots, 3))
        if num_input_plots == 1: 
            axes_seq = [axes_seq] 

        for j in range(num_input_plots):
            axes_seq[j].imshow(X_single[j, :, :, 0], cmap='viridis')  
            axes_seq[j].set_title(f'Input Month {j+1}')
            axes_seq[j].axis('off')
        plt.suptitle(f'Input Sequence for Sample {i+1}')
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        
        input_seq_plot_path = os.path.join(results_directory, f'input_sequence_{i+1}.png')
        plt.savefig(input_seq_plot_path)
        print(f"Input sequence visualization {i+1} saved to {input_seq_plot_path}")
        plt.close(fig_seq)

def clean_cache(cache_dir_to_clean, keep_best_model=True):
    """Clean up cache files to free up space"""
    files_to_keep = []
    if keep_best_model:
        files_to_keep.append('best_model.pth')

    if not os.path.exists(cache_dir_to_clean):
        print(f"Cache directory {cache_dir_to_clean} does not exist. Nothing to clean.")
        return

    cache_files = glob.glob(os.path.join(cache_dir_to_clean, '*'))
    if not cache_files:
        print(f"Cache directory {cache_dir_to_clean} is empty.")
        return
        
    total_size = sum(os.path.getsize(f) for f in cache_files if os.path.isfile(f))

    deleted_size = 0
    for file_path in cache_files:
        if not os.path.isfile(file_path): 
            continue
        file_name = os.path.basename(file_path)
        if file_name not in files_to_keep:
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                deleted_size += file_size
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")

    print(f"Cleaned {deleted_size / (1024 ** 2):.2f} MB of {total_size / (1024 ** 2):.2f} MB from cache {cache_dir_to_clean}")


if __name__ == "__main__":
    # --- Directory Definitions ---
    
    # User-specified path for XML data files
    xml_input_data_path = "/data/" 

    # Base path for outputs (cache, results, models) - current directory by default
    output_base_path = "." 

    # Path to the XML data files
    data_directory = xml_input_data_path
    
    # Cache directory for processed chunks and best model during training
    cache_directory = os.path.join(output_base_path, 'swe_cache')
    os.makedirs(cache_directory, exist_ok=True)
    
    # Results directory for plots
    results_directory = os.path.join(output_base_path, 'results')
    os.makedirs(results_directory, exist_ok=True)
    
    # Model save directory (final model)
    model_save_dir = os.path.join(output_base_path, 'models') # Final model saved here
    os.makedirs(model_save_dir, exist_ok=True)

    # Mount Google Drive if in Colab (e.g., for saving final model to a specific Drive path if desired later)
    if IN_COLAB:
        google_drive_mount_path = '/content/drive'
        mount_drive(google_drive_mount_path)
        # Example: if you want to save final model to Drive instead of local './models'
        # model_save_dir = os.path.join(google_drive_mount_path, 'MyDrive/IMSS_SWE_Output/models')
        # os.makedirs(model_save_dir, exist_ok=True)


    print(f"XML Data input directory: {data_directory}")
    print(f"Cache directory: {cache_directory}")
    print(f"Results directory: {results_directory}")
    print(f"Model save directory (for final model): {model_save_dir}")

    # --- 3. Initialize the XML Processor and Preprocess Data ---
    try:
        processor = XMLProcessor(
            directory_path=data_directory,
            cache_dir=cache_directory,
            chunk_size=10 
        )
    except FileNotFoundError as e:
        print(f"Error initializing XMLProcessor: {e}")
        print("Please ensure the data_directory is correctly set and contains XML files.")
        exit()


    # Get dataset metadata
    metadata = processor.get_metadata()
    print("Dataset metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Preprocess all chunks in parallel (creates cache files)
    processor.preprocess_all_chunks(num_workers=2) 

    # --- 4. Visualize Sample Data ---
    initial_sample_save_path = os.path.join(results_directory, "initial_sample_visualization.png")
    processor.visualize_sample(sensor=0, month=0, row=0, col=0, save_path=initial_sample_save_path)

    # --- Create Datasets and DataLoaders ---
    sequence_length_data = 12
    train_dataset = ChunkedSWEDataset(processor, split='train', sequence_length=sequence_length_data)
    val_dataset = ChunkedSWEDataset(processor, split='val', sequence_length=sequence_length_data)
    test_dataset = ChunkedSWEDataset(processor, split='test', sequence_length=sequence_length_data)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    if len(train_loader) > 0:
        for X_sample, y_sample in train_loader:
            print(f"Batch shapes - X: {X_sample.shape}, y: {y_sample.shape}")
            break
    else:
        print("Train loader is empty. Check dataset split, data availability, and sequence_length.")


    # --- Initialize Model ---
    model = SWETransUNet().to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # --- 8. Train the Model ---
    num_epochs_train = 5 
    learning_rate_train = 1e-3
    
    training_metrics_history = {} # Initialize
    if len(train_dataset) > 0 and len(val_dataset) > 0:
        print(f"\nStarting training for {num_epochs_train} epochs...")
        model, training_metrics_history = train_model(model, train_loader, val_loader, 
                                            num_epochs=num_epochs_train, lr=learning_rate_train,
                                            results_directory=results_directory,
                                            cache_dir_model=cache_directory) # best model saved in cache
    elif len(train_dataset) == 0:
        print("Skipping training: Training dataset is empty.")
    elif len(val_dataset) == 0:
        print("Skipping training: Validation dataset is empty (required for saving best model and LR scheduling).")


    # --- 9. Evaluate on Test Set ---
    test_mse, test_mae, test_r2 = 0.0, 0.0, 0.0 # Initialize
    if len(test_dataset) > 0:
        print("\nEvaluating on test set...")
        test_mse, test_mae, test_r2 = evaluate_model(model, test_loader)
    else:
        print("Skipping evaluation: Test dataset is empty.")


    # --- 10. Visualize Predictions ---
    if len(test_dataset) > 0:
        print("\nVisualizing predictions...")
        visualize_predictions(model, test_loader, num_samples_to_plot=3, results_directory=results_directory)
    else:
        print("Skipping prediction visualization: Test dataset is empty.")


    # --- 11. Save the Trained Model ---
    final_model_save_path = os.path.join(model_save_dir, 'swe_transunet_chunked_model_final.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(), # Save current state of the model (could be best or last epoch)
        'test_metrics': {
            'mse': test_mse,
            'mae': test_mae,
            'r2': test_r2
        },
        'training_metrics_history': training_metrics_history, 
        'processor_metadata': processor.get_metadata()
    }, final_model_save_path)
    print(f"Final model state saved to {final_model_save_path}")

    # --- 12. Clean Up Cache (Optional) ---
    # print("\nCleaning up cache directory (keeps best_model.pth if training occurred)...")
    # clean_cache(cache_directory, keep_best_model=True)

    print("\nScript finished.")
