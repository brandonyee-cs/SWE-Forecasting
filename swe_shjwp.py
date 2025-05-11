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
from concurrent.futures import ThreadPoolExecutor
import time
import shutil # For cache cleaning if uncommented

# --- BEGIN HDF5 CONFIGURATION ---
# Only one dataset for SWE
HDF5_SWE_DATASET_NAME = "HDFEOS/GRIDS/Northern Hemisphere/Data Fields/SWE_NorthernPentad"
# --- END HDF5 CONFIGURATION ---

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

# --- 2. XML and HDF5 Data Processing ---
class XMLProcessor:
    def __init__(self, xml_directory_path, h5_data_dir, cache_dir=None, chunk_size=5,
                 default_patch_height=64, default_patch_width=64,
                 default_grid_rows=10, default_grid_cols=9, default_channels=1):  # Changed default_channels to 1
        self.xml_directory_path = xml_directory_path
        self.h5_data_dir = h5_data_dir
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.xml_files = []
        self.h5_files = [] # List to store corresponding HDF5 file paths

        # These dimensions now relate to patches from HDF5 files
        self.dimensions = {
            'patch_height': default_patch_height,
            'patch_width': default_patch_width,
            'grid_rows_in_hdf5': default_grid_rows, # Number of rows of patches to extract from an HDF5
            'grid_cols_in_hdf5': default_grid_cols, # Number of columns of patches
            'channels': default_channels,  # Changed to 1 for SWE only
            'num_xml_files': 0
        }
        self._scan_files_and_metadata()

    def _scan_files_and_metadata(self):
        if not os.path.exists(self.xml_directory_path):
            raise FileNotFoundError(f"XML directory not found: {self.xml_directory_path}")
        if not os.path.exists(self.h5_data_dir):
            raise FileNotFoundError(f"HDF5 data directory not found: {self.h5_data_dir}")

        all_xml_paths = sorted(glob.glob(os.path.join(self.xml_directory_path, "*.xml")))
        if not all_xml_paths:
            raise FileNotFoundError(f"No XML files found in directory: {self.xml_directory_path}")

        print(f"Scanning {len(all_xml_paths)} XML files...")
        first_valid_xml_processed = False
        for xml_path in all_xml_paths:
            try:
                h5_filename, extracted_dims = self._extract_dimensions_and_h5_filename(xml_path)
                if h5_filename:
                    h5_full_path = os.path.join(self.h5_data_dir, h5_filename)
                    if os.path.exists(h5_full_path):
                        self.xml_files.append(xml_path)
                        self.h5_files.append(h5_full_path)
                        if not first_valid_xml_processed:
                            # Update dimensions based on the first valid XML (if it provides them)
                            # For now, we primarily use defaults unless XML structure is well-known to provide these
                            if extracted_dims.get('patch_height'): self.dimensions['patch_height'] = extracted_dims['patch_height']
                            if extracted_dims.get('patch_width'): self.dimensions['patch_width'] = extracted_dims['patch_width']
                            if extracted_dims.get('grid_rows_in_hdf5'): self.dimensions['grid_rows_in_hdf5'] = extracted_dims['grid_rows_in_hdf5']
                            if extracted_dims.get('grid_cols_in_hdf5'): self.dimensions['grid_cols_in_hdf5'] = extracted_dims['grid_cols_in_hdf5']
                            if extracted_dims.get('channels'): self.dimensions['channels'] = extracted_dims['channels']
                            print(f"Dimensions set from defaults or first valid XML '{os.path.basename(xml_path)}':")
                            for k, v in self.dimensions.items(): print(f"  {k}: {v}")
                            first_valid_xml_processed = True
                    else:
                        print(f"Warning: HDF5 file '{h5_filename}' (from XML '{os.path.basename(xml_path)}') not found in '{self.h5_data_dir}'. Skipping XML.")
                else:
                    print(f"Warning: Could not extract HDF5 filename from XML '{os.path.basename(xml_path)}'. Skipping.")
            except ET.ParseError as e:
                print(f"Warning: Could not parse XML file '{os.path.basename(xml_path)}': {e}. Skipping.")
            except Exception as e:
                print(f"Warning: Error processing XML file '{os.path.basename(xml_path)}': {e}. Skipping.")

        if not self.xml_files:
            raise ValueError("No valid XML/HDF5 pairs found. Check XMLs point to existing HDF5s and paths are correct.")

        self.dimensions['num_xml_files'] = len(self.xml_files)
        print(f"Found {self.dimensions['num_xml_files']} valid XML/HDF5 pairs.")
        print(f"Final data processing dimensions: {self.dimensions}")

    def _extract_dimensions_and_h5_filename(self, xml_file_path):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        h5_filename = None
        h5_file_elem = root.find('.//DataFileContainer/DistributedFileName')
        if h5_file_elem is None: # Try another common path
            h5_file_elem = root.find('.//ECSDataGranule/LocalGranuleID')
        if h5_file_elem is not None and h5_file_elem.text:
            h5_filename = h5_file_elem.text.strip()

        # Placeholder for extracting dimensions from XML if they exist
        # For now, we rely on defaults or what's set from the first XML.
        extracted_dims = {} # Populate this if XML contains usable dimension info

        return h5_filename, extracted_dims

    def _read_hdf5_patch(self, h5_filepath, patch_row_idx, patch_col_idx,
                         patch_h, patch_w, channels):
        """
        Reads SWE dataset from HDF5, extracts/crops a patch, with added normalization.
        Returns: np.array of shape (patch_h, patch_w, channels) or None on error.
        """
        if HDF5_SWE_DATASET_NAME == "PLACEHOLDER_SWE_DATASET_NAME":
            raise ValueError("HDF5 dataset name is not configured. Please set it at the top of the script.")

        try:
            with h5py.File(h5_filepath, 'r') as f:
                swe_full = f[HDF5_SWE_DATASET_NAME][()]

                # Get full dataset dimensions
                full_h, full_w = swe_full.shape

                start_row = patch_row_idx * patch_h
                end_row = start_row + patch_h
                start_col = patch_col_idx * patch_w
                end_col = start_col + patch_w

                if end_row > full_h or end_col > full_w:
                    print(f"Warning: Patch ({patch_row_idx},{patch_col_idx}) for {os.path.basename(h5_filepath)} "
                          f"from ({start_row}:{end_row}, {start_col}:{end_col}) "
                          f"exceeds HDF5 dims ({full_h},{full_w}). Returning zeros.")
                    return np.zeros((patch_h, patch_w, channels), dtype=np.float32)

                swe_patch = swe_full[start_row:end_row, start_col:end_col]
                
                # Ensure extracted patch matches desired patch_h, patch_w
                if swe_patch.shape != (patch_h, patch_w):
                    print(f"Error: Extracted patch shape mismatch for {os.path.basename(h5_filepath)}. "
                          f"Expected {(patch_h, patch_w)}, got SWE:{swe_patch.shape}. "
                          "Check HDF5 dimensions, patch size, and grid_rows/cols settings.")
                    return np.zeros((patch_h, patch_w, channels), dtype=np.float32)

                # --- CHANGE 1: Filter extreme values ---
                # First, replace NaN or invalid values
                swe_patch = np.nan_to_num(swe_patch, nan=0.0, posinf=1000.0, neginf=0.0)
                
                # Clip to reasonable SWE range (adjust based on your data)
                # Typical SWE values might range from 0 to 1000 mm
                swe_patch = np.clip(swe_patch, 0.0, 1000.0)
                
                # --- CHANGE 2: Normalize the data ---
                # Option 1: Min-max normalization to [0,1] range
                # Using fixed range based on typical SWE values
                SWE_MAX = 1000.0  # Maximum expected SWE value in mm
                SWE_MIN = 0.0     # Minimum expected SWE value in mm
                swe_patch = (swe_patch - SWE_MIN) / (SWE_MAX - SWE_MIN + 1e-8)  # Add small epsilon to avoid div by zero
                
                # Option 2: Standardization (if you prefer this approach)
                # Commented out but you can uncomment if you prefer standardization
                # SWE_MEAN = 200.0  # Estimated mean SWE value
                # SWE_STD = 150.0   # Estimated standard deviation 
                # swe_patch = (swe_patch - SWE_MEAN) / (SWE_STD + 1e-8)

                # Create data patch with just SWE data
                data_patch = np.zeros((patch_h, patch_w, channels), dtype=np.float32)
                data_patch[..., 0] = swe_patch
                
                return data_patch

        except FileNotFoundError:
            print(f"Error: HDF5 file not found during read: {h5_filepath}")
            return np.zeros((patch_h, patch_w, channels), dtype=np.float32)
        except KeyError as e:
            print(f"Error: Dataset not found in {h5_filepath}: {e}. Check HDF5_SWE_DATASET_NAME constant.")
            return np.zeros((patch_h, patch_w, channels), dtype=np.float32)
        except Exception as e:
            print(f"Error reading or processing HDF5 file {h5_filepath}: {e}")
            return np.zeros((patch_h, patch_w, channels), dtype=np.float32)

    def _get_cache_path(self, chunk_idx):
        if self.cache_dir is None: return None
        return os.path.join(self.cache_dir, f"chunk_{chunk_idx}.h5")

    def _process_chunk(self, chunk_idx):
        cache_path = self._get_cache_path(chunk_idx)

        patch_h = self.dimensions['patch_height']
        patch_w = self.dimensions['patch_width']
        grid_r = self.dimensions['grid_rows_in_hdf5']
        grid_c = self.dimensions['grid_cols_in_hdf5']
        ch = self.dimensions['channels']
        
        start_xml_idx = chunk_idx * self.chunk_size
        num_xml_in_this_chunk = min(self.chunk_size, len(self.xml_files) - start_xml_idx)

        if num_xml_in_this_chunk <= 0:
             return np.array([])

        # Shape of data for one chunk: (num_xml_files, grid_r, grid_c, patch_h, patch_w, ch)
        expected_chunk_shape = (num_xml_in_this_chunk, grid_r, grid_c, patch_h, patch_w, ch)

        if cache_path and os.path.exists(cache_path):
            is_cache_valid = False
            try:
                with h5py.File(cache_path, 'r') as f:
                    if 'data' in f and f['data'].shape == expected_chunk_shape:
                        is_cache_valid = True
                    else:
                        reason = "shape mismatch" if 'data' in f else "no 'data' key"
                        print(f"Chunk {chunk_idx}: Stale cache {os.path.basename(cache_path)} ({reason}). Deleting.")
            except Exception as e:
                print(f"Chunk {chunk_idx}: Error reading cache {os.path.basename(cache_path)}: {e}. Deleting.")
            
            if is_cache_valid: return cache_path
            else:
                try: os.remove(cache_path)
                except OSError as e: print(f"Error deleting stale cache file {cache_path}: {e}")
        
        current_chunk_data = np.zeros(expected_chunk_shape, dtype=np.float32)

        for i in range(num_xml_in_this_chunk):
            xml_file_global_idx = start_xml_idx + i
            h5_filepath = self.h5_files[xml_file_global_idx]
            for r_idx in range(grid_r):
                for c_idx in range(grid_c):
                    patch_data = self._read_hdf5_patch(h5_filepath, r_idx, c_idx,
                                                      patch_h, patch_w, ch)
                    current_chunk_data[i, r_idx, c_idx, ...] = patch_data
        
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            try:
                with h5py.File(cache_path, 'w') as f: f.create_dataset('data', data=current_chunk_data)
                return cache_path
            except Exception as e:
                print(f"Error writing to cache file {cache_path}: {e}. Returning data directly.")
                return current_chunk_data
        
        return current_chunk_data

    def get_num_chunks(self):
        if not self.xml_files: return 0
        return (len(self.xml_files) + self.chunk_size - 1) // self.chunk_size

    def get_metadata(self):
        return {'dimensions': self.dimensions, 
                'num_xml_files': len(self.xml_files), 
                'num_chunks': self.get_num_chunks(), 
                'chunk_size': self.chunk_size}

    def preprocess_all_chunks(self, num_workers=4):
        if self.cache_dir is None: print("Caching is disabled. Skipping preprocessing."); return
        if not self.xml_files: print("No XML/HDF5 files found. Skipping preprocessing."); return
        os.makedirs(self.cache_dir, exist_ok=True)
        num_chunks = self.get_num_chunks()
        if num_chunks == 0: print("No chunks to preprocess."); return

        print(f"Preprocessing {num_chunks} chunks (validating/creating cache) with {num_workers} workers...")
        # Force num_workers=0 for debugging HDF5 issues, as h5py might not be thread-safe in all configs
        # num_workers = 0 # Uncomment for debugging
        if num_workers > 0:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._process_chunk, i) for i in range(num_chunks)]
                for future in tqdm(futures, total=num_chunks, desc="Preprocessing chunks"):
                    try: future.result()
                    except Exception as e: print(f"Error during threaded preprocessing of a chunk: {e}")
        else: # Sequential processing
            for i in tqdm(range(num_chunks), desc="Preprocessing chunks (sequential)"):
                try: self._process_chunk(i)
                except Exception as e: print(f"Error during sequential preprocessing of chunk {i}: {e}")

        print("All chunks preprocessed.")

    def get_chunk_data(self, chunk_idx): # Renamed from get_chunk to avoid confusion
        if chunk_idx >= self.get_num_chunks():
            raise ValueError(f"Chunk index {chunk_idx} out of range.")
        processed_output = self._process_chunk(chunk_idx) 
        if isinstance(processed_output, str): # Path to cache file
            try:
                with h5py.File(processed_output, 'r') as f: return f['data'][:]
            except Exception as e:
                print(f"Error loading chunk {chunk_idx} from cache {processed_output}: {e}. Reprocessing.")
                original_cache_dir = self.cache_dir
                self.cache_dir = None # Temporarily disable caching for this call
                data_array = self._process_chunk(chunk_idx)
                self.cache_dir = original_cache_dir
                return data_array
        elif isinstance(processed_output, np.ndarray): return processed_output
        else: raise RuntimeError(f"Unexpected result from _process_chunk for chunk {chunk_idx}.")

    def get_hdf5_data_for_xml_index(self, xml_file_idx):
        """
        Retrieves all patches for a single HDF5 file (specified by xml_file_idx).
        Returns: np.array of shape (grid_rows, grid_cols, patch_h, patch_w, channels)
        """
        if not self.xml_files or xml_file_idx >= len(self.xml_files):
            raise ValueError(f"XML file index {xml_file_idx} is out of range.")
        
        chunk_idx = xml_file_idx // self.chunk_size
        idx_in_chunk = xml_file_idx % self.chunk_size
        
        chunk_data = self.get_chunk_data(chunk_idx) # Shape: (num_xml_in_chunk, grid_r, grid_c, pH, pW, ch)
        return chunk_data[idx_in_chunk, ...]


    def get_sample(self, xml_start_idx, patch_row_idx, patch_col_idx, sequence_length, xml_target_idx):
        """
        Generates a training sample (X, y).
        X: Input sequence of patches, shape (sequence_length, patch_h, patch_w, channels)
        y: Target SWE patch, shape (patch_h, patch_w, 1)
        """
        patch_h = self.dimensions['patch_height']
        patch_w = self.dimensions['patch_width']
        channels = self.dimensions['channels']

        X_sequence_patches = np.zeros((sequence_length, patch_h, patch_w, channels), dtype=np.float32)

        for i in range(sequence_length):
            current_xml_idx = xml_start_idx + i
            if current_xml_idx >= len(self.xml_files):
                raise IndexError(f"Sequence generation out of bounds: current_xml_idx {current_xml_idx}")
            
            # Get all patches for the HDF5 file corresponding to current_xml_idx
            all_patches_for_current_xml = self.get_hdf5_data_for_xml_index(current_xml_idx)
            # Shape: (grid_r, grid_c, patch_h, patch_w, channels)
            
            # Select the specific patch
            X_sequence_patches[i, ...] = all_patches_for_current_xml[patch_row_idx, patch_col_idx, ...]

        # Get target patch
        all_patches_for_target_xml = self.get_hdf5_data_for_xml_index(xml_target_idx)
        # Extract SWE channel (assuming it's the first one, index 0)
        y_patch = all_patches_for_target_xml[patch_row_idx, patch_col_idx, ..., 0:1]

        return X_sequence_patches, y_patch

    def visualize_sample(self, xml_file_idx=0, patch_row_idx=0, patch_col_idx=0, save_path=None):
        if save_path is None: print("Warning: save_path not provided for visualize_sample."); return
        if not self.xml_files: print("Cannot visualize: No XML/HDF5 files."); return
        if xml_file_idx >= len(self.xml_files):
            print(f"Cannot visualize: XML file index {xml_file_idx} out of bounds."); return

        try:
            all_patches = self.get_hdf5_data_for_xml_index(xml_file_idx)
            
            if patch_row_idx >= all_patches.shape[0] or patch_col_idx >= all_patches.shape[1]:
                print(f"Cannot visualize: Patch index ({patch_row_idx},{patch_col_idx}) out of bounds "
                      f"for HDF5 file {xml_file_idx} (max: {all_patches.shape[0]-1},{all_patches.shape[1]-1}).")
                return

            sample_patch = all_patches[patch_row_idx, patch_col_idx, ...] # Shape (patch_h, patch_w, channels)

            plt.figure(figsize=(5, 5))
            im = plt.imshow(sample_patch[:,:,0], cmap='viridis')
            plt.title('SWE')
            plt.colorbar(im)
            
            plt.suptitle(f"HDF5: {os.path.basename(self.h5_files[xml_file_idx])}, Patch ({patch_row_idx},{patch_col_idx})")
            plt.tight_layout(rect=[0,0,1,0.96])
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            plt.savefig(save_path); plt.close()
            print(f"Sample patch visualization saved to {save_path}")
        except IndexError as e: 
            print(f"Error visualizing sample (xml_idx={xml_file_idx}, patch_row={patch_row_idx}, patch_col={patch_col_idx}): {e}")
        except Exception as e: 
            print(f"Unexpected error during visualization: {e}")


# --- 5. Memory-Efficient Dataset Implementation ---
class ChunkedSWEDataset(Dataset):
    def __init__(self, xml_processor, sequence_length=12, split='train'):
        self.processor = xml_processor
        self.sequence_length = sequence_length
        self.split = split

        metadata = self.processor.get_metadata()
        self.dimensions = metadata['dimensions']
        self.num_xml_files = metadata['num_xml_files']
        
        self.indices = self._get_indices()

        if not self.indices:
            print(f"Warning: No samples generated for split '{split}'. "
                  f"Total XML files: {self.num_xml_files}, Sequence length: {self.sequence_length}. "
                  f"Patch grid: {self.dimensions.get('grid_rows_in_hdf5')}x{self.dimensions.get('grid_cols_in_hdf5')}. "
                  f"Check data availability and split logic.")

    def _get_indices(self):
        indices = []
        if self.num_xml_files == 0 or \
           self.dimensions.get('grid_rows_in_hdf5', 0) == 0 or \
           self.dimensions.get('grid_cols_in_hdf5', 0) == 0:
            return indices

        # Max start index for an XML file to form a sequence
        max_start_xml_idx = self.num_xml_files - self.sequence_length - 1 # -1 because target is one step ahead
        if max_start_xml_idx < 0:
            return []
        
        # Splitting based on the available start XML indices
        train_split_end_xml_idx = int(0.7 * (max_start_xml_idx + 1))
        val_split_end_xml_idx = int(0.85 * (max_start_xml_idx + 1))
        
        start_xml_for_split, end_xml_for_split = 0, 0
        if self.split == 'train':
            start_xml_for_split, end_xml_for_split = 0, train_split_end_xml_idx
        elif self.split == 'val':
            start_xml_for_split, end_xml_for_split = train_split_end_xml_idx, val_split_end_xml_idx
        else:  # test
            start_xml_for_split, end_xml_for_split = val_split_end_xml_idx, max_start_xml_idx + 1
        
        xml_indices_for_split = range(start_xml_for_split, end_xml_for_split)

        for xml_start_idx in xml_indices_for_split:
            for patch_r_idx in range(self.dimensions['grid_rows_in_hdf5']):
                for patch_c_idx in range(self.dimensions['grid_cols_in_hdf5']):
                    indices.append((xml_start_idx, patch_r_idx, patch_c_idx))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if not self.indices: raise IndexError("Dataset empty or not initialized.")
        
        xml_start_idx, patch_row_idx, patch_col_idx = self.indices[idx]
        
        # Target XML index is one step after the end of the input sequence
        xml_target_idx = xml_start_idx + self.sequence_length 
        
        if xml_target_idx >= self.num_xml_files:
             raise RuntimeError(f"Target XML index {xml_target_idx} (from start {xml_start_idx} + seq_len {self.sequence_length}) "
                                f"is out of bounds ({self.num_xml_files} files). Error with sample index: {self.indices[idx]}")
        
        X, y = self.processor.get_sample(
            xml_start_idx, patch_row_idx, patch_col_idx,
            self.sequence_length, xml_target_idx
        )
        # X shape: (sequence_length, patch_h, patch_w, channels)
        # y shape: (patch_h, patch_w, 1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# --- 6. Model Definition (Updated for Single Channel) ---
class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation='relu', batch_first=True 
        )
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x): return self.norm(self.encoder_layer(x))

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim, self.hidden_dim, self.kernel_size, self.bias = input_dim, hidden_dim, kernel_size, bias
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, 
                             padding=self.padding, bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Ensure dimensions match
        if input_tensor.shape[2:] != h_cur.shape[2:]:
            # Resize h_cur and c_cur to match input_tensor spatial dimensions
            h_cur = F.interpolate(h_cur, size=input_tensor.shape[2:], mode='bilinear', align_corners=True)
            c_cur = F.interpolate(c_cur, size=input_tensor.shape[2:], mode='bilinear', align_corners=True)
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split along channel dimension
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Update cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """Initialize hidden state for a batch of samples"""
        h, w = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, return_sequences=True):
        super(ConvLSTM, self).__init__()
        self.input_dim, self.num_layers = input_dim, num_layers
        self.batch_first, self.return_sequences = batch_first, return_sequences
        self.hidden_dim = [hidden_dim] * num_layers if isinstance(hidden_dim, int) else hidden_dim
        self.kernel_size = [kernel_size] * num_layers if isinstance(kernel_size, int) else kernel_size
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            self.cells.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i]))

    def forward(self, x, hidden_state=None):
        if not self.batch_first: x = x.permute(1, 0, 2, 3, 4)
        bs, seq_len, _, h, w = x.size()
        
        # Initialize hidden state - ensure dimensions match input
        if hidden_state is None: 
            hidden_state = []
            for cell in self.cells:
                # Get the expected output size from the cell's convolution
                conv_out_h = h
                conv_out_w = w
                hidden_state.append(cell.init_hidden(bs, (conv_out_h, conv_out_w)))
                
        outputs, last_h_t = [], None
        for t in range(seq_len):
            h_t_input = x[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                h_prev, c_prev = hidden_state[layer_idx]
                
                # Ensure h_t_input and h_prev have the same spatial dimensions
                if h_t_input.shape[2:] != h_prev.shape[2:]:
                    # If there's a mismatch, resize h_prev to match h_t_input
                    # This is a hacky solution but should work for this specific case
                    h_prev = F.interpolate(h_prev, size=h_t_input.shape[2:], mode='nearest')
                    c_prev = F.interpolate(c_prev, size=h_t_input.shape[2:], mode='nearest')
                    hidden_state[layer_idx] = (h_prev, c_prev)
                
                h_t, c_t = self.cells[layer_idx](h_t_input, (h_prev, c_prev))
                hidden_state[layer_idx] = (h_t, c_t)
                h_t_input = h_t
                
            last_h_t = h_t
            if self.return_sequences: outputs.append(last_h_t)
            
        if self.return_sequences: return torch.stack(outputs, dim=1), hidden_state
        else: return last_h_t, hidden_state

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,3,padding=1); self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,padding=1); self.bn2=nn.BatchNorm2d(out_channels)
    def forward(self, x): x=self.relu(self.bn1(self.conv1(x))); x=self.relu(self.bn2(self.conv2(x))); return x

class SWETransUNet(nn.Module):
    def __init__(self, in_channels=1, emb_dim=128, num_heads=4, hidden_dim=256, dropout=0.1, num_transformer_layers=1):
        super(SWETransUNet, self).__init__()
        # Use a simpler architecture to avoid dimension issues
        # Start with a standard convolution for embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(emb_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim//2, emb_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(inplace=True)
        )
        
        # Transformer encoder
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(emb_dim, num_heads, hidden_dim, dropout) 
            for _ in range(num_transformer_layers)
        ])
        
        # ConvLSTM for temporal processing
        self.conv_lstm = ConvLSTM(emb_dim, emb_dim, 3, 1, batch_first=True, return_sequences=False)
        
        # Decoder path - use a more flexible architecture
        self.decoder = nn.Sequential(
            nn.Conv2d(emb_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        bs, seq, H_in, W_in, C_in = x.shape
        
        # First, process each time step through the embedding
        embeddings = []
        for t in range(seq):
            x_t = x[:, t].permute(0, 3, 1, 2)  # [B, C, H, W]
            emb_t = self.patch_embed(x_t)       # [B, emb_dim, H', W']
            embeddings.append(emb_t)
        
        # Stack along sequence dimension
        embeddings = torch.stack(embeddings, dim=1)  # [B, seq, emb_dim, H', W']
        
        # Apply ConvLSTM directly on embedded sequence
        lstm_out, _ = self.conv_lstm(embeddings)  # [B, emb_dim, H', W']
        
        # Decode to final output
        decoded = self.decoder(lstm_out)
        
        # Ensure output size matches input size using interpolation
        if decoded.shape[2:] != (H_in, W_in):
            decoded = F.interpolate(decoded, size=(H_in, W_in), mode='bilinear', align_corners=True)
        
        # Final convolution and permute to match expected output format
        output = self.final_conv(decoded)    # [B, 1, H_in, W_in]
        return output.permute(0, 2, 3, 1)    # [B, H_in, W_in, 1]

# --- 7. Training and Evaluation Functions ---
def compute_metrics(y_true, y_pred):
    y_true_np=y_true.cpu().detach().numpy().reshape(-1); y_pred_np=y_pred.cpu().detach().numpy().reshape(-1)
    mse=mean_squared_error(y_true_np,y_pred_np); mae=mean_absolute_error(y_true_np,y_pred_np)
    try: r2=r2_score(y_true_np,y_pred_np) if np.var(y_true_np) >= 1e-9 else (1.0 if np.allclose(y_true_np,y_pred_np) else 0.0)
    except ValueError: r2=0.0 
    return mse,mae,r2

def train_model(model, train_loader, val_loader, num_epochs, lr, results_directory, cache_dir_model):
    # --- CHANGE 3: Use a more robust loss function ---
    # criterion = nn.MSELoss()  # Original loss
    criterion = nn.SmoothL1Loss(beta=0.2)  # Huber loss - less sensitive to outliers
    
    # You can also try L1Loss if Huber doesn't work well
    # criterion = nn.L1Loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # FIX: Use only the basic parameters supported by older PyTorch versions
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=3
    )
    
    # Print message to show what scheduler is doing
    print(f"Using ReduceLROnPlateau scheduler with factor=0.5, patience=3")
    
    metrics_hist={'train_mse':[],'train_mae':[],'train_r2':[],'val_mse':[],'val_mae':[],'val_r2':[],'learning_rates':[]}
    best_val_mse=float('inf')
    best_model_path=os.path.join(cache_dir_model,'best_model.pth')
    os.makedirs(cache_dir_model,exist_ok=True)

    for epoch in range(num_epochs):
        start_time=time.time(); model.train(); tr_mse,tr_mae,tr_r2,tr_loss,tr_samples=0.0,0.0,0.0,0.0,0
        prog_bar_tr=tqdm(train_loader,desc=f'Epoch {epoch+1}/{num_epochs} Train',leave=False)
        for X_b,y_b in prog_bar_tr: # X_b: (bs,seq,H,W,C), y_b: (bs,H,W,1)
            try:
                X_b,y_b=X_b.to(device),y_b.to(device); optimizer.zero_grad(); out=model(X_b) # model output: (bs,H,W,1)
                loss=criterion(out,y_b); loss.backward() 
                
                # --- CHANGE 4: More aggressive gradient clipping ---
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Reduced from 1.0 to 0.5
                
                optimizer.step()
                
                # Even though we're using Huber loss for training, we still compute MSE for monitoring
                with torch.no_grad():
                    mse_loss = F.mse_loss(out, y_b).item()
                
                m,a,r=compute_metrics(y_b,out); bs=X_b.size(0)
                tr_loss+=loss.item()*bs; tr_mse+=m*bs; tr_mae+=a*bs; tr_r2+=r*bs; tr_samples+=bs
                prog_bar_tr.set_postfix(loss=loss.item(), mse=mse_loss, r2=f"{r:.3f}")
                
                # Detect and report extreme loss values
                if loss.item() > 100:
                    print(f"Warning: High loss detected: {loss.item():.2f}")
                
                if tr_samples>0 and tr_samples%(len(train_loader)//4+1)==0: gc.collect(); torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"Error during training batch: {e}")
                # Skip this batch and continue with the next one
                optimizer.zero_grad()  # Clear any partial gradients
                torch.cuda.empty_cache()
                continue
        
        if tr_samples > 0:  # Only update metrics if we processed at least one batch
            metrics_hist['train_mse'].append(tr_mse/tr_samples); metrics_hist['train_mae'].append(tr_mae/tr_samples)
            metrics_hist['train_r2'].append(tr_r2/tr_samples)
        else:
            print("Warning: No training samples were processed successfully in this epoch")
            metrics_hist['train_mse'].append(float('inf')); metrics_hist['train_mae'].append(float('inf'))
            metrics_hist['train_r2'].append(0.0)

        model.eval(); val_mse,val_mae,val_r2,val_loss,val_samples=0.0,0.0,0.0,0.0,0
        prog_bar_val=tqdm(val_loader,desc=f'Epoch {epoch+1}/{num_epochs} Val',leave=False)
        with torch.no_grad():
            for X_b,y_b in prog_bar_val:
                try:
                    X_b,y_b=X_b.to(device),y_b.to(device); out=model(X_b); loss=criterion(out,y_b)
                    m,a,r=compute_metrics(y_b,out); bs=X_b.size(0)
                    val_loss+=loss.item()*bs; val_mse+=m*bs; val_mae+=a*bs; val_r2+=r*bs; val_samples+=bs
                    prog_bar_val.set_postfix(loss=loss.item(),mse=m,r2=f"{r:.3f}")
                except RuntimeError as e:
                    print(f"Error during validation batch: {e}")
                    continue
        
        if val_samples > 0:
            curr_val_mse=val_mse/val_samples
            metrics_hist['val_mse'].append(curr_val_mse); metrics_hist['val_mae'].append(val_mae/val_samples)
            metrics_hist['val_r2'].append(val_r2/val_samples)
        else:
            print("Warning: No validation samples were processed successfully in this epoch")
            curr_val_mse = float('inf')
            metrics_hist['val_mse'].append(float('inf')); metrics_hist['val_mae'].append(float('inf'))
            metrics_hist['val_r2'].append(0.0)
            
        metrics_hist['learning_rates'].append(optimizer.param_groups[0]['lr'])

        if val_samples > 0: scheduler.step(curr_val_mse)
        if val_samples > 0 and curr_val_mse < best_val_mse:
            best_val_mse=curr_val_mse
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                        'val_mse':curr_val_mse,'metrics_history':metrics_hist},best_model_path)

        print(f'E{epoch+1}/{num_epochs} ({time.time()-start_time:.1f}s) LR:{optimizer.param_groups[0]["lr"]:.1e} | ' +
              f'Tr MSE:{metrics_hist["train_mse"][-1]:.4f} R2:{metrics_hist["train_r2"][-1]:.2f} | ' +
              f'Val MSE:{metrics_hist["val_mse"][-1]:.4f} R2:{metrics_hist["val_r2"][-1]:.2f}')
        gc.collect(); torch.cuda.empty_cache()

    # Create visualization plots
    try:
        fig,ax=plt.subplots(2,2,figsize=(12,8))
        ax[0,0].plot(metrics_hist['train_mse'],label='Tr MSE'); ax[0,0].plot(metrics_hist['val_mse'],label='Val MSE')
        ax[0,1].plot(metrics_hist['train_mae'],label='Tr MAE'); ax[0,1].plot(metrics_hist['val_mae'],label='Val MAE')
        ax[1,0].plot(metrics_hist['train_r2'],label='Tr R2'); ax[1,0].plot(metrics_hist['val_r2'],label='Val R2')
        ax[1,1].plot(metrics_hist['learning_rates'],label='LR')
        titles=['MSE','MAE','R2','Learning Rate']
        for r_idx in range(2): 
            for c_idx in range(2): ax[r_idx,c_idx].legend(); ax[r_idx,c_idx].set_xlabel('Epoch'); ax[r_idx,c_idx].set_title(titles[r_idx*2+c_idx])
        ax[1,1].set_yscale('log'); plt.tight_layout(); os.makedirs(results_directory,exist_ok=True)
        plot_path=os.path.join(results_directory,'training_metrics.png'); plt.savefig(plot_path); plt.close(fig)
        print(f"Training metrics plot: {plot_path}")
    except Exception as e:
        print(f"Error creating training plots: {e}")

    if os.path.exists(best_model_path):
        try:
            ckpt=torch.load(best_model_path,map_location=device); model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded best model from E{ckpt['epoch']+1} (Val MSE: {ckpt['val_mse']:.6f})")
        except Exception as e:
            print(f"Error loading best model: {e}")
    return model,metrics_hist

def evaluate_model(model, test_loader):
    model.eval(); t_mse,t_mae,t_r2,t_samples=0.0,0.0,0.0,0
    prog_bar_test=tqdm(test_loader,desc="Evaluating Test",leave=False)
    with torch.no_grad():
        for X_b,y_b in prog_bar_test:
            try:
                X_b,y_b=X_b.to(device),y_b.to(device); out=model(X_b)
                m,a,r=compute_metrics(y_b,out); bs=X_b.size(0)
                t_mse+=m*bs; t_mae+=a*bs; t_r2+=r*bs; t_samples+=bs
                prog_bar_test.set_postfix(mse=m,r2=f"{r:.3f}")
            except RuntimeError as e:
                print(f"Error during evaluation batch: {e}")
                continue
    res_mse=t_mse/t_samples if t_samples else 0; res_mae=t_mae/t_samples if t_samples else 0; res_r2=t_r2/t_samples if t_samples else 0
    print(f"Test Results - MSE: {res_mse:.5f}, MAE: {res_mae:.5f}, R²: {res_r2:.3f}")
    return res_mse,res_mae,res_r2

def visualize_predictions(model, test_loader, num_samples, results_dir):
    model.eval()
    if len(test_loader.dataset)==0: print("Test dataset empty for viz."); return
    samples_plt,count=[],0
    with torch.no_grad():
        for X_b,y_b in test_loader: # X_b: (bs,seq,H,W,C), y_b: (bs,H,W,1)
            if count>=num_samples: break
            X_b_dev,y_b_dev=X_b.to(device),y_b.to(device)
            try:
                pred_b=model(X_b_dev) # pred_b: (bs,H,W,1)
                for i in range(X_b.size(0)):
                    if count>=num_samples: break
                    samples_plt.append((X_b[i].cpu(),y_b[i].cpu(),pred_b[i].cpu())); count+=1 # Store original X,y from loader
            except Exception as e:
                print(f"Error during visualization forward pass: {e}")
                continue
                
    if not samples_plt: print("No samples for viz."); return
    os.makedirs(results_dir,exist_ok=True)
    for i,(X_s,y_s,p_s) in enumerate(samples_plt): # X_s:(seq,H,W,C), y_s:(H,W,1), p_s:(H,W,1)
        try:
            m,a,r=compute_metrics(y_s.unsqueeze(0),p_s.unsqueeze(0)) # Add batch dim for metrics
            fig,ax=plt.subplots(1,2,figsize=(10,4));
            im0=ax[0].imshow(y_s.squeeze(),cmap='viridis'); ax[0].set_title('GT SWE'); plt.colorbar(im0,ax=ax[0],fraction=0.046,pad=0.04)
            im1=ax[1].imshow(p_s.squeeze(),cmap='viridis',vmin=y_s.min().item(),vmax=y_s.max().item()); ax[1].set_title('Pred SWE'); plt.colorbar(im1,ax=ax[1],fraction=0.046,pad=0.04)
            plt.suptitle(f'Sample {i+1} - MSE:{m:.3f} R2:{r:.2f}',fontsize=10)
            plt.tight_layout(rect=[0,0,1,0.93]); pred_path=os.path.join(results_dir,f'pred_sample_{i+1}.png')
            plt.savefig(pred_path); plt.close(fig);

            seq_len,_,_,C_in=X_s.shape; n_in_plots=min(seq_len,2)
            fig_s,ax_s=plt.subplots(1,n_in_plots,figsize=(3*n_in_plots,3))
            if n_in_plots==1: ax_s=[ax_s]
            for j in range(n_in_plots):
                # Visualize SWE channel from input
                im_sq=ax_s[j].imshow(X_s[j,:,:,0].squeeze(),cmap='viridis'); ax_s[j].set_title(f'In M-{seq_len-j} (SWE)',fontsize=8)
                ax_s[j].axis('off'); plt.colorbar(im_sq,ax=ax_s[j],fraction=0.046,pad=0.04)
            plt.suptitle(f'Input SWE Sample {i+1} (Last {n_in_plots})',fontsize=10)
            plt.tight_layout(rect=[0,0,1,0.90]); in_path=os.path.join(results_dir,f'input_seq_sample_{i+1}.png')
            plt.savefig(in_path); plt.close(fig_s);
        except Exception as e:
            print(f"Error visualizing sample {i+1}: {e}")
            continue
    print(f"Saved {len(samples_plt)} prediction visualizations to {results_dir}")

def clean_cache(cache_dir_to_clean, keep_best_model=True): # Unchanged
    if not os.path.exists(cache_dir_to_clean): print(f"Cache {cache_dir_to_clean} not found."); return
    files=glob.glob(os.path.join(cache_dir_to_clean,'*'))
    if not files: print(f"Cache {cache_dir_to_clean} empty."); return
    total_size_before = sum(os.path.getsize(f) for f in files if os.path.isfile(f)); deleted_size=0
    print(f"Cleaning cache: {cache_dir_to_clean}")
    for fp in files:
        if not os.path.isfile(fp): continue
        fn=os.path.basename(fp)
        if keep_best_model and fn=='best_model.pth': print(f"  Keeping: {fn}"); continue
        try: fs=os.path.getsize(fp); os.remove(fp); deleted_size+=fs
        except OSError as e: print(f"  Err deleting {fp}: {e}")
    print(f"Cleaned {deleted_size/(1024**2):.2f}MB. Original: {total_size_before/(1024**2):.2f}MB.")


if __name__ == "__main__":
    # --- Main Configuration ---
    xml_input_dir = "/home/ubuntu/SWE-Forecasting/data"  # Directory containing XML metadata files
    hdf5_data_dir = "/home/ubuntu/SWE-Forecasting/data" # Directory containing .he5 data files 
                                                       # Can be same as xml_input_dir if .he5 are there

    output_base_path = "." 
    cache_directory = os.path.join(output_base_path, 'swe_cache_h5_swe_only') # New cache dir name
    results_directory = os.path.join(output_base_path, 'results_h5_swe_only')
    model_save_dir = os.path.join(output_base_path, 'models_h5_swe_only')
    # --- End Main Configuration ---

    os.makedirs(cache_directory,exist_ok=True); os.makedirs(results_directory,exist_ok=True); os.makedirs(model_save_dir,exist_ok=True)

    if IN_COLAB: mount_drive() # Not relevant here

    print(f"XML Metadata input: {xml_input_dir}")
    print(f"HDF5 Data input: {hdf5_data_dir}")
    print(f"Cache: {cache_directory}\nResults: {results_directory}\nModels: {model_save_dir}")
    print("--- Processing HDF5 data referenced by XMLs ---")
    print("--- IMPORTANT: Manually delete the cache directory if you change HDF5 dataset names or patch parameters! ---")
    # Uncomment to always clean cache at start:
    if os.path.exists(cache_directory): 
       print(f"Initial cleaning of cache directory: {cache_directory}...")
       shutil.rmtree(cache_directory)
       os.makedirs(cache_directory, exist_ok=True)

    processor = None
    try:
        # XMLProcessor defaults: patch_height=64, patch_width=64, grid_rows=10, grid_cols=9, channels=1 (changed for SWE only)
        # chunk_size can be adjusted based on memory
        processor = XMLProcessor(xml_directory_path=xml_input_dir,
                                 h5_data_dir=hdf5_data_dir, 
                                 cache_dir=cache_directory,
                                 default_channels=1,  # Set to 1 for SWE only
                                 chunk_size=10) 
    except FileNotFoundError as e: print(f"Fatal: {e}\nCheck input directories."); exit(1)
    except ValueError as e: print(f"Fatal: {e}\nCheck XML/HDF5 consistency or dataset names."); exit(1)
    except Exception as e: print(f"Unexpected XMLProcessor init error: {e}"); exit(1)

    metadata = processor.get_metadata()
    print("Dataset metadata (from HDF5 processing):")
    for k,v in metadata.items(): print(f"  {k}: {v}")
    if metadata.get('num_xml_files',0)==0: print("No valid XML/HDF5 files processed. Exiting."); exit(1)

    # Preprocess (validates/creates cache).
    processor.preprocess_all_chunks(num_workers=0) # Start with 0 workers for HDF5 stability

    # Visualize a sample patch
    vis_xml_idx = min(0, metadata.get('num_xml_files', 1)-1)
    vis_patch_r = min(0, metadata['dimensions'].get('grid_rows_in_hdf5', 1)-1)
    vis_patch_c = min(0, metadata['dimensions'].get('grid_cols_in_hdf5', 1)-1)
    initial_sample_path=os.path.join(results_directory,"initial_data_patch_sample.png")
    processor.visualize_sample(xml_file_idx=vis_xml_idx, 
                               patch_row_idx=vis_patch_r, patch_col_idx=vis_patch_c, 
                               save_path=initial_sample_path)

    # Training parameters
    sequence_len = 6
    batch_s = 8  
    num_eps = 5
    lr_val = 1e-5  # Reduced from 1e-4 to 1e-5

    train_ds=ChunkedSWEDataset(processor,split='train',sequence_length=sequence_len)
    val_ds=ChunkedSWEDataset(processor,split='val',sequence_length=sequence_len)
    test_ds=ChunkedSWEDataset(processor,split='test',sequence_length=sequence_len)
    print(f"Dataset sizes - Tr:{len(train_ds)}, Vl:{len(val_ds)}, Ts:{len(test_ds)}")
    if not(len(train_ds) or len(val_ds) or len(test_ds)): print(f"All datasets empty. Check data processing and split logic. Exit."); exit(1)

    # DataLoader workers: h5py can be tricky with multiprocessing. Start with 0.
    dl_workers=0
    print(f"Using {dl_workers} DataLoader workers (0 recommended for HDF5).")
    pin_mem = torch.cuda.is_available() if dl_workers == 0 else False # Pin memory usually better with workers > 0
    
    train_loader=DataLoader(train_ds,batch_s,shuffle=True,num_workers=dl_workers,pin_memory=pin_mem,drop_last=True if len(train_ds)>batch_s else False)
    val_loader=DataLoader(val_ds,batch_s,shuffle=False,num_workers=dl_workers,pin_memory=pin_mem)
    test_loader=DataLoader(test_ds,batch_s,shuffle=False,num_workers=dl_workers,pin_memory=pin_mem)
    
    C_in_data = metadata['dimensions']['channels']
    patch_H_data = metadata['dimensions']['patch_height']
    patch_W_data = metadata['dimensions']['patch_width']

    if len(train_loader)>0:
        try:
            X_samp,y_samp=next(iter(train_loader)) # X:(bs,seq,pH,pW,C), y:(bs,pH,pW,1)
            print(f"Sample batch shapes - X:{X_samp.shape}, y:{y_samp.shape}")
            if X_samp.shape[-1] != C_in_data or \
               X_samp.shape[2] != patch_H_data or \
               X_samp.shape[3] != patch_W_data :
                print(f"WARNING: Data shape mismatch from DataLoader vs Metadata!")
                print(f"  X channels: {X_samp.shape[-1]} vs Meta: {C_in_data}")
                print(f"  X Patch H: {X_samp.shape[2]} vs Meta: {patch_H_data}")
                print(f"  X Patch W: {X_samp.shape[3]} vs Meta: {patch_W_data}")
                # Update metadata if dataloader is source of truth, though it should match
                C_in_data = X_samp.shape[-1]
                patch_H_data = X_samp.shape[2]
                patch_W_data = X_samp.shape[3]
                metadata['dimensions']['channels'] = C_in_data
                metadata['dimensions']['patch_height'] = patch_H_data
                metadata['dimensions']['patch_width'] = patch_W_data
        except Exception as e:
            print(f"Error getting sample batch from train_loader: {e}. Check dataset and loader setup.")
            if len(train_ds) == 0: print("Train dataset is empty, cannot get sample batch."); exit(1)
            # Fallback if sample can't be loaded, use metadata directly
            print("Using dimensions directly from metadata due to loader error.")

    # Check if input dimensions are suitable for patch embedding
    if patch_H_data % 2 != 0 or patch_W_data % 2 != 0:
        print(f"WARNING: Input dimensions ({patch_H_data}x{patch_W_data}) are not even numbers. "
              f"This might cause dimension mismatches. Consider resizing your patches.")
    
    # Create model with 1 input channel instead of 3
    model=SWETransUNet(in_channels=C_in_data,emb_dim=128,num_heads=4,hidden_dim=256,num_transformer_layers=1).to(device)
    print(f"Model initialized with input C={C_in_data}, H={patch_H_data}, W={patch_W_data} (from DataLoader/Metadata).")
    print(f"Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params.")

    train_hist={}
    if len(train_ds)>0 and len(val_ds)>0:
        print(f"\nTraining for {num_eps} epochs..."); model,train_hist=train_model(model,train_loader,val_loader,num_eps,lr_val,results_directory,cache_directory)
    else: print(f"Skip training: Tr empty({len(train_ds)==0}), Vl empty({len(val_ds)==0})")

    t_mse,t_mae,t_r2=0.0,0.0,0.0
    if len(test_ds)>0: print("\nEvaluating..."); t_mse,t_mae,t_r2=evaluate_model(model,test_loader)
    else: print("Skip eval: Test dataset empty.")

    if len(test_ds)>0: print("\nVisualizing preds..."); visualize_predictions(model,test_loader,min(3,len(test_ds)),results_directory)
    else: print("Skip pred viz: Test dataset empty.")

    final_model_path=os.path.join(model_save_dir,'swe_transunet_final_model_swe_only.pth')
    torch.save({'model_state_dict':model.state_dict(),'test_metrics':{'mse':t_mse,'mae':t_mae,'r2':t_r2},
                'training_metrics_history':train_hist,'processor_metadata':metadata,'sequence_length':sequence_len,
                'model_params':{'in_channels':C_in_data,'emb_dim':128,'num_heads':4,'hidden_dim':256,'num_transformer_layers':1}},
               final_model_path)
    print(f"Final model state saved: {final_model_path}")
    print("\nScript finished.")