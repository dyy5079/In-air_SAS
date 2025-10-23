import numpy as np
from typing import Union, Tuple, Optional, List
from enum import Enum

class CFARMethod(Enum):
    """CFAR detection methods"""
    CA = "CA"      # Cell Averaging
    SOCA = "SOCA"  # Smallest Of Cell Averaging  
    GOCA = "GOCA"  # Greatest Of Cell Averaging
    OS = "OS"      # Order Statistic

class OutputFormat(Enum):
    """Output format options"""
    CUT_RESULT = "CUT result"
    DETECTION_INDEX = "Detection index"

class NumDetectionsSource(Enum):
    """Number of detections source options"""
    AUTO = "Auto"
    PROPERTY = "Property"

class ThresholdFactorSource(Enum):
    """Threshold factor source options"""
    AUTO = "Auto"
    PROPERTY = "Property" 
    INPUT_PORT = "Input port"

class CFARDetector2D:
    """
    2D Constant False Alarm Rate (CFAR) detector for matrix data.
    
    This class performs CFAR detection on 2D matrix input data using various
    algorithms including Cell Averaging (CA), Smallest/Greatest of Cell 
    Averaging (SOCA/GOCA), and Order Statistic (OS) methods.
    """
    
    def __init__(self,
                 method: Union[str, CFARMethod] = CFARMethod.CA,
                 guard_band_size: Union[int, List[int]] = [1, 1],
                 training_band_size: Union[int, List[int]] = [1, 1],
                 rank: int = 1,
                 threshold_factor: Union[str, ThresholdFactorSource] = ThresholdFactorSource.AUTO,
                 probability_false_alarm: float = 1e-6,
                 custom_threshold_factor: float = 1.0,
                 output_format: Union[str, OutputFormat] = OutputFormat.CUT_RESULT,
                 threshold_output_port: bool = False,
                 noise_power_output_port: bool = False,
                 num_detections_source: Union[str, NumDetectionsSource] = NumDetectionsSource.AUTO,
                 num_detections: int = 100):
        """
        Initialize CFARDetector2D.
        
        Parameters:
        -----------
        method : str or CFARMethod
            CFAR algorithm ('CA', 'SOCA', 'GOCA', 'OS')
        guard_band_size : int or list
            Size in cells of the guard region band [rows, cols]
        training_band_size : int or list  
            Size in cells of the training region band [rows, cols]
        rank : int
            Rank of order statistic (for OS method)
        threshold_factor : str or ThresholdFactorSource
            Threshold factor method
        probability_false_alarm : float
            Probability of false alarm
        custom_threshold_factor : float
            Custom threshold factor
        output_format : str or OutputFormat
            Output reporting format
        threshold_output_port : bool
            Output detection threshold
        noise_power_output_port : bool
            Output noise power
        num_detections_source : str or NumDetectionsSource
            Source of the number of detections
        num_detections : int
            Maximum number of detections
        """
        
        # Convert string inputs to enums if needed
        if isinstance(method, str):
            method = CFARMethod(method)
        if isinstance(threshold_factor, str):
            threshold_factor = ThresholdFactorSource(threshold_factor)
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format)
        if isinstance(num_detections_source, str):
            num_detections_source = NumDetectionsSource(num_detections_source)
            
        self.method = method
        self.guard_band_size = np.array(guard_band_size) if isinstance(guard_band_size, list) else np.array([guard_band_size, guard_band_size])
        self.training_band_size = np.array(training_band_size) if isinstance(training_band_size, list) else np.array([training_band_size, training_band_size])
        self.rank = rank
        self.threshold_factor = threshold_factor
        self.probability_false_alarm = probability_false_alarm
        self.custom_threshold_factor = custom_threshold_factor
        self.output_format = output_format
        self.threshold_output_port = threshold_output_port
        self.noise_power_output_port = noise_power_output_port
        self.num_detections_source = num_detections_source
        self.num_detections = num_detections
        
        # Private properties
        self._maximum_cell_row_index = None
        self._maximum_cell_col_index = None
        self._num_pages = 1
        self._num_channels = 1
        self._size_initialized = False
        self._factor = None
        
        # Validate inputs
        self._validate_properties()
        
    def _validate_properties(self):
        """Validate property values"""
        if not np.all(self.guard_band_size >= 0):
            raise ValueError("GuardBandSize must be non-negative")
        if not np.all(self.training_band_size > 0):
            raise ValueError("TrainingBandSize must be positive")
        if self.rank < 1:
            raise ValueError("Rank must be positive")
        if not 0 <= self.probability_false_alarm <= 1:
            raise ValueError("ProbabilityFalseAlarm must be between 0 and 1")
            
    def _get_guard_region_size(self) -> np.ndarray:
        """Get guard region size"""
        return 2 * self.guard_band_size + 1
        
    def _get_training_region_size(self) -> np.ndarray:
        """Get training region size"""
        guard_region_size = self._get_guard_region_size()
        return 2 * self.training_band_size + guard_region_size
        
    def _get_num_training_cells(self) -> int:
        """Get number of training cells"""
        guard_region_size = self._get_guard_region_size()
        training_region_size = self._get_training_region_size()
        return (training_region_size[0] * training_region_size[1] - 
                guard_region_size[0] * guard_region_size[1])
                
    def _calculate_threshold_factor(self) -> float:
        """Calculate threshold factor based on probability of false alarm"""
        if self.threshold_factor == ThresholdFactorSource.AUTO:
            # Calculate threshold factor for given PFA
            num_training_cells = self._get_num_training_cells()
            # For exponentially distributed noise (common assumption)
            factor = num_training_cells * (self.probability_false_alarm ** (-1/num_training_cells) - 1)
            return factor
        elif self.threshold_factor == ThresholdFactorSource.PROPERTY:
            return self.custom_threshold_factor
        else:  # INPUT_PORT
            return None  # Will be provided as input
            
    def _get_2d_training_indices(self, X: np.ndarray, idx: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculate 2D training cell indices"""
        num_rows, num_cols = X.shape[:2]
        
        guard_region_size = self._get_guard_region_size()
        training_region_size = self._get_training_region_size()
        
        # Compute training and guard cell indices
        row_idx = idx[0]
        col_idx = idx[1]
        
        # Training region indices
        train_row_start = int(row_idx - (training_region_size[0] - 1) // 2)
        train_row_end = int(row_idx + (training_region_size[0] - 1) // 2 + 1)
        train_col_start = int(col_idx - (training_region_size[1] - 1) // 2)
        train_col_end = int(col_idx + (training_region_size[1] - 1) // 2 + 1)
        
        # Guard region indices  
        guard_row_start = int(row_idx - (guard_region_size[0] - 1) // 2)
        guard_row_end = int(row_idx + (guard_region_size[0] - 1) // 2 + 1)
        guard_col_start = int(col_idx - (guard_region_size[1] - 1) // 2)
        guard_col_end = int(col_idx + (guard_region_size[1] - 1) // 2 + 1)
        
        # Create meshgrids for training and guard regions
        train_rows, train_cols = np.meshgrid(range(train_row_start, train_row_end),
                                           range(train_col_start, train_col_end), indexing='ij')
        guard_rows, guard_cols = np.meshgrid(range(guard_row_start, guard_row_end),
                                           range(guard_col_start, guard_col_end), indexing='ij')
        
        # Convert to linear indices
        train_linear = np.ravel_multi_index((train_rows.ravel(), train_cols.ravel()), (num_rows, num_cols))
        guard_linear = np.ravel_multi_index((guard_rows.ravel(), guard_cols.ravel()), (num_rows, num_cols))
        cut_linear = np.ravel_multi_index((row_idx, col_idx), (num_rows, num_cols))
        
        # Get training cells (excluding guard region and CUT)
        training_indices = np.setdiff1d(train_linear, guard_linear)
        
        if self.method in [CFARMethod.SOCA, CFARMethod.GOCA]:
            # Split into front and rear training cells
            front_indices = training_indices[training_indices > cut_linear]
            rear_indices = training_indices[training_indices < cut_linear]
            return front_indices, rear_indices
        else:
            return training_indices
            
    def _validate_cuts(self, X: np.ndarray, idx: np.ndarray):
        """Validate that training regions fit within input data"""
        num_rows, num_cols = X.shape[:2]
        training_region_size = self._get_training_region_size()
        
        # Check if training region falls outside of X
        for i in range(idx.shape[1]):
            row_idx, col_idx = idx[:, i]
            
            row_min = row_idx - (training_region_size[0] - 1) // 2
            row_max = row_idx + (training_region_size[0] - 1) // 2
            col_min = col_idx - (training_region_size[1] - 1) // 2
            col_max = col_idx + (training_region_size[1] - 1) // 2
            
            if (row_min < 0 or row_max >= num_rows or 
                col_min < 0 or col_max >= num_cols):
                raise ValueError(f"Training cells fall outside input data for CUT at ({row_idx}, {col_idx})")
                
    def __call__(self, X: np.ndarray, idx: np.ndarray, 
                 threshold_factor: Optional[float] = None) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Perform CFAR detection.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix (MxN or MxNxP)
        idx : np.ndarray  
            2xL matrix of cell under test indices
        threshold_factor : float, optional
            Threshold factor (when ThresholdFactor is 'Input port')
            
        Returns:
        --------
        Union[np.ndarray, Tuple[np.ndarray, ...]]
            Detection results and optional threshold/noise power outputs
        """
        return self.step(X, idx, threshold_factor)
        
    def step(self, X: np.ndarray, idx: np.ndarray,
             threshold_factor: Optional[float] = None) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Perform CFAR detection step.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data matrix (MxN or MxNxP)
        idx : np.ndarray
            2xL matrix of cell under test indices  
        threshold_factor : float, optional
            Threshold factor (when ThresholdFactor is 'Input port')
            
        Returns:
        --------
        Union[np.ndarray, Tuple[np.ndarray, ...]]
            Detection results and optional threshold/noise power outputs
        """
        
        # Validate inputs
        if len(X.shape) < 2 or len(X.shape) > 3:
            raise ValueError("Input X must be 2D or 3D array")
        if idx.shape[0] != 2:
            raise ValueError("idx must have 2 rows")
            
        # Setup dimensions
        if len(X.shape) == 3:
            self._num_pages = X.shape[2]
            self._num_channels = X.shape[2]
        else:
            self._num_pages = 1
            self._num_channels = 1
            X = X[:, :, np.newaxis]  # Add channel dimension
            
        self._maximum_cell_row_index = X.shape[0]
        self._maximum_cell_col_index = X.shape[1]
        
        # Validate CUTs
        self._validate_cuts(X, idx)
        
        # Get threshold factor
        if self.threshold_factor == ThresholdFactorSource.INPUT_PORT:
            if threshold_factor is None:
                raise ValueError("Threshold factor must be provided when ThresholdFactor is 'Input port'")
            th_factor = threshold_factor
        else:
            if self._factor is None:
                self._factor = self._calculate_threshold_factor()
            th_factor = self._factor
            
        num_cuts = idx.shape[1]
        thresholds = np.zeros((num_cuts, self._num_channels), dtype=X.dtype)
        noise_powers = np.zeros((num_cuts, self._num_channels), dtype=X.dtype)
        
        # Process each CUT
        for m in range(num_cuts):
            cut_idx = idx[:, m]
            
            if self.method == CFARMethod.CA:
                # Cell Averaging
                training_indices = self._get_2d_training_indices(X, cut_idx)
                rows, cols = np.unravel_index(training_indices, X.shape[:2])
                noise_estimate = np.mean(X[rows, cols, :], axis=0)
                
            elif self.method == CFARMethod.SOCA:
                # Smallest of Cell Averaging
                front_indices, rear_indices = self._get_2d_training_indices(X, cut_idx)
                front_rows, front_cols = np.unravel_index(front_indices, X.shape[:2])
                rear_rows, rear_cols = np.unravel_index(rear_indices, X.shape[:2])
                front_avg = np.mean(X[front_rows, front_cols, :], axis=0)
                rear_avg = np.mean(X[rear_rows, rear_cols, :], axis=0)
                noise_estimate = np.minimum(front_avg, rear_avg)
                
            elif self.method == CFARMethod.GOCA:
                # Greatest of Cell Averaging
                front_indices, rear_indices = self._get_2d_training_indices(X, cut_idx)
                front_rows, front_cols = np.unravel_index(front_indices, X.shape[:2])
                rear_rows, rear_cols = np.unravel_index(rear_indices, X.shape[:2])
                front_avg = np.mean(X[front_rows, front_cols, :], axis=0)
                rear_avg = np.mean(X[rear_rows, rear_cols, :], axis=0)
                noise_estimate = np.maximum(front_avg, rear_avg)
                
            elif self.method == CFARMethod.OS:
                # Order Statistic
                training_indices = self._get_2d_training_indices(X, cut_idx)
                rows, cols = np.unravel_index(training_indices, X.shape[:2])
                training_data = X[rows, cols, :]
                sorted_data = np.sort(training_data, axis=0)
                noise_estimate = sorted_data[self.rank - 1, :]
                
            thresholds[m, :] = noise_estimate * th_factor
            noise_powers[m, :] = noise_estimate
            
        # Get CUT values
        cut_values = X[idx[0, :], idx[1, :], :]
        
        # Perform detection
        detections = cut_values > thresholds
        
        # Format outputs
        outputs = []
        
        if self.output_format == OutputFormat.CUT_RESULT:
            outputs.append(detections)
            if self.threshold_output_port:
                outputs.append(thresholds)
            if self.noise_power_output_port:
                outputs.append(noise_powers)
                
        else:  # Detection index format
            detection_indices = np.where(detections)
            num_detections = len(detection_indices[0])
            
            if self.num_detections_source == NumDetectionsSource.AUTO:
                max_detections = num_detections
            else:
                max_detections = self.num_detections
                
            if self._num_channels == 1:
                detection_output = np.full((2, max_detections), np.nan, dtype=X.dtype)
                threshold_output = np.full((1, max_detections), np.nan, dtype=X.dtype)
                noise_output = np.full((1, max_detections), np.nan, dtype=X.dtype)
            else:
                detection_output = np.full((3, max_detections), np.nan, dtype=X.dtype)
                threshold_output = np.full((1, max_detections), np.nan, dtype=X.dtype)
                noise_output = np.full((1, max_detections), np.nan, dtype=X.dtype)
                
            num_available = min(max_detections, num_detections)
            if num_available > 0:
                if self._num_channels == 1:
                    detection_output[0, :num_available] = idx[0, detection_indices[0][:num_available]]
                    detection_output[1, :num_available] = idx[1, detection_indices[0][:num_available]]
                else:
                    detection_output[0, :num_available] = idx[0, detection_indices[0][:num_available]]
                    detection_output[1, :num_available] = idx[1, detection_indices[0][:num_available]]
                    detection_output[2, :num_available] = detection_indices[1][:num_available]
                    
                threshold_output[0, :num_available] = thresholds[detection_indices][:num_available]
                noise_output[0, :num_available] = noise_powers[detection_indices][:num_available]
                
            outputs.append(detection_output)
            if self.threshold_output_port:
                outputs.append(threshold_output)
            if self.noise_power_output_port:
                outputs.append(noise_output)
                
        # Remove channel dimension if input was 2D
        if len(X.shape) == 2:
            outputs = [out.squeeze(axis=-1) if out.ndim > 2 else out for out in outputs]
            
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


# Example usage
if __name__ == "__main__":
    # Example 1: Basic CFAR detection
    np.random.seed(42)
    
    # Create test data with noise
    data = np.random.exponential(1.0, (20, 20))
    
    # Add some targets
    data[10, 10] = 15  # Strong target
    data[5, 15] = 8    # Weaker target
    
    # Create CFAR detector
    cfar = CFARDetector2D(
        method=CFARMethod.CA,
        guard_band_size=[1, 1],
        training_band_size=[2, 2],
        probability_false_alarm=1e-3
    )
    
    # Define cells under test
    cut_indices = np.array([
        [10, 5],   # Row indices
        [10, 15]   # Column indices  
    ])
    
    # Perform detection
    detections = cfar(data, cut_indices)
    print("Detections:", detections)
    
    # Example with threshold and noise power outputs
    cfar.threshold_output_port = True
    cfar.noise_power_output_port = True
    
    detections, thresholds, noise_powers = cfar(data, cut_indices)
    print("Detections:", detections)
    print("Thresholds:", thresholds)
    print("Noise powers:", noise_powers)