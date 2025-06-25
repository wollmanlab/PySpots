# PySpots

Code developed in the Wollman Lab for processing images and calling spots from MERFISH/seqFISH style experiments.

## Overview

PySpots is a comprehensive Python package for analyzing multiplexed fluorescence in situ hybridization (MERFISH/seqFISH) data. It provides tools for:

- Image processing and registration
- Spot detection and classification
- Cell segmentation
- Data analysis and visualization
- Automated pipeline execution

## Installation

### Prerequisites

- Python 3.6 or higher
- Conda package manager
- CUDA-compatible GPU (optional, for GPU acceleration)

### Creating a Conda Environment

1. **Create a new conda environment:**
   ```bash
   conda create -n pyspots python=3.9
   conda activate pyspots
   ```

2. **Install PySpots and dependencies:**
   ```bash
   # Clone the repository
   git clone https://github.com/wollmanlab/PySpots.git
   cd PySpots
   
   # Install the package in development mode
   pip install -e .
   
   # Install additional dependencies
   pip install -r requirements.txt
   ```

### Alternative Installation Methods

**Using pip directly:**
```bash
pip install git+https://github.com/wollmanlab/PySpots.git
```

**For GPU support (TensorFlow):**
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow-gpu==2.7.0
```

## Configuration

PySpots uses configuration files to define experiment parameters. Example configurations are provided in the root directory:

- `merfish_config_TBI_Hippocampus.py` - TBI hippocampus analysis
- `merfish_config_Inflammation.py` - Inflammation studies
- `merfish_config_zebrafinch.py` - Zebra finch brain analysis
- `smfish_config_elegans.py` - C. elegans analysis

### Key Configuration Parameters

- **bitmap**: Defines the hybridization order and fluorophores
- **codebook**: Specifies the barcode-to-gene mapping
- **parameters**: Contains processing parameters for each analysis step

## Usage

### Basic Workflow

1. **Setup your data directory structure:**
   ```
   /path/to/your/data/
   ├── hybe1/
   ├── hybe2/
   ├── ...
   └── hybeN/
   ```

2. **Create a configuration file** or modify an existing one to match your experiment.

3. **Initialize the analysis:**
   ```python
   from MERFISH_Objects.Analyze import *
   
   # Import your configuration
   import merfish_config_your_experiment as config
   
   # Set up paths
   metadata_path = '/path/to/your/data/'
   daemon_path = config.parameters['daemon_path']
   ```

4. **Run the analysis pipeline:**
   ```bash
   # Initialize dataset
   python MERFISH_Objects/Analyze.py /path/to/data merfish_config_your_experiment
   
   # Run processing steps (in order)
   python MERFISH_Objects/Daemon_wrapper.py dataset -i 60 -n 1
   python MERFISH_Objects/Daemon_wrapper.py position -i 60 -n 1
   python MERFISH_Objects/Daemon_wrapper.py hybe -i 60 -n 1
   python MERFISH_Objects/Daemon_wrapper.py stack -i 60 -n 1
   python MERFISH_Objects/Daemon_wrapper.py registration -i 60 -n 20
   python MERFISH_Objects/Daemon_wrapper.py segmentation -i 60 -n 10
   python MERFISH_Objects/Daemon_wrapper.py image -i 60 -n 30
   python MERFISH_Objects/Daemon_wrapper.py classification -i 60 -n 20
   ```

### Monitoring Progress

Use the Progress module to monitor analysis status:
```bash
python MERFISH_Objects/Progress.py /path/to/data/ merfish_config_your_experiment
```

### Example Jupyter Notebook

See `MERFISH_Objects/Example.ipynb` for a complete workflow example.

## Pipeline Steps

1. **Dataset**: Initializes the dataset structure
2. **Position**: Processes position-specific data
3. **Hybe**: Handles hybridization cycle data
4. **Stack**: Manages image stacks
5. **Registration**: Aligns images across cycles
6. **Segmentation**: Identifies cell boundaries
7. **Image**: Processes and analyzes images
8. **Classification**: Classifies detected spots

## Key Components

- **MERFISH_Objects/**: Core analysis modules
- **hybescope_config/**: Microscope-specific configurations
- **Old_configs/**: Legacy configuration files
- **Configuration files**: Experiment-specific parameters

## Dependencies

Major dependencies include:
- NumPy, SciPy, Pandas
- TensorFlow (for GPU acceleration)
- OpenCV, scikit-image
- Cellpose (for cell segmentation)
- PyQt5, pyqtgraph (for visualization)

See `requirements.txt` for the complete list.

## Troubleshooting

### Common Issues

1. **CUDA/GPU errors**: Ensure TensorFlow-GPU is properly installed
2. **Memory issues**: Reduce batch sizes in configuration
3. **File permission errors**: Check write permissions for daemon and utilities paths

### Getting Help

- Check the example notebook for usage patterns
- Review configuration files for parameter descriptions
- Contact the Wollman Lab for support

## Citation

If you use PySpots in your research, please cite the appropriate publications from the Wollman Lab.

## License

MIT License - see LICENSE file for details.

## Contact

- **Lab**: Wollman Lab, UCLA
- **Email**: wollmanlab@g.ucla.edu
- **GitHub**: https://github.com/wollmanlab/PySpots
