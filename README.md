# Spatial Axis Analysis for Tissue Zonation

A computational framework for analyzing spatial gradients and tissue zonation patterns using  spatial transcriptomics data.

## Features

- **Marker-based scoring**: Calculate periportal (PV) and pericentral (CV) marker scores
- **Dynamic seed detection**: Automatically identify PV and CV seed regions
- **Bidirectional axis computation**: Generate a continuous axis representing tissue zonation
- **Flow field analysis**: Visualize directional flow between zones
- **Differential expression**: Identify genes associated with zonation patterns
- **Comprehensive visualization**: Generate publication-ready plots

## Installation

```bash


# Install dependencies
pip install -r requirements.txt
=======
# liver_axis
A computational framework for analyzing spatial gradients and tissue zonation patterns using  spatial transcriptomics data.

#Usage
python main.py --input data.h5ad --sample_name liver_sample --output_dir results

The default marker is human genes, which can be modified in config.py