# XOR-FPGA-Implementation
This repository contains my current work on implementing and validating a small XOR learning model across Vivado and replicating the same in Jupyter Notebook:

- Python / TensorFlow
- Vitis HLS
- Vivado simulation

## Current progress

### 1. Floating-point HLS version
- XOR model reimplemented using `float`
- compared against TensorFlow/Python more easily
- tested in Vivado using updated wrapper and VHDL testbench

### 2. Python / TensorFlow verification
- float reference notebook
- quantized replica path for HLS-oriented comparison


## Repository structure

- `hls/` → HLS source files and C testbenches
- `vivado/` → VHDL testbenches and wrapper-related notes
- `notebooks/` → Jupyter notebooks for verification
- `docs/` → progress notes and task tracking
