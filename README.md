# ACCEL-v1: INT8 CNN Accelerator

A high-performance INT8 CNN accelerator implemented with a row-stationary systolic array architecture, designed for FPGA deployment. This project demonstrates end-to-end deep learning inference acceleration from algorithm to hardware implementation.

## 🚀 Key Features

- **INT8 Quantized CNN Accelerator** with systolic array compute engine
- **Row-stationary dataflow** optimized for CNN workloads
- **Double-buffered SRAM controller** for efficient memory access
- **UART-based CSR interface** for host communication and control
- **Complete MNIST CNN implementation** with training and inference
- **Comprehensive verification** with Python golden models
- **FPGA-ready design** with synthesis and timing closure on Cyclone V

## 📋 Architecture Overview

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Host System   │◄──►│   UART Interface │◄──►│   CSR Control   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Weight Buffer  │──►│  Systolic Array  │◄──►│ Activation Buf  │
│   (INT8 Wgts)   │    │   (N×M PEs)      │    │  (INT8 Acts)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Output Pipeline │
                       │ (Clamp/Shift)   │
                       └─────────────────┘
```

### Dataflow Architecture
- **Weights**: Pre-quantized INT8 values streamed into weight buffer
- **Activations**: Im2col-transformed tiles loaded as INT8 into activation buffer  
- **Compute**: Systolic array with INT32 accumulators for high precision
- **Post-processing**: Quantization with clamp/shift operations
- **Output**: INT8 results returned via UART interface

## 🗂️ Project Structure

```
accel v1/
├── data/                    # Training data and model checkpoints
│   ├── checkpoints/         # Trained MNIST models
│   └── MNIST/              # MNIST dataset
├── docs/                   # Architecture and design documentation
│   ├── ARCHITECTURE.md     # Detailed system architecture
│   ├── QUANTIZATION.md     # INT8 quantization methodology
│   ├── VERIFICATION.md     # Verification and testing strategy
│   └── HOST_RS_TILER.md    # 📚 Complete Host RS Tiler Documentation
├── python/                 # 🚀 Host software and golden models
│   ├── host_uart/          # 🎯 HOST RS TILER (Production Ready)
│   │   ├── run_gemm.py     # Main Host RS Tiler implementation
│   │   ├── uart_driver.py  # UART communication layer
│   │   └── csr_map.py      # CSR register definitions
│   ├── tests/              # 🧪 Comprehensive test suite (26 tests - 100%)
│   │   └── test_integration.py # Complete validation framework
│   ├── golden_models/      # Reference implementations
│   ├── MNIST CNN/          # CNN training and inference
│   └── utils/              # Utility functions
├── verilog/                # RTL implementation
│   ├── systolic/           # Systolic array modules
│   ├── buffer/             # Memory interface modules
│   ├── control/            # Control and CSR modules
│   ├── uart/               # UART communication
│   └── top/                # Top-level integration
├── tb/                     # Testbenches and verification
│   ├── integration/        # System-level testbenches
│   ├── unit/               # Module-level testbenches
│   └── uart/               # UART protocol verification
└── tests/                  # C++ verification framework
    ├── unit/               # Unit tests
    ├── integration/        # Integration tests
    └── verilator/          # Verilator-based simulation
```

## ✨ NEW: Host RS Tiler - Production Ready!

### 🎯 **Complete Host-Side Software Stack**

The ACCEL-v1 project now includes a **production-ready Host RS (Row-Stationary) Tiler** that provides complete orchestration for matrix multiplication operations on the systolic array accelerator.

**Key Features:**
- **🔄 Row-Stationary Dataflow:** Optimized for systolic array efficiency
- **📡 UART Communication:** Robust packet-based protocol with CRC validation
- **🧩 Matrix Tiling:** Automatic partitioning for arbitrary matrix dimensions
- **🧪 100% Test Coverage:** 26 comprehensive tests validating all functionality
- **⚡ Performance Optimized:** Efficient bandwidth utilization and PE utilization

**Quick Start:**
```bash
# Navigate to host software
cd "accel v1/python/host_uart"

# Run golden model verification (no hardware required)
python run_gemm.py --verify-only --M 8 --N 8 --K 8 --verbose

# Execute comprehensive test suite
cd ../tests
python test_integration.py --verbose  # 26/26 tests passing!

# Run on real hardware
cd ../host_uart
python run_gemm.py --M 16 --N 16 --K 16 --Tm 4 --Tn 4 --Tk 4 --verbose
```

📚 **[Complete Documentation](docs/HOST_RS_TILER.md)** | 🧪 **[Test Results](accel%20v1/python/tests/test_integration.py)** | 🚀 **[Quick Start Guide](accel%20v1/python/README.md)**

---

## 🛠️ Getting Started

### Prerequisites

- **Hardware**: Cyclone V FPGA development board (or compatible)
- **Software**: 
  - Quartus Prime (for FPGA synthesis)
  - ModelSim/QuestaSim (for simulation)
  - Python 3.8+ with PyTorch
  - CMake 3.10+
  - GCC/Clang compiler

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/joshuathomascarter/ACCEL-v1.git
   cd ACCEL-v1
   ```

2. **Install Python dependencies**
   ```bash
   cd "accel v1/python"
   pip install torch torchvision numpy matplotlib
   ```

3. **Train the MNIST CNN model**
   ```bash
   cd "MNIST CNN"
   python train_mnist.py
   ```

4. **Run software tests**
   ```bash
   cd ../../scripts
   ./run_tests.sh
   ```

5. **Run RTL simulation**
   ```bash
   cd ../tb/integration
   # Use your preferred simulator (ModelSim, QuestaSim, etc.)
   ```

## 🧮 Quantization Scheme

The accelerator uses **symmetric INT8 quantization**:

- **Format**: 8-bit signed integers with symmetric range
- **Scale factors**: Per-tensor quantization scale `S`
- **Zero point**: Always 0 (symmetric quantization)
- **Calibration**: Min/max percentile-based range estimation
- **Math**: `y = clamp(round((x_fp32 / Sx) * (W_fp32 / Sw)) * Sacc) >> shift`

## 🔧 Hardware Implementation

### Systolic Array Design
- **Processing Elements (PEs)**: INT8×INT8 → INT32 MAC units
- **Dataflow**: Row-stationary with weight reuse
- **Scalability**: Configurable N×M array dimensions
- **Pipeline**: Multi-stage with optimal throughput

### Memory Hierarchy
- **Weight Buffer**: Dedicated SRAM for filter weights
- **Activation Buffer**: Double-buffered input feature maps
- **Output Buffer**: Post-processed results staging

### Interface Protocol
- **UART**: 8N1 format at configurable baud rate
- **Framing**: `[HEADER|PAYLOAD|CRC]` structure
- **CSR Map**: Memory-mapped control and status registers

## 📊 Performance Results

- **FPGA Target**: Cyclone V (5CGXFC7C7F23C8)
- **Operating Frequency**: 100 MHz (achieved timing closure)
- **Resource Utilization**: 
  - Logic Elements: ~15K (optimized for area)
  - Memory Blocks: 85% (double-buffered design)
  - DSP Blocks: 90% (dedicated MAC units)

## 🧪 Verification Strategy

### Multi-level Testing
1. **Unit Tests**: Individual component verification
2. **Integration Tests**: System-level functionality
3. **Golden Model**: Bit-accurate Python reference
4. **FPGA Validation**: Hardware-in-the-loop testing

### Test Coverage
- Functional verification of all modules
- Corner case and edge condition testing
- Performance benchmarking and timing analysis
- Power consumption characterization

## 📚 Documentation

- [`HOST_RS_TILER.md`](docs/HOST_RS_TILER.md) - Complete Host RS Tiler implementation guide
- [`PROJECT_COMPLETION_SUMMARY.md`](docs/PROJECT_COMPLETION_SUMMARY.md) - Project status summary
- Hardware docs in `accel v1/docs/` - Basic documentation stubs (minimal coverage)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Performance optimizations
- Additional CNN layer support
- Enhanced quantization schemes
- Documentation improvements

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🏆 Achievements

- ✅ Complete end-to-end CNN accelerator implementation
- ✅ Successful FPGA synthesis and timing closure
- ✅ Bit-accurate functional verification
- ✅ Optimized INT8 quantization with minimal accuracy loss
- ✅ Scalable systolic array architecture
- ✅ Production-ready UART communication protocol

---

**Note**: This accelerator is designed for educational and research purposes, demonstrating modern CNN acceleration techniques and FPGA implementation best practices.
