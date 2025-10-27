# ACCEL-v1 System Architecture```markdown

# Architecture

## Overview

## Dataflow (v1)

The ACCEL-v1 is an INT8 CNN accelerator built around a systolic array architecture optimized for matrix multiplication operations. This document provides a comprehensive view of the system design, component interactions, and dataflow implementation.- **Weights**: pre-quantized INT8, streamed into `wgt_buffer`.

- **Activations**: im2col tiles as INT8 into `act_buffer`.

## System Block Diagram- **Compute**: systolic array (N x M PEs), int32 accumulators.

- **Post**: clamp/shift → INT8, return via UART.

```

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐## Interfaces

│   Host System   │◄──►│   UART Interface │◄──►│   CSR Control   │- **CSR**: start, dims (M,N,K), strides, tile sizes, scale/shift.

└─────────────────┘    └──────────────────┘    └─────────────────┘- **UART**: 8N1 @ configurable baud; simple framing: [HDR|PAYLOAD|CRC].

                                                         │

                                                         ▼*(Diagram lives in README; final fig export to `docs/figs/` later.)*
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

## Core Components

### 1. Systolic Array
- **Configuration**: N×M Processing Elements (PEs)
- **Dataflow**: Row-Stationary (weights stationary, activations flow)
- **Precision**: INT8 inputs, INT32 internal accumulators
- **Throughput**: N×M MAC operations per cycle

### 2. Processing Element (PE)
```verilog
module pe (
    input clk, rst_n,
    input [7:0] act_in, wgt_in,
    input [31:0] acc_in,
    output [7:0] act_out,
    output [31:0] acc_out,
    output reg [7:0] wgt_stored
);
```

**PE Operation:**
- Stores weight value locally
- Performs MAC: `acc_out = acc_in + (act_in * wgt_stored)`
- Forwards activation to next PE in row
- Maintains weight for multiple activation streams

### 3. Buffer Architecture

#### Weight Buffer (`wgt_buffer.v`)
- **Capacity**: Configurable depth for weight storage
- **Width**: 8-bit INT8 weights
- **Access Pattern**: Sequential read during weight loading
- **Interface**: Simple FIFO-style with read enable

#### Activation Buffer (`act_buffer.v`)
- **Capacity**: Tile-sized activation storage
- **Width**: 8-bit INT8 activations
- **Access Pattern**: Simultaneous broadcast to PE array
- **Interface**: Dual-port for load/compute overlap

### 4. Control and Status Registers (CSR)

The CSR module (`csr.v`) provides software-visible configuration:

```verilog
// Register Map
0x00: CONTROL       // Start, reset, mode control
0x04: STATUS        // Ready, busy, error flags  
0x08: M_DIM         // Matrix M dimension
0x0C: N_DIM         // Matrix N dimension
0x10: K_DIM         // Matrix K dimension
0x14: TILE_M        // M-dimension tile size
0x18: TILE_N        // N-dimension tile size
0x1C: SCALE_SHIFT   // Post-processing parameters
```

**Control Flow:**
1. Host writes matrix dimensions and tile sizes
2. Host writes scale/shift parameters for quantization
3. Host sets START bit in CONTROL register
4. Hardware sets BUSY flag and begins computation
5. Hardware clears BUSY and sets READY when complete

### 5. UART Communication Interface

#### Protocol Stack
```
┌─────────────────────────────────────┐
│           Host Software             │
├─────────────────────────────────────┤
│         UART Driver Layer           │
├─────────────────────────────────────┤
│      Hardware UART (8N1)           │
├─────────────────────────────────────┤
│         Physical Layer              │
└─────────────────────────────────────┘
```

#### Packet Format
```
[HEADER|LENGTH|PAYLOAD|CRC16]
 1 byte 1 byte N bytes 2 bytes
```

- **HEADER**: Command type (0x01=Write, 0x02=Read, 0x03=Data)
- **LENGTH**: Payload length (0-255 bytes)
- **PAYLOAD**: Command data or matrix values
- **CRC16**: Error detection checksum

## Dataflow Architecture

### Row-Stationary Dataflow

The ACCEL-v1 implements Row-Stationary (RS) dataflow for optimal weight reuse:

```
Weights (stationary):     Activations (flowing):
┌─────┬─────┬─────┐      ┌─────┐ ┌─────┐ ┌─────┐
│ W00 │ W01 │ W02 │ ←──  │ A00 │→│ A01 │→│ A02 │
├─────┼─────┼─────┤      ├─────┤ ├─────┤ ├─────┤
│ W10 │ W11 │ W12 │ ←──  │ A10 │→│ A11 │→│ A12 │
├─────┼─────┼─────┤      ├─────┤ ├─────┤ ├─────┤
│ W20 │ W21 │ W22 │ ←──  │ A20 │→│ A21 │→│ A22 │
└─────┴─────┴─────┘      └─────┘ └─────┘ └─────┘
                             ↓       ↓       ↓
                        [Partial Sums Flow Down]
```

**Advantages:**
- Maximum weight reuse (each weight used for full activation column)
- Minimal weight memory bandwidth
- Natural tiling for large matrices
- Efficient for CNN convolution layers

### Data Movement Patterns

#### 1. Weight Loading Phase
```verilog
for (int row = 0; row < N; row++) begin
    for (int col = 0; col < M; col++) begin
        pe_array[row][col].load_weight(weights[row][col]);
    end
end
```

#### 2. Activation Streaming Phase
```verilog
for (int k = 0; k < K; k++) begin
    for (int m = 0; m < M; m++) begin
        pe_array[0][m].act_in <= activations[k][m];
        // Activations flow through array
    end
end
```

#### 3. Accumulation Collection
```verilog
for (int n = 0; n < N; n++) begin
    for (int m = 0; m < M; m++) begin
        results[n][m] = pe_array[n][M-1].acc_out;
    end
end
```

## Memory Hierarchy

### 1. Host Memory
- **Capacity**: System RAM (GB scale)
- **Content**: Full model weights and activations
- **Access**: UART transfer to accelerator

### 2. On-Chip Buffers
- **Weight Buffer**: 2KB - 8KB (configurable)
- **Activation Buffer**: 1KB - 4KB (configurable)
- **Output Buffer**: 512B - 2KB (configurable)

### 3. PE Local Storage
- **Weight Registers**: 8-bit × N×M PEs
- **Accumulator Registers**: 32-bit × N×M PEs

## Timing and Performance

### Clock Domains
- **System Clock**: 50-100 MHz target frequency
- **UART Clock**: Derived from system clock
- **PE Array Clock**: Same as system clock (synchronous design)

### Performance Characteristics
- **Peak Throughput**: N×M×f_clk MACs/second
- **Effective Utilization**: 80-95% (depending on tile sizes)
- **Memory Bandwidth**: Limited by UART interface (~1MB/s)

### Pipeline Stages
1. **Weight Load**: 1-2 cycles per weight
2. **Activation Stream**: 1 cycle per activation
3. **Computation**: 1 cycle per MAC operation
4. **Output Collection**: 1-2 cycles per result

## Interface Specifications

### CSR Interface
- **Bus Width**: 32-bit
- **Address Space**: 64 registers (256 bytes)
- **Access Latency**: 1 cycle read, 1 cycle write
- **Endianness**: Little-endian

### UART Interface
- **Baud Rate**: 115200 bps (configurable)
- **Frame Format**: 8N1 (8 data bits, no parity, 1 stop bit)
- **Flow Control**: None (software managed)
- **Buffer Depth**: 16-byte RX FIFO, 16-byte TX FIFO

## Power and Area Estimates

### Area Breakdown (for 4×4 array)
- **PE Array**: ~60% of total area
- **Buffers**: ~25% of total area
- **Control Logic**: ~10% of total area
- **I/O Interface**: ~5% of total area

### Power Breakdown
- **Compute**: ~70% of total power
- **Memory**: ~20% of total power
- **I/O**: ~10% of total power

## Design Considerations

### Scalability
- **Array Size**: Parameterizable N×M configuration
- **Buffer Sizes**: Configurable based on target applications
- **Clock Frequency**: Scalable based on technology node

### Optimization Opportunities
1. **Sparsity Support**: Zero-skipping for sparse weights/activations
2. **Mixed Precision**: 4-bit/8-bit/16-bit support
3. **Advanced Dataflows**: Weight-stationary, output-stationary
4. **Memory Hierarchy**: Multi-level buffer hierarchy

### Verification Strategy
- **Unit Testing**: Individual PE, buffer, CSR verification
- **Integration Testing**: Full array with golden model comparison
- **System Testing**: Host software + hardware co-verification
- **Performance Testing**: Throughput and latency characterization

## Future Enhancements

### Short Term (v1.1)
- Enhanced error detection and recovery
- Improved UART throughput with compression
- Power management modes

### Medium Term (v2.0)
- Support for different quantization schemes
- Multiple array configurations
- Hardware-software co-design optimizations

### Long Term (v3.0)
- Multi-accelerator systems
- Advanced neural network layer support
- Real-time inference capabilities

---

*This architecture document serves as the authoritative reference for the ACCEL-v1 system design. For implementation details, refer to the individual Verilog modules in the `verilog/` directory.*