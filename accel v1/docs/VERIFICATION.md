# Verification Plan

- **Unit**: MAC (signed sat), PE (weight/act flow), buffers (read/write).
- **Integration**: NxM array vs. golden GEMM for random seeds & dims.
- **HW/SW**: UART loopback sim → host driver → compare outputs.
- **Coverage**: sign combos, saturation cases, tiling edges, K not multiple of tile.
- **Pass/Fail**: bit-exact INT8 match to golden; timing of ready/valid handshakes.
