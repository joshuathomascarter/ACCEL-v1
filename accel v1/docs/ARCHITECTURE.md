```markdown
# Architecture

## Dataflow (v1)
- **Weights**: pre-quantized INT8, streamed into `wgt_buffer`.
- **Activations**: im2col tiles as INT8 into `act_buffer`.
- **Compute**: systolic array (N x M PEs), int32 accumulators.
- **Post**: clamp/shift → INT8, return via UART.

## Interfaces
- **CSR**: start, dims (M,N,K), strides, tile sizes, scale/shift.
- **UART**: 8N1 @ configurable baud; simple framing: [HDR|PAYLOAD|CRC].

*(Diagram lives in README; final fig export to `docs/figs/` later.)*