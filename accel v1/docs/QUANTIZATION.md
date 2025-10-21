# Quantization (v1)

- **Scheme**: symmetric INT8, per-tensor scale S, zero-point = 0.
- **Calib**: min/max percentile (TBD).
- **Math**: y = clamp(round((x_fp32 / Sx) * (W_fp32 / Sw)) * Sacc → shift → INT8)
- **Next**: per-channel for W, optional zp, bias folding.
