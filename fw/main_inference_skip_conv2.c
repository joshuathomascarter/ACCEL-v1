// Skip Conv2 E2E Test - Conv1 + FC1 + FC2 only
// Jumps Conv2 entirely, uses pre-computed Conv2 output from DRAM
// Tests FC1+FC2 correctness without 60M+ Conv2 cycles
// Estimated runtime: ~5M cycles (Conv1) + ~1M (FC1) + ~100K (FC2) + overhead

#include <stdint.h>
#include <string.h>
#include "hal_accel.h"

// Pre-computed Conv2 outputs (8x8x64 -> maxpool 8x8) in DRAM
// At address: 0x80000000 + 0x10000 (after DRAM init data)
extern uint32_t dram_init_conv2_output[];

// Golden reference digit
extern uint32_t golden_inference_digit;

int main(void) {
    uart_init(50_000_000, 115_200);
    uart_puts(".MNIST\r\n");
    uart_puts("Skip Conv2: Conv1 + FC1 + FC2\r\n");

    // ====== Conv1: 1x28x28 -> 32x26x26 ======
    uart_puts("Conv1 start\r\n");
    accel_set_mode(ACCEL_MODE_CONV);
    accel_set_output_channels(32);
    accel_set_input_h(28);
    accel_set_input_w(28);
    accel_set_kernel_h(3);
    accel_set_kernel_w(3);
    accel_set_stride(1);

    accel_run();
    while (!accel_done());
    uart_puts("Conv1 done\r\n");

    // ====== SKIP Conv2 - use pre-computed output from DRAM ======
    // Conv2 output already computed and stored at DRAM address 0x80010000
    // 64 channels x 8x8 = 4096 values (16KB)
    // Maxpool to 8x8 = 64 values (256B)
    uart_puts("Conv2 skipped (using pre-computed from DRAM)\r\n");

    // Load Conv2 output into tile local memory for FC1
    // Address: 0x80010000 (Conv2 output in DRAM)
    uint32_t conv2_addr = 0x80010000;

    // ====== FC1: 64 -> 140 (sparse GEMV) ======
    uart_puts("FC1 start\r\n");
    accel_set_mode(ACCEL_MODE_GEMV_SPARSE);
    accel_set_output_dim(140);
    accel_set_input_dim(64);
    accel_set_input_addr(conv2_addr);  // Load from Conv2 output

    accel_run();
    while (!accel_done());
    uart_puts("FC1 done\r\n");

    // ====== FC2: 140 -> 10 (dense GEMV) ======
    uart_puts("FC2 start\r\n");
    accel_set_mode(ACCEL_MODE_GEMV_DENSE);
    accel_set_output_dim(10);
    accel_set_input_dim(140);

    accel_run();
    while (!accel_done());
    uart_puts("FC2 done\r\n");

    // Verify result
    uint32_t predicted = accel_get_prediction();

    if (predicted == golden_inference_digit) {
        uart_puts("PASS: matches golden\r\n");
        gpio_write(0xF0 | predicted);
    } else {
        uart_puts("FAIL: wrong digit\r\n");
        gpio_write(0xE0 | predicted);
    }

    uart_puts("Done.\r\n");
    return 0;
}
