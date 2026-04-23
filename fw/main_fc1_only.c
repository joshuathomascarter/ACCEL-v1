// FC1 Unit Test - Load pre-computed Conv2 output, run FC1 only
// Tests the 140x9216 sparse GEMV on actual hardware

#include <stdint.h>
#include <string.h>
#include "hal_accel.h"

// Conv2 output (8x8x64 = 4096 values, padded to 9216 for alignment)
extern uint32_t dram_init_fc1_input[];  // Pre-computed Conv2 output in DRAM

// FC1 weights: 140x9216 sparse BSR matrix
extern uint32_t dram_init_fc1_weights[];

// FC1 bias: 140 values
extern uint32_t dram_init_fc1_bias[];

// Golden FC1 output: 140 values
extern uint32_t dram_init_fc1_golden[];

int main(void) {
    uart_init(50_000_000, 115_200);
    uart_puts("FC1 Unit Test\r\n");

    // Configure accelerator for FC1 (systolic 8x8 grid, single invocation)
    accel_set_mode(ACCEL_MODE_GEMV);  // or appropriate FC1 mode
    accel_set_output_channels(140);
    accel_set_input_dim(9216);

    uint32_t start_cycle = gpio_read();  // Mark start
    uart_puthex32(start_cycle);
    uart_puts(" : FC1 start\r\n");

    // Issue FC1 on accelerator
    // Load input from DRAM address 0x80000000 + input_offset
    // Weights at 0x80000000 + weight_offset
    // Store output at 0x80000000 + output_offset
    accel_run();

    // Poll for completion
    uint32_t timeout = 10_000_000;
    while (!accel_done() && timeout-- > 0) {
        __asm__("nop");
    }

    uint32_t end_cycle = gpio_read();
    uart_puthex32(end_cycle);
    uart_puts(" : FC1 done\r\n");

    // Verify output matches golden
    uint32_t errors = 0;
    for (int i = 0; i < 140; i++) {
        uint32_t output = dram_read(0x80000000 + output_offset + i*4);
        uint32_t golden = dram_init_fc1_golden[i];
        if (output != golden) {
            errors++;
        }
    }

    if (errors == 0) {
        uart_puts("PASS: FC1 output matches golden\r\n");
        gpio_write(0xF7);  // Done, pass
    } else {
        uart_puthex32(errors);
        uart_puts(" errors\r\nFAIL: FC1 mismatch\r\n");
        gpio_write(0xF8);  // Done, fail
    }

    uart_puts("Done.\r\n");
    return 0;
}
