// Reduced Conv2 E2E Test - Conv1 + Single-channel Conv2 + FC1 + FC2
// Tests all hardware paths without Conv2's full 64-channel bloat
// Estimated runtime: ~30M cycles vs 180M for full Conv1+Conv2+FC1+FC2

#include <stdint.h>
#include <string.h>
#include "hal_accel.h"

// Golden reference output (digit 0-9)
extern uint32_t golden_inference_digit;

int main(void) {
    uart_init(50_000_000, 115_200);
    uart_puts(".MNIST\r\n");
    uart_puts("Reduced E2E: Conv1 + 1-ch Conv2 + FC1 + FC2\r\n");

    // ====== Conv1: 1x28x28 -> 32x26x26 ======
    uart_puts("Conv1 start\r\n");
    accel_set_mode(ACCEL_MODE_CONV);
    accel_set_output_channels(32);  // Full 32 channels
    accel_set_input_h(28);
    accel_set_input_w(28);
    accel_set_kernel_h(3);
    accel_set_kernel_w(3);
    accel_set_stride(1);

    accel_run();
    while (!accel_done());
    uart_puts("Conv1 done\r\n");

    // ====== Conv2: 32x26x26 -> 1x24x24 (SINGLE CHANNEL) ======
    // This is the key reduction: 64 channels -> 1 channel
    // Still exercises all systolic/NoC paths but 1/64th the work
    uart_puts("Conv2 start (1 channel only)\r\n");
    accel_set_mode(ACCEL_MODE_CONV);
    accel_set_output_channels(1);  // REDUCED from 64
    accel_set_input_h(26);
    accel_set_input_w(26);
    accel_set_kernel_h(3);
    accel_set_kernel_w(3);
    accel_set_stride(1);

    accel_run();
    while (!accel_done());
    uart_puts("Conv2 done\r\n");

    // Conv2 output: 1x24x24 = 576 values, maxpool to 8x8 = 64
    // Feed into FC1

    // ====== FC1: 64 -> 140 (sparse GEMV) ======
    uart_puts("FC1 start\r\n");
    accel_set_mode(ACCEL_MODE_GEMV_SPARSE);
    accel_set_output_dim(140);
    accel_set_input_dim(64);  // Pooled Conv2 output

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
    uint32_t predicted = accel_get_prediction();  // Argmax of FC2 output

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
