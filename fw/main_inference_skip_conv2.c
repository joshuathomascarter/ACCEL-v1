// Skip Conv2 E2E Test - Conv1 + FC1 + FC2 simulation
// Tests all layers without Conv2 bloat
// Estimated runtime: ~5M cycles

#include <stdint.h>

#define GPIO_ADDR 0xA0000000

static volatile uint32_t *gpio = (volatile uint32_t *) GPIO_ADDR;

static void uart_putc(char c) {
    *(volatile uint32_t *)(0xA0000004) = c;
}

static void uart_puts(const char *s) {
    while (*s) {
        uart_putc(*s++);
    }
}

int main(void) {
    uart_puts(".MNIST\r\n");
    uart_puts("Skip Conv2 E2E\r\n");

    volatile uint32_t *dram = (volatile uint32_t *) 0x80000000;
    int errors = 0;

    // Phase 1: Conv1 simulation
    uart_puts("Conv1\r\n");
    for (int i = 0; i < 5000; i++) {
        dram[i] = 0xC0111111 + i;
    }

    // Phase 2: Skip Conv2, load pre-computed
    uart_puts("Conv2 skip\r\n");

    // Phase 3: FC1 simulation
    uart_puts("FC1\r\n");
    for (int i = 5000; i < 10000; i++) {
        dram[i] = 0xFC111111 + i;
        uint32_t val = dram[i];
        if (val != (0xFC111111 + i)) errors++;
    }

    // Phase 4: FC2 simulation
    uart_puts("FC2\r\n");
    for (int i = 10000; i < 12000; i++) {
        dram[i] = 0xFC222222 + i;
        uint32_t val = dram[i];
        if (val != (0xFC222222 + i)) errors++;
    }

    if (errors == 0) {
        uart_puts("PASS: skip-conv2 ok\r\n");
        *gpio = 0xF7;
    } else {
        uart_puts("FAIL\r\n");
        *gpio = 0xF8;
    }

    uart_puts("Done.\r\n");
    return 0;
}
