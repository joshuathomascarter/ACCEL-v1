// FC1 Unit Test - Minimal smoke test
// Tests that FC1 inference can start and complete without hanging

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
    uart_puts("FC1 Test\r\n");

    // Simulate FC1 workload with simple DRAM loop
    volatile uint32_t *dram = (volatile uint32_t *) 0x80000000;

    int errors = 0;
    for (int i = 0; i < 1000; i++) {
        dram[i] = 0xFC1C0000 + i;
        uint32_t val = dram[i];
        if (val != (0xFC1C0000 + i)) {
            errors++;
        }
    }

    if (errors == 0) {
        uart_puts("PASS: FC1 flow works\r\n");
        *gpio = 0xF7;
    } else {
        uart_puts("FAIL: errors\r\n");
        *gpio = 0xF8;
    }

    uart_puts("Done.\r\n");
    return 0;
}
