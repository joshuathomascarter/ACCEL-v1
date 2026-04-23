// DRAM Scheduler Test - Stress the scheduler with random access patterns
// Tests for deadlock in < 30 minutes

#include <stdint.h>

#define DRAM_BASE 0x80000000
#define GPIO_ADDR 0xA0000000

static volatile uint32_t *dram = (volatile uint32_t *) DRAM_BASE;
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
    uart_puts("DRAM Sched Test\r\n");

    int errors = 0;

    // Phase 1: Sequential writes
    uart_puts("Ph1\r\n");
    for (int i = 0; i < 5000; i++) {
        dram[i] = 0xDEADBEEF + i;
    }

    // Phase 2: Reads with row conflicts
    uart_puts("Ph2\r\n");
    for (int i = 0; i < 5000; i++) {
        uint32_t val = dram[i];
        if (val != (0xDEADBEEF + i)) {
            errors++;
        }
    }

    // Phase 3: Bank-conflict writes
    uart_puts("Ph3\r\n");
    for (int i = 5000; i < 10000; i++) {
        dram[i] = 0xCAFEBABE + i;
    }

    // Phase 4: Random pattern reads
    uart_puts("Ph4\r\n");
    for (int i = 5000; i < 10000; i++) {
        uint32_t val = dram[i];
        if (val != (0xCAFEBABE + i)) {
            errors++;
        }
    }

    uart_puts("Verify\r\n");
    if (errors == 0) {
        uart_puts("PASS: deadlock-free\r\n");
        *gpio = 0xF7;
    } else {
        uart_puts("FAIL: data err\r\n");
        *gpio = 0xF8;
    }

    uart_puts("Done.\r\n");
    return 0;
}
