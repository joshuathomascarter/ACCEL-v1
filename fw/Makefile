# Makefile for RISC-V Bare-Metal Firmware

CROSS_COMPILE ?= riscv64-elf-
CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump

EXTRA_CFLAGS ?=
CFLAGS  = -march=rv32i_zicsr -mabi=ilp32 -nostdlib -ffreestanding -O2 -I. $(EXTRA_CFLAGS)
ASFLAGS = -march=rv32i_zicsr -mabi=ilp32
LDFLAGS = -T link.ld -nostdlib -m elf32lriscv

SOURCES = startup.S main.c
OBJECTS = startup.o main.o

INF_SOURCES = startup.S main_inference.c
INF_OBJECTS = startup.o main_inference.o

FULL_INF_SOURCES = startup.S main_inference_full.c
FULL_INF_OBJECTS = startup.o main_inference_full.o

MT_INF_SOURCES = startup.S main_inference_multitile.c
MT_INF_OBJECTS = startup.o main_inference_multitile.o

CONV_INF_SOURCES = startup.S main_inference_conv.c
CONV_INF_OBJECTS = startup.o main_inference_conv.o

CONV2_SMOKE_SOURCES = startup.S main_conv2_smoke.c
CONV2_SMOKE_OBJECTS = startup.o main_conv2_smoke.o

DRAM_SCHED_TEST_SOURCES = startup.S main_dram_scheduler_test.c
DRAM_SCHED_TEST_OBJECTS = startup.o main_dram_scheduler_test.o

FC1_ONLY_SOURCES = startup.S main_fc1_only.c
FC1_ONLY_OBJECTS = startup.o main_fc1_only.o

SKIP_CONV2_SOURCES = startup.S main_inference_skip_conv2.c
SKIP_CONV2_OBJECTS = startup.o main_inference_skip_conv2.o

REDUCED_CONV2_SOURCES = startup.S main_inference_reduced_conv2.c
REDUCED_CONV2_OBJECTS = startup.o main_inference_reduced_conv2.o

.PHONY: all clean disasm inference full-inference multitile conv conv2-smoke \
        dram-sched-test fc1-only skip-conv2 reduced-conv2

all: firmware.hex

inference: firmware_inference.hex

full-inference: firmware_inference_full.hex

multitile: firmware_inference_multitile.hex

conv: firmware_inference_conv.hex

conv2-smoke: firmware_conv2_smoke.hex

%.o: %.S
	$(CC) $(ASFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

firmware.elf: $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) -o $@

firmware.hex: firmware.elf
	$(OBJCOPY) -O verilog firmware.elf firmware.hex
	@echo "Generated firmware.hex"

firmware_inference.elf: $(INF_OBJECTS)
	$(LD) $(LDFLAGS) $(INF_OBJECTS) -o $@

firmware_inference.hex: firmware_inference.elf
	$(OBJCOPY) -O verilog firmware_inference.elf firmware_inference.hex
	@echo "Generated firmware_inference.hex"

firmware_inference_full.elf: $(FULL_INF_OBJECTS)
	$(LD) $(LDFLAGS) $(FULL_INF_OBJECTS) -o $@

firmware_inference_full.hex: firmware_inference_full.elf
	$(OBJCOPY) -O verilog firmware_inference_full.elf firmware_inference_full.hex
	@echo "Generated firmware_inference_full.hex"

firmware_inference_multitile.elf: $(MT_INF_OBJECTS)
	$(LD) $(LDFLAGS) $(MT_INF_OBJECTS) -o $@

firmware_inference_multitile.hex: firmware_inference_multitile.elf
	$(OBJCOPY) -O verilog firmware_inference_multitile.elf firmware_inference_multitile.hex
	@echo "Generated firmware_inference_multitile.hex"

firmware_inference_conv.elf: $(CONV_INF_OBJECTS)
	$(LD) $(LDFLAGS) $(CONV_INF_OBJECTS) -o $@

firmware_inference_conv.hex: firmware_inference_conv.elf
	$(OBJCOPY) -O verilog firmware_inference_conv.elf firmware_inference_conv.hex
	@echo "Generated firmware_inference_conv.hex"

firmware_conv2_smoke.elf: $(CONV2_SMOKE_OBJECTS)
	$(LD) $(LDFLAGS) $(CONV2_SMOKE_OBJECTS) -o $@

firmware_conv2_smoke.hex: firmware_conv2_smoke.elf
	$(OBJCOPY) -O verilog firmware_conv2_smoke.elf firmware_conv2_smoke.hex
	@echo "Generated firmware_conv2_smoke.hex"

firmware_dram_scheduler_test.elf: $(DRAM_SCHED_TEST_OBJECTS)
	$(LD) $(LDFLAGS) $(DRAM_SCHED_TEST_OBJECTS) -o $@

firmware_dram_scheduler_test.hex: firmware_dram_scheduler_test.elf
	$(OBJCOPY) -O verilog firmware_dram_scheduler_test.elf firmware_dram_scheduler_test.hex
	@echo "Generated firmware_dram_scheduler_test.hex"

dram-sched-test: firmware_dram_scheduler_test.hex

firmware_fc1_only.elf: $(FC1_ONLY_OBJECTS)
	$(LD) $(LDFLAGS) $(FC1_ONLY_OBJECTS) -o $@

firmware_fc1_only.hex: firmware_fc1_only.elf
	$(OBJCOPY) -O verilog firmware_fc1_only.elf firmware_fc1_only.hex
	@echo "Generated firmware_fc1_only.hex"

fc1-only: firmware_fc1_only.hex

firmware_skip_conv2.elf: $(SKIP_CONV2_OBJECTS)
	$(LD) $(LDFLAGS) $(SKIP_CONV2_OBJECTS) -o $@

firmware_skip_conv2.hex: firmware_skip_conv2.elf
	$(OBJCOPY) -O verilog firmware_skip_conv2.elf firmware_skip_conv2.hex
	@echo "Generated firmware_skip_conv2.hex"

skip-conv2: firmware_skip_conv2.hex

firmware_reduced_conv2.elf: $(REDUCED_CONV2_OBJECTS)
	$(LD) $(LDFLAGS) $(REDUCED_CONV2_OBJECTS) -o $@

firmware_reduced_conv2.hex: firmware_reduced_conv2.elf
	$(OBJCOPY) -O verilog firmware_reduced_conv2.elf firmware_reduced_conv2.hex
	@echo "Generated firmware_reduced_conv2.hex"

reduced-conv2: firmware_reduced_conv2.hex

clean:
	rm -f $(OBJECTS) $(INF_OBJECTS) $(FULL_INF_OBJECTS) $(MT_INF_OBJECTS) $(CONV_INF_OBJECTS) $(CONV2_SMOKE_OBJECTS) \
	      $(DRAM_SCHED_TEST_OBJECTS) $(FC1_ONLY_OBJECTS) $(SKIP_CONV2_OBJECTS) $(REDUCED_CONV2_OBJECTS) \
	      firmware.elf firmware.hex \
	      firmware_inference.elf firmware_inference.hex \
	      firmware_inference_full.elf firmware_inference_full.hex \
	      firmware_inference_multitile.elf firmware_inference_multitile.hex \
	      firmware_inference_conv.elf firmware_inference_conv.hex \
	      firmware_conv2_smoke.elf firmware_conv2_smoke.hex \
	      firmware_dram_scheduler_test.elf firmware_dram_scheduler_test.hex \
	      firmware_fc1_only.elf firmware_fc1_only.hex \
	      firmware_skip_conv2.elf firmware_skip_conv2.hex \
	      firmware_reduced_conv2.elf firmware_reduced_conv2.hex
.PRECIOUS: firmware.elf
