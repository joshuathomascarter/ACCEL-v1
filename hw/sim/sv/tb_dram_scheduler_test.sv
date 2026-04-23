// =============================================================================
// tb_dram_scheduler_test.sv - DRAM Scheduler Unit Test
// =============================================================================
// Stress-tests DRAM scheduler with random access patterns
// Expected runtime: < 10 minutes (< 500K cycles)
// Tests for deadlock, data corruption, bank conflicts
// =============================================================================
`timescale 1ns/1ps
/* verilator lint_off PROCASSINIT */
/* verilator lint_off UNUSEDSIGNAL  */

module tb_dram_scheduler_test;

  localparam int    DEFAULT_MAX_CYCLES       = 2_000_000;
  localparam int    DEFAULT_PROGRESS_CYCLES  = 100_000;
  localparam string FW_HEX           = "fw/firmware_dram_scheduler_test.hex";
  localparam string DRAM_HEX         = "data/dram_init_blank.hex";
  localparam int    CLK_HALF_NS      = 10;       // 50 MHz
  localparam int    UART_BIT_CYCLES  = 50_000_000 / 115_200;

  logic clk   = 1'b0;
  logic rst_n = 1'b0;
  always #(CLK_HALF_NS) clk = ~clk;

  logic        uart_rx          = 1'b1;
  logic        uart_tx;
  logic [7:0]  gpio_o;
  logic [7:0]  gpio_i           = 8'h0;
  logic [7:0]  gpio_oe;
  logic        irq_external     = 1'b0;
  logic        irq_timer        = 1'b0;
  logic        accel_busy;
  logic        accel_done;

  logic [7:0]  dram_phy_act;
  logic [7:0]  dram_phy_read;
  logic [7:0]  dram_phy_write;
  logic [7:0]  dram_phy_pre;
  logic [13:0] dram_phy_row;
  logic [9:0]  dram_phy_col;
  logic        dram_phy_ref;
  logic [31:0] dram_phy_wdata;
  logic [3:0]  dram_phy_wstrb;
  logic [31:0] dram_phy_rdata;
  logic        dram_phy_rdata_valid;
  logic        dram_ctrl_busy;

  soc_top_v2 #(
    .BOOT_ROM_FILE (FW_HEX),
    .CLK_FREQ      (50_000_000),
    .UART_BAUD     (115_200),
    .MESH_ROWS     (4),
    .MESH_COLS     (4)
  ) dut (
    .clk                  (clk),
    .rst_n                (rst_n),
    .uart_rx              (uart_rx),
    .uart_tx              (uart_tx),
    .gpio_o               (gpio_o),
    .gpio_i               (gpio_i),
    .gpio_oe              (gpio_oe),
    .irq_external         (irq_external),
    .irq_timer            (irq_timer),
    .accel_busy           (accel_busy),
    .accel_done           (accel_done),
    .dram_phy_act         (dram_phy_act),
    .dram_phy_read        (dram_phy_read),
    .dram_phy_write       (dram_phy_write),
    .dram_phy_pre         (dram_phy_pre),
    .dram_phy_row         (dram_phy_row),
    .dram_phy_col         (dram_phy_col),
    .dram_phy_ref         (dram_phy_ref),
    .dram_phy_wdata       (dram_phy_wdata),
    .dram_phy_wstrb       (dram_phy_wstrb),
    .dram_phy_rdata       (dram_phy_rdata),
    .dram_phy_rdata_valid (dram_phy_rdata_valid),
    .dram_ctrl_busy       (dram_ctrl_busy)
  );

  dram_phy_simple_mem #(
    .NUM_BANKS  (8),
    .ROW_BITS   (14),
    .COL_BITS   (10),
    .DATA_W     (32),
    .MEM_WORDS  (524288),
    .INIT_FILE  (DRAM_HEX)
  ) u_dram (
    .clk                  (clk),
    .rst_n                (rst_n),
    .dram_phy_act         (dram_phy_act),
    .dram_phy_read        (dram_phy_read),
    .dram_phy_write       (dram_phy_write),
    .dram_phy_pre         (dram_phy_pre),
    .dram_phy_row         (dram_phy_row),
    .dram_phy_col         (dram_phy_col),
    .dram_phy_ref         (dram_phy_ref),
    .dram_phy_wdata       (dram_phy_wdata),
    .dram_phy_wstrb       (dram_phy_wstrb),
    .dram_phy_rdata       (dram_phy_rdata),
    .dram_phy_rdata_valid (dram_phy_rdata_valid)
  );

  int cycle_count = 0;
  int max_cycles = DEFAULT_MAX_CYCLES;
  int progress_interval_cycles = DEFAULT_PROGRESS_CYCLES;
  int stall_counter = 0;

  initial begin
    if (!$value$plusargs("max_cycles=%d", max_cycles))
      max_cycles = DEFAULT_MAX_CYCLES;
    $display("[TB] DRAM Scheduler Test - max_cycles=%0d", max_cycles);
  end

  always_ff @(posedge clk) begin
    cycle_count <= cycle_count + 1;
    stall_counter <= stall_counter + 1;

    if (cycle_count > 0 && (cycle_count % progress_interval_cycles) == 0) begin
      $display("[TB] cycle=%0d gpio=0x%02x dram_busy=%0b", cycle_count, gpio_o, dram_ctrl_busy);
      $fflush();
      stall_counter <= 0;
    end

    // Deadlock detection
    if (stall_counter > (progress_interval_cycles * 3)) begin
      $display("[TB] DEADLOCK: no progress for %0d cycles", stall_counter);
      $display("[TB] RESULT: FAIL (deadlock)");
      $fflush();
      $finish;
    end

    if (cycle_count >= max_cycles) begin
      $display("[TB] TIMEOUT at %0d cycles", cycle_count);
      $display("[TB] RESULT: FAIL (timeout)");
      $fflush();
      $finish;
    end
  end

  logic        uart_capturing  = 1'b0;
  int          uart_bit_cnt    = 0;
  int          uart_sample_pt;
  logic [7:0]  uart_shift      = 8'h0;
  int          uart_rx_cnt     = 0;
  logic        uart_prev_tx    = 1'b1;

  string uart_line      = "";
  bit    uart_pass_seen = 0;
  bit    uart_fail_seen = 0;
  bit    uart_done_seen = 0;

  initial uart_sample_pt = UART_BIT_CYCLES / 2;

  always_ff @(posedge clk) begin
    uart_prev_tx <= uart_tx;
    if (!uart_capturing && uart_prev_tx && !uart_tx) begin
      uart_capturing <= 1'b1;
      uart_bit_cnt   <= UART_BIT_CYCLES + uart_sample_pt;
      uart_rx_cnt    <= 0;
      uart_shift     <= 8'h0;
    end
    if (uart_capturing) begin
      if (uart_bit_cnt == 0) begin
        uart_bit_cnt <= UART_BIT_CYCLES - 1;
        if (uart_rx_cnt < 8) begin
          uart_shift  <= {uart_tx, uart_shift[7:1]};
          uart_rx_cnt <= uart_rx_cnt + 1;
        end else begin
          uart_capturing <= 1'b0;
          if (uart_shift == 8'h0A) begin
            $display("[UART] %s", uart_line);
            $fflush();
            if (uart_line == "PASS: No deadlock, all data correct") begin
              uart_pass_seen = 1;
            end
            if (uart_line.substr(0,4) == "FAIL:") begin
              uart_fail_seen = 1;
            end
            if (uart_line == "Done.") begin
              uart_done_seen = 1;
            end
            uart_line = "";
          end else if (uart_shift != 8'h0D && uart_shift != 8'h00) begin
            uart_line = {uart_line, string'(uart_shift)};
          end
        end
      end else begin
        uart_bit_cnt <= uart_bit_cnt - 1;
      end
    end
  end

  initial begin
    $display("=================================================================");
    $display("  DRAM Scheduler Stress Test");
    $display("  Tests: bank conflicts, row hits/misses, deadlock freedom");
    $display("=================================================================");
    $fflush();

    rst_n = 1'b0;
    repeat (40) @(posedge clk);
    rst_n = 1'b1;

    // Wait for test to complete
    while (!uart_done_seen && cycle_count < max_cycles)
      @(posedge clk);

    repeat (500) @(posedge clk);

    $display("\n=================================================================");
    if (uart_pass_seen && !uart_fail_seen)
      $display("  RESULT: PASS - DRAM scheduler is deadlock-free");
    else if (uart_fail_seen)
      $display("  RESULT: FAIL - Data corruption or timeout");
    else
      $display("  RESULT: FAIL - Test incomplete");
    $display("  Total cycles: %0d", cycle_count);
    $display("=================================================================");
    $fflush();
    $finish;
  end

endmodule
