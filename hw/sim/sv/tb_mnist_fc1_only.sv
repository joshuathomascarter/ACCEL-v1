// =============================================================================
// tb_mnist_fc1_only.sv - FC1 Unit Test Testbench
// =============================================================================
// Boots soc_top_v2 with main_fc1_only.hex
// Runs FC1 inference on pre-computed Conv2 output
// Verifies output matches golden reference
// Expected runtime: < 5 minutes (< 500K cycles)
// =============================================================================
`timescale 1ns/1ps
/* verilator lint_off PROCASSINIT */
/* verilator lint_off UNUSEDSIGNAL  */
/* verilator lint_off SYNCASYNCNET  */

module tb_mnist_fc1_only;

  localparam int    DEFAULT_MAX_CYCLES       = 1_000_000;
  localparam int    DEFAULT_PROGRESS_CYCLES  = 100_000;
  localparam string FW_HEX           = "fw/firmware_fc1_only.hex";
  localparam string DRAM_HEX         = "data/dram_init_fc1_only.hex";
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
    if (!$value$plusargs("progress_cycles=%d", progress_interval_cycles))
      progress_interval_cycles = DEFAULT_PROGRESS_CYCLES;
    if (progress_interval_cycles <= 0)
      progress_interval_cycles = DEFAULT_PROGRESS_CYCLES;
    $display("[TB] limits: max_cycles=%0d progress_cycles=%0d",
             max_cycles, progress_interval_cycles);
  end

  always_ff @(posedge clk) begin
    cycle_count <= cycle_count + 1;
    stall_counter <= stall_counter + 1;

    if (cycle_count > 0 && (cycle_count % progress_interval_cycles) == 0) begin
      $display("[TB] progress: cycle=%0d gpio=0x%02x accel_busy=%0b accel_done=%0b",
               cycle_count, gpio_o, accel_busy, accel_done);
      $fflush();
      stall_counter <= 0;
    end

    if (stall_counter > (progress_interval_cycles * 2)) begin
      $display("[TB] DEADLOCK DETECTED: no progress for %0d cycles", stall_counter);
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
  int          uart_chars_rx   = 0;

  string uart_line      = "";
  bit    uart_pass_seen = 0;
  bit    uart_fail_seen = 0;
  bit    uart_done_seen = 0;

  int t_fc1_start = 0;
  int t_fc1_done  = 0;

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
          uart_chars_rx  <= uart_chars_rx + 1;
          if (uart_shift == 8'h0A) begin
            $display("[UART @%0d] %s", cycle_count, uart_line);
            $fflush();
            if (uart_line == "PASS: FC1 output matches golden") begin
              uart_pass_seen = 1;
              t_fc1_done = cycle_count;
            end
            if (uart_line.substr(0,4) == "FAIL:") begin
              uart_fail_seen = 1;
              t_fc1_done = cycle_count;
            end
            if (uart_line == "Done.")
              uart_done_seen = 1;
            uart_line = "";
          end else if (uart_shift != 8'h0D && uart_shift != 8'h00) begin
            uart_line = {uart_line,
                         string'(((uart_shift >= 8'h20) && (uart_shift <= 8'h7e))
                                  ? uart_shift : 8'h2e)};
          end
        end
      end else begin
        uart_bit_cnt <= uart_bit_cnt - 1;
      end
    end
  end

  always_ff @(posedge clk)
    if (gpio_o[0] && t_fc1_start == 0 && rst_n)
      t_fc1_start <= cycle_count;

  int tests_passed = 0;
  int tests_failed = 0;

  task automatic check(
    input string test_name,
    input logic  cond,
    input string fail_msg = ""
  );
    if (cond) begin
      $display("[PASS] %-55s @cycle %0d", test_name, cycle_count);
      $fflush();
      tests_passed++;
    end else begin
      $display("[FAIL] %-55s @cycle %0d  %s", test_name, cycle_count, fail_msg);
      $fflush();
      tests_failed++;
    end
  endtask

  initial begin
    $display("=================================================================");
    $display("  tb_mnist_fc1_only - FC1 Unit Test");
    $display("  Firmware: %s", FW_HEX);
    $display("  DRAM:     %s", DRAM_HEX);
    $display("=================================================================");
    $fflush();

    rst_n = 1'b0;
    repeat (40) @(posedge clk);
    rst_n = 1'b1;
    $display("[TB] Reset released @cycle %0d", cycle_count);
    $fflush();

    begin
      int bwait = 0;
      while (uart_chars_rx == 0 && bwait < 500_000) begin
        @(posedge clk); bwait++;
      end
      check("UART boot message received", uart_chars_rx > 0,
            "No UART output after 500k cycles");
    end

    begin
      while (!uart_done_seen && cycle_count < max_cycles)
        @(posedge clk);
      check("FC1 completed (Done. on UART)", uart_done_seen,
            "Firmware never printed Done.");
    end

    repeat (500) @(posedge clk);

    $display("\n=== FC1 UNIT TEST RESULTS ===");
    check("UART PASS: output matches golden", uart_pass_seen,
          uart_fail_seen ? "Firmware printed FAIL" : "Neither PASS nor FAIL seen");
    check("GPIO done flag set", gpio_o[7:4] == 4'hF,
          $sformatf("GPIO=0x%02x", gpio_o));

    if (t_fc1_start > 0 && t_fc1_done > 0) begin
      automatic int total_cyc = t_fc1_done - t_fc1_start;
      $display("\n=== PERFORMANCE (FC1 inference) ===");
      $display("  Total inference cycles : %0d", total_cyc);
      $display("  @ 50 MHz               : %.2f us", real'(total_cyc) / 50.0);
      $display("  Throughput             : %.4f inferences/sec",
               50_000_000.0 / real'(total_cyc));
    end

    $display("\n=================================================================");
    $display("  TESTS PASSED: %0d / %0d", tests_passed, tests_passed + tests_failed);
    if (tests_failed == 0)
      $display("  RESULT: PASS - FC1 unit test verified");
    else
      $display("  RESULT: FAIL - %0d check(s) failed", tests_failed);
    $display("=================================================================");
    $fflush();
    $finish;
  end

endmodule
