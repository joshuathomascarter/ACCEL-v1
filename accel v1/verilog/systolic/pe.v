/*
  pe.v - Systolic Processing Element (PE)
  --------------------------------------
  Features (project-standard)
   - Pass-through of a/b to neighbors with a configurable 1-stage skew pipeline
   - Holds partial-sum locally using mac8 (RS-compatible)
   - Pk = 1 lane (single MAC per PE)
   - Parameters:
       PIPE = 1 -> enable internal pipeline registers that create one-cycle
                 skew between inputs and forwarded outputs (default = 1)
       SAT  = 0 -> forwarded to mac8 for saturation behaviour
   - Controls:
       clk, rst_n  : clock / active-low reset
       clr         : synchronous clear of local partial-sum (drains to zero)
       en          : enable for accumulation this cycle
   - IO:
       a_in,b_in   : incoming INT8 activation/weight
       a_out,b_out : forwarded values to neighbor PEs (skewed if PIPE=1)
       acc         : local 32-bit partial-sum (from mac8)
*/
`ifndef PE_V
`define PE_V
`default_nettype none
// -----------------------------------------------------------------------------
// Title      : pe
// File       : pe.v
// Description: Systolic Processing Element (single MAC lane, RS dataflow).
//              Verilog-2001 compliant; forwards activation (a) horizontally
//              and weight (b) vertically with optional one-cycle pipeline skew.
//
// Requirements Trace:
//   REQ-ACCEL-PE-01: Pass-through a_in/b_in to neighbors with PIPE selectable skew.
//   REQ-ACCEL-PE-02: Accumulate partial sum locally via mac8 (Pk=1).
//   REQ-ACCEL-PE-03: Support synchronous clear of partial sum (clr).
//   REQ-ACCEL-PE-04: Provide deterministic hold when en=0.
// -----------------------------------------------------------------------------
// Parameters:
//   PIPE (0/1): 1 inserts a pipeline register stage introducing one-cycle skew.
//   SAT  (0/1): passed to mac8 for saturation behavior.
// -----------------------------------------------------------------------------
module pe #(parameter PIPE = 1, parameter SAT = 0)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire signed [7:0] a_in,
    input  wire signed [7:0] b_in,
    input  wire              en,
    input  wire              clr,
    output wire signed [7:0] a_out,
    output wire signed [7:0] b_out,
    output wire signed [31:0] acc
);

    // Internal registers for pipeline
    reg signed [7:0] a_reg, b_reg;
    reg signed [7:0] a_del, b_del;
    wire sat_internal;

    // Sequential pipeline (Verilog-2001 style)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg <= 8'sd0; b_reg <= 8'sd0; a_del <= 8'sd0; b_del <= 8'sd0;
        end else begin
            if (clr) begin
                a_reg <= 8'sd0; b_reg <= 8'sd0; // zero MAC operands on clear
            end else begin
                a_reg <= a_in;  b_reg <= b_in;
            end
            a_del <= a_reg; b_del <= b_reg; // forward chain
        end
    end

    // Select signals depending on PIPE parameter
    wire signed [7:0] mac_a = (PIPE) ? a_reg : a_in;
    wire signed [7:0] mac_b = (PIPE) ? b_reg : b_in;
    assign a_out = (PIPE) ? a_del : a_in;
    assign b_out = (PIPE) ? b_del : b_in;

    mac8 #( .SAT(SAT) ) u_mac (
        .clk(clk), .rst_n(rst_n),
        .a(mac_a), .b(mac_b),
        .clr(clr), .en(en),
        .acc(acc), .sat_flag(sat_internal)
    );

endmodule
`default_nettype wire
`endif

