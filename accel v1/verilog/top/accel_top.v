// accel_top.v
// Top-level scaffold for integration testing of C_tile accelerator.
// This is a lightweight top that exposes memory interfaces for the TB to
// preload A/B data and read back C results. It instantiates a placeholder
// compute unit (behavioral) which performs a blocking matrix multiply using
// combinational logic for the purposes of integration TBs. Replace with
// actual systolic tile_array later.

module accel_top #(
    parameter M = 8,
    parameter N = 8,
    parameter K = 8
)(
    input  wire clk,
    input  wire rst_n,
    // simple memory-mapped ports for TB
    input  wire start, // one-shot pulse to start compute
    output reg  done,
    // simple linear memories: tb will preload A_mem and B_mem via $readmemh
    input  wire [31:0] A_addr,
    input  wire [31:0] B_addr,
    input  wire [31:0] C_addr,
    input  wire [31:0] mem_wr_data,
    input  wire mem_wr_en
);

    // sizes
    localparam A_WORDS = M * K;
    localparam B_WORDS = K * N;
    localparam C_WORDS = M * N;

    // Simple memories
    reg [31:0] A_mem [0:A_WORDS-1];
    reg [31:0] B_mem [0:B_WORDS-1];
    reg [31:0] C_mem [0:C_WORDS-1];

    // Control/Status Registers (CSR)
    reg [31:0] csr_start;
    reg [31:0] csr_done;
    reg [31:0] csr_tile_dims; // [M, N, K]

    // Buffers for A_tile, B_tile, and C_tile
    reg [31:0] A_tile [0:M*K-1];
    reg [31:0] B_tile [0:K*N-1];
    reg [31:0] C_tile [0:M*N-1];

    // Instantiate the PE array (systolic array)
    tile_array #(
        .M(M),
        .N(N),
        .K(K)
    ) pe_array (
        .clk(clk),
        .rst_n(rst_n),
        .A_tile(A_tile),
        .B_tile(B_tile),
        .C_tile(C_tile),
        .start(csr_start[0]),
        .done(csr_done[0])
    );

    // Instantiate the scheduler
    scheduler #(
        .M(M),
        .N(N),
        .K(K)
    ) scheduler_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(csr_start[0]),
        .done(csr_done[0]),
        .A_tile(A_tile),
        .B_tile(B_tile),
        .C_tile(C_tile)
    );

    // UART interface
    uart_rx uart_rx_inst (
        .clk(clk),
        .rst_n(rst_n),
        .rx(rx),
        .data_out(uart_data_out),
        .valid(uart_valid)
    );

    uart_tx uart_tx_inst (
        .clk(clk),
        .rst_n(rst_n),
        .tx(tx),
        .data_in(uart_data_in),
        .ready(uart_ready)
    );

    // CSR logic to handle UART commands
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            csr_start <= 0;
            csr_done <= 0;
            csr_tile_dims <= 0;
        end else if (uart_valid) begin
            // Decode UART commands and update CSRs
            case (uart_data_out[7:0])
                8'h01: csr_start <= uart_data_out[31:0];
                8'h02: csr_tile_dims <= uart_data_out[31:0];
                default: ;
            endcase
        end
    end

    // Done signal logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 0;
        end else begin
            done <= csr_done[0];
        end
    end

    // TB may write directly into A/B via mem_wr_en and address mapping.
    // We'll map mem_wr_en writes to A, B, or C regions depending on provided addresses.
    // For simplicity, treat A_addr/B_addr/C_addr as base addresses; TB should
    // supply addresses within small ranges (0..A_WORDS-1, etc.).

    always @(posedge clk) begin
        if (!rst_n) begin
            done <= 0;
        end else begin
            // host TB write into memories
            if (mem_wr_en) begin
                if (A_addr < A_WORDS)
                    A_mem[A_addr] <= mem_wr_data;
                else if (B_addr < B_WORDS)
                    B_mem[B_addr] <= mem_wr_data;
                else if (C_addr < C_WORDS)
                    C_mem[C_addr] <= mem_wr_data;
            end

            // start pulse triggers compute
            if (start) begin
                integer i,j,k;
                reg signed [63:0] acc;
                for (i = 0; i < M; i = i + 1) begin
                    for (j = 0; j < N; j = j + 1) begin
                        acc = 0;
                        for (k = 0; k < K; k = k + 1) begin
                            acc = acc + $signed(A_mem[i*K + k]) * $signed(B_mem[k*N + j]);
                        end
                        C_mem[i*N + j] <= acc[31:0];
                    end
                end
                done <= 1'b1;
            end else begin
                done <= 1'b0;
            end
        end
    end

    // Export a small read function for TB via tasks (SystemVerilog $readmemh used by TB)
    // Provide explicit accessors via functions for convenience in TB.

    // Readback tasks used by TB (via hierarchical access) if needed.

endmodule
