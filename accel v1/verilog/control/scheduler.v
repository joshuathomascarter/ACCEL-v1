//------------------------------------------------------------------------------
// scheduler.v
// Tiled RS (Row-Stationary) GEMM scheduler for systolic array
//
// Responsibilities:
//  - Implements tile loop-nest over (m_tile, n_tile, k_tile)
//  - Drives buffers/array: clr, en, rd_en, k_idx, bank_sel_rd_A/B
//  - Honors 1-cycle SRAM read latency (see PREPRIME parameter)
//  - Generates row/col enable masks for edge tiles (Tm_eff/Tn_eff)
//  - Ping/Pong bank policy: k_tile parity
//  - Status + perf: busy, done_tile, tile coords, cycles, stall_cycles
//
// Latency note (per tile, ideal):
//   cycles_tile ≈ Tk_eff + (Tm-1) + (Tn-1)
//   +1 cycle bubble if PREPRIME=0 (documented below)
//
// Copyright:
//   Accel v1 (INT8 GEMM IP). MIT/Apache as you choose.
//------------------------------------------------------------------------------

module scheduler #(
  // Dimension widths (log2 maxima)
  parameter int M_W  = 10,  // log2(max M)
  parameter int N_W  = 10,  // log2(max N)
  parameter int K_W  = 12,  // log2(max K) -- K is typically larger
  parameter int TM_W = 6,   // log2(max Tm)
  parameter int TN_W = 6,   // log2(max Tn)
  parameter int TK_W = 6,   // log2(max Tk)

  // Enable pre-prime of SRAM read latency:
  //  PREPRIME = 1 -> perform a 1-cycle "dummy read" before STREAM_K, so first compute cycle has valid a_vec/b_vec (no bubble).
  //  PREPRIME = 0 -> simpler: first STREAM_K cycle is a bubble (rd_en=1/k_idx=0, en=0), compute starts next cycle.
  parameter bit PREPRIME = 0,

  // Set to 1 to require host-provided MT/NT/KT
  parameter bit USE_CSR_COUNTS = 1
)(
  input  logic                 clk,
  input  logic                 rst_n,

  // -------- CSR/config inputs (programmed by host) --------
  input  logic                 start,     // pulse to start (level acceptable; latched internally)
  input  logic                 abort,     // synchronous abort
  input  logic [M_W-1:0]       M,         // problem dims
  input  logic [N_W-1:0]       N,
  input  logic [K_W-1:0]       K,
  input  logic [TM_W-1:0]      Tm,        // tile dims
  input  logic [TN_W-1:0]      Tn,
  input  logic [TK_W-1:0]      Tk,

  // Optional: precomputed tile counts from CSR (to avoid div). If zero, will compute internally.
  input  logic [M_W-1:0]       MT_csr,    // ceil(M/Tm)
  input  logic [N_W-1:0]       NT_csr,    // ceil(N/Tn)
  input  logic [K_W-1:0]       KT_csr,    // ceil(K/Tk)
  // input  logic                 use_csr_counts, // 1: use *_csr; 0: compute internally (synth div)

  // Bank readiness (set by host/DMA when ping/pong banks hold the needed tile)
  input  logic                 valid_A_ping,
  input  logic                 valid_A_pong,
  input  logic                 valid_B_ping,
  input  logic                 valid_B_pong,

  // -------- Drives buffers / array --------
  output logic                 rd_en, 
           // to both A/B buffers
  output logic [TK_W-1:0]      k_idx,          // 0..Tk_eff-1
  output logic                 bank_sel_rd_A,  // 0: ping, 1: pong
  output logic                 bank_sel_rd_B,  // typically mirror A
  output logic                 clr,            // 1-cycle pulse at tile start
  output logic                 en,             // MAC enable (AND with row/col masks externally if desired)
  output logic [MAX_TM-1:0]    en_mask_row,    // bit i = 1 -> row i valid
  output logic [MAX_TN-1:0]    en_mask_col,    // bit j = 1 -> col j valid

  // -------- Status / perf --------
  output logic                 busy,
  output logic                 done_tile,      // 1-cycle pulse at end of C tile
  output logic [M_W-1:0]       m_tile,         // current tile row index (0..MT-1)
  output logic [N_W-1:0]       n_tile,         // current tile col index (0..NT-1) (driven from n_tile_r)
  output logic [K_W-1:0]       k_tile,         // current tile depth index (0..KT-1)
  output logic [31:0]          cycles_tile,    // counts cycles within tile
  output logic [31:0]          stall_cycles    // cycles stalled waiting for bank valid
);

  // ------------------------
  // Internal registers/state
  // ------------------------
  typedef enum logic [2:0] {
    S_IDLE,
    S_PREP_TILE,
    S_WAIT_READY,
    S_PREPRIME_RD,  // optional, only used if PREPRIME=1
    S_STREAM_K,
    S_TILE_DONE,
    S_DONE
  } state_e;

  state_e state, state_n;

  // Tile counts & effective sizes
  logic [M_W-1:0] MT; // number of m tiles
  logic [N_W-1:0] NT; // number of n tiles
  logic [K_W-1:0] KT; // number of k tiles

  // Internal registered tile indices (explicit internal ownership)
  logic [M_W-1:0] m_tile_r;
  logic [N_W-1:0] n_tile_r; // existing uses already refer to n_tile_r
  logic [K_W-1:0] k_tile_r;

  // Effective dims for the *current* edges
  logic [TM_W-1:0] Tm_eff;
  logic [TN_W-1:0] Tn_eff;
  logic [TK_W-1:0] Tk_eff;

  // k loop counter within current k-tile
  logic [TK_W-1:0] k_ctr;

  // Scratch for remainder computations
  logic [M_W+TM_W:0] m_off; // m_tile*Tm
  logic [N_W+TN_W:0] n_off; // n_tile*Tn
  logic [K_W+TK_W:0] k_off; // k_tile*Tk

  // Ping/pong selects (simple policy: bank = k_tile[0])
  logic bank_sel_k;      // 0 ping / 1 pong for this k_tile
  logic A_ready, B_ready;

  // Latches
  logic start_latched;

  // Concrete mask sizes derived from log2 parameters (number of PEs per dim)
  localparam int MAX_TM = (1 << TM_W);
  localparam int MAX_TN = (1 << TN_W);

  // Perf counters
  logic [31:0] cycles_tile_r, stall_cycles_r;

  // Outputs default
  always_comb begin
    rd_en          = 1'b0;
    k_idx          = '0;
    bank_sel_rd_A  = bank_sel_k;
    bank_sel_rd_B  = bank_sel_k;
    clr            = 1'b0;
    en             = 1'b0;
    done_tile      = 1'b0;
    busy           = (state != S_IDLE) && (state != S_DONE);
  end

  // ------------------------
  // Helpers (combinational)
  // ------------------------

  // Ceil-div helper (be aware: synthesizes a divider if use_csr_counts=0)
  function automatic [M_W-1:0] ceil_div_M (input logic [M_W-1:0] a, input logic [TM_W-1:0] b);
    ceil_div_M = (b==0) ? '0 : (a + b - 1) / b;
  endfunction
  function automatic [N_W-1:0] ceil_div_N (input logic [N_W-1:0] a, input logic [TN_W-1:0] b);
    ceil_div_N = (b==0) ? '0 : (a + b - 1) / b;
  endfunction
  function automatic [K_W-1:0] ceil_div_K (input logic [K_W-1:0] a, input logic [TK_W-1:0] b);
    ceil_div_K = (b==0) ? '0 : (a + b - 1) / b;
  endfunction

  // Compute effective sizes for edge tiles: eff = min(T*, remaining)
  always_comb begin
    // Offsets in full dims
    m_off  = m_tile_r * Tm;
    n_off  = n_tile_r * Tn;
    k_off  = k_tile_r * Tk;

    // Remaining
    logic [M_W-1:0] m_rem = (M > m_off[M_W-1:0]) ? (M - m_off[M_W-1:0]) : '0;
    logic [N_W-1:0] n_rem = (N > n_off[N_W-1:0]) ? (N - n_off[N_W-1:0]) : '0;
    logic [K_W-1:0] k_rem = (K > k_off[K_W-1:0]) ? (K - k_off[K_W-1:0]) : '0;

    // Effective sizes (clamp to tile sizes)
    Tm_eff = (m_rem > Tm) ? Tm : m_rem[TM_W-1:0];
    Tn_eff = (n_rem > Tn) ? Tn : n_rem[TN_W-1:0];
    Tk_eff = (k_rem > Tk) ? Tk : k_rem[TK_W-1:0];
  end

  // Row/col masks: bit i/j set if within eff sizes
  // Masks are packed LSB=row0/col0..; consumer can AND with 'en' per PE
  always_comb begin
    en_mask_row = '0;
    en_mask_col = '0;
    for (int i = 0; i < MAX_TM; i++) begin
      if (i < Tm_eff) en_mask_row[i] = 1'b1;
    end
    for (int j = 0; j < MAX_TN; j++) begin
      if (j < Tn_eff) en_mask_col[j] = 1'b1;
    end
  end

  // Bank select policy & readiness
  always_comb begin
    bank_sel_k = k_tile_r[0]; // even k_tile -> ping(0), odd -> pong(1)
    A_ready    = bank_sel_k ? valid_A_pong : valid_A_ping;
    B_ready    = bank_sel_k ? valid_B_pong : valid_B_ping;
  end

  // ------------------------
  // Start latch & tile counts
  // ------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      start_latched   <= 1'b0;
    end else begin
      if (start)      start_latched <= 1'b1;
      // cleared when entering S_PREP_TILE
      if (state == S_PREP_TILE) start_latched <= 1'b0;
    end
  end

  // Compute or load tile counts once per session (could also be latched in PREP_TILE)
  always_comb begin
    // Compute fallback counts (safe combinational helpers)
    logic [M_W-1:0] mt_calc;
    logic [N_W-1:0] nt_calc;
    logic [K_W-1:0] kt_calc;

    mt_calc = ceil_div_M(M, Tm);
    nt_calc = ceil_div_N(N, Tn);
    kt_calc = ceil_div_K(K, Tk);

    // Use CSR-provided counts when requested, but fall back to computed values
    // if the CSR fields are zero (defensive: prevents a fatal zero-tile situation).
    if (USE_CSR_COUNTS) begin
      MT = (MT_csr != '0) ? MT_csr : mt_calc;
      NT = (NT_csr != '0) ? NT_csr : nt_calc;
      KT = (KT_csr != '0) ? KT_csr : kt_calc;
    end else begin
      MT = mt_calc;
      NT = nt_calc;
      KT = kt_calc;
    end
  end

  // ------------------------
  // Tile indices & counters
  // ------------------------
  // m_tile, n_tile, k_tile advance in TILE_DONE / PREP next k
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_tile_r <= '0;
      n_tile_r <= '0;
      k_tile_r <= '0;
      k_ctr  <= '0;
    end else if (abort) begin
      m_tile_r <= '0;
      n_tile_r <= '0;
      k_tile_r <= '0;
      k_ctr  <= '0;
    end else begin
      case (state)
        S_PREP_TILE: begin
          // entering a new C tile (m,n), ensure k_tile=0
          k_tile_r <= '0;
          k_ctr  <= '0;
        end
        S_PREPRIME_RD: begin
          // Prime read happens in combinational block; set k_ctr=1 so first
          // STREAM_K cycle reads k_idx=1 (we already read k_idx=0 here)
          k_ctr <= 6'd1;  // Start at 1, not 0
        end
        S_STREAM_K: begin
          if (Tk_eff != '0) begin
            // STREAM_K body: drive rd_en/k_idx; advance local k_ctr
            // For PREPRIME=0: count 0..Tk_eff (extra cycle to consume last read)
            // For PREPRIME=1: count 1..Tk_eff (k_ctr starts at 1 from PREPRIME_RD)
            if (k_ctr < Tk_eff) begin
              k_ctr <= k_ctr + 1'b1;
            end
          end
        end
        S_TILE_DONE: begin
          // Advance n/m; k resets next S_PREP_TILE
          if (n_tile_r + 1 < NT) begin
            n_tile_r <= n_tile_r + 1'b1;
          end else begin
            n_tile_r <= '0;
            if (m_tile_r + 1 < MT) begin
              m_tile_r <= m_tile_r + 1'b1;
            end else begin
              m_tile_r <= '0;
            end
          end
        end
        default: ;
      endcase

      // Advance k_tile when we finish a k-slice
      if (state == S_STREAM_K && (Tk_eff == '0 || (k_ctr == Tk_eff))) begin
        if (k_tile_r + 1 < KT) begin
          k_tile_r <= k_tile_r + 1'b1;
        end
      end
    end
  end

  // ------------------------
  // FSM next-state & outputs
  // ------------------------
  // Note: Outputs rd_en/k_idx/en/clr are driven in this block for clarity.
  always_comb begin
    state_n        = state;

    // Default outputs already zeroed above; we drive deltas per state.
    // k_idx mirrors k_ctr during STREAM_K.
    k_idx          = k_ctr;

    case (state)

      S_IDLE: begin
        if (start_latched) state_n = S_PREP_TILE;
      end

      S_PREP_TILE: begin
        // Fire a 1-cycle clr at the start of each (m,n) tile
        clr     = 1'b1;

        // Reset k_ctr; k_tile is already 0 here
        // Decide whether to pre-prime or go wait for bank ready
        state_n = S_WAIT_READY;
      end

      S_WAIT_READY: begin
        // Wait until both A and B banks for this k_tile are declared ready.
        if (A_ready && B_ready) begin
          if (PREPRIME) state_n = S_PREPRIME_RD;
          else          state_n = S_STREAM_K;
        end
        // Count stall cycles while waiting
      end

      S_PREPRIME_RD: begin
        // PREPRIME path: issue one dummy read so the first STREAM_K cycle has valid data
        // rd_en=1, k_idx=0; BUT en=0 (no MAC enable yet)
        rd_en   = (Tk_eff != '0);
        k_idx   = '0;
        // Next cycle we enter STREAM_K, and compute starts immediately (no bubble).
        state_n = S_STREAM_K;
      end

      S_STREAM_K: begin
        // Bubble-start alternative when PREPRIME=0 (documented):
        //   First STREAM_K cycle asserts rd_en=1, k_idx=0, but en=0
        //   -> buffers output valid vectors next cycle, when en becomes 1.
        //   This introduces a single-cycle compute bubble at start of k-slice.
        //   k_ctr runs 0..Tk_eff (extra cycle at end to consume last read).
        //
        // PREPRIME=1: k_ctr starts at 1 (set in PREPRIME_RD) and runs 1..Tk_eff.
        //   k_idx=1..Tk_eff-1 are read (k_idx=0 was already read in PREPRIME_RD).
        //   Final cycle (k_ctr=Tk_eff) just computes with rd_en=0.

        // Stop reading when k_ctr reaches Tk_eff (final cycle just computes)
        rd_en = (Tk_eff != '0) && (k_ctr < Tk_eff);
        k_idx = k_ctr;

        // Enable MACs except:
        //  - PREPRIME: en=1 always (data is primed from PREPRIME_RD)
        //  - BUBBLE:   en=0 only for the first cycle (k_ctr==0), else en=1
        if (Tk_eff != '0) begin
          if (PREPRIME) begin
            en = 1'b1;
          end else begin
            en = (k_ctr != '0); // bubble on first cycle
          end
        end

        // If we just finished last k step, decide next:
        if (Tk_eff == '0) begin
          // No work in this k-slice (edge case), treat as done
          if (k_tile_r + 1 < KT) state_n = S_WAIT_READY; // next k-slice
          else                 state_n = S_TILE_DONE;
        end else if (k_ctr == Tk_eff) begin
          // End of k-slice (consumed last read); flip bank (via k_tile parity)
          if (k_tile_r + 1 < KT) state_n = S_WAIT_READY; // next k-slice
          else                 state_n = S_TILE_DONE;  // all k done for this (m,n)
        end
      end

      S_TILE_DONE: begin
        done_tile = 1'b1;
        // Advance n/m indices happens in the index block; decide if more tiles remain
        if ((n_tile_r + 1 < NT) || (m_tile_r + 1 < MT)) state_n = S_PREP_TILE;
        else                                         state_n = S_DONE;
      end

      S_DONE: begin
        // Hold until a new start
        if (start) state_n = S_PREP_TILE;
      end

      default: state_n = S_IDLE;
    endcase

    // Abort handling (synchronous)
    if (abort) begin
      state_n = S_IDLE;
    end
  end

  // ------------------------
  // State / perf registers
  // ------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state           <= S_IDLE;
      cycles_tile_r   <= '0;
      stall_cycles_r  <= '0;
    end else begin
      state <= state_n;

      // Per-tile cycle counters
      if (state == S_PREP_TILE) begin
        cycles_tile_r  <= 32'd0;
        stall_cycles_r <= 32'd0;
      end else if (state == S_STREAM_K) begin
        cycles_tile_r  <= cycles_tile_r + 32'd1;
      end else if (state == S_WAIT_READY) begin
        stall_cycles_r <= stall_cycles_r + 32'd1;
      end
    end
  end

  assign cycles_tile  = cycles_tile_r;
  assign stall_cycles = stall_cycles_r;
  assign m_tile = m_tile_r;
  assign n_tile = n_tile_r;
  assign k_tile = k_tile_r;

  // ------------------------
  // Synthesis-time safety notes (SVA in TB recommended)
  // ------------------------
  // 1) No read from a bank unless A_ready/B_ready for that bank asserted.
  // 2) PREPRIME=0: expect a 1-cycle compute bubble at k-start (en=0 on first cycle).
  // 3) PREPRIME=1: expect no bubble; first STREAM_K cycle computes immediately.
  // 4) en_mask_row/en_mask_col should be ANDed inside array or at MAC enable granularity.
  // 5) Optional: assert Tk != 0, Tm != 0, Tn != 0 at start (or treat zero as no-op edges).
  // 6) Document that bank_sel_rd_A/B = k_tile[0] policy; host must preload the opposite bank.

endmodule
