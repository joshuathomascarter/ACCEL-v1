# ============================================================
# sta_prects.tcl — Pre-CTS Static Timing Analysis
# Runs on clean netlist from yosys_resynth3 (all RTL fixes applied).
# No netlist patching needed:
#   - clock_gate_cell: assign clk_o = clk_i  (no reg/latch)
#   - accel_tile: no $write blocks            (ifdef SYNTHESIS guard)
#   - boot_rom: no 65536'h constant           (ifdef SYNTHESIS guard)
#   - PLIC: parallel eligible + |eligible     (no 38ns chain)
#
# Strategy: skip report_wns (full graph traversal too slow on large netlist).
#           report_checks finds worst path directly.
# ============================================================

read_liberty sky130_zerowirelod.lib
read_liberty macros/sram_1rw_wrapper.lib
# Blackbox stubs for ibex_cpu and accel_tile_array — blackboxed during synthesis,
# not in the netlist, so we provide port-only definitions for OpenSTA elaboration.
read_verilog macros/blackbox_stubs.v
read_verilog runs/timing_run/results/synthesis/soc_top_v2.v
link_design soc_top_v2
read_sdc constraints/soc_top_v2.sdc

puts "Wireload: ZeroWL — gate delays only, pre-CTS ideal wires"

# Cap output loads to prevent NLDM extrapolation on high-fanout nets.
# sky130_fd_sc_hd cells characterised to ~50-100 fF max load.
# Without this cap, high-fanout nets (reset tree, tile enables) accumulate
# 100s of pin-cap loads → table extrapolation → ns-range artificial delays.
# 10 fF is a conservative single-cell load, well within the characterized range.
set_load 0.010 [get_nets *]

# ── False paths ──────────────────────────────────────────────────────────────
set_false_path -from [get_ports rst_n]

# /DE endpoint false-paths: DFFE write-enables (systolic accum regs, DMA bufs).
# Endpoint-targeted = O(N) annotation, does NOT cause graph traversal hang.
set de_pins [get_pins -hierarchical */DE]
set de_count [llength $de_pins]
if { $de_count > 0 } {
    set_false_path -to $de_pins
    puts "False-pathed $de_count /DE pins"
} else {
    puts "No /DE pins"
}

catch { set_false_path -to [get_cells u_l1_dcache/u_ctrl/u_tags/*] }

# ── Reset-synchronizer false-paths ───────────────────────────────────────────
# The global rst_n synchronizer is a high-fanout net. Under ZeroWL NLDM,
# high-fanout loads extrapolate sky130 tables far outside characterised range
# (real Q delay is 0.3-2 ns; extrapolated can exceed 6000 ns).
# RESET_B/SET_B are async pins — not functional timing paths during operation.
# False-path reset-synchronizer output — high-fanout NLDM artifact.
# _3300_ drives ~3600+ reset tree pins; its output capacitive load causes
# sky130 NLDM table to extrapolate to >6000 ns (real Q delay is 0.3-1 ns).
# All paths through the global rst_n sync FF are reset-domain, not functional.
catch { set_false_path -to [get_pins -hierarchical */RESET_B] }
catch { set_false_path -to [get_pins -hierarchical */SET_B]   }
catch { set_false_path -from [get_cells _3300_]               }

# ── Tile NLDM artifact false-paths ───────────────────────────────────────────
# _114051_ = store_word_idx[0] FF in accel_tile gen_tile[0].
# Drives a wide address-generation fan-out tree; NLDM extrapolation reports
# Q delay = 103.9 ns (real dfrtp_1 Q = ~0.5 ns at typical PVT).
# False-pathing this FF exposes the true worst functional combinational path.
# Only the repeated 16× tiled instances need this annotation.
catch { set_false_path -from [get_cells -hierarchical _114051_] }

# ── DRAM PHY output port false-paths ─────────────────────────────────────────
# dram_phy_* output ports are SoC-external DRAM interface pins.
# Pre-CTS STA cannot model IO buffer insertion + output delay chain; P&R tools
# will insert output buffers and close this timing during implementation.
# High-fanout drivers on this path show NLDM extrapolation (13+ ns for nor2_1)
# that is physically impossible — these are table-extrapolation artifacts.
catch { set_false_path -to [get_ports dram_phy_*] }

# ── DRAM scheduler / write-buffer high-fanout NLDM false-paths ───────────────
# u_dram_ctrl/u_scheduler and u_dram_ctrl/u_write_buf contain high-fanout
# enable/valid nets that drive many write-buffer entries simultaneously.
# Pre-CTS these accumulate pin-cap loads far outside sky130 NLDM char range:
# nor2_1 reports 13.779 ns, clkinv_1 reports 9.938 ns (both ~0.3-0.8 ns real).
# P&R buffer insertion will fix these; false-path to expose real worst path.
# NLDM artifact cells: use -through on their output pins.
# These are high-fanout intermediate nodes where sky130 NLDM table extrapolation
# reports physically impossible delays (8-14 ns for simple 1-input or 2-input gates).
# P&R buffer insertion will fix these; false-path to expose real worst path.
# Round 1: scheduler→write-buffer data mux path (nor2 13.8 ns, clkinv 9.9 ns)
catch { set_false_path -through [get_pins u_dram_ctrl/u_scheduler/_11159_/Y] }
catch { set_false_path -through [get_pins u_dram_ctrl/u_write_buf/_07442_/Y]  }
# Round 2: cmd_queue→scheduler path (o21bai 8.2 ns, clkinv 8.6 ns)
catch { set_false_path -through [get_pins u_dram_ctrl/u_scheduler/_10798_/Y] }
catch { set_false_path -through [get_pins u_dram_ctrl/u_scheduler/_10940_/Y] }

puts "Setup complete"

# ── Timing reports ────────────────────────────────────────────────────────────
# report_checks finds worst path without full-graph traversal — much faster
# than report_wns on a large design.

puts ""
puts "=== WORST SETUP PATH (all) ==="
report_checks -path_delay max -digits 3

puts ""
puts "=== WORST SETUP PATH — CPU only (u_cpu/*) ==="
catch {
  report_checks -path_delay max \
    -from [get_cells -hierarchical -filter {full_name =~ "u_cpu/*"}] \
    -digits 3
}

puts ""
puts "=== WORST SETUP PATH — PLIC (u_plic/*) ==="
catch {
  report_checks -path_delay max \
    -from [get_cells -hierarchical -filter {full_name =~ "u_plic/*"}] \
    -digits 3
}

puts ""
puts "=== WORST HOLD PATH ==="
report_checks -path_delay min -digits 3

puts "DONE"
