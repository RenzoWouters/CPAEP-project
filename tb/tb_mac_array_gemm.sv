
module tb_mac_array_gemm;

 timeunit 1ns;
 timeprecision 1ps;

 //---------------------------
 // Design-time parameters
 //---------------------------

 parameter int unsigned InDataWidth    = 8;
 parameter int unsigned OutDataWidth   = 32;
 parameter int unsigned NumCores       = 16;
 parameter int unsigned NumInputs      = 8;
 parameter int unsigned SramADataWidth = 512;
 parameter int unsigned SramBDataWidth = 480;
 parameter int unsigned DataDepthA     = 131072;
 parameter int unsigned DataDepthB     = 512;
 parameter int unsigned DataDepthC     = 4096;
 parameter int unsigned AddrWidth      = 17;
 parameter int unsigned SizeAddrWidth  = 12;
 parameter int unsigned RandomSeed     = 32'hCAFE_2027;

 localparam int unsigned AElements = SramADataWidth / InDataWidth;
 localparam int unsigned BElements = SramBDataWidth / InDataWidth;
 localparam int unsigned TileElements = NumCores * NumInputs;
 localparam int unsigned AReadWords = (TileElements + AElements - 1) / AElements;
 localparam int unsigned BReadWords = (TileElements + BElements - 1) / BElements;

 //---------------------------
 // Test cases: A[M x K] * B[K x N] = C[M x N]
 //---------------------------

 typedef struct {
   string       name;
   int unsigned M;
   int unsigned K;
   int unsigned N;
   int unsigned pattern;
 } test_case_t;

 // Only the first six cases contribute to the project latency score.
 localparam int unsigned NumWorkloads = 6;
 localparam int unsigned NumCases     = 37;

 typedef enum int unsigned {
   ResetTopStart,
   ResetDuringLoad,
   ResetTopFlush,
   ResetTopIssue,
   ResetControllerDrain,
   ResetDuringCWrite
 } reset_target_t;

 localparam int unsigned NumResetTargets = 6;
 localparam int unsigned NumResetCases   = 2 * NumResetTargets;

 // Pattern IDs: 0=random, 1=zero, 2=+1, 3=-1, 4=A:-1/B:+1,
 //              5=+127, 6=-128, 7=alternating A / B:+1.
 test_case_t test_cases [NumCases] = '{
   // name              M     K     N  pattern
   '{"GeMM 1",           64,   64, 32, 0},
   '{"GeMM 2",           64,  128, 32, 1},
   '{"GeMM 3",           64,  576, 32, 2},
   '{"GeMV 1",         1536,  576,  1, 3},
   '{"GeMV 2",         3072, 1024,  1, 4},
   '{"GeMV 3",         2816, 2048,  1, 5},

   // Existing integration tests
   '{"GeMM tails",        3,   65, 17, 6},
   '{"GeMV tails",       17,   33,  1, 7},
   '{"Cache A first",     1,    1, 32, 0},
   '{"Cache A repeat",    1,    1, 32, 1},
   '{"Single block",      1,    1,  1, 2},
   '{"GeMM 2x18",         2,   33, 18, 3},
   '{"GeMM 1x17",         1,   65, 17, 4},
   '{"GeMV 1x65",         1,   65,  1, 5},
   '{"GeMV 16x64",       16,   64,  1, 6},
   '{"GeMM alt",          4,   34, 18, 7},
   '{"GeMV alt",          5,   34,  1, 0},

   // K boundary tests
   '{"GeMM K31",          2,   31, 16, 1},
   '{"GeMM K32",          2,   32, 16, 2},
   '{"GeMM K60",          2,   60, 16, 3},
   '{"GeMM K61",          2,   61, 16, 4},

   // N boundary tests
   '{"GeMM N16",          2,   64, 16, 5},
   '{"GeMM N33",          2,   64, 33, 6},

   // M boundary tests
   '{"GeMV M15",         15,   64,  1, 7},
   '{"GeMV M32",         32,   64,  1, 0},
   '{"GeMV M33",         33,   64,  1, 1},

   // GeMV B-vector packing boundary
   '{"GeMV K60",         16,   60,  1, 2},

   // Maximum A SRAM usage while respecting 12-bit size inputs.
   // M=4095 is the largest representable dimension (M=4096 would wrap to zero).
   '{"GeMV max SRAM",  4095, 2048,  1, 3},

   // All six required workloads again, each with fresh random A and B data.
   // These are correctness regressions; only the first six rows are scored.
   '{"GeMM 1 random",    64,   64, 32, 0},
   '{"GeMM 2 random",    64,  128, 32, 0},
   '{"GeMM 3 random",    64,  576, 32, 0},
   '{"GeMV 1 random",  1536,  576,  1, 0},
   '{"GeMV 2 random",  3072, 1024,  1, 0},
   '{"GeMV 3 random",  2816, 2048,  1, 0},

   // Target the three functional-coverage gaps measured in iteration 5.
   '{"GeMM N15",          2,   64, 15, 0},
   '{"GeMV K32",         16,   32,  1, 0},
   '{"GeMV K61",         16,   61,  1, 0}
 };

 int unsigned     case_cycles [NumCases];
 real             case_energy [NumCases];
 real             case_sram_energy [NumCases];

 real             case_spatial_util [NumCases];
 real             case_temporal_util [NumCases];
 real             case_total_util [NumCases];

 int unsigned     case_errors [NumCases];

 longint unsigned case_a_reads [NumCases];
 longint unsigned case_b_reads [NumCases];
 longint unsigned case_c_writes [NumCases];

 //---------------------------
 // Wires
 //---------------------------

 logic clk_i, rst_ni, start, done;

 logic [SizeAddrWidth-1:0] M_i, K_i, N_i;

 logic [AddrWidth-1:0] sram_a_addr, sram_b_addr, sram_c_addr;

 logic sram_a_re, sram_b_re, sram_c_we;

 logic [SramADataWidth-1:0] sram_a_rdata;
 logic [SramBDataWidth-1:0] sram_b_rdata;
 logic signed [OutDataWidth-1:0] sram_c_wdata;

 logic signed [InDataWidth-1:0] A_ref[];
 logic signed [InDataWidth-1:0] B_ref[];
 logic signed [OutDataWidth-1:0] G_memory[];

 bit c_written[];

 string active_case_name;

 int unsigned a_used_words, b_used_words, c_used_words;

 longint unsigned a_reads, b_reads, c_writes;
 longint unsigned mac_compute_cycles;

 bit monitor_active, case_finished;

 bit expect_reset_quiet;
 int unsigned reset_cases_passed;

 int unsigned seed_state;
 int unsigned active_pattern;

 //----------------------------------------------------------------------------
 // Functional coverage
 //----------------------------------------------------------------------------

 covergroup gemm_coverage;

   cp_mode: coverpoint (N_i == 1) {
     bins GeMV = {1'b1};
     bins GeMM = {1'b0};
   }

   cp_K: coverpoint K_i {
     bins below_32 = {[1:31]};
     bins exact_32 = {32};
     bins below_60 = {[33:59]};
     bins exact_60 = {60};
     bins above_60 = {[61:63]};
     bins exact_64 = {64};
     bins above_64 = {[65:2048]};
   }

   cp_M: coverpoint M_i {
     bins below_16 = {[1:15]};
     bins exact_16 = {16};
     bins below_32 = {[17:31]};
     bins exact_32 = {32};
     bins above_32 = {[33:4095]};
   }

   cp_N: coverpoint N_i {
     bins single = {1};
     bins below_16 = {[2:15]};
     bins exact_16 = {16};
     bins below_32 = {[17:31]};
     bins exact_32 = {32};
     bins above_32 = {[33:4095]};
   }

   cp_pattern: coverpoint active_pattern {
     bins random_data = {0};
     bins zero_data = {1};
     bins positive_data = {2};
     bins negative_data = {3};
     bins opposite_signs = {4};
     bins maximum_value = {5};
     bins minimum_value = {6};
     bins alternating = {7};
   }

   mode_k_cross: cross cp_mode, cp_K;

 endgroup

 gemm_coverage coverage = new();

 //---------------------------
 // The supplied SRAM modules, with fixed widths/depths for ALL cases
 //---------------------------

 single_port_memory #(
   .DataWidth (SramADataWidth),
   .DataDepth (DataDepthA),
   .AddrWidth (AddrWidth)
 ) i_sram_a (
   .clk_i(clk_i), .rst_ni(rst_ni),
   .mem_addr_i(sram_a_addr), .mem_re_i(sram_a_re), .mem_we_i(1'b0),
   .mem_wr_data_i('0), .mem_rd_data_o(sram_a_rdata)
 );

 single_port_memory #(
   .DataWidth (SramBDataWidth),
   .DataDepth (DataDepthB),
   .AddrWidth (AddrWidth)
 ) i_sram_b (
   .clk_i(clk_i), .rst_ni(rst_ni),
   .mem_addr_i(sram_b_addr), .mem_re_i(sram_b_re), .mem_we_i(1'b0),
   .mem_wr_data_i('0), .mem_rd_data_o(sram_b_rdata)
 );

 single_port_memory #(
   .DataWidth (OutDataWidth),
   .DataDepth (DataDepthC),
   .AddrWidth (AddrWidth)
 ) i_sram_c (
   .clk_i(clk_i), .rst_ni(rst_ni),
   .mem_addr_i(sram_c_addr), .mem_re_i(1'b0), .mem_we_i(sram_c_we),
   .mem_wr_data_i(sram_c_wdata), .mem_rd_data_o()
 );

 //---------------------------
 // DUT instantiation
 //---------------------------

 gemm_accelerator_top_mac_array #(
   .InDataWidth(InDataWidth), .OutDataWidth(OutDataWidth),
   .AddrWidth(AddrWidth), .SizeAddrWidth(SizeAddrWidth),
   .NumCores(NumCores), .NumInputs(NumInputs),
   .SramADataWidth(SramADataWidth), .SramBDataWidth(SramBDataWidth)
 ) i_dut (
   .clk_i(clk_i), .rst_ni(rst_ni), .start_i(start),
   .M_size_i(M_i), .K_size_i(K_i), .N_size_i(N_i),
   .sram_a_addr_o(sram_a_addr), .sram_b_addr_o(sram_b_addr),
   .sram_c_addr_o(sram_c_addr),
   .sram_a_re_o(sram_a_re), .sram_b_re_o(sram_b_re),
   .sram_a_rdata_i(sram_a_rdata), .sram_b_rdata_i(sram_b_rdata),
   .sram_c_wdata_o(sram_c_wdata), .sram_c_we_o(sram_c_we),
   .done_o(done)
 );

 //---------------------------
 // Tasks and functions
 //---------------------------

 `include "includes/common_tasks.svh"

 function automatic string pattern_name(input int unsigned pattern);
   case (pattern)
     0: return "random int8 + explicit extremes";
     1: return "all zero";
     2: return "all +1";
     3: return "all -1";
     4: return "A=-1, B=+1";
     5: return "all +127";
     6: return "all -128";
     7: return "alternating A, B=+1";
     default: return "unknown";
   endcase
 endfunction

 task automatic initialize_memories(
   input int unsigned test_id,
   input int unsigned m_sz,
   input int unsigned k_sz,
   input int unsigned n_sz,
   input bit force_random = 1'b0
 );

   int unsigned k_blocks, a_words_per_row;
   int unsigned word_addr, lane, element;

   if (m_sz == 0 || k_sz == 0 || n_sz == 0 ||
       longint'(m_sz) >= (64'd1 << SizeAddrWidth) ||
       longint'(k_sz) >= (64'd1 << SizeAddrWidth) ||
       longint'(n_sz) >= (64'd1 << SizeAddrWidth))
     $fatal(1, "Case dimensions must be positive and fit SizeAddrWidth.");

   k_blocks = (k_sz + NumInputs - 1) / NumInputs;
   a_words_per_row = (k_sz + AElements - 1) / AElements;

   if (n_sz == 1) begin
     a_used_words = ((m_sz + NumCores - 1) / NumCores) * k_blocks * AReadWords;
     b_used_words = k_blocks;
   end else begin
     a_used_words = m_sz * a_words_per_row;
     b_used_words = ((n_sz + NumCores - 1) / NumCores) * k_blocks * BReadWords;
   end

   c_used_words = m_sz * n_sz;

   if (a_used_words > DataDepthA || b_used_words > DataDepthB || c_used_words > DataDepthC)
     $fatal(1, "Case does not fit SRAMs: A=%0d, B=%0d, C=%0d words.",
               a_used_words, b_used_words, c_used_words);

   if (longint'(a_used_words) > (64'd1 << AddrWidth) ||
       longint'(b_used_words) > (64'd1 << AddrWidth) ||
       longint'(c_used_words) > (64'd1 << AddrWidth))
     $fatal(1, "Used SRAM word addresses do not fit AddrWidth.");

   A_ref = new[m_sz*k_sz];
   B_ref = new[k_sz*n_sz];
   G_memory = new[c_used_words];
   c_written = new[c_used_words];

   a_reads = 0;
   b_reads = 0;
   c_writes = 0;
   mac_compute_cycles = 0;

   active_pattern = force_random ? 0 : test_cases[test_id].pattern;

   foreach (A_ref[i]) begin
     case (active_pattern)
       0: A_ref[i] = InDataWidth'($urandom());
       1: A_ref[i] = 0;
       2: A_ref[i] = 1;
       3: A_ref[i] = -1;
       4: A_ref[i] = -1;
       5: A_ref[i] = 127;
       6: A_ref[i] = -128;
       7: A_ref[i] = ((i % 2) == 0) ? 1 : -1;
       default: A_ref[i] = 0;
     endcase
   end

   foreach (B_ref[i]) begin
     case (active_pattern)
       0: B_ref[i] = InDataWidth'($urandom());
       1: B_ref[i] = 0;
       2: B_ref[i] = 1;
       3: B_ref[i] = -1;
       4: B_ref[i] = 1;
       5: B_ref[i] = 127;
       6: B_ref[i] = -128;
       7: B_ref[i] = 1;
       default: B_ref[i] = 0;
     endcase
   end

   if (test_id == 8 || test_id == 9) begin
     A_ref[0] = (test_id % 2 == 0) ? 8'sh80 : 8'sd127;
     B_ref[0] = (test_id % 2 == 0) ? 8'sd127 : 8'sh80;
   end

   if (active_pattern == 0) begin
     if ((m_sz * k_sz) > 1)
       A_ref[m_sz*k_sz - 1] = (test_id % 2 == 0) ? 8'sd127 : 8'sh80;
     if ((k_sz * n_sz) > 1)
       B_ref[k_sz*n_sz - 1] = (test_id % 2 == 0) ? 8'sh80 : 8'sd127;
   end

   for (int unsigned w = 0; w < a_used_words; w++) i_sram_a.memory[w] = '0;
   for (int unsigned w = 0; w < b_used_words; w++) i_sram_b.memory[w] = '0;
   for (int unsigned w = 0; w < DataDepthC; w++) i_sram_c.memory[w] = 'x;

   // Pack A by placing each logical matrix element into its word and lane.
   for (int unsigned m = 0; m < m_sz; m++) begin
     for (int unsigned k = 0; k < k_sz; k++) begin
       if (n_sz == 1) begin
         element = (m % NumCores) * NumInputs + (k % NumInputs);
         word_addr = ((m / NumCores) * k_blocks + k / NumInputs) * AReadWords +
                     element / AElements;
         lane = element % AElements;
       end else begin
         word_addr = m * a_words_per_row + k / AElements;
         lane = k % AElements;
       end
       i_sram_a.memory[word_addr][lane*InDataWidth +: InDataWidth] = A_ref[m*k_sz+k];
     end
   end

   // Pack B: column fragments for GeMM, a shared vector fragment for GeMV.
   for (int unsigned k = 0; k < k_sz; k++) begin
     for (int unsigned n = 0; n < n_sz; n++) begin
       if (n_sz == 1) begin
         word_addr = k / NumInputs;
         lane = k % NumInputs;
       end else begin
         element = (n % NumCores) * NumInputs + (k % NumInputs);
         word_addr = ((n / NumCores) * k_blocks + k / NumInputs) * BReadWords +
                     element / BElements;
         lane = element % BElements;
       end
       i_sram_b.memory[word_addr][lane*InDataWidth +: InDataWidth] = B_ref[k*n_sz+n];
     end
   end

 endtask

 task automatic gemm_golden(
   input int unsigned m_sz,
   input int unsigned k_sz,
   input int unsigned n_sz
 );

   longint signed acc;

   for (int unsigned m = 0; m < m_sz; m++) begin
     for (int unsigned n = 0; n < n_sz; n++) begin
       acc = 0;
       for (int unsigned k = 0; k < k_sz; k++) begin
         acc += longint'($signed(A_ref[m*k_sz+k])) *
                longint'($signed(B_ref[k*n_sz+n]));
       end

       if (acc > 64'sd2147483647 || acc < -64'sd2147483648)
         $fatal(1, "Golden result exceeds int32 for C[%0d,%0d].", m, n);

       G_memory[m*n_sz+n] = OutDataWidth'(acc);
     end
   end

 endtask

 function automatic int unsigned timeout_cycles(
   input int unsigned m_sz,
   input int unsigned k_sz,
   input int unsigned n_sz
 );

   int unsigned blocks, k_blocks, read_words;

   k_blocks = (k_sz + NumInputs - 1) / NumInputs;

   if (n_sz == 1) begin
     blocks = ((m_sz + NumCores - 1) / NumCores) * k_blocks;
     read_words = AReadWords;
   end else begin
     blocks = m_sz * ((n_sz + NumCores - 1) / NumCores) * k_blocks;
     read_words = BReadWords;
   end

   return 100 + blocks * (read_words + NumCores + 8);

 endfunction

 task automatic start_and_wait_gemm(
   output int unsigned elapsed_cycles,
   output real measured_energy,
   input int unsigned max_cycles
 );

   @(negedge clk_i);

   if (done !== 1'b0) $fatal(1, "done must be low before a new start.");

   metrics_pkg::energy = 0;

   monitor_active = 1'b1;
   case_finished = 1'b0;
   start = 1'b1;

   @(posedge clk_i);
   #1;

   elapsed_cycles = 0;

   @(negedge clk_i);
   start = 1'b0;

   while (done !== 1'b1) begin
     @(posedge clk_i);
     #1;

     elapsed_cycles++;

     if ($isunknown(done)) $fatal(1, "done contains X/Z.");

     if (elapsed_cycles > max_cycles)
       $fatal(1, "[%0s] Timeout after %0d cycles.", active_case_name, elapsed_cycles);
   end

   measured_energy = metrics_pkg::energy;
   case_finished = 1'b1;

 endtask

 task automatic verify_result_c(output int unsigned num_errors);

   num_errors = 0;

   for (int unsigned addr = 0; addr < c_used_words; addr++) begin
     if (!c_written[addr]) begin
       if (num_errors < 10) $display("ERROR: C[%0d] was not written.", addr);
       num_errors++;
     end else if (G_memory[addr] !== i_sram_c.memory[addr]) begin
       if (num_errors < 10)
         $display("ERROR: C[%0d]: expected %0d (0x%08h), got %0d (0x%08h).",
                  addr, $signed(G_memory[addr]), G_memory[addr],
                  $signed(i_sram_c.memory[addr]), i_sram_c.memory[addr]);
       num_errors++;
     end
   end

   if (c_writes != c_used_words) begin
     $display("ERROR: C write count=%0d, expected=%0d.", c_writes, c_used_words);
     num_errors++;
   end

 endtask

 //---------------------------
 // Directed reset tests
 //---------------------------

 function automatic string reset_target_name(input reset_target_t target);
   case (target)
     ResetTopStart:        return "TopStart";
     ResetDuringLoad:      return "mid-load after first K block";
     ResetTopFlush:        return "TopFlush";
     ResetTopIssue:        return "TopIssue of second K block";
     ResetControllerDrain: return "ControllerDrain";
     ResetDuringCWrite:    return "partial C write in TopDrain";
     default:              return "unknown reset target";
   endcase
 endfunction

 function automatic bit reset_target_reached(input reset_target_t target);
   case (target)
     ResetTopStart:
       return (i_dut.state_q === i_dut.TopStart);

     ResetDuringLoad:
       return (i_dut.state_q === i_dut.TopLoad) &&
              (i_dut.K_count == NumInputs) &&
              (i_dut.read_beat_q != '0) && (mac_compute_cycles > 0);

     ResetTopFlush:
       return (i_dut.state_q === i_dut.TopFlush);

     ResetTopIssue:
       return (i_dut.state_q === i_dut.TopIssue) &&
              (i_dut.K_count == NumInputs) && (mac_compute_cycles > 0);

     ResetControllerDrain:
       return (i_dut.state_q === i_dut.TopDrain) &&
              (i_dut.i_gemm_controller_mac_array.current_state ===
               i_dut.i_gemm_controller_mac_array.ControllerDrain);

     ResetDuringCWrite:
       return (i_dut.state_q === i_dut.TopDrain) &&
              (sram_c_we === 1'b1) && (i_dut.c_core_q != '0) &&
              (c_writes > 0) && (c_writes < c_used_words);

     default: return 1'b0;
   endcase
 endfunction

 task automatic check_reset_idle(input string phase_label);

   if ({sram_a_re, sram_b_re, sram_c_we, done} !== 4'b0000)
     $fatal(1, "[%0s] Nonzero/unknown memory control or done during %0s.",
            active_case_name, phase_label);

   if ((i_dut.state_q !== i_dut.TopIdle) ||
       (i_dut.i_gemm_controller_mac_array.current_state !==
        i_dut.i_gemm_controller_mac_array.ControllerIdle))
     $fatal(1, "[%0s] FSM did not return to idle during %0s.",
            active_case_name, phase_label);

   if ({i_dut.valid_data, i_dut.init_save, i_dut.result_valid,
        i_dut.controller_busy, i_dut.controller_done,
        i_dut.controller_done_seen_q,
        i_dut.a_response_valid_q, i_dut.b_response_valid_q,
        i_dut.a_cache_valid_q, i_dut.output_slot_busy_q,
        i_dut.c_valid_q} !== '0)
     $fatal(1, "[%0s] Stale valid/busy/cache state during %0s.",
            active_case_name, phase_label);

   if ({i_dut.M_count, i_dut.K_count, i_dut.N_count} !== '0)
     $fatal(1, "[%0s] Block counters not cleared during %0s.",
            active_case_name, phase_label);

   if (i_dut.array_c_out !== '0)
     $fatal(1, "[%0s] MAC accumulators not cleared during %0s.",
            active_case_name, phase_label);

 endtask

 task automatic run_reset_case(
   input reset_target_t target,
   input bit abort_is_gemv
 );

   int unsigned abort_m, abort_k, abort_n;
   int unsigned recovery_m, recovery_k, recovery_n;
   int unsigned waited_cycles, max_cycles;
   int unsigned recovery_cycles, recovery_errors;
   real recovery_energy;
   string reset_label;
   logic signed [OutDataWidth-1:0] c_snapshot[];

   reset_label = $sformatf("%0s / %0s",
                         abort_is_gemv ? "GeMV" : "GeMM",
                         reset_target_name(target));
   active_case_name = {reset_label, " abort"};

   abort_m = abort_is_gemv ? (2 * NumCores) : 2;
   abort_k = 2 * NumInputs;
   abort_n = abort_is_gemv ? 1 : (2 * NumCores);

   M_i = SizeAddrWidth'(abort_m);
   K_i = SizeAddrWidth'(abort_k);
   N_i = SizeAddrWidth'(abort_n);

   initialize_memories(0, abort_m, abort_k, abort_n, 1'b1);
   max_cycles = timeout_cycles(abort_m, abort_k, abort_n);
   waited_cycles = 0;

   $display("\n[RESET %0s] Start abort job M=%0d K=%0d N=%0d.",
            reset_label, abort_m, abort_k, abort_n);

   @(negedge clk_i);

   if (done !== 1'b0) $fatal(1, "[%0s] done high before abort job.", active_case_name);

   monitor_active = 1'b1;
   case_finished = 1'b0;
   start = 1'b1;

   @(negedge clk_i);
   start = 1'b0;

   while (!reset_target_reached(target)) begin
     if (done !== 1'b0)
       $fatal(1, "[%0s] Job completed before reset target was reached.", active_case_name);

     if (waited_cycles >= max_cycles)
       $fatal(1, "[%0s] Timeout waiting for reset target.", active_case_name);

     @(negedge clk_i);
     waited_cycles++;
   end

   if (done !== 1'b0)
     $fatal(1, "[%0s] Reset target was reached after job completion.", active_case_name);

   c_snapshot = new[DataDepthC];

   for (int unsigned w = 0; w < DataDepthC; w++)
     c_snapshot[w] = i_sram_c.memory[w];

   $display("[RESET %0s] Assert reset after %0d compute cycles and %0d C writes.",
            reset_label, mac_compute_cycles, c_writes);

   rst_ni = 1'b0;
   monitor_active = 1'b0;
   case_finished = 1'b0;

   #1;

   expect_reset_quiet = 1'b1;
   check_reset_idle("reset assertion");

   repeat (2) begin
     @(posedge clk_i);
     #1;
     check_reset_idle("reset hold");
   end

   @(negedge clk_i);
   rst_ni = 1'b1;

   #1;
   check_reset_idle("reset release");

   repeat (NumCores + 4) begin
     @(posedge clk_i);
     #1;
     check_reset_idle("idle after reset");
   end

   for (int unsigned w = 0; w < DataDepthC; w++) begin
     if (i_sram_c.memory[w] !== c_snapshot[w])
       $fatal(1, "[%0s] C[%0d] changed during reset/idle: before=%h after=%h.",
              active_case_name, w, c_snapshot[w], i_sram_c.memory[w]);
   end

   recovery_m = abort_is_gemv ? 3 : (NumCores + 1);
   recovery_k = NumInputs + 1;
   recovery_n = abort_is_gemv ? (NumCores + 1) : 1;

   active_case_name = {reset_label, " recovery"};

   M_i = SizeAddrWidth'(recovery_m);
   K_i = SizeAddrWidth'(recovery_k);
   N_i = SizeAddrWidth'(recovery_n);

   initialize_memories(0, recovery_m, recovery_k, recovery_n, 1'b1);
   gemm_golden(recovery_m, recovery_k, recovery_n);

   expect_reset_quiet = 1'b0;

   start_and_wait_gemm(recovery_cycles, recovery_energy,
                       timeout_cycles(recovery_m, recovery_k, recovery_n));

   verify_result_c(recovery_errors);

   if (recovery_errors != 0)
     $fatal(1, "[%0s] Recovery result failed with %0d errors.",
            active_case_name, recovery_errors);

   repeat (2) begin
     clk_delay(1);
     if (done !== 1'b0)
       $fatal(1, "[%0s] Recovery done must be a one-cycle pulse.", active_case_name);
   end

   monitor_active = 1'b0;
   case_finished = 1'b0;

   reset_cases_passed++;

   $display("[RESET %0s] PASS: C preserved and recovery M=%0d K=%0d N=%0d correct (%0d cycles).",
            reset_label, recovery_m, recovery_k, recovery_n, recovery_cycles);

 endtask

 task automatic run_reset_tests();

   if (NumCores < 2 || AReadWords < 2 || BReadWords < 2)
     $fatal(1, "Reset scenarios require at least two cores and two read beats per A/B tile.");

   $display("\n==========================================");
   $display(" PHASE 2: RESET / ABORT / RESTART TESTS");
   $display("==========================================");

   reset_cases_passed = 0;

   for (int unsigned mode = 0; mode < 2; mode++) begin
     for (int unsigned r = 0; r < NumResetTargets; r++) begin
       run_reset_case(reset_target_t'(r), mode == 1);
     end
   end

 endtask

 //---------------------------
 // Monitor physical SRAM transactions and MAC utilization at rising clock edges.
 //---------------------------

 always @(posedge clk_i) begin

   if (expect_reset_quiet &&
       ({sram_a_re, sram_b_re, sram_c_we, done} !== 4'b0000))
     $fatal(1, "[%0s] Unexpected transaction/done at a reset or idle clock edge.",
            active_case_name);

   if (rst_ni) begin

     if ($isunknown({sram_a_re, sram_b_re, sram_c_we, done}))
       $fatal(1, "Unknown SRAM control or done signal.");

     if (!monitor_active) begin

       if (sram_a_re || sram_b_re || sram_c_we)
         $fatal(1, "Unexpected SRAM transaction outside an active case.");

     end else begin

       if (case_finished && (sram_a_re || sram_b_re || sram_c_we))
         $fatal(1, "[%0s] SRAM transaction after done.", active_case_name);

       if (!case_finished && i_dut.valid_data)
         mac_compute_cycles++;

       if (sram_a_re) begin

         if ($isunknown(sram_a_addr) || sram_a_addr >= a_used_words)
           $fatal(1, "[%0s] Invalid A word address %0h (used words=%0d).",
                     active_case_name, sram_a_addr, a_used_words);

         a_reads++;

       end

       if (sram_b_re) begin

         if ($isunknown(sram_b_addr) || sram_b_addr >= b_used_words)
           $fatal(1, "[%0s] Invalid B word address %0h (used words=%0d).",
                     active_case_name, sram_b_addr, b_used_words);

         b_reads++;

       end

       if (sram_c_we) begin

         if ($isunknown(sram_c_addr) || sram_c_addr >= c_used_words)
           $fatal(1, "[%0s] Invalid C word address %0h (used words=%0d).",
                     active_case_name, sram_c_addr, c_used_words);

         if ($isunknown(sram_c_wdata))
           $fatal(1, "[%0s] Unknown C write data at address %0d.",
                     active_case_name, sram_c_addr);

         if (c_written[sram_c_addr])
           $fatal(1, "[%0s] Duplicate write to C[%0d].", active_case_name, sram_c_addr);

         c_written[sram_c_addr] = 1'b1;
         c_writes++;

       end

     end

   end

 end

 //---------------------------
 // Clock and sequence driver
 //---------------------------

 initial begin
   clk_i = 1'b0;
   forever #5 clk_i = ~clk_i;
 end

 initial begin

   start = 1'b0;
   rst_ni = 1'b0;

   M_i = '0;
   K_i = '0;
   N_i = '0;

   monitor_active = 1'b0;
   case_finished = 1'b0;
   active_case_name = "startup";

   expect_reset_quiet = 1'b0;
   reset_cases_passed = 0;

   a_used_words = 0;
   b_used_words = 0;
   c_used_words = 0;

   a_reads = 0;
   b_reads = 0;
   c_writes = 0;

   mac_compute_cycles = 0;

   seed_state = RandomSeed;
   void'($urandom(seed_state));

   // The only reset in phase 1. Release it away from a rising edge.
   repeat (5) @(negedge clk_i);
   rst_ni = 1'b1;

   clk_delay(1);

   $display("Random seed: 0x%08h", RandomSeed);
   $display("PHASE 1: %0d matrix cases without an intermediate DUT reset.", NumCases);

   foreach (test_cases[c]) begin

     active_case_name = test_cases[c].name;

     M_i = SizeAddrWidth'(test_cases[c].M);
     K_i = SizeAddrWidth'(test_cases[c].K);
     N_i = SizeAddrWidth'(test_cases[c].N);

     $display("\n[%0s] A (%0d x %0d) * B (%0d x %0d) -> C (%0d x %0d)",
              test_cases[c].name, test_cases[c].M, test_cases[c].K,
              test_cases[c].K, test_cases[c].N,
              test_cases[c].M, test_cases[c].N);

     initialize_memories(c, test_cases[c].M, test_cases[c].K, test_cases[c].N);

     $display("  Data pattern: %0s (id=%0d)", pattern_name(active_pattern), active_pattern);

     if (c == 8 || c == 9)
       $display("  Cache check: A[0] and B[0] replaced with signed extremes.");

     coverage.sample();

     gemm_golden(test_cases[c].M, test_cases[c].K, test_cases[c].N);

     clk_delay(1);

     start_and_wait_gemm(case_cycles[c], case_energy[c],
                        timeout_cycles(test_cases[c].M, test_cases[c].K, test_cases[c].N));

     case_a_reads[c] = a_reads;
     case_b_reads[c] = b_reads;
     case_c_writes[c] = c_writes;

     case_sram_energy[c] = 2.0 * (real'(a_reads) * SramADataWidth +
                                  real'(b_reads) * SramBDataWidth +
                                  real'(c_writes) * OutDataWidth);

     verify_result_c(case_errors[c]);

     begin

       real peak_macs_per_cycle;
       real useful_mac_operations;
       real compute_mac_slots;
       real total_mac_slots;
       real spatial_util_pct;
       real temporal_util_pct;
       real total_util_pct;

       peak_macs_per_cycle = real'(NumCores) * real'(NumInputs);

       useful_mac_operations = real'(test_cases[c].M) *
                               real'(test_cases[c].K) *
                               real'(test_cases[c].N);

       compute_mac_slots = real'(mac_compute_cycles) * peak_macs_per_cycle;
       total_mac_slots = real'(case_cycles[c]) * peak_macs_per_cycle;

       spatial_util_pct = (compute_mac_slots > 0.0) ?
                          (100.0 * useful_mac_operations) / compute_mac_slots : 0.0;

       temporal_util_pct = (case_cycles[c] > 0) ?
                           (100.0 * real'(mac_compute_cycles)) / real'(case_cycles[c]) : 0.0;

       total_util_pct = (total_mac_slots > 0.0) ?
                        (100.0 * useful_mac_operations) / total_mac_slots : 0.0;

       case_spatial_util[c] = spatial_util_pct;
       case_temporal_util[c] = temporal_util_pct;
       case_total_util[c] = total_util_pct;

       $display("[%0s] %0s: %0d cycles, %0.2f logged energy units.",
                test_cases[c].name, (case_errors[c] == 0) ? "PASS" : "FAIL",
                case_cycles[c], case_energy[c]);

       $display("  SRAM words: A=%0d, B=%0d, C=%0d; SRAM energy=%0.0f.",
                a_reads, b_reads, c_writes, case_sram_energy[c]);

       $display("  Spatial utilization:  %0.2f%% (useful MACs=%0.0f / compute MAC slots=%0.0f).",
                spatial_util_pct, useful_mac_operations, compute_mac_slots);

       $display("  Temporal utilization: %0.2f%% (%0d valid compute cycles / %0d total cycles).",
                temporal_util_pct, mac_compute_cycles, case_cycles[c]);

       $display("  Total utilization:    %0.2f%% (useful MACs=%0.0f / total MAC slots=%0.0f).",
                total_util_pct, useful_mac_operations, total_mac_slots);

     end

     repeat (2) begin
       clk_delay(1);
       if (done !== 1'b0) $fatal(1, "done must be a one-cycle pulse.");
     end

     monitor_active = 1'b0;
     case_finished = 1'b0;

     clk_delay(8);

   end

   run_reset_tests();

   print_report();

   if (reset_cases_passed != NumResetCases)
     $fatal(1, "Only %0d of %0d reset/restart tests completed.",
            reset_cases_passed, NumResetCases);

   $display("Reset/restart tests: %0d/%0d PASS.", reset_cases_passed, NumResetCases);

   $display("\n==========================================");
   $display(" FUNCTIONAL COVERAGE");
   $display("==========================================");

   $display("Covergroup total: %0.2f%%", coverage.get_coverage());
   $display("Mode coverage:    %0.2f%%", coverage.cp_mode.get_coverage());
   $display("K coverage:       %0.2f%%", coverage.cp_K.get_coverage());
   $display("M coverage:       %0.2f%%", coverage.cp_M.get_coverage());
   $display("N coverage:       %0.2f%%", coverage.cp_N.get_coverage());
   $display("Pattern coverage: %0.2f%%", coverage.cp_pattern.get_coverage());
   $display("Mode x K cross:   %0.2f%%", coverage.mode_k_cross.get_coverage());

   $display("==========================================");

   $display("All %0d matrix cases and %0d reset/restart tests passed!",
            NumCases, reset_cases_passed);

   $finish;

 end

 //---------------------------
 // Final report
 //---------------------------

 function automatic string pad(string str, int unsigned width);

   while (str.len() < width) str = {str, " "};

   return str;

 endfunction

 task automatic print_report();

   int unsigned num_failed;
   int unsigned workload_failures;

   real normalized, normalized_sum;
   real spatial_util_sum, temporal_util_sum, total_util_sum;

   num_failed = 0;
   workload_failures = 0;

   normalized_sum = 0.0;

   spatial_util_sum = 0.0;
   temporal_util_sum = 0.0;
   total_util_sum = 0.0;

   $display("\n===========================================================================================================================================");
   $display(" RESULTS -- first six rows are the required workloads; remaining rows are extra tests");
   $display("-------------------------------------------------------------------------------------------------------------------------------------------");

   $display(" %s | %4s %4s %3s | %8s | %12s | %8s | %9s | %8s | %12s | %12s | %s",
            pad("Case", 14), "M", "K", "N", "Cycles", "Cycles/MKN",
            "Spatial", "Temporal", "Total", "Energy", "SRAM energy", "Check");

   foreach (test_cases[c]) begin

     normalized = real'(case_cycles[c]) /
                  (real'(test_cases[c].M) * real'(test_cases[c].K) * real'(test_cases[c].N));

     $display(" %s | %4d %4d %3d | %8d | %12.9f | %7.2f%% | %8.2f%% | %7.2f%% | %12.2f | %12.0f | %s",
              pad(test_cases[c].name, 14),
              test_cases[c].M, test_cases[c].K, test_cases[c].N,
              case_cycles[c], normalized,
              case_spatial_util[c], case_temporal_util[c], case_total_util[c],
              case_energy[c], case_sram_energy[c],
              (case_errors[c] == 0) ? "PASS" : "FAIL");

     if (case_errors[c] != 0) num_failed++;

     if (c < NumWorkloads) begin

       normalized_sum += normalized;

       spatial_util_sum += case_spatial_util[c];
       temporal_util_sum += case_temporal_util[c];
       total_util_sum += case_total_util[c];

       if (case_errors[c] != 0) workload_failures++;

     end

   end

   $display("-------------------------------------------------------------------------------------------------------------------------------------------");

   if (workload_failures == 0) begin

     $display(" Mean normalized latency (ONLY six required workloads): %.9f cycles/MAC",
              normalized_sum / real'(NumWorkloads));

     $display(" Mean spatial utilization  (ONLY six required workloads): %0.2f%%",
              spatial_util_sum / real'(NumWorkloads));

     $display(" Mean temporal utilization (ONLY six required workloads): %0.2f%%",
              temporal_util_sum / real'(NumWorkloads));

     $display(" Mean total utilization    (ONLY six required workloads): %0.2f%%",
              total_util_sum / real'(NumWorkloads));

   end else begin

     $display(" Mean normalized latency: INVALID -- required workloads failed.");

   end

   $display(" Logged area: %0.2f area units", metrics_pkg::area);
   $display(" Energy/area totals depend on the metric hooks implemented in the RTL modules.");
   $display("===========================================================================================================================================\n");

   if (num_failed != 0) $fatal(1, "%0d of %0d cases FAILED.", num_failed, NumCases);

   $display("All %0d matrix cases passed!", NumCases);

 endtask

endmodule
