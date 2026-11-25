//-----------------------------------------------------------------------------
// Integration regression for the 8 x128-bit banked 512-MAC accelerator.
// All SRAM instances retain the supplied synchronous single-port behavior.
// Each physical read OR write consumes its complete 128-bit bank word.
// Phase 1: all matrix jobs, including all six random-data reruns, without reset.
// Phase 2: directed abort/reset/restart tests on the SAME DUT and memories.
// Phase 3: SRAM checks, including an explicit protocol-negative test.
// Final report: only the six required workloads and the complete test status.
// Logical row-major golden matrices are independent of the banked layout.
//-----------------------------------------------------------------------------
module tb_mac_array_gemm;
  timeunit 1ns;
  timeprecision 1ps;

  parameter int unsigned InDataWidth = 8;
  parameter int unsigned OutDataWidth = 32;
  parameter int unsigned NumCores = 16;
  parameter int unsigned NumInputs = 32;
  parameter int unsigned GemmRows = 2;
  parameter int unsigned MReuse = 64;
  parameter int unsigned OutputQueueDepth = 20;
  parameter int unsigned NumBanks = 8;
  parameter int unsigned BankDataWidth = 128;
  parameter int unsigned GemvFragmentElements = 128;
  // Eight banks preserve exactly the old aggregate SRAM storage capacity:
  // 131072*104 +4096*8 +4096*16 =8*107264*16 bytes.
  parameter int unsigned DataDepth = 107264;
  parameter int unsigned AddrWidth = 17;
  parameter int unsigned SizeAddrWidth = 12;
  parameter int unsigned RandomSeed = 32'hCAFE_2027;

  localparam int unsigned BankBytes = BankDataWidth / InDataWidth;
  localparam int unsigned BankResults = BankDataWidth / OutDataWidth;
  localparam int unsigned BusBytes = NumBanks * BankBytes;
  localparam int unsigned GemmCols = NumCores / GemmRows;
  localparam int unsigned NumContexts = MReuse / GemmRows;
  localparam int unsigned GemvReuseRows = NumContexts * NumCores;
  localparam int unsigned TileElements = NumCores * NumInputs;
  localparam int unsigned AReadWords = TileElements / BusBytes;
  localparam int unsigned BReadWords = (GemmCols*NumInputs) / (2*BankBytes);
  // Retain the dimensions of every previous test case. These two constants
  // describe the OLD boundary tests, not the new physical memory widths.
  localparam int unsigned CElements = 4;
  localparam int unsigned BElements = 8;
  localparam int unsigned GemmCElements = 2*BankResults;
  localparam int unsigned NumWorkloads = 6;
  localparam int unsigned NumCases = 81;

  typedef struct {
    string name;
    int unsigned M, K, N, pattern;
  } test_case_t;
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

   // Large GeMV memory case at the maximum 12-bit M dimension.
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
   '{"GeMM N15",          2,   64, 15, 0}, // cp_N.below_16
   '{"GeMV K32",         16,   32,  1, 0}, // GeMV x cp_K.exact_32
   '{"GeMV K61",         16,   61,  1, 0}, // GeMV x cp_K.above_60

   // M-row reuse and FIFO boundaries. All use fresh random operands.
   '{"Reuse below", (MReuse > 1) ? MReuse-1 : 1,
                      NumInputs+1, NumCores+1, 0},
   '{"Reuse exact", MReuse, NumInputs, 2*NumCores, 0},
   '{"Reuse above", MReuse+1, 2*NumInputs, NumCores, 0},
   '{"Reuse multi tails", 2*MReuse+1, 2*NumInputs+1, 2*NumCores+1, 0},

   // Every K block finishes an output group; the final M group is partial.
   // This exercises successive initialized results with the merged GeMV ISSUE.
   '{"GeMV consecutive", 2*NumCores+1, NumInputs, 1, 0},

   // Pair and N-tile boundaries, including one inactive M row in a pair.
   '{"GeMM pair below", GemmRows-1, NumInputs-1, GemmCols-1, 0},
   '{"GeMM pair exact", GemmRows, NumInputs, GemmCols, 0},
   '{"GeMM pair above", GemmRows+1, NumInputs+1, GemmCols+1, 0},

   // C word boundaries in both rows of one result group.
   '{"GeMM C below", GemmRows+1, NumInputs-1, CElements-1, 0},
   '{"GeMM C exact", GemmRows+1, NumInputs, CElements, 0},
   '{"GeMM C above", GemmRows+1, NumInputs+1, CElements+1, 0},

   // Multiple B64 responses assemble one shared GeMV vector fragment.
   '{"GeMV B below", NumCores+1, BElements-1, 1, 0},
   '{"GeMV B exact", NumCores+1, BElements, 1, 0},
   '{"GeMV B above", NumCores+1, BElements+1, 1, 0},

   // Nonzero input padding makes missing operand masks visible numerically.
   '{"GeMM poison tails", GemmRows+1, NumInputs+1, GemmCols+1, 0},
   '{"GeMV poison tails", NumCores+1, NumInputs+1, 1, 0},

   // With A832, 13 x512-byte tiles occupy exactly 64 x104-byte words.
   // Exercise the short read interval, its following tile, and end-of-job
   // handling immediately before/on/after that physical alignment boundary.
   '{"GeMV stream 12", NumCores, 12*NumInputs, 1, 0},
   '{"GeMV stream 13", NumCores, 13*NumInputs, 1, 0},
   '{"GeMV stream 14", NumCores, 14*NumInputs, 1, 0},

   // Same alignment period across M groups; every tile produces a C group.
   '{"GeMV stream M", 14*NumCores, NumInputs, 1, 0},

   // Carry must preserve stream order across both K/M boundaries. Nonzero
   // padding checks that partial K lanes and partial M rows remain masked.
   '{"GeMV stream tails", 2*NumCores+1, 5*NumInputs-1, 1, 0},

   // Change the operands at identical SRAM addresses without a DUT reset.
   // Each job ends with an incomplete physical word, exposing stale carry.
   '{"GeMV carry first", NumCores+1, 7*NumInputs-1, 1, 0},
   '{"GeMV carry repeat", NumCores+1, 7*NumInputs-1, 1, 1},
   '{"GeMV carry changed", NumCores+1, 7*NumInputs-1, 1, 0},

   // Bank/fragment boundaries and the old 2048-byte-cache boundary.
   // Large K now follows the same fragment schedule as every other GeMV.
   '{"GeMV cache K127", 33, 127, 1, 0},
   '{"GeMV cache K128", 33, 128, 1, 0},
   '{"GeMV cache K129", 33, 129, 1, 0},
   '{"GeMV cache first", 33, 129, 1, 0},
   '{"GeMV cache changed", 33, 129, 1, 5},
   '{"GeMV cache limit", 17, 2048, 1, 0},
   '{"GeMV fallback", 17, 2049, 1, 0},
   '{"GeMV fallback repeat", 17, 2049, 1, 6},

   // GeMV uses the existing accumulator contexts for groups of 512 M rows.
   // Cross the context-group and 128-byte fragment boundaries independently.
   '{"GeMV ctx below", GemvReuseRows-1, GemvFragmentElements+1, 1, 0},
   '{"GeMV ctx exact", GemvReuseRows, GemvFragmentElements+1, 1, 0},
   '{"GeMV ctx above", GemvReuseRows+1, GemvFragmentElements+1, 1, 0},
   '{"GeMV ctx cache", GemvReuseRows+1, GemvFragmentElements, 1, 0},
   '{"GeMV ctx multi", 2*GemvReuseRows+1, NumInputs+1, 1, 0},
   '{"GeMV ctx changed", GemvReuseRows+1, GemvFragmentElements+1, 1, 7},

   // Directed coverage cases from iteration 15. Append them so the previous
   // cases, random-data sequence and six-workload performance score stay intact.
   // Complete the full job: a short final M group can wait for B while C is full.
   '{"GeMM Cfull Bwait", 65, 32, 128, 0},
   // High N bits and C addresses, with seven valid columns in the final N tile.
   '{"GeMM high N Caddr", 129, 1, 4095, 0},
   // ceil(N/8)=511 complements the 512-word C-row stride of the previous case.
   '{"GeMM Cstride511", 2, 1, 4088, 0},

   // Remaining dimension/address gaps after iteration 16. Keep all earlier
   // cases in their original order; these jobs also use fresh random operands.
   // Cross A-address bit 16 and high K bits; the last K block has 31 lanes.
   '{"GeMM high K Aaddr", 1025, 4095, 2, 0},
   // Exercise the high M bits in the GeMM producers, including the final tail.
   '{"GeMM high M", 4095, 1, 7, 0},
   // 4032/64=63 M groups sets complementary ceiling bits; N has six columns.
   '{"GeMM Mgroups63", 4032, 1, 6, 0}
 };

  int unsigned case_cycles[NumCases], case_errors[NumCases];
  real case_energy[NumCases], case_sram_energy[NumCases];
  real case_spatial_util[NumCases], case_temporal_util[NumCases], case_total_util[NumCases];
  longint unsigned case_a_reads[NumCases], case_b_reads[NumCases], case_c_writes[NumCases];
  longint unsigned case_output_stalls[NumCases], case_write_cycles[NumCases];
  int unsigned case_queue_peak[NumCases];
  longint unsigned case_b_overlap_cycles[NumCases], case_b_bank_switches[NumCases];

  logic clk_i, rst_ni, start, done;
  logic [SizeAddrWidth-1:0] M_i, K_i, N_i;
  logic [NumBanks-1:0][AddrWidth-1:0] bank_addr;
  logic [NumBanks-1:0] bank_re, bank_we;
  logic [NumBanks-1:0][BankDataWidth-1:0] bank_rdata, bank_wdata;

  // SRAM coverage controls stay inactive throughout the matrix/reset tests.
  logic sram_tb_override;
  logic sram_negative_active;
  int unsigned sram_negative_cycles;
  logic [NumBanks-1:0][AddrWidth-1:0] tb_bank_addr;
  logic [NumBanks-1:0] tb_bank_re, tb_bank_we;
  logic [NumBanks-1:0][BankDataWidth-1:0] tb_bank_wdata;

  logic signed [InDataWidth-1:0] A_ref[], B_ref[];
  logic signed [OutDataWidth-1:0] G_memory[];
  bit c_written[];
  string active_case_name;
  int unsigned a_region_rows, b_region_rows, c_region_rows, b_base, c_base;
  int unsigned a_used_words, b_used_words, c_used_words;
  longint unsigned a_reads, b_reads, c_writes, mac_compute_cycles;
  longint unsigned output_stall_cycles, c_write_cycles, b_overlap_cycles, b_bank_switches;
  int unsigned output_queue_peak;
  bit previous_b_bank, seen_gemm_issue;
  bit monitor_active, case_finished, expect_reset_quiet;
  int unsigned reset_cases_passed, seed_state, active_pattern;
  bit queue_observation_active, check_queue_update;
  int unsigned expected_head_q, expected_tail_q, expected_used_q;
  longint unsigned queue_head_wraps, queue_tail_wraps, queue_full_replacements;
  longint unsigned paired_c_writes, gemv_c_pause_events, bank_direction_switches;
  longint unsigned paired_queue_wraps, prefetch_write_conflicts, response_write_overlap;
  logic [NumBanks-1:0] previous_bank_re, previous_bank_we;

  typedef enum int unsigned {
    ResetTopStart, ResetDuringLoad, ResetTopFlush, ResetTopIssue,
    ResetControllerDrain, ResetDuringCWrite, ResetPrefetchPending,
    ResetBPrefetchPending, ResetBankSwitch, ResetGemvWritePause,
    ResetCacheLoad, ResetCacheResponse, ResetFragmentReload, ResetOutputBackpressure
  } reset_target_t;
  localparam int unsigned NumResetTargets = 6;
  localparam int unsigned NumResetCases = 2*NumResetTargets + 8;

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

  for (genvar b = 0; b < NumBanks; b++) begin : gen_sram_banks
    single_port_memory #(
      .DataWidth(BankDataWidth), .DataDepth(DataDepth), .AddrWidth(AddrWidth)
    ) i_sram (
      .clk_i(clk_i), .rst_ni(rst_ni),
      .mem_addr_i   (sram_tb_override ? tb_bank_addr[b]  : bank_addr[b]),
      .mem_re_i     (sram_tb_override ? tb_bank_re[b]    : bank_re[b]),
      .mem_we_i     (sram_tb_override ? tb_bank_we[b]    : bank_we[b]),
      .mem_wr_data_i(sram_tb_override ? tb_bank_wdata[b] : bank_wdata[b]),
      .mem_rd_data_o(bank_rdata[b])
    );
  end

  gemm_accelerator_top_mac_array #(
    .InDataWidth(InDataWidth), .OutDataWidth(OutDataWidth),
    .AddrWidth(AddrWidth), .SizeAddrWidth(SizeAddrWidth),
    .NumCores(NumCores), .NumInputs(NumInputs), .GemmRows(GemmRows), .MReuse(MReuse),
    .OutputQueueDepth(OutputQueueDepth), .NumBanks(NumBanks),
    .BankDataWidth(BankDataWidth), .GemvFragmentElements(GemvFragmentElements)
  ) i_dut (
    .clk_i(clk_i), .rst_ni(rst_ni), .start_i(start),
    .M_size_i(M_i), .K_size_i(K_i), .N_size_i(N_i),
    .bank_addr_o(bank_addr), .bank_re_o(bank_re), .bank_we_o(bank_we),
    .bank_rdata_i(bank_rdata), .bank_wdata_o(bank_wdata), .done_o(done)
  );

  `include "includes/common_tasks.svh"

  // SystemVerilog requires constant generated-instance indices. Keep these
  // backdoor helpers explicit; they affect only TB initialization/inspection.
  function automatic logic [BankDataWidth-1:0] memory_read(
    input int unsigned bank, address
  );
    if (bank >= NumBanks || address >= DataDepth)
      $fatal(1, "TB backdoor read out of range: bank=%0d address=%0d", bank, address);
    case (bank)
      0: return gen_sram_banks[0].i_sram.memory[address];
      1: return gen_sram_banks[1].i_sram.memory[address];
      2: return gen_sram_banks[2].i_sram.memory[address];
      3: return gen_sram_banks[3].i_sram.memory[address];
      4: return gen_sram_banks[4].i_sram.memory[address];
      5: return gen_sram_banks[5].i_sram.memory[address];
      6: return gen_sram_banks[6].i_sram.memory[address];
      7: return gen_sram_banks[7].i_sram.memory[address];
      default: return 'x;
    endcase
  endfunction

  task automatic memory_write(
    input int unsigned bank, address,
    input logic [BankDataWidth-1:0] value
  );
    if (bank >= NumBanks || address >= DataDepth)
      $fatal(1, "TB backdoor write out of range: bank=%0d address=%0d", bank, address);
    case (bank)
      0: gen_sram_banks[0].i_sram.memory[address] = value;
      1: gen_sram_banks[1].i_sram.memory[address] = value;
      2: gen_sram_banks[2].i_sram.memory[address] = value;
      3: gen_sram_banks[3].i_sram.memory[address] = value;
      4: gen_sram_banks[4].i_sram.memory[address] = value;
      5: gen_sram_banks[5].i_sram.memory[address] = value;
      6: gen_sram_banks[6].i_sram.memory[address] = value;
      7: gen_sram_banks[7].i_sram.memory[address] = value;
      default: $fatal(1, "Invalid SRAM bank.");
    endcase
  endtask

  task automatic put_input_byte(input int unsigned bank, address, lane,
                                input logic [InDataWidth-1:0] value);
    logic [BankDataWidth-1:0] word_data;
    word_data = memory_read(bank, address);
    word_data[lane*InDataWidth +: InDataWidth] = value;
    memory_write(bank, address, word_data);
  endtask

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

  // Physical C word index used only by the independent scoreboard.
  function automatic int unsigned c_bank_from_index(input int unsigned index);
    return (N_i == 1) ? index % NumBanks : 6 + index % 2;
  endfunction
  function automatic int unsigned c_addr_from_index(input int unsigned index);
    return (N_i == 1) ? c_base + index / NumBanks : index / 2;
  endfunction

  task automatic initialize_memories(
    input int unsigned test_id, m_sz, k_sz, n_sz,
    input bit force_random = 1'b0
  );
    int unsigned k_blocks, element, address, lane, bank;
    bit poison_padding;
    if (m_sz == 0 || k_sz == 0 || n_sz == 0 ||
        longint'(m_sz) >= (64'd1 << SizeAddrWidth) ||
        longint'(k_sz) >= (64'd1 << SizeAddrWidth) ||
        longint'(n_sz) >= (64'd1 << SizeAddrWidth))
      $fatal(1, "Case dimensions must be positive and fit SizeAddrWidth.");
    k_blocks = (k_sz + NumInputs - 1) / NumInputs;
    poison_padding = !force_random &&
      ((test_cases[test_id].name == "GeMM poison tails") ||
       (test_cases[test_id].name == "GeMV poison tails") ||
       (test_cases[test_id].name == "GeMV stream tails") ||
       (test_cases[test_id].name == "GeMV cache K129") ||
       (test_cases[test_id].name == "GeMV fallback") ||
       (test_cases[test_id].name == "GeMV ctx above") ||
       (test_cases[test_id].name == "GeMV ctx multi"));
    if (n_sz == 1) begin
      a_region_rows = ((m_sz+NumCores-1)/NumCores)*k_blocks*AReadWords;
      b_region_rows = (k_sz + BusBytes-1) / BusBytes;
      c_region_rows = (m_sz + NumBanks*BankResults-1)/(NumBanks*BankResults);
      b_base = a_region_rows;
      c_base = b_base + b_region_rows;
      a_used_words = a_region_rows*NumBanks;
      b_used_words = (k_sz+BankBytes-1)/BankBytes;
      c_used_words = (m_sz+BankResults-1)/BankResults;
      if (c_base+c_region_rows >= DataDepth)
        $fatal(1, "GeMV regions and guard row exceed physical bank depth.");
    end else begin
      a_region_rows = ((m_sz+GemmRows-1)/GemmRows)*k_blocks;
      b_region_rows = ((n_sz+GemmCols-1)/GemmCols)*k_blocks*BReadWords;
      c_region_rows = m_sz*((n_sz+GemmCElements-1)/GemmCElements);
      b_base = 0; c_base = 0;
      a_used_words = 4*a_region_rows;
      b_used_words = 2*b_region_rows;
      c_used_words = 2*c_region_rows;
      if (a_region_rows >= DataDepth || b_region_rows >= DataDepth || c_region_rows >= DataDepth)
        $fatal(1, "GeMM regions and guard rows exceed physical bank depth.");
    end
    if ((64'd1 << AddrWidth) < DataDepth)
      $fatal(1, "SRAM address width does not cover DataDepth.");
    A_ref = new[m_sz*k_sz]; B_ref = new[k_sz*n_sz]; G_memory = new[m_sz*n_sz];
    c_written = new[c_used_words];
    a_reads = 0; b_reads = 0; c_writes = 0; mac_compute_cycles = 0;
    output_stall_cycles = 0; output_queue_peak = 0; c_write_cycles = 0;
    b_overlap_cycles = 0; b_bank_switches = 0; previous_b_bank = 0; seen_gemm_issue = 0;
    active_pattern = force_random ? 0 : test_cases[test_id].pattern;
    foreach (A_ref[i]) begin
      case (active_pattern)
        0: A_ref[i] = InDataWidth'($urandom());
        1: A_ref[i] = 0;
        2: A_ref[i] = 1;
        3,4: A_ref[i] = -1;
        5: A_ref[i] = 127;
        6: A_ref[i] = -128;
        7: A_ref[i] = (i%2 == 0) ? 1 : -1;
        default: A_ref[i] = 0;
      endcase
    end
    foreach (B_ref[i]) begin
      case (active_pattern)
        0: B_ref[i] = InDataWidth'($urandom());
        1: B_ref[i] = 0;
        2,4,7: B_ref[i] = 1;
        3: B_ref[i] = -1;
        5: B_ref[i] = 127;
        6: B_ref[i] = -128;
        default: B_ref[i] = 0;
      endcase
    end
    if (test_id == 8 || test_id == 9) begin
      A_ref[0] = (test_id%2 == 0) ? 8'sh80 : 8'sd127;
      B_ref[0] = (test_id%2 == 0) ? 8'sd127 : 8'sh80;
    end
    if (active_pattern == 0) begin
      if (m_sz*k_sz > 1) A_ref[m_sz*k_sz-1] = (test_id%2 == 0) ? 8'sd127 : 8'sh80;
      if (k_sz*n_sz > 1) B_ref[k_sz*n_sz-1] = (test_id%2 == 0) ? 8'sh80 : 8'sd127;
    end
    // Initialize complete input regions, including every physical padding bit.
    // Poison C and a guard row, then verify writes independently in the monitor.
    if (n_sz == 1) begin
      for (int b=0; b<NumBanks; b++) begin
        for (int w=0; w<a_region_rows; w++)
          memory_write(b,w,poison_padding ? {BankBytes{8'h5a}} : '0);
        for (int w=0; w<b_region_rows; w++)
          memory_write(b,b_base+w,poison_padding ? {BankBytes{8'ha5}} : '0);
        for (int w=0; w<=c_region_rows; w++) memory_write(b,c_base+w,'x);
      end
    end else begin
      for (int b=0; b<4; b++)
        for (int w=0; w<a_region_rows; w++)
          memory_write(b,w,poison_padding ? {BankBytes{8'h5a}} : '0);
      for (int b=4; b<6; b++)
        for (int w=0; w<b_region_rows; w++)
          memory_write(b,w,poison_padding ? {BankBytes{8'ha5}} : '0);
      for (int b=6; b<8; b++)
        for (int w=0; w<=c_region_rows; w++) memory_write(b,w,'x);
    end
    for (int m=0; m<m_sz; m++) begin
      for (int k=0; k<k_sz; k++) begin
        if (n_sz == 1) begin
          element = ((m/NumCores)*k_blocks+k/NumInputs)*TileElements +
                    (m%NumCores)*NumInputs+k%NumInputs;
          address = element/BusBytes; bank = (element/BankBytes)%NumBanks; lane = element%BankBytes;
        end else begin
          address = (m/GemmRows)*k_blocks+k/NumInputs;
          element = (m%GemmRows)*NumInputs+k%NumInputs;
          bank = element/BankBytes; lane = element%BankBytes;
        end
        put_input_byte(bank,address,lane,A_ref[m*k_sz+k]);
      end
    end
    for (int k=0; k<k_sz; k++) begin
      for (int n=0; n<n_sz; n++) begin
        if (n_sz == 1) begin
          address = b_base+k/BusBytes; bank = (k/BankBytes)%NumBanks; lane = k%BankBytes;
        end else begin
          element = (n%GemmCols)*NumInputs+k%NumInputs;
          address = ((n/GemmCols)*k_blocks+k/NumInputs)*BReadWords+element/(2*BankBytes);
          bank = 4+(element/BankBytes)%2; lane = element%BankBytes;
        end
        put_input_byte(bank,address,lane,B_ref[k*n_sz+n]);
      end
    end
  endtask

  task automatic gemm_golden(input int unsigned m_sz,k_sz,n_sz);
    longint signed acc;
    for (int m=0; m<m_sz; m++) begin
      for (int n=0; n<n_sz; n++) begin
        acc = 0;
        for (int k=0; k<k_sz; k++)
          acc += longint'($signed(A_ref[m*k_sz+k])) * longint'($signed(B_ref[k*n_sz+n]));
        if (acc > 64'sd2147483647 || acc < -64'sd2147483648)
          $fatal(1,"Golden result exceeds int32 at C[%0d,%0d].",m,n);
        G_memory[m*n_sz+n] = OutDataWidth'(acc);
      end
    end
  endtask

  function automatic longint unsigned expected_compute_steps(input int unsigned m_sz,k_sz,n_sz);
    longint unsigned kb,mg,ng;
    kb=(longint'(k_sz)+NumInputs-1)/NumInputs;
    mg=(longint'(m_sz)+((n_sz==1)?NumCores:GemmRows)-1)/((n_sz==1)?NumCores:GemmRows);
    ng=(longint'(n_sz)+GemmCols-1)/GemmCols;
    return (n_sz==1)?mg*kb:mg*kb*ng;
  endfunction
  function automatic longint unsigned expected_a_reads(input int unsigned m_sz,k_sz,n_sz);
    return expected_compute_steps(m_sz,k_sz,n_sz)*((n_sz==1)?NumBanks*AReadWords:4);
  endfunction
  function automatic longint unsigned expected_b_reads(input int unsigned m_sz,k_sz,n_sz,reuse_rows);
    longint unsigned kb,mg,ng;
    if(n_sz==1) begin
      // One full logical vector per M supergroup; if it fits one fragment,
      // keep that fragment across the supergroup boundary as well.
      mg=(k_sz<=GemvFragmentElements)?1:(longint'(m_sz)+GemvReuseRows-1)/GemvReuseRows;
      return mg*((longint'(k_sz)+BankBytes-1)/BankBytes);
    end
    kb=(longint'(k_sz)+NumInputs-1)/NumInputs;
    ng=(longint'(n_sz)+GemmCols-1)/GemmCols;
    mg=(longint'(m_sz)+reuse_rows-1)/reuse_rows;
    return mg*ng*kb*BReadWords*2;
  endfunction
  function automatic longint unsigned expected_gemm_a_address(
    input longint unsigned read_index,input int unsigned m_sz,k_sz,n_sz
  );
    longint unsigned kb,ng,group_issues,group,within_group,remaining,active_pairs,pair_index,k_block;
    kb=(longint'(k_sz)+NumInputs-1)/NumInputs;
    ng=(longint'(n_sz)+GemmCols-1)/GemmCols;
    group_issues=longint'(NumContexts)*ng*kb;
    group=read_index/group_issues; within_group=read_index%group_issues;
    remaining=longint'(m_sz)-group*MReuse;
    active_pairs=((remaining<MReuse?remaining:MReuse)+GemmRows-1)/GemmRows;
    pair_index=within_group%active_pairs; k_block=(within_group/active_pairs)%kb;
    return (group*NumContexts+pair_index)*kb+k_block;
  endfunction
  function automatic longint unsigned expected_gemv_a_address(
    input longint unsigned burst_index,input int unsigned m_sz,k_sz
  );
    longint unsigned kb,tile_index,supergroup,within_group,row_tiles,active_contexts;
    longint unsigned context_index,k_block,physical_tile;
    // Logical accepted order is [M supergroup][K block][16-row context].
    // Decode the request ordinal independently of DUT counters and prefetches,
    // then map it into the unchanged [M tile][K block] physical A layout.
    kb=(longint'(k_sz)+NumInputs-1)/NumInputs;
    row_tiles=(longint'(m_sz)+NumCores-1)/NumCores;
    tile_index=burst_index/AReadWords;
    supergroup=tile_index/(NumContexts*kb);
    within_group=tile_index%(NumContexts*kb);
    active_contexts=row_tiles-supergroup*NumContexts;
    if(active_contexts>NumContexts) active_contexts=NumContexts;
    context_index=within_group%active_contexts;
    k_block=within_group/active_contexts;
    physical_tile=(supergroup*NumContexts+context_index)*kb+k_block;
    return physical_tile*AReadWords+burst_index%AReadWords;
  endfunction
  function automatic int unsigned timeout_cycles(input int unsigned m_sz,k_sz,n_sz);
    return 200+int'(expected_compute_steps(m_sz,k_sz,n_sz))*(BReadWords+NumCores+12)+k_sz;
  endfunction

  task automatic start_and_wait_gemm(output int unsigned elapsed_cycles,
                                    output real measured_energy,input int unsigned max_cycles);
    @(negedge clk_i);
    if(done!==1'b0) $fatal(1,"done must be low before start.");
    metrics_pkg::energy=0;
    monitor_active=1'b1; case_finished=1'b0; start=1'b1;
    @(posedge clk_i); #1;
    // Accepted start is edge zero; count all preload, compute and final writes.
    elapsed_cycles=0;
    @(negedge clk_i); start=1'b0;
    while(done!==1'b1) begin
      @(posedge clk_i); #1; elapsed_cycles++;
      if($isunknown(done)) $fatal(1,"done contains X/Z.");
      if(elapsed_cycles>max_cycles) $fatal(1,"[%0s] Timeout after %0d cycles.",active_case_name,elapsed_cycles);
    end
    measured_energy=metrics_pkg::energy; case_finished=1'b1;
  endtask

  task automatic verify_result_c(output int unsigned num_errors);
    int unsigned addr,bank,row,column,logical_index,row_words;
    logic [BankDataWidth-1:0] word_data;
    logic signed [OutDataWidth-1:0] expected_value,actual_value;
    bit padding;
    num_errors=0;
    row_words=(int'(N_i)+GemmCElements-1)/GemmCElements;
    for(int idx=0;idx<c_used_words;idx++) begin
      addr=c_addr_from_index(idx); bank=c_bank_from_index(idx);
      if(!c_written[idx]) begin
        if(num_errors<10) $display("ERROR: C word bank%0d address%0d was not written.",bank,addr);
        num_errors++;
      end else begin
        word_data=memory_read(bank,addr);
        for(int lane=0;lane<BankResults;lane++) begin
          if(N_i==1) begin
            logical_index=idx*BankResults+lane; padding=(logical_index>=int'(M_i));
          end else begin
            row=(idx/2)/row_words;
            column=((idx/2)%row_words)*GemmCElements+(idx%2)*BankResults+lane;
            logical_index=row*int'(N_i)+column; padding=(column>=int'(N_i));
          end
          expected_value=padding?'0:G_memory[logical_index];
          actual_value=$signed(word_data[lane*OutDataWidth+:OutDataWidth]);
          if(actual_value!==expected_value) begin
            if(num_errors<10) $display("ERROR C bank%0d addr%0d lane%0d (%0s): got%0d expected%0d.",bank,addr,lane,padding?"padding":"result",actual_value,expected_value);
            num_errors++;
          end
        end
      end
    end
    if(c_writes!=c_used_words) begin $display("ERROR C writes=%0d expected=%0d.",c_writes,c_used_words);num_errors++;end
    if(mac_compute_cycles!=expected_compute_steps(M_i,K_i,N_i)) begin
      $display("ERROR compute steps=%0d expected=%0d.",mac_compute_cycles,expected_compute_steps(M_i,K_i,N_i));num_errors++;
    end
    if(a_reads!=expected_a_reads(M_i,K_i,N_i)) begin
      $display("ERROR A bank-word reads=%0d expected=%0d.",a_reads,expected_a_reads(M_i,K_i,N_i));num_errors++;
    end
    if(b_reads!=expected_b_reads(M_i,K_i,N_i,MReuse)) begin
      $display("ERROR B bank-word reads=%0d expected=%0d.",b_reads,expected_b_reads(M_i,K_i,N_i,MReuse));num_errors++;
    end
  endtask

  // Every access is observed at the accepting clock edge, before SRAM NBAs.
  // Reference addresses are decoded from the logical stream, never DUT counters.
  always @(posedge clk_i) begin : monitor_banks
    int unsigned active_banks, idx, expected_bank, expected_addr;
    longint unsigned logical_word;
    bit any_a, any_b;
    if(expect_reset_quiet && ({bank_re,bank_we,done}!=='0))
      $fatal(1,"[%0s] Transaction or done during reset/idle.",active_case_name);
    if(rst_ni) begin
      if($isunknown({bank_re,bank_we,done})) $fatal(1,"Unknown bank enables or done.");
      active_banks=0; any_a=0; any_b=0;
      for(int b=0;b<NumBanks;b++) begin
        if(bank_re[b] && bank_we[b]) $fatal(1,"Bank%0d reads and writes in the same cycle.",b);
        if(bank_re[b] || bank_we[b]) active_banks++;
      end
      if(active_banks*BankDataWidth>1024) $fatal(1,"Bandwidth exceeded1024bits/cycle.");
      if(!monitor_active && (bank_re!='0 || bank_we!='0))
        $fatal(1,"Memory transaction outside an active case.");
      if(monitor_active) begin
        if(case_finished && (bank_re!='0 || bank_we!='0))
          $fatal(1,"[%0s] Memory transaction after done.",active_case_name);
        if(!case_finished) begin
          if(i_dut.valid_data) mac_compute_cycles++;
          if($isunknown({i_dut.output_stall,i_dut.output_queue_used_q}))
            $fatal(1,"Unknown output queue state.");
          if(int'(i_dut.output_queue_used_q)>OutputQueueDepth)
            $fatal(1,"Output reservations exceed capacity.");
          if(i_dut.output_stall) output_stall_cycles++;
          if(int'(i_dut.output_queue_used_q)>output_queue_peak)
            output_queue_peak=int'(i_dut.output_queue_used_q);
          if((N_i!=1) && i_dut.issue_block) begin
            if(seen_gemm_issue && i_dut.b_compute_bank_q!=previous_b_bank) b_bank_switches++;
            previous_b_bank=i_dut.b_compute_bank_q; seen_gemm_issue=1;
          end
          if(bank_we!='0) c_write_cycles++;
          if(queue_observation_active) begin
            if(i_dut.gemv_pair_write) begin
              paired_c_writes++;
              if(int'(i_dut.c_head_q)==OutputQueueDepth-1) paired_queue_wraps++;
            end
            if(i_dut.is_gemv && i_dut.issue_block && i_dut.gemv_c_write) prefetch_write_conflicts++;
            if(i_dut.is_gemv && i_dut.a_response_valid_q && i_dut.gemv_c_write) response_write_overlap++;
            if((N_i==1) && (bank_we!='0)) gemv_c_pause_events++;
            for(int b=0;b<NumBanks;b++)
              if((previous_bank_re[b] && bank_we[b]) || (previous_bank_we[b] && bank_re[b]))
                bank_direction_switches++;
          end
        end
        if((N_i==1) && bank_we!='0 && bank_re!='0)
          $fatal(1,"GeMV C arbitration must pause the shared A/B read bus.");
        for(int b=0;b<NumBanks;b++) begin
          if(bank_re[b] || bank_we[b]) begin
            if($isunknown(bank_addr[b]) || int'(bank_addr[b])>=DataDepth)
              $fatal(1,"[%0s] Bank%0d invalid address%0h.",active_case_name,b,bank_addr[b]);
          end
          if(bank_re[b]) begin
            if((N_i!=1 && b<4) || (N_i==1 && int'(bank_addr[b])<b_base)) begin
              any_a=1;
              if(N_i==1) begin
                expected_bank=int'(a_reads%NumBanks);
                expected_addr=int'(expected_gemv_a_address(a_reads/NumBanks,M_i,K_i));
              end else begin
                expected_bank=int'(a_reads%4);
                expected_addr=int'(expected_gemm_a_address(a_reads/4,M_i,K_i,N_i));
              end
              if(b!=expected_bank || int'(bank_addr[b])!=expected_addr)
                $fatal(1,"[%0s] A read%0d bank/addr=%0d/%0d expected=%0d/%0d.",
                       active_case_name,a_reads,b,bank_addr[b],expected_bank,expected_addr);
              if(int'(bank_addr[b])>=a_region_rows)
                $fatal(1,"A read outside initialized input region.");
              a_reads++;
            end else if((N_i!=1 && b>=4 && b<6) ||
                        (N_i==1 && int'(bank_addr[b])>=b_base && int'(bank_addr[b])<c_base)) begin
              any_b=1;
              logical_word=b_reads%b_used_words;
              expected_bank=(N_i==1)?int'(logical_word%NumBanks):(4+int'(logical_word%2));
              expected_addr=(N_i==1)?(b_base+int'(logical_word/NumBanks)):int'(logical_word/2);
              if(b!=expected_bank || int'(bank_addr[b])!=expected_addr)
                $fatal(1,"[%0s] B read%0d bank/addr=%0d/%0d expected=%0d/%0d.",
                       active_case_name,b_reads,b,bank_addr[b],expected_bank,expected_addr);
              if(N_i!=1 && int'(bank_addr[b])>=b_region_rows)
                $fatal(1,"B read outside initialized input region.");
              b_reads++;
            end else $fatal(1,"[%0s] Read from a non-input region: bank%0d addr%0d.",active_case_name,b,bank_addr[b]);
          end
          if(bank_we[b]) begin
            if($isunknown(bank_wdata[b])) $fatal(1,"Unknown C word on bank%0d.",b);
            if(N_i==1) begin
              if(int'(bank_addr[b])<c_base || int'(bank_addr[b])>=c_base+c_region_rows)
                $fatal(1,"GeMV write outside C region: bank%0d addr%0d.",b,bank_addr[b]);
              idx=(int'(bank_addr[b])-c_base)*NumBanks+b;
            end else begin
              if(b<6 || int'(bank_addr[b])>=c_region_rows)
                $fatal(1,"GeMM write outside C region: bank%0d addr%0d.",b,bank_addr[b]);
              idx=int'(bank_addr[b])*2+b-6;
            end
            if(idx>=c_used_words) $fatal(1,"C write to unused physical word%0d.",idx);
            if(c_written[idx]) $fatal(1,"Duplicate C write to physical word%0d.",idx);
            c_written[idx]=1; c_writes++;
          end
        end
        if(N_i!=1) begin
          if(any_a && bank_re[3:0]!=4'b1111) $fatal(1,"GeMM A row pair requires four banks.");
          if(any_b && bank_re[5:4]!=2'b11) $fatal(1,"GeMM B beat requires two banks.");
          if(bank_we!='0 && bank_we!=8'hc0) $fatal(1,"GeMM C must write both physical banks.");
          if(any_b && i_dut.valid_data) b_overlap_cycles++;
        end else if(any_a && bank_re!=8'hff) $fatal(1,"GeMV A beat requires all eight banks.");
      end
    end
    previous_bank_re=bank_re; previous_bank_we=bank_we;
  end

  // Head may pop TWO entries in one cycle. Check the actual modulo-depth
  // successor, including a pair spanning slot24 ->0 and simultaneous reserve.
  always @(posedge clk_i) begin : monitor_queue
    int unsigned pop_count;
    if(!rst_ni || !queue_observation_active) begin
      check_queue_update=0;
    end else begin
      if(check_queue_update) begin
        if(int'(i_dut.c_head_q)!=expected_head_q || int'(i_dut.c_tail_q)!=expected_tail_q ||
           int'(i_dut.output_queue_used_q)!=expected_used_q)
          $fatal(1,"Queue pointer/occupancy update did not match accepted pop/reserve.");
      end
      pop_count=int'(i_dut.c_pop_count);
      if(pop_count>int'(i_dut.output_queue_used_q)) $fatal(1,"Queue underflow.");
      if(int'(i_dut.c_head_q)>=OutputQueueDepth || int'(i_dut.c_tail_q)>=OutputQueueDepth)
        $fatal(1,"Queue pointer out of range.");
      expected_head_q=(int'(i_dut.c_head_q)+pop_count)%OutputQueueDepth;
      expected_tail_q=(int'(i_dut.c_tail_q)+(i_dut.reserve_output?1:0))%OutputQueueDepth;
      expected_used_q=int'(i_dut.output_queue_used_q)+(i_dut.reserve_output?1:0)-pop_count;
      check_queue_update=!i_dut.accept_start;
      if(monitor_active && !case_finished) begin
        if(int'(i_dut.c_head_q)+pop_count>=OutputQueueDepth && pop_count!=0) queue_head_wraps++;
        if(i_dut.reserve_output && int'(i_dut.c_tail_q)==OutputQueueDepth-1) queue_tail_wraps++;
        if(int'(i_dut.output_queue_used_q)==OutputQueueDepth && pop_count!=0 && i_dut.reserve_output)
          queue_full_replacements++;
      end
    end
  end

  function automatic string reset_target_name(input reset_target_t target);
    case(target)
      ResetTopStart:return "TopStart";
      ResetDuringLoad:return "operand load after first compute";
      ResetTopFlush:return "TopFlush";
      ResetTopIssue:return "second K block issue";
      ResetControllerDrain:return "controller drain";
      ResetDuringCWrite:return "C write during drain";
      ResetPrefetchPending:return "A prefetch pending during MAC consumption";
      ResetBPrefetchPending:return "GeMM B prefetch overlapping compute";
      ResetBankSwitch:return "GeMM B buffer switch";
      ResetGemvWritePause:return "GeMV C write stealing A read bandwidth";
      ResetCacheLoad:return "initial B fragment request";
      ResetCacheResponse:return "B fragment response pending";
      ResetFragmentReload:return "prefetched B fragment response with MAC pending";
      ResetOutputBackpressure:return "output queue backpressure";
      default:return "unknown reset target";
    endcase
  endfunction

  function automatic bit reset_target_reached(input reset_target_t target);
    case(target)
      ResetTopStart:return (i_dut.state_q===i_dut.TopStart);
      ResetDuringLoad:return (mac_compute_cycles>0) &&
        ((N_i==1)?((i_dut.state_q===i_dut.TopLoad) && i_dut.K_count==NumInputs && i_dut.read_beat_q!='0):
                    (bank_re[5:4]!=0));
      ResetTopFlush:return (i_dut.state_q===i_dut.TopFlush);
      ResetTopIssue:return i_dut.issue_block && i_dut.K_count==NumInputs && mac_compute_cycles>0;
      ResetControllerDrain:return (i_dut.state_q===i_dut.TopDrain) &&
        (i_dut.i_gemm_controller_mac_array.current_state===i_dut.i_gemm_controller_mac_array.ControllerDrain);
      ResetDuringCWrite:return (i_dut.state_q===i_dut.TopDrain) && bank_we!='0 &&
        c_writes>0 && c_writes<c_used_words;
      ResetPrefetchPending:return i_dut.is_gemv && i_dut.a_response_valid_q &&
        i_dut.a_response_beat_q=='0 && i_dut.valid_data;
      ResetBPrefetchPending:return !i_dut.is_gemv && i_dut.b_response_valid_q && i_dut.valid_data &&
        (i_dut.b_fetch_bank_q!=i_dut.b_compute_bank_q);
      ResetBankSwitch:return !i_dut.is_gemv && i_dut.issue_block && seen_gemm_issue &&
        (i_dut.b_compute_bank_q!=previous_b_bank);
      ResetGemvWritePause:return i_dut.is_gemv && i_dut.gemv_c_write && bank_re=='0 &&
        (i_dut.state_q!==i_dut.TopDrain);
      ResetCacheLoad:return i_dut.is_gemv && i_dut.cache_request && mac_compute_cycles==0;
      ResetCacheResponse:return i_dut.is_gemv && i_dut.cache_response_valid_q &&
        (i_dut.state_q===i_dut.TopCacheFlush);
      ResetFragmentReload:return i_dut.is_gemv && i_dut.cache_response_valid_q &&
        i_dut.valid_data && mac_compute_cycles>0;
      ResetOutputBackpressure:return !i_dut.is_gemv && i_dut.output_stall;
      default:return 0;
    endcase
  endfunction

  task automatic check_reset_idle(input string phase_label);
    if({bank_re,bank_we,done}!=='0) $fatal(1,"[%0s] Live bank controls in %0s.",active_case_name,phase_label);
    if(i_dut.state_q!==i_dut.TopIdle ||
       i_dut.i_gemm_controller_mac_array.current_state!==i_dut.i_gemm_controller_mac_array.ControllerIdle)
      $fatal(1,"[%0s] FSM not idle in %0s.",active_case_name,phase_label);
    if({i_dut.valid_data,i_dut.init_save,i_dut.result_valid,i_dut.controller_busy,
        i_dut.controller_done,i_dut.controller_done_seen_q,
        i_dut.a_response_valid_q,i_dut.b_response_valid_q,i_dut.cache_response_valid_q,
        i_dut.cache_ready_q,i_dut.b_valid_q,i_dut.a_ready_q,i_dut.a_used_q,
        i_dut.a_fetch_done_q,i_dut.b_fetch_done_q,i_dut.output_slot_busy_q,
        i_dut.c_valid_q,i_dut.output_stall,i_dut.output_queue_used_q,
        i_dut.fifo_pending_valid_q,i_dut.c_ready_q}!=='0)
      $fatal(1,"[%0s] Stale valid/cache/FIFO state in %0s.",active_case_name,phase_label);
    if({i_dut.M_count,i_dut.K_count,i_dut.N_count}!=='0 || i_dut.array_c_out!=='0)
      $fatal(1,"[%0s] Counters or MACs not reset in %0s.",active_case_name,phase_label);
  endtask

  task automatic run_reset_case(input reset_target_t target,input bit abort_is_gemv);
    int unsigned am,ak,an,rm,rk,rn,waited,maximum,recovery_cycles,recovery_errors;
    real recovery_energy;
    string label_text;
    logic [BankDataWidth-1:0] c_snapshot[];
    label_text=$sformatf("%0s / %0s",abort_is_gemv?"GeMV":"GeMM",reset_target_name(target));
    active_case_name={label_text," abort"};
    am=abort_is_gemv?(3*NumCores+1):GemmRows;
    ak=2*NumInputs; an=abort_is_gemv?1:2*NumCores;
    if(target==ResetBPrefetchPending || target==ResetBankSwitch) begin am=MReuse;an=4*GemmCols;end
    if(target==ResetOutputBackpressure) begin am=MReuse;ak=NumInputs;an=4*GemmCols;end
    if(target==ResetGemvWritePause) am=4*NumCores+1;
    if(target==ResetCacheLoad || target==ResetCacheResponse) ak=257;
    if(target==ResetFragmentReload) begin ak=GemvFragmentElements+1;am=NumCores+1;end
    M_i=SizeAddrWidth'(am); K_i=SizeAddrWidth'(ak); N_i=SizeAddrWidth'(an);
    initialize_memories(0,am,ak,an,1'b1); maximum=timeout_cycles(am,ak,an);waited=0;
    @(negedge clk_i);
    if(done!==1'b0) $fatal(1,"done high before reset test.");
    monitor_active=1;case_finished=0;start=1;
    @(negedge clk_i);start=0;
    while(!reset_target_reached(target)) begin
      if(done!==1'b0) $fatal(1,"[%0s] Job completed before reset target.",active_case_name);
      if(waited>=maximum) $fatal(1,"[%0s] Reset target timeout.",active_case_name);
      @(negedge clk_i);waited++;
    end
    c_snapshot=new[c_used_words];
    for(int w=0;w<c_used_words;w++) c_snapshot[w]=memory_read(c_bank_from_index(w),c_addr_from_index(w));
    rst_ni=0;monitor_active=0;case_finished=0;
    #1;expect_reset_quiet=1;check_reset_idle("reset assertion");
    repeat(2) begin @(posedge clk_i);#1;check_reset_idle("reset hold");end
    @(negedge clk_i);rst_ni=1;#1;check_reset_idle("reset release");
    repeat(NumCores+4) begin @(posedge clk_i);#1;check_reset_idle("idle after reset");end
    for(int w=0;w<c_used_words;w++)
      if(memory_read(c_bank_from_index(w),c_addr_from_index(w))!==c_snapshot[w])
        $fatal(1,"[%0s] C word%0d changed during reset/idle.",active_case_name,w);
    // Recovery always changes mode and data, and includes partial tiles.
    rm=abort_is_gemv?3:NumCores+1;rk=NumInputs+1;rn=abort_is_gemv?NumCores+1:1;
    active_case_name={label_text," recovery"};
    M_i=SizeAddrWidth'(rm);K_i=SizeAddrWidth'(rk);N_i=SizeAddrWidth'(rn);
    initialize_memories(0,rm,rk,rn,1'b1);gemm_golden(rm,rk,rn);expect_reset_quiet=0;
    start_and_wait_gemm(recovery_cycles,recovery_energy,timeout_cycles(rm,rk,rn));
    verify_result_c(recovery_errors);
    if(recovery_errors!=0) $fatal(1,"[%0s] Recovery result failed.",active_case_name);
    repeat(2) begin clk_delay(1);if(done!==1'b0) $fatal(1,"Recovery done must pulse once.");end
    monitor_active=0;case_finished=0;reset_cases_passed++;
  endtask

  task automatic run_reset_tests();
    reset_cases_passed=0;
    for(int mode=0;mode<2;mode++)
      for(int r=0;r<NumResetTargets;r++) run_reset_case(reset_target_t'(r),mode==1);
    run_reset_case(ResetPrefetchPending,1);
    run_reset_case(ResetBPrefetchPending,0);
    run_reset_case(ResetBankSwitch,0);
    run_reset_case(ResetGemvWritePause,1);
    run_reset_case(ResetCacheLoad,1);
    run_reset_case(ResetCacheResponse,1);
    run_reset_case(ResetFragmentReload,1);
    run_reset_case(ResetOutputBackpressure,0);
  endtask

  //---------------------------------------------------------------------------
  // Phase 3: SRAM coverage after the matrix and reset/restart tests.
  // Normal accesses remain half-duplex. One explicitly marked negative cycle
  // exercises the unchanged SRAM module's simultaneous read/write $error check.
  // The markers identify the eight expected SRAM diagnostics in the transcript.
  //---------------------------------------------------------------------------
  always @(posedge clk_i) begin : monitor_sram_override
    int unsigned active_banks;
    if (sram_tb_override) begin
      // These are the actual SRAM port controls selected by the TB muxes.
      if (monitor_active || queue_observation_active || start ||
          bank_re !== '0 || bank_we !== '0 || done !== 1'b0)
        $fatal(1, "SRAM test override overlaps accelerator activity.");
      if ($isunknown({tb_bank_re, tb_bank_we}))
        $fatal(1, "Unknown SRAM test enables.");
      if (sram_negative_active) begin
        if (rst_ni !== 1'b1 || tb_bank_re !== '1 || tb_bank_we !== '1 ||
            tb_bank_addr !== '0)
          $fatal(1, "Malformed SRAM negative-test stimulus.");
        sram_negative_cycles++;
        if (sram_negative_cycles != 1)
          $fatal(1, "SRAM negative test lasted more than one clock cycle.");
      end
      active_banks = 0;
      for (int b = 0; b < NumBanks; b++) begin
        if (tb_bank_re[b] && tb_bank_we[b] && !sram_negative_active)
          $fatal(1, "SRAM test bank %0d reads and writes in the same cycle.", b);
        if (tb_bank_re[b] || tb_bank_we[b]) begin
          active_banks++;
          if ($isunknown(tb_bank_addr[b]) || int'(tb_bank_addr[b]) >= DataDepth)
            $fatal(1, "SRAM test bank %0d has an invalid active address.", b);
        end
        if (tb_bank_we[b] && $isunknown(tb_bank_wdata[b]))
          $fatal(1, "SRAM test bank %0d writes unknown data.", b);
      end
      if (active_banks * BankDataWidth > 1024)
        $fatal(1, "SRAM test exceeds 1024 bits per cycle.");
    end
  end

  task automatic check_sram_read_data(
    input logic [BankDataWidth-1:0] expected,
    input string phase_label
  );
    for (int b = 0; b < NumBanks; b++) begin
      if (bank_rdata[b] !== expected)
        $fatal(1, "SRAM bank %0d mismatch during %0s: got %0h, expected %0h.",
               b, phase_label, bank_rdata[b], expected);
    end
  endtask

  task automatic run_sram_coverage_exercise();
    logic [BankDataWidth-1:0] pat;
    logic [NumBanks-1:0][BankDataWidth-1:0] held_rdata;

    @(negedge clk_i);
    if (rst_ni !== 1'b1 || i_dut.state_q !== i_dut.TopIdle ||
        monitor_active || queue_observation_active || start ||
        bank_re !== '0 || bank_we !== '0 || done !== 1'b0)
      $fatal(1, "SRAM coverage exercise requires an idle accelerator.");
    active_case_name = "SRAM coverage exercise";
    tb_bank_addr = '0;
    tb_bank_re = '0;
    tb_bank_we = '0;
    tb_bank_wdata = '0;
    sram_tb_override = 1'b1;

    // Toggle all address-port bits without accessing memory. The all-ones
    // address may exceed DataDepth; both enables remain zero throughout.
    held_rdata = bank_rdata;
    @(posedge clk_i); #1;
    @(negedge clk_i); tb_bank_addr = '1;
    @(posedge clk_i); #1;
    if (bank_rdata !== held_rdata)
      $fatal(1, "SRAM output changed during a disabled address toggle.");
    @(negedge clk_i); tb_bank_addr = '0;
    @(posedge clk_i); #1;
    if (bank_rdata !== held_rdata)
      $fatal(1, "SRAM output changed while both enables were low.");

    // Write/read/hold zeros, ones and both alternating patterns in every bank.
    for (int unsigned step = 0; step < 5; step++) begin
      case (step)
        0: pat = '0;
        1: pat = '1;
        2: pat = {(BankDataWidth/2){2'b10}};
        3: pat = {(BankDataWidth/2){2'b01}};
        default: pat = '0;
      endcase
      held_rdata = bank_rdata;

      @(negedge clk_i);
      tb_bank_addr = '0;
      tb_bank_re = '0;
      tb_bank_we = '1;
      for (int b = 0; b < NumBanks; b++) tb_bank_wdata[b] = pat;
      @(posedge clk_i); #1;
      if (bank_rdata !== held_rdata)
        $fatal(1, "SRAM read output changed during a write-only cycle.");

      @(negedge clk_i);
      tb_bank_we = '0;
      tb_bank_re = '1;
      #1;
      if (bank_rdata !== held_rdata)
        $fatal(1, "SRAM read output changed before the accepting clock edge.");
      @(posedge clk_i); #1;
      check_sram_read_data(pat, $sformatf("read pattern %0d", step));

      @(negedge clk_i);
      tb_bank_re = '0;
      for (int b = 0; b < NumBanks; b++) begin
        tb_bank_addr[b] = AddrWidth'(DataDepth-1);
        tb_bank_wdata[b] = ~pat;
      end
      @(posedge clk_i); #1;
      check_sram_read_data(pat, $sformatf("hold pattern %0d", step));
    end

    // Negative test: exactly one active edge with RE=WE=1 on every bank.
    // These are testbench-generated invalid requests, never accelerator traffic.
    // Do not suppress or modify the SRAM $errors: the transcript must contain
    // exactly one original error from each bank, and no unrelated errors.
    @(negedge clk_i);
    sram_negative_active = 1'b1;
    tb_bank_addr = '0;
    tb_bank_wdata = '1;
    tb_bank_re = '1;
    tb_bank_we = '1;
    $display("[SRAM NEGATIVE BEGIN] expected_errors=%0d address=0", NumBanks);
    @(posedge clk_i); #1;
    if (sram_negative_cycles != 1)
      $fatal(1, "SRAM negative-test cycle was not observed.");
    @(negedge clk_i);
    tb_bank_re = '0;
    tb_bank_we = '0;
    sram_negative_active = 1'b0;
    $display("[SRAM NEGATIVE END] cycles=%0d", sram_negative_cycles);

    // Re-establish known contents using legal cycles; do not assume any
    // defined read-during-write behavior from the invalid negative cycle.
    @(negedge clk_i);
    tb_bank_addr = '0;
    tb_bank_wdata = '1;
    tb_bank_we = '1;
    @(posedge clk_i); #1;
    @(negedge clk_i);
    tb_bank_we = '0;
    tb_bank_re = '1;
    @(posedge clk_i); #1;
    check_sram_read_data('1, "read before reset");

    // Reset clears the registered output asynchronously, not the SRAM contents.
    @(negedge clk_i);
    tb_bank_re = '0;
    rst_ni = 1'b0;
    #1;
    check_sram_read_data('0, "asynchronous reset assertion");
    @(posedge clk_i); #1;
    check_sram_read_data('0, "reset hold");
    @(negedge clk_i); rst_ni = 1'b1;
    @(posedge clk_i); #1;
    check_sram_read_data('0, "reset release without read");

    @(negedge clk_i); tb_bank_re = '1;
    @(posedge clk_i); #1;
    check_sram_read_data('1, "memory contents retained across reset");

    @(negedge clk_i);
    tb_bank_re = '0;
    tb_bank_we = '0;
    tb_bank_addr = '0;
    tb_bank_wdata = '0;
    sram_tb_override = 1'b0;
    @(posedge clk_i); #1;
    check_reset_idle("after SRAM coverage exercise");
  endtask

  function automatic string pad(input string str,input int unsigned width);
    while(str.len()<width) str={str," "};
    return str;
  endfunction

  task automatic print_report();
    int unsigned num_failed,workload_failures;
    real normalized,norm_sum,spatial_sum,temporal_sum,total_sum;
    num_failed=0;workload_failures=0;norm_sum=0;spatial_sum=0;temporal_sum=0;total_sum=0;

    // Check the complete regression, even though only the scored workloads
    // appear in the performance table and contribute to its averages.
    foreach(test_cases[c])
      if(case_errors[c]!=0) num_failed++;

    $display("\n=============================================================================================================");
    $display(" RESULTS -- six required workloads");
    $display("-------------------------------------------------------------------------------------------------------------");
    $display(" %s | %4s %4s %3s | %7s | %11s | %7s | %8s | %7s | %12s | %s",pad("Case",8),"M","K","N","Cycles","Cycles/MKN","Spatial","Temporal","Total","Energy","Check");
    $display("-------------------------------------------------------------------------------------------------------------");
    for(int c=0;c<NumWorkloads;c++) begin
      normalized=real'(case_cycles[c])/(real'(test_cases[c].M)*real'(test_cases[c].K)*real'(test_cases[c].N));
      $display(" %s | %4d %4d %3d | %7d | %11.9f | %6.2f%% | %7.2f%% | %6.2f%% | %12.2f | %s",pad(test_cases[c].name,8),test_cases[c].M,test_cases[c].K,test_cases[c].N,case_cycles[c],normalized,case_spatial_util[c],case_temporal_util[c],case_total_util[c],case_energy[c],(case_errors[c]==0)?"PASS":"FAIL");
      norm_sum+=normalized;spatial_sum+=case_spatial_util[c];temporal_sum+=case_temporal_util[c];total_sum+=case_total_util[c];
      if(case_errors[c]!=0) workload_failures++;
    end
    $display("-------------------------------------------------------------------------------------------------------------");
    $display(" Averages over the six required workloads:");
    if(workload_failures==0) begin
      $display(" Mean normalized latency   : %.9f cycles/MKN",norm_sum/NumWorkloads);
      $display(" Mean spatial utilization  : %0.2f%%",spatial_sum/NumWorkloads);
      $display(" Mean temporal utilization : %0.2f%%",temporal_sum/NumWorkloads);
      $display(" Mean total utilization    : %0.2f%%",total_sum/NumWorkloads);
    end else $display(" Mean normalized latency INVALID: required workload failure.");
    $display("\n Area                      : %0.2f area units",metrics_pkg::area);
    $display("=============================================================================================================\n");
    if(num_failed!=0) $fatal(1,"%0d of%0d cases FAILED.",num_failed,NumCases);
  endtask

  initial begin
    clk_i=0; forever #5 clk_i=~clk_i;
  end
  initial begin : regression
    if(NumBanks!=8 || BankDataWidth!=128 || NumCores!=16 || NumInputs!=32 || GemmRows!=2)
      $fatal(1,"This banked regression uses8x128bit banks and a2Mx8Nx32K mapping.");
    if(SizeAddrWidth<12 || GemvFragmentElements!=BusBytes || 2*GemvReuseRows+1>=4096)
      $fatal(1,"Regression requires12-bit dimensions, one128-byte B fragment and representable context-tail tests.");
    start=0;rst_ni=0;M_i='0;K_i='0;N_i='0;monitor_active=0;case_finished=0;
    sram_tb_override=0;tb_bank_addr='0;tb_bank_re='0;tb_bank_we='0;tb_bank_wdata='0;
    sram_negative_active=0;sram_negative_cycles=0;
    active_case_name="startup";expect_reset_quiet=0;reset_cases_passed=0;
    queue_observation_active=1;check_queue_update=0;queue_head_wraps=0;queue_tail_wraps=0;queue_full_replacements=0;
    paired_c_writes=0;gemv_c_pause_events=0;bank_direction_switches=0;
    paired_queue_wraps=0;prefetch_write_conflicts=0;response_write_overlap=0;previous_bank_re='0;previous_bank_we='0;
    a_region_rows=0;b_region_rows=0;c_region_rows=0;b_base=0;c_base=0;
    a_used_words=0;b_used_words=0;c_used_words=0;a_reads=0;b_reads=0;c_writes=0;mac_compute_cycles=0;
    output_stall_cycles=0;output_queue_peak=0;c_write_cycles=0;b_overlap_cycles=0;b_bank_switches=0;
    previous_b_bank=0;seen_gemm_issue=0;seed_state=RandomSeed;void'($urandom(seed_state));
    repeat(5) @(negedge clk_i);rst_ni=1;clk_delay(1);
    $display("Running %0d matrix cases and %0d reset/restart tests (seed 0x%08h)...",NumCases,NumResetCases,RandomSeed);
    foreach(test_cases[c]) begin
      real ops,peak,compute_slots,total_slots;
      active_case_name=test_cases[c].name;
      M_i=SizeAddrWidth'(test_cases[c].M);K_i=SizeAddrWidth'(test_cases[c].K);N_i=SizeAddrWidth'(test_cases[c].N);
      initialize_memories(c,test_cases[c].M,test_cases[c].K,test_cases[c].N);
      coverage.sample();
      gemm_golden(test_cases[c].M,test_cases[c].K,test_cases[c].N);clk_delay(1);
      start_and_wait_gemm(case_cycles[c],case_energy[c],timeout_cycles(test_cases[c].M,test_cases[c].K,test_cases[c].N));
      case_a_reads[c]=a_reads;case_b_reads[c]=b_reads;case_c_writes[c]=c_writes;
      case_output_stalls[c]=output_stall_cycles;case_queue_peak[c]=output_queue_peak;case_write_cycles[c]=c_write_cycles;
      case_b_overlap_cycles[c]=b_overlap_cycles;case_b_bank_switches[c]=b_bank_switches;
      case_sram_energy[c]=2.0*real'(BankDataWidth)*(real'(a_reads)+real'(b_reads)+real'(c_writes));
      verify_result_c(case_errors[c]);
      peak=real'(NumCores)*real'(NumInputs);ops=real'(test_cases[c].M)*real'(test_cases[c].K)*real'(test_cases[c].N);
      compute_slots=real'(mac_compute_cycles)*peak;total_slots=real'(case_cycles[c])*peak;
      case_spatial_util[c]=(compute_slots>0)?100.0*ops/compute_slots:0;
      case_temporal_util[c]=(case_cycles[c]>0)?100.0*real'(mac_compute_cycles)/real'(case_cycles[c]):0;
      case_total_util[c]=(total_slots>0)?100.0*ops/total_slots:0;
      if(case_errors[c]!=0) $fatal(1,"[%0s] Matrix/check failure.",active_case_name);
      repeat(2) begin clk_delay(1);if(done!==1'b0) $fatal(1,"done must pulse for one cycle.");end
      monitor_active=0;case_finished=0;clk_delay(8);
    end
    queue_observation_active=0;
    if(queue_head_wraps==0 || queue_tail_wraps==0 || gemv_c_pause_events==0 || bank_direction_switches==0)
      $fatal(1,"Banked regression missed a required queue/arbitration transition.");
    if(OutputQueueDepth>1 && paired_c_writes==0) $fatal(1,"No paired GeMV write observed.");
    run_reset_tests();
    if(reset_cases_passed!=NumResetCases) $fatal(1,"Incomplete reset regression.");
    run_sram_coverage_exercise();
    print_report();
    $display("All %0d matrix cases and %0d reset/restart tests passed!",NumCases,reset_cases_passed);
    $finish;
  end
endmodule
