//-----------------------------------------------------------------------------
// Controller unit test for the USER'S original interface, with the Idle fix:
//   next_state = last_counter_last_value ? ControllerDrain : ControllerBusy;
//
// This testbench does NOT use the alternative input_ready/result_done interface.
// Keep your current gemm_controller_mac_array.sv and ceiling_counter.sv.
//
// What is checked:
//   - every M/K/N index, using an independent linear request-number reference;
//   - the first request in Idle, ordinary requests, and no extra requests;
//   - data_valid/init_save/result_valid timing, including cycles without reads;
//   - input stalls at start, inside K, and before final K/global steps;
//   - busy/done timing, one-cycle done, and return to Idle;
//   - reset, X/Z detection, rounding boundaries, and all six required workloads;
//   - all cases execute consecutively without resetting the DUT between cases.
//
// Sampling: drive on negedge; inspect requests/counters at posedge BEFORE NBA;
// inspect the updated registers #1 after that edge. No hierarchical DUT probes.
// This tests controller scheduling, not SRAM bandwidth or matrix arithmetic.
//-----------------------------------------------------------------------------

module tb_gemm_controller_mac_array;
  timeunit 1ns;
  timeprecision 1ps;

  parameter int unsigned AddrWidth = 16;
  parameter int unsigned NumCores  = 16;
  parameter int unsigned NumInputs = 32;

  logic clk_i, rst_ni, start_i, input_valid_i;
  logic result_valid_o, mem_re_o, data_valid_o, init_save_o;
  logic busy_o, done_o, is_gemv_o;
  logic [AddrWidth-1:0] M_size_i, K_size_i, N_size_i;
  logic [AddrWidth-1:0] M_count_o, K_count_o, N_count_o;
  int unsigned passed_cases = 0;

  gemm_controller_mac_array #(
    .AddrWidth(AddrWidth), .NumCores(NumCores), .NumInputs(NumInputs)
  ) i_dut (
    .clk_i(clk_i), .rst_ni(rst_ni), .start_i(start_i),
    .input_valid_i(input_valid_i),
    .result_valid_o(result_valid_o), .mem_re_o(mem_re_o),
    .data_valid_o(data_valid_o), .init_save_o(init_save_o),
    .busy_o(busy_o), .done_o(done_o), .is_gemv_o(is_gemv_o),
    .M_size_i(M_size_i), .K_size_i(K_size_i), .N_size_i(N_size_i),
    .M_count_o(M_count_o), .K_count_o(K_count_o), .N_count_o(N_count_o)
  );

  `include "includes/common_tasks.svh"

  //---------------------------
  // Reusable check functions
  //---------------------------
  task automatic check_bit(
    input int unsigned test_id,
    input int unsigned cycle,
    input string label,
    input logic actual,
    input logic expected
  );
    if (actual !== expected)
      $fatal(1, "Test %0d, cycle %0d: %s=%b, expected=%b",
             test_id, cycle, label, actual, expected);
  endtask

  task automatic check_index(
    input int unsigned test_id,
    input int unsigned cycle,
    input string label,
    input logic [AddrWidth-1:0] actual,
    input int unsigned expected
  );
    if (actual !== AddrWidth'(expected))
      $fatal(1, "Test %0d, cycle %0d: %s=%0d, expected=%0d",
             test_id, cycle, label, actual, expected);
  endtask

  task automatic check_idle(input string phase);
    if ({busy_o, done_o, mem_re_o, data_valid_o, init_save_o,
         result_valid_o} !== 6'b000000)
      $fatal(1, "%s: controller/pipeline is not fully idle", phase);
    if ((M_count_o !== '0) || (K_count_o !== '0) || (N_count_o !== '0))
      $fatal(1, "%s: expected zero counters, got M/K/N=%0d/%0d/%0d",
             phase, M_count_o, K_count_o, N_count_o);
    // is_gemv_o follows N_size_i combinationally; it need not be zero in Idle.
  endtask

  // Independent reference: derive the full loop position from request number.
  task automatic reference_indices(
    input int unsigned request_index,
    input int unsigned k_blocks,
    input int unsigned n_steps,
    input bit gemv,
    output int unsigned m_ref,
    output int unsigned k_ref,
    output int unsigned n_ref
  );
    int unsigned tile_ref;
    k_ref = (request_index % k_blocks) * NumInputs;
    tile_ref = request_index / k_blocks;
    m_ref = gemv ? tile_ref * NumCores : tile_ref / n_steps;
    n_ref = gemv ? 0 : (tile_ref % n_steps) * NumCores;
  endtask

  //---------------------------
  // One complete test case
  //---------------------------
  // Optional arguments:
  //   start_stalls: hold input_valid low for this many clocks from start.
  //   stall_before_request: ZERO-BASED request number before which to pause.
  //   stall_cycles: length of that one directed pause (0 disables it).
  // Existing run_case(id, M, K, N) calls remain valid.
  task automatic run_case(
    input int unsigned test_id,
    input int unsigned m_sz,
    input int unsigned k_sz,
    input int unsigned n_sz,
    input int unsigned start_stalls = 0,
    input int unsigned stall_before_request = 32'hffffffff,
    input int unsigned stall_cycles = 0
  );
    int unsigned k_blocks, m_steps, n_steps, expected_steps;
    int unsigned request_index, result_cycles, cycle, max_cycles;
    int unsigned m_ref, k_ref, n_ref, next_reference_index;
    int unsigned stall_remaining;
    bit stall_started, gemv;
    bit expected_request, expected_first, expected_last_k, expected_final;
    bit previous_last_k, previous_final;
    logic request_sample;

    if ((m_sz == 0) || (k_sz == 0) || (n_sz == 0))
      $fatal(1, "This test suite uses positive M/K/N only");
    if ((AddrWidth'(m_sz) != m_sz) || (AddrWidth'(k_sz) != k_sz) ||
        (AddrWidth'(n_sz) != n_sz))
      $fatal(1, "Test %0d: dimensions do not fit AddrWidth=%0d", test_id, AddrWidth);

    gemv = (n_sz == 1);
    k_blocks = k_sz / NumInputs + ((k_sz % NumInputs) != 0);
    m_steps = gemv ? m_sz / NumCores + ((m_sz % NumCores) != 0) : m_sz;
    n_steps = gemv ? 1 : n_sz / NumCores + ((n_sz % NumCores) != 0);
    expected_steps = m_steps * n_steps * k_blocks;
    max_cycles = expected_steps + start_stalls + stall_cycles + 20;

    if ((stall_cycles != 0) && (stall_before_request >= expected_steps))
      $fatal(1, "Test %0d: requested stall lies beyond the last request", test_id);

    request_index = 0;
    result_cycles = 0;
    cycle = 0;
    stall_remaining = 0;
    stall_started = 0;
    previous_last_k = 0;
    previous_final = 0;

    check_idle("before start");

    forever begin
      // Drive half a cycle before the DUT samples the signals.
      @(negedge clk_i);
      if (cycle == 0) begin
        M_size_i = AddrWidth'(m_sz);
        K_size_i = AddrWidth'(k_sz);
        N_size_i = AddrWidth'(n_sz);
      end
      start_i = (cycle == 0);

      if (cycle < start_stalls) begin
        input_valid_i = 1'b0;
      end else begin
        if (!stall_started && (stall_cycles != 0) &&
            (request_index == stall_before_request)) begin
          stall_started = 1;
          stall_remaining = stall_cycles;
        end
        if (stall_remaining != 0) begin
          input_valid_i = 1'b0;
          stall_remaining--;
        end else begin
          // Keep valid high even after the expected last request. The FSM must
          // stop requests itself; this catches the old one-step Idle bug.
          input_valid_i = 1'b1;
        end
      end

      // This expectation depends on our stimulus and reference, not DUT flags.
      expected_request = (request_index < expected_steps) && input_valid_i;
      expected_first = expected_request && ((request_index % k_blocks) == 0);
      expected_last_k = expected_request &&
                        ((request_index % k_blocks) == (k_blocks - 1));
      expected_final = expected_request && (request_index == expected_steps - 1);

      // Sample the request and current indices BEFORE the DUT's NBA updates.
      @(posedge clk_i);
      request_sample = mem_re_o;
      check_bit(test_id, cycle, "mem_re_o", request_sample, expected_request);
      check_bit(test_id, cycle, "is_gemv_o", is_gemv_o, gemv);

      next_reference_index = (request_index == expected_steps) ? 0 : request_index;
      reference_indices(next_reference_index, k_blocks, n_steps, gemv,
                        m_ref, k_ref, n_ref);
      check_index(test_id, cycle, "M_count_o before edge", M_count_o, m_ref);
      check_index(test_id, cycle, "K_count_o before edge", K_count_o, k_ref);
      check_index(test_id, cycle, "N_count_o before edge", N_count_o, n_ref);

      // Now inspect the NEW pipeline/state values after nonblocking updates.
      #1;
      check_bit(test_id, cycle, "data_valid_o", data_valid_o, expected_request);
      check_bit(test_id, cycle, "init_save_o", init_save_o, expected_first);
      check_bit(test_id, cycle, "result_valid_o", result_valid_o, previous_last_k);
      check_bit(test_id, cycle, "done_o", done_o, previous_final);
      // Busy is high after the start edge, including Drain and Finish.
      // Before that first edge a read may occur while busy is still low.
      check_bit(test_id, cycle, "busy_o", busy_o, 1'b1);

      if (expected_request) request_index++;
      if (result_valid_o) result_cycles++;

      // Also check the indices AFTER the edge. Stalls must preserve them.
      // After the final request the nested counters wrap back to zero.
      next_reference_index = (request_index == expected_steps) ? 0 : request_index;
      reference_indices(next_reference_index, k_blocks, n_steps, gemv,
                        m_ref, k_ref, n_ref);
      check_index(test_id, cycle, "M_count_o after edge", M_count_o, m_ref);
      check_index(test_id, cycle, "K_count_o after edge", K_count_o, k_ref);
      check_index(test_id, cycle, "N_count_o after edge", N_count_o, n_ref);

      // The expected completion cycle is derived from the final request.
      if (previous_final) break;

      // Update history only AFTER the pipeline checks above.
      previous_last_k = expected_last_k;
      previous_final = expected_final;
      cycle++;
      if (cycle > max_cycles)
        $fatal(1, "Test %0d: timeout after %0d cycles", test_id, cycle);
    end

    if ((request_index != expected_steps) || (result_cycles != m_steps * n_steps))
      $fatal(1, "Test %0d: counts are reads=%0d/%0d, results=%0d/%0d",
             test_id, request_index, expected_steps, result_cycles, m_steps*n_steps);
    if ((stall_cycles != 0) && !stall_started)
      $fatal(1, "Test %0d: directed stall was not exercised", test_id);

    // Verify done lasts exactly one cycle, the pipeline clears and the DUT can
    // start the next case WITHOUT reset. Keep one extra idle clock as a check.
    @(negedge clk_i);
    start_i = 1'b0;
    input_valid_i = 1'b0;
    clk_delay(1);
    check_idle("one cycle after done");
    clk_delay(1);
    check_idle("second idle cycle");

    passed_cases++;
    $display("PASS test %0d: M/K/N=%0d/%0d/%0d, reads=%0d, result cycles=%0d, clocks=%0d",
             test_id, m_sz, k_sz, n_sz, request_index, result_cycles, cycle+1);
  endtask

  //---------------------------
  // Clock and stimulus
  //---------------------------
  initial begin
    clk_i = 1'b0;
    forever #5 clk_i = ~clk_i;  // 100 MHz with the explicit timeunit above.
  end

  initial begin
    rst_ni = 1'b0;
    start_i = 1'b0;
    input_valid_i = 1'b0;
    M_size_i = '0;
    K_size_i = '0;
    N_size_i = '0;

    clk_delay(3);
    check_idle("during reset");
    @(negedge clk_i);
    rst_ni = 1'b1;
    clk_delay(1);
    check_idle("after reset release");

    // Your original four cases and the two one-step regressions.
    run_case( 0, 32, 64,  1);
    run_case( 1, 20, 40,  1);
    run_case( 2,  2, 64, 32);
    run_case( 3,  3, 32, 24);
    run_case( 4, 16, 32,  1);
    run_case( 5,  1, 32, 16);

    // Boundaries around K=32 (K=32 is already covered above).
    run_case( 6, 16, 31,  1);
    run_case( 7, 16, 33,  1);
    // Boundaries around M=16 in GeMV and N=16 in GeMM.
    run_case( 8, 15, 64,  1);
    run_case( 9, 16, 64,  1);
    run_case(10, 17, 64,  1);
    run_case(11,  1, 64, 15);
    run_case(12,  1, 64, 16);
    run_case(13,  1, 64, 17);
    run_case(14,  1,  1,  1);

    // Arguments after N: start_stalls, stall_before_request, stall_cycles.
    run_case(15, 16, 32,  1, 3);        // Wait three clocks from start.
    run_case(16, 32, 96,  1, 0, 1, 3); // Stall inside first K reduction.
    run_case(17, 32, 96,  1, 0, 2, 3); // Stall before first tile's last K step.
    run_case(18,  2, 64, 32, 0, 7, 3); // Stall before the very last request.
    run_case(19,  1, 32, 16, 3);        // One-step GeMM after waiting at start.
    run_case(20,  3, 32, 24, 0, 3, 2); // Interrupt consecutive valid results.

    // All six required workloads, consecutively, still with no DUT reset.
    run_case(21,   64,   64, 32);
    run_case(22,   64,  128, 32);
    run_case(23,   64,  576, 32);
    run_case(24, 1536,  576,  1);
    run_case(25, 3072, 1024,  1);
    run_case(26, 2816, 2048,  1);

    $display("GLOBAL PASS: all %0d controller cases passed.", passed_cases);
    $finish;
  end
endmodule