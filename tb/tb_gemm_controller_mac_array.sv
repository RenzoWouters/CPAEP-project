//--------------------------
// GEMM / GeMV Controller MAC Array Testbench
// - Unit test to check counter stepping, GeMV vs GeMM mode,
//   init_save_o, result_valid_o, and FSM handshake signals.
//--------------------------

module tb_gemm_controller_mac_array;

  //---------------------------
  // Design Time Parameters
  //---------------------------
  parameter int unsigned AddrWidth = 16;
  parameter int unsigned NumCores  = 16;
  parameter int unsigned NumInputs = 32;

  //---------------------------
  // Wires
  //---------------------------
  logic                   clk_i, rst_ni;
  logic                   start_i;
  logic                   input_valid_i;
  logic                   result_valid_o;
  logic                   mem_re_o;
  logic                   data_valid_o;
  logic                   init_save_o;
  logic                   busy_o;
  logic                   done_o;
  logic                   is_gemv_o;

  logic [AddrWidth-1:0]   M_size_i;
  logic [AddrWidth-1:0]   K_size_i;
  logic [AddrWidth-1:0]   N_size_i;

  logic [AddrWidth-1:0]   M_count_o;
  logic [AddrWidth-1:0]   K_count_o;
  logic [AddrWidth-1:0]   N_count_o;

  // Instantiate DUT
  gemm_controller_mac_array #(
    .AddrWidth     ( AddrWidth     ),
    .NumCores      ( NumCores      ),
    .NumInputs     ( NumInputs     )
  ) i_dut (
    .clk_i         ( clk_i         ),
    .rst_ni        ( rst_ni        ),
    .start_i       ( start_i       ),
    .input_valid_i ( input_valid_i ),
    .result_valid_o( result_valid_o),
    .mem_re_o      ( mem_re_o      ),
    .data_valid_o  ( data_valid_o  ),
    .init_save_o   ( init_save_o   ),
    .busy_o        ( busy_o        ),
    .done_o        ( done_o        ),
    .is_gemv_o     ( is_gemv_o     ),
    .M_size_i      ( M_size_i      ),
    .K_size_i      ( K_size_i      ),
    .N_size_i      ( N_size_i      ),
    .M_count_o     ( M_count_o     ),
    .K_count_o     ( K_count_o     ),
    .N_count_o     ( N_count_o     )
  );

  //---------------------------
  // Tasks and functions
  //---------------------------
  `include "includes/common_tasks.svh"

  task automatic run_case(
    input int unsigned test_id,
    input int unsigned m_sz,
    input int unsigned k_sz,
    input int unsigned n_sz
  );
    int unsigned expected_steps;
    int unsigned active_cycles;
    int unsigned result_pulses;
    int unsigned k_blocks;
    int unsigned m_steps;
    int unsigned n_steps;
    int unsigned guard_cycles;

    k_blocks = (k_sz + NumInputs - 1) / NumInputs;
    if (k_blocks == 0) k_blocks = 1;

    if (n_sz == 1) begin
      m_steps = (m_sz + NumCores - 1) / NumCores;
      if (m_steps == 0) m_steps = 1;
      n_steps = 1;
    end else begin
      m_steps = (m_sz == 0) ? 1 : m_sz;
      n_steps = (n_sz + NumCores - 1) / NumCores;
      if (n_steps == 0) n_steps = 1;
    end

    expected_steps = k_blocks * m_steps * n_steps;
    active_cycles  = 0;
    result_pulses  = 0;
    guard_cycles   = 0;

    M_size_i = AddrWidth'(m_sz);
    K_size_i = AddrWidth'(k_sz);
    N_size_i = AddrWidth'(n_sz);

    // Pulse start_i with valid input stream
    start_i       = 1'b1;
    input_valid_i = 1'b1;
    #1;
    if (mem_re_o) begin
      active_cycles++;
    end
    clk_delay(1);
    start_i       = 1'b0;

    while (!done_o) begin
      if (mem_re_o) begin
        active_cycles++;
        if (n_sz == 1) begin
          if (!is_gemv_o) begin
            $display("Error in test %0d: expected is_gemv_o=1 for N=%0d", test_id, n_sz);
            $fatal;
          end
          if (N_count_o != '0) begin
            $display("Error in test %0d: expected N_count_o=0 in GeMV, got %0d", test_id, N_count_o);
            $fatal;
          end
          if ((M_count_o % NumCores) != 0) begin
            $display("Error in test %0d: M_count_o=%0d not multiple of %0d in GeMV", test_id, M_count_o, NumCores);
            $fatal;
          end
        end else begin
          if (is_gemv_o) begin
            $display("Error in test %0d: expected is_gemv_o=0 for N=%0d", test_id, n_sz);
            $fatal;
          end
          if ((N_count_o % NumCores) != 0) begin
            $display("Error in test %0d: N_count_o=%0d not multiple of %0d in GeMM", test_id, N_count_o, NumCores);
            $fatal;
          end
        end

        if ((K_count_o % NumInputs) != 0) begin
          $display("Error in test %0d: K_count_o=%0d not multiple of %0d", test_id, K_count_o, NumInputs);
          $fatal;
        end
      end

      if (result_valid_o) begin
        result_pulses++;
      end

      clk_delay(1);
      guard_cycles++;
      if (guard_cycles > (expected_steps * 4 + 20)) begin
        $display("Error in test %0d: timeout waiting for done_o", test_id);
        $fatal;
      end
    end

    // Sample final cycle's result_valid_o if asserted on done_o cycle
    if (result_valid_o) begin
      result_pulses++;
    end

    if (active_cycles != expected_steps) begin
      $display("Error in test %0d (M=%0d, K=%0d, N=%0d): active_cycles=%0d, expected=%0d",
        test_id, m_sz, k_sz, n_sz, active_cycles, expected_steps);
      $fatal;
    end

    if (result_pulses != (m_steps * n_steps)) begin
      $display("Error in test %0d (M=%0d, K=%0d, N=%0d): result_pulses=%0d, expected=%0d",
        test_id, m_sz, k_sz, n_sz, result_pulses, m_steps * n_steps);
      $fatal;
    end

    input_valid_i = 1'b0;
    clk_delay(2);
    $display("Test %0d passed (M=%0d, K=%0d, N=%0d): active_cycles=%0d, result_pulses=%0d",
      test_id, m_sz, k_sz, n_sz, active_cycles, result_pulses);
  endtask

  //---------------------------
  // Start of testbench
  //---------------------------

  // Clock generation
  initial begin
    clk_i = 1'b0;
    forever #5 clk_i = ~clk_i;  // 100MHz clock
  end

  // Test control
  initial begin
    clk_i         = 1'b0;
    rst_ni        = 1'b0;
    start_i       = 1'b0;
    input_valid_i = 1'b0;
    M_size_i      = '0;
    K_size_i      = '0;
    N_size_i      = '0;

    clk_delay(3);

    // Release reset
    #1;
    rst_ni = 1'b1;
    clk_delay(1);

    // Test 0: GeMV (N=1), M=32 (2 steps of 16), K=64 (2 blocks of 32)
    run_case(0, 32, 64, 1);

    // Test 1: GeMV (N=1) with non-multiples: M=20 (2 steps), K=40 (2 blocks)
    run_case(1, 20, 40, 1);

    // Test 2: GeMM (N=32 -> 2 steps of 16), M=2, K=64 (2 blocks of 32)
    run_case(2, 2, 64, 32);

    // Test 3: GeMM (N=24 -> 2 steps), M=3, K=32 (1 block)
    run_case(3, 3, 32, 24);

    run_case(4, 16, 32, 1);   // GeMV: precies één totale rekenstap
    
    run_case(5, 1, 32, 16);   // GeMM: precies één totale rekenstap

    clk_delay(5);
    $display("All tests passed!");
    $finish;
  end

endmodule