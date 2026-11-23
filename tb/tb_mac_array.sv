//-----------------------------------------------------------------------------
// MAC Array Testbench
// - Unit test to check functionality of the hierarchical 16x32 MAC array
//-----------------------------------------------------------------------------

module tb_mac_array;

  //---------------------------
  // Design Time Parameters
  //---------------------------
  parameter int unsigned InDataWidth  = 8;
  parameter int unsigned NumCores     = 16;
  parameter int unsigned NumInputs    = 32;
  parameter int unsigned OutDataWidth = 32;

  //---------------------------
  // Test Parameters
  //---------------------------
  parameter int unsigned NumTests = 10;
  parameter int unsigned KSteps   = 4;   // aantal 32-brede K-stappen per accumulatie

  //---------------------------
  // Wires
  //---------------------------
  logic clk_i, rst_ni;

  logic signed [NumCores-1:0][NumInputs-1:0][InDataWidth-1:0] a_i, b_i;
  logic a_valid_i, b_valid_i;
  logic init_save_i;
  logic acc_clr_i;

  logic signed [NumCores-1:0][OutDataWidth-1:0] c_o;
  logic signed [NumCores-1:0][OutDataWidth-1:0] golden_c_o;

  // Instantiate the MAC Array module (16 cores x 32 inputs = 512 MACs)
  mac_array #(
    .InDataWidth  ( InDataWidth  ),
    .NumCores     ( NumCores     ),
    .NumInputs    ( NumInputs    ),
    .OutDataWidth ( OutDataWidth )
  ) i_dut (
    .clk_i        ( clk_i        ),
    .rst_ni       ( rst_ni       ),
    .a_i          ( a_i          ),
    .b_i          ( b_i          ),
    .a_valid_i    ( a_valid_i    ),
    .b_valid_i    ( b_valid_i    ),
    .init_save_i  ( init_save_i  ),
    .acc_clr_i    ( acc_clr_i    ),
    .c_o          ( c_o          )
  );

  //---------------------------
  // Tasks and functions
  //---------------------------
  `include "includes/common_tasks.svh"

  function automatic logic signed [OutDataWidth-1:0] dot_product_32(
    input logic signed [NumInputs-1:0][InDataWidth-1:0] A_vec,
    input logic signed [NumInputs-1:0][InDataWidth-1:0] B_vec
  );
    logic signed [OutDataWidth-1:0] acc;
    acc = '0;
    for (int j = 0; j < NumInputs; j++) begin
      acc += $signed(A_vec[j]) * $signed(B_vec[j]);
    end
    return acc;
  endfunction

  //---------------------------
  // Start of testbench
  //---------------------------

  // Clock generation (100 MHz)
  initial begin
    clk_i = 1'b0;
    forever #5 clk_i = ~clk_i;
  end

  // Test control
  initial begin
    clk_i       = 1'b0;
    rst_ni      = 1'b0;
    a_valid_i   = 1'b0;
    b_valid_i   = 1'b0;
    init_save_i = 1'b0;
    acc_clr_i   = 1'b0;

    for (int p = 0; p < NumCores; p++) begin
      golden_c_o[p] = '0;
      for (int j = 0; j < NumInputs; j++) begin
        a_i[p][j] = '0;
        b_i[p][j] = '0;
      end
    end

    clk_delay(3);

    // Release reset
    #1;
    rst_ni = 1'b1;
    clk_delay(1);

    // Deel 1: Single-cycle dot-product per core (init_save_i = 1)
    for (int t = 0; t < NumTests; t++) begin
      for (int p = 0; p < NumCores; p++) begin
        for (int j = 0; j < NumInputs; j++) begin
          a_i[p][j] = $urandom();
          b_i[p][j] = $urandom();
        end
        golden_c_o[p] = dot_product_32(a_i[p], b_i[p]);
      end

      a_valid_i   = 1'b1;
      b_valid_i   = 1'b1;
      init_save_i = 1'b1;
      clk_delay(1);

      for (int p = 0; p < NumCores; p++) begin
        if (golden_c_o[p] !== c_o[p]) begin
          $display("Error in single-cycle test %0d, core %0d", t, p);
          $display("OUT: %0d, GOLDEN: %0d", $signed(c_o[p]), $signed(golden_c_o[p]));
          $fatal;
        end
      end
      $display("Single-cycle test %0d passed (all %0d cores).", t, NumCores);
    end

    // Deel 2: Multi-step accumulatie langs K (KSteps x 32 elementen)
    for (int t = 0; t < NumTests; t++) begin
      for (int p = 0; p < NumCores; p++) begin
        golden_c_o[p] = '0;
      end

      for (int k = 0; k < KSteps; k++) begin
        for (int p = 0; p < NumCores; p++) begin
          for (int j = 0; j < NumInputs; j++) begin
            a_i[p][j] = $urandom();
            b_i[p][j] = $urandom();
          end

          if (k == 0) begin
            golden_c_o[p] = dot_product_32(a_i[p], b_i[p]);
          end else begin
            golden_c_o[p] += dot_product_32(a_i[p], b_i[p]);
          end
        end

        a_valid_i   = 1'b1;
        b_valid_i   = 1'b1;
        init_save_i = (k == 0) ? 1'b1 : 1'b0;
        clk_delay(1);
      end

      // Stop geldige invoer na de laatste K-stap
      a_valid_i   = 1'b0;
      b_valid_i   = 1'b0;
      init_save_i = 1'b0;

      for (int p = 0; p < NumCores; p++) begin
        if (golden_c_o[p] !== c_o[p]) begin
          $display("Error in multi-step test %0d, core %0d", t, p);
          $display("OUT: %0d, GOLDEN: %0d", $signed(c_o[p]), $signed(golden_c_o[p]));
          $fatal;
        end
      end
      $display("Multi-step (%0d x %0d) test %0d passed (all %0d cores).",
               KSteps, NumInputs, t, NumCores);
    end

    clk_delay(5);
    $display("All MAC array tests passed!");
    $finish;
  end

endmodule