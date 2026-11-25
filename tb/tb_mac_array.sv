//-----------------------------------------------------------------------------
// MAC Array Testbench
// - Unit test to check functionality of the hierarchical 16x32 MAC array
//-----------------------------------------------------------------------------

module tb_mac_array;

  timeunit 1ns;
  timeprecision 1ps;

  //---------------------------
  // Design Time Parameters
  //---------------------------
  parameter int unsigned InDataWidth  = 8;
  parameter int unsigned NumCores     = 16;
  parameter int unsigned NumInputs    = 32;
  parameter int unsigned OutDataWidth = 32;
  // Run with NumContexts=1 for the original PE, and 4/8 for row reuse.
  parameter int unsigned NumContexts  = 1;
  localparam int unsigned ContextWidth = (NumContexts <= 1) ? 1 : $clog2(NumContexts);

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
  logic [ContextWidth-1:0] acc_context_i;

  logic signed [NumCores-1:0][OutDataWidth-1:0] c_o;
  logic signed [NumCores-1:0][OutDataWidth-1:0] golden_c_o;
  logic signed [NumCores-1:0][OutDataWidth-1:0] context_golden [NumContexts];
  logic signed [NumCores-1:0][OutDataWidth-1:0] held_result;

  // Instantiate the MAC Array module (16 cores x 32 inputs = 512 MACs)
  mac_array #(
    .InDataWidth  ( InDataWidth  ),
    .NumCores     ( NumCores     ),
    .NumInputs    ( NumInputs    ),
    .OutDataWidth ( OutDataWidth ),
    .NumContexts  ( NumContexts  )
  ) i_dut (
    .clk_i        ( clk_i        ),
    .rst_ni       ( rst_ni       ),
    .a_i          ( a_i          ),
    .b_i          ( b_i          ),
    .a_valid_i    ( a_valid_i    ),
    .b_valid_i    ( b_valid_i    ),
    .init_save_i  ( init_save_i  ),
    .acc_clr_i    ( acc_clr_i    ),
    .acc_context_i( acc_context_i),
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

  // A distinct signed operand pattern per core, lane, row and K block catches
  // accidental context aliasing. Consecutive calls accept one block per clock.
  task automatic context_block(
    input int unsigned context_id,
    input bit first_block,
    input int unsigned step_id
  );
    @(negedge clk_i);
    acc_context_i = ContextWidth'(context_id);
    acc_clr_i = 1'b0;
    a_valid_i = 1'b1;
    b_valid_i = 1'b1;
    init_save_i = first_block;
    for (int p = 0; p < NumCores; p++) begin
      for (int j = 0; j < NumInputs; j++) begin
        a_i[p][j] = InDataWidth'(((context_id+1)*13 + (p+1)*7 + j*3 + step_id*5) % 255 - 127);
        b_i[p][j] = InDataWidth'(((context_id+1)*11 + (p+1)*3 + j*7 + step_id*9) % 255 - 127);
      end
      if (first_block)
        context_golden[context_id][p] = dot_product_32(a_i[p], b_i[p]);
      else
        context_golden[context_id][p] = $signed(context_golden[context_id][p]) +
                                         dot_product_32(a_i[p], b_i[p]);
    end
    @(posedge clk_i);
    #1;
    for (int p = 0; p < NumCores; p++) begin
      if (c_o[p] !== context_golden[context_id][p])
        $fatal(1, "Context %0d, step %0d, core %0d: got %0d, expected %0d",
               context_id, step_id, p, $signed(c_o[p]),
               $signed(context_golden[context_id][p]));
    end
  endtask

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
    rst_ni      = 1'b0;
    a_valid_i   = 1'b0;
    b_valid_i   = 1'b0;
    init_save_i = 1'b0;
    acc_clr_i   = 1'b0;
    acc_context_i = '0;
    for (int ctx = 0; ctx < NumContexts; ctx++) context_golden[ctx] = '0;

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

    // Interleave rows, reversing their order on alternate K blocks. The output
    // must follow the last COMPLETED block, not the current input row selector.
    for (int step = 0; step < KSteps; step++) begin
      for (int row = 0; row < NumContexts; row++) begin
        context_block((step % 2 == 0) ? row : NumContexts-1-row,
                      step == 0, step);
      end
    end

    @(negedge clk_i);
    held_result = c_o;
    a_valid_i = 1'b0;
    b_valid_i = 1'b0;
    init_save_i = 1'b0;
    acc_context_i = ContextWidth'((int'(acc_context_i)+1) % NumContexts);
    #1;
    if (c_o !== held_result)
      $fatal(1, "Changing the input context changed the last completed result.");
    clk_delay(1);
    if (c_o !== held_result) $fatal(1, "Invalid block changed an accumulator.");

    // Each mismatched valid combination must also hold all accumulated data.
    for (int which = 0; which < 2; which++) begin
      @(negedge clk_i);
      a_valid_i = (which == 0);
      b_valid_i = (which == 1);
      clk_delay(1);
      if (c_o !== held_result) $fatal(1, "Mismatched valid inputs changed a result.");
    end

    // Clear every context and prove it by accumulating without init_save.
    @(negedge clk_i);
    a_valid_i = 1'b0;
    b_valid_i = 1'b0;
    acc_clr_i = 1'b1;
    clk_delay(1);
    if (c_o !== '0) $fatal(1, "acc_clr_i did not clear the result.");
    for (int ctx = 0; ctx < NumContexts; ctx++) context_golden[ctx] = '0;
    for (int ctx = 0; ctx < NumContexts; ctx++) context_block(ctx, 1'b0, 51);

    // Initializing one row must not destroy the remaining row contexts.
    context_block(0, 1'b1, 97);
    for (int ctx = 0; ctx < NumContexts; ctx++) context_block(ctx, 1'b0, 101);

    // Asynchronous reset with live contexts, followed by fresh accumulation.
    @(negedge clk_i);
    a_valid_i = 1'b0;
    b_valid_i = 1'b0;
    init_save_i = 1'b0;
    rst_ni = 1'b0;
    #1;
    if (c_o !== '0) $fatal(1, "Asynchronous reset did not clear the result.");
    for (int ctx = 0; ctx < NumContexts; ctx++) context_golden[ctx] = '0;
    clk_delay(2);
    @(negedge clk_i);
    rst_ni = 1'b1;
    for (int ctx = 0; ctx < NumContexts; ctx++) context_block(ctx, 1'b0, 113);
    @(negedge clk_i);
    a_valid_i = 1'b0;
    b_valid_i = 1'b0;
    init_save_i = 1'b0;
    clk_delay(2);
    $display("GLOBAL PASS: MAC array arithmetic, %0d contexts, clear and reset passed.",
             NumContexts);
    $finish;
  end

endmodule
