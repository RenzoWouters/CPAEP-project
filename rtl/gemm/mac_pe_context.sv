//-----------------------------------------------------------------------------
// Array-specific dot-product PE with independent accumulator contexts.
//
// This is the MReuse extension of the supplied MAC arithmetic. The original
// general_mac_pe.sv and its baseline users/tests remain unchanged.
// All contexts share NumInputs products, one reduction and one accumulator
// adder. Only accumulator storage and selection grow with NumContexts.
// c_o exposes the last completed result, independent of the incoming selector.
//-----------------------------------------------------------------------------
module mac_pe_context #(
  parameter int unsigned InDataWidth  = 8,
  parameter int unsigned NumInputs    = 32,
  parameter int unsigned OutDataWidth = 32,
  parameter int unsigned NumContexts  = 4
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic signed [NumInputs-1:0][InDataWidth-1:0] a_i,
  input  logic signed [NumInputs-1:0][InDataWidth-1:0] b_i,
  input  logic a_valid_i,
  input  logic b_valid_i,
  input  logic init_save_i,
  input  logic acc_clr_i,
  input  logic [((NumContexts <= 1) ? 1 : $clog2(NumContexts))-1:0] acc_context_i,
  output logic signed [OutDataWidth-1:0] c_o
);
  localparam int unsigned ContextWidth = (NumContexts <= 1) ? 1 : $clog2(NumContexts);
  logic acc_valid;
  logic signed [OutDataWidth-1:0] mult_result;
  logic signed [NumContexts-1:0][OutDataWidth-1:0] accumulator_q;
  logic [ContextWidth-1:0] completed_context_q;

  assign acc_valid = a_valid_i && b_valid_i;
  always_comb begin
    mult_result = '0;
    for (int i=0; i<NumInputs; i++)
      mult_result += $signed(a_i[i]) * $signed(b_i[i]);
  end

  // Separate input and completed-result indices support consecutive rows.
  // The top captures the preceding result before the next edge's NBA update.
  assign c_o = $signed(accumulator_q[completed_context_q]);
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      accumulator_q <= '0;
      completed_context_q <= '0;
    end else if (init_save_i) begin
      accumulator_q[acc_context_i] <= mult_result;
      completed_context_q <= acc_context_i;
    end else if (acc_valid) begin
      accumulator_q[acc_context_i] <=
        $signed(accumulator_q[acc_context_i]) + mult_result;
      completed_context_q <= acc_context_i;
    end else if (acc_clr_i) begin
      accumulator_q <= '0;
      completed_context_q <= '0;
    end
  end

  // Same normalized arithmetic and value-change register-energy convention as
  // the supplied PE, with the actual context-bank storage included once.
  // synthesis translate_off
  localparam int unsigned AccumulatorBits = NumContexts*OutDataWidth + ContextWidth;
  initial begin
    #0;
    if (NumContexts < 2)
      $fatal(1, "Use the supplied general_mac_pe for a single accumulator context.");
    metrics_pkg::area += (3*NumInputs) + real'(AccumulatorBits)/32.0;
    $display("%m added %0.3f to area", (3*NumInputs) + real'(AccumulatorBits)/32.0);
  end
  for (genvar i=0; i<NumInputs; i++) begin : gen_arithmetic_energy
    always @(a_i[i] or b_i[i]) metrics_pkg::energy += 2 + 1;
  end
  for (genvar ctx=0; ctx<NumContexts; ctx++) begin : gen_context_energy
    always @(accumulator_q[ctx]) metrics_pkg::energy += real'(OutDataWidth)/32.0;
  end
  always @(completed_context_q) metrics_pkg::energy += real'(ContextWidth)/32.0;
  always @(posedge clk_i) begin
    if (rst_ni && (init_save_i || acc_valid) &&
        ($isunknown(acc_context_i) || int'(acc_context_i) >= NumContexts))
      $fatal(1, "%m: invalid accumulator context %0d", acc_context_i);
  end
  // synthesis translate_on
endmodule
