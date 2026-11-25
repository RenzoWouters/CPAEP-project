//-----------------------------------------------------------------------------
// Hierarchical MAC Array: NumCores parallel dot-product cores of NumInputs MACs
// Default: 16 cores x 32 MACs = 512 MACs
//-----------------------------------------------------------------------------

module mac_array #(
  parameter int unsigned InDataWidth  = 8,
  parameter int unsigned NumCores     = 16,
  parameter int unsigned NumInputs    = 32,
  parameter int unsigned OutDataWidth = 32,
  parameter int unsigned NumContexts  = 1
) (
  input  logic                                                      clk_i,
  input  logic                                                      rst_ni,
  input  logic signed [NumCores-1:0][NumInputs-1:0][InDataWidth-1:0]  a_i,
  input  logic signed [NumCores-1:0][NumInputs-1:0][InDataWidth-1:0]  b_i,
  input  logic                                                      a_valid_i,
  input  logic                                                      b_valid_i,
  input  logic                                                      init_save_i,
  input  logic                                                      acc_clr_i,
  input  logic [((NumContexts <= 1) ? 1 : $clog2(NumContexts))-1:0]     acc_context_i,
  output logic signed [NumCores-1:0][OutDataWidth-1:0]              c_o
);

  genvar p;
  generate
    for (p = 0; p < NumCores; p++) begin : gen_mac_cores
      if (NumContexts == 1) begin : gen_supplied_pe
        // Duplicate the supplied PE without modifying its interface or source.
        general_mac_pe #(
          .InDataWidth(InDataWidth), .NumInputs(NumInputs),
          .OutDataWidth(OutDataWidth)
        ) i_mac_pe (
          .clk_i(clk_i), .rst_ni(rst_ni), .a_i(a_i[p]), .b_i(b_i[p]),
          .a_valid_i(a_valid_i), .b_valid_i(b_valid_i),
          .init_save_i(init_save_i), .acc_clr_i(acc_clr_i), .c_o(c_o[p])
        );
      end else begin : gen_context_pe
        // MReuse uses an explicitly separate array-specific PE.
        mac_pe_context #(
          .InDataWidth(InDataWidth), .NumInputs(NumInputs),
          .OutDataWidth(OutDataWidth), .NumContexts(NumContexts)
        ) i_mac_pe (
          .clk_i(clk_i), .rst_ni(rst_ni), .a_i(a_i[p]), .b_i(b_i[p]),
          .a_valid_i(a_valid_i), .b_valid_i(b_valid_i),
          .init_save_i(init_save_i), .acc_clr_i(acc_clr_i),
          .acc_context_i(acc_context_i), .c_o(c_o[p])
        );
      end
    end
  endgenerate

endmodule
