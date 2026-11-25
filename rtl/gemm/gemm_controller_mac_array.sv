//---------------------------
// The GeMM / GeMV Controller Module for MAC Array (16 cores x 32 inputs = 512 MACs)
// File: gemm_controller_mac_array.sv
//
// Description:
// Controller for a 16x32 MAC array:
// - Each K step advances by NumInputs (32) elements along K.
// - When N_size_i == 1 (GeMV), NumCores (16) cores process 16 rows of M
//   in parallel. The existing NumContexts accumulator contexts cover
//   NumContexts*NumCores rows (512 by default). Loop order is M group ->
//   K block -> row-group context, so a small B fragment is reused over rows.
// - When N_size_i > 1 (GeMM), the cores process GemmRows M rows and
//   NumCores/GemmRows N columns in parallel (2 M x 8 N by default).
//   MReuse is the number of M ROWS sharing a B tile, not the number of
//   accumulator contexts. MReuse=64 and GemmRows=2 require 32 contexts.
//   The loop order is M group -> N group -> K block -> row-pair context.
//   M_count_o is the base M row of the accepted row pair.
// - GemmRows=1, MReuse=1 preserves the original controller's sequence.
//
// input_valid_i accepts ONE COMPLETE operand block, already loaded by the top.
// mem_re_o is the accepted-block pulse, not a physical SRAM read enable.
// data_valid_o/init_save_o follow acceptance by one register stage;
// result_valid_o follows a last-K acceptance by two stages. The top captures
// the registered MAC result and serializes it through the physical C port.
//---------------------------

module gemm_controller_mac_array #(
  parameter int unsigned AddrWidth = 16,
  parameter int unsigned NumCores  = 16,
  parameter int unsigned NumInputs = 32,
  parameter int unsigned MReuse    = 64,
  parameter int unsigned GemmRows  = 2
)(
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   start_i,
  input  logic                   input_valid_i,
  output logic                   result_valid_o,
  output logic                   mem_re_o,
  output logic                   data_valid_o,
  output logic                   init_save_o,
  output logic                   busy_o,
  output logic                   done_o,
  output logic                   is_gemv_o,
  // The target M, K, and N sizes
  input  logic [AddrWidth-1:0]   M_size_i,
  input  logic [AddrWidth-1:0]   K_size_i,
  input  logic [AddrWidth-1:0]   N_size_i,
  // Current M, K, and N base indices (in elements)
  output logic [AddrWidth-1:0]   M_count_o,
  output logic [AddrWidth-1:0]   K_count_o,
  output logic [AddrWidth-1:0]   N_count_o
);

  //-----------------------
  // Wires and logic
  //-----------------------
  logic move_K_counter;
  logic move_N_counter;
  logic move_M_counter;
  logic move_counter;

  logic clear_counters;
  logic last_counter_last_value;

  // Raw step counters (0 .. ceiling - 1)
  logic [AddrWidth-1:0] k_step_count;
  logic [AddrWidth-1:0] n_step_count;
  logic [AddrWidth-1:0] m_step_count;

  localparam int unsigned GemmCols = (GemmRows == 0) ? 1 : NumCores/GemmRows;
  localparam int unsigned NumContexts = (GemmRows == 0) ? 1 : MReuse/GemmRows;
  localparam int unsigned GemvGroupRows = NumContexts*NumCores;
  // A ceiling must represent NumContexts itself, not only NumContexts-1.
  localparam int unsigned RowWidth = (NumContexts <= 1) ? 1 : $clog2(NumContexts + 1);
  localparam int unsigned CalcWidth = ((AddrWidth > 32) ? AddrWidth : 32) + 1;
  logic [RowWidth-1:0] row_step_count, row_ceiling;
  logic [CalcWidth-1:0] m_group_base, rows_remaining, rows_in_group;

  // Ceilings (number of steps per dimension)
  logic [AddrWidth-1:0] k_ceiling;
  logic [AddrWidth-1:0] n_ceiling;
  logic [AddrWidth-1:0] m_ceiling;

  // Mode detection: GeMV when N_size_i == 1, otherwise GeMM
  logic is_gemv;
  assign is_gemv   = (N_size_i == {{(AddrWidth-1){1'b0}}, 1'b1});
  assign is_gemv_o = is_gemv;

  // Pipeline registers that follow the read latency of the SRAMs
  logic data_valid_q;    // A/B data at the MAC inputs is valid
  logic first_k_q;       // ... and it is the first K block of an output
  logic last_k_q;        // ... and it is the last K block of an output
  logic result_valid_q;  // MAC outputs hold finished C elements

  // State machine states
  typedef enum logic [1:0] {
    ControllerIdle,
    ControllerBusy,
    ControllerDrain,
    ControllerFinish
  } controller_state_t;

  controller_state_t current_state, next_state;

  assign busy_o = (current_state == ControllerBusy  ) ||
                  (current_state == ControllerDrain ) ||
                  (current_state == ControllerFinish);
  // Keep acceptance independent of the next-state calculation. Terminal
  // counter pulses feed next_state, but never feed acceptance back into itself.
  assign move_counter = input_valid_i &&
    ((current_state == ControllerBusy) ||
     ((current_state == ControllerIdle) && start_i));

  // Division plus remainder avoids a size+(span-1) overflow at the largest
  // representable M/K/N. The span is widened too, so narrow size ports do not
  // truncate a design-time parallelism constant to zero.
  function automatic logic [AddrWidth-1:0] ceil_steps(
    input logic [AddrWidth-1:0] size,
    input int unsigned span
  );
    logic [CalcWidth-1:0] wide_size, wide_span;
    wide_size = CalcWidth'(size);
    wide_span = CalcWidth'((span == 0) ? 1 : span);
    if (size == '0) return AddrWidth'(1);
    return AddrWidth'((wide_size / wide_span) +
                      CalcWidth'((wide_size % wide_span) != '0));
  endfunction

  //-----------------------
  // Ceilings for 16 cores x 32 inputs
  //-----------------------
  always_comb begin
    m_group_base = CalcWidth'(m_step_count) *
                   CalcWidth'(is_gemv ? GemvGroupRows : MReuse);
    rows_remaining = CalcWidth'(M_size_i) - m_group_base;
    rows_in_group = (rows_remaining < CalcWidth'(is_gemv ? GemvGroupRows : MReuse)) ?
                     rows_remaining : CalcWidth'(is_gemv ? GemvGroupRows : MReuse);
    row_ceiling = RowWidth'(1);
    // K advances by NumInputs (32) per step
    k_ceiling = ceil_steps(K_size_i, NumInputs);

    if (is_gemv) begin
      // Each existing context accumulates a separate group of NumCores rows.
      // Visit all active contexts before advancing K and reusing the fragment.
      m_ceiling = ceil_steps(M_size_i, GemvGroupRows);
      row_ceiling = RowWidth'(ceil_steps(AddrWidth'(rows_in_group), NumCores));
      if (row_ceiling == '0) row_ceiling = RowWidth'(1);
      n_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};
    end else begin
      // A context represents GemmRows independently accumulated rows. Only
      // the final M group may have fewer contexts or a partially valid pair.
      m_ceiling = ceil_steps(M_size_i, MReuse);
      row_ceiling = RowWidth'(ceil_steps(AddrWidth'(rows_in_group), GemmRows));
      if (row_ceiling == '0) row_ceiling = RowWidth'(1);
      n_ceiling = ceil_steps(N_size_i, GemmCols);
    end
  end

  // Convert step indices to element offsets
  assign K_count_o = k_step_count * AddrWidth'(NumInputs);
  assign M_count_o = AddrWidth'(m_group_base + CalcWidth'(row_step_count) *
                                CalcWidth'(is_gemv ? NumCores : GemmRows));
  assign N_count_o = is_gemv ? n_step_count : (n_step_count * AddrWidth'(GemmCols));

  //-----------------------
  // Counters for M, K, N steps
  //-----------------------

  // Visit all row contexts in either mode before K advances: one context
  // covers GemmRows rows in GeMM and NumCores rows in GeMV.
  generate
    if (NumContexts <= 1) begin : gen_one_row
      assign row_step_count = '0;
      assign move_K_counter = move_counter;
    end else begin : gen_reuse_rows
      ceiling_counter #(.Width(RowWidth), .HasCeiling(1)) i_row_counter (
        .clk_i(clk_i), .rst_ni(rst_ni), .tick_i(move_counter),
        .clear_i(clear_counters), .ceiling_i(row_ceiling),
        .count_o(row_step_count), .last_value_o(move_K_counter)
      );
      // Account only for this newly introduced register; existing controller
      // accounting conventions otherwise stay unchanged.
      // synthesis translate_off
      initial begin
        #0;
        metrics_pkg::area += real'(RowWidth) / 32.0;
      end
      always @(row_step_count) begin
        if (rst_ni) metrics_pkg::energy += real'(RowWidth) / 32.0;
      end
      // synthesis translate_on
    end
  endgenerate

  // K Step Counter (each step = 32 elements along K)
  ceiling_counter #(
    .Width        ( AddrWidth  ),
    .HasCeiling   ( 1          )
  ) i_K_counter (
    .clk_i        ( clk_i          ),
    .rst_ni       ( rst_ni         ),
    .tick_i       ( move_K_counter ),
    .clear_i      ( clear_counters ),
    .ceiling_i    ( k_ceiling      ),
    .count_o      ( k_step_count   ),
    .last_value_o ( move_N_counter )
  );

  // N Step Counter (each step = GemmCols columns in GeMM, 1 in GeMV)
  ceiling_counter #(
    .Width        ( AddrWidth  ),
    .HasCeiling   ( 1          )
  ) i_N_counter (
    .clk_i        ( clk_i          ),
    .rst_ni       ( rst_ni         ),
    .tick_i       ( move_N_counter ),
    .clear_i      ( clear_counters ),
    .ceiling_i    ( n_ceiling      ),
    .count_o      ( n_step_count   ),
    .last_value_o ( move_M_counter )
  );

  // M Step Counter (each step = GemvGroupRows in GeMV, MReuse rows in GeMM)
  ceiling_counter #(
    .Width        ( AddrWidth               ),
    .HasCeiling   ( 1                       )
  ) i_M_counter (
    .clk_i        ( clk_i                   ),
    .rst_ni       ( rst_ni                  ),
    .tick_i       ( move_M_counter          ),
    .clear_i      ( clear_counters          ),
    .ceiling_i    ( m_ceiling               ),
    .count_o      ( m_step_count            ),
    .last_value_o ( last_counter_last_value )
  );

  //-----------------------
  // Pipeline control signals
  //-----------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      data_valid_q   <= 1'b0;
      first_k_q      <= 1'b0;
      last_k_q       <= 1'b0;
      result_valid_q <= 1'b0;
    end else begin
      data_valid_q   <= move_counter;
      first_k_q      <= move_counter && (k_step_count == '0);
      // Every context has results at its last K block. K only wraps after
      // the final context, so move_N_counter would miss earlier results.
      last_k_q       <= move_counter && (k_step_count == k_ceiling - AddrWidth'(1));
      result_valid_q <= last_k_q;
    end
  end

  assign mem_re_o       = move_counter;
  assign data_valid_o   = data_valid_q;
  assign init_save_o    = first_k_q;
  assign result_valid_o = result_valid_q;

  //-----------------------
  // Main controller state machine
  //-----------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      current_state <= ControllerIdle;
    end else begin
      current_state <= next_state;
    end
  end

  always_comb begin
    next_state     = current_state;
    clear_counters = 1'b0;
    done_o         = 1'b0;

    case (current_state)
      ControllerIdle: begin
        if (start_i) begin
          next_state   = last_counter_last_value ? ControllerDrain : ControllerBusy;
        end
      end

      ControllerBusy: begin
        if (last_counter_last_value) begin
          next_state = ControllerDrain;
        end
      end

      ControllerDrain: begin
        next_state = ControllerFinish;
      end

      ControllerFinish: begin
        done_o         = 1'b1;
        clear_counters = 1'b1;
        next_state     = ControllerIdle;
      end

      default: begin
        next_state = ControllerIdle;
      end
    endcase
  end

  // synthesis translate_off
  initial begin
    if (MReuse == 0 || NumCores == 0 || NumInputs == 0 || GemmRows == 0 || AddrWidth == 0)
      $fatal(1, "Controller dimensions must be positive.");
    if (GemmRows > NumCores || (NumCores % GemmRows) != 0 ||
        (MReuse % GemmRows) != 0)
      $fatal(1, "GemmRows must divide both NumCores and MReuse.");
  end
  // synthesis translate_on

endmodule
