//---------------------------
// The GeMM / GeMV Controller Module for MAC Array (16 cores x 32 inputs = 512 MACs)
// File: gemm_controller_mac_array.sv
//
// Description:
// Controller for a 16x32 MAC array:
// - Each K step advances by NumInputs (32) elements along K.
// - When N_size_i == 1 (GeMV), NumCores (16) cores process 16 rows of M
//   in parallel (M ceiling = ceil(M_size_i / 16), M_count_o stepped by 16).
// - When N_size_i > 1 (GeMM), NumCores (16) cores process 16 columns of N
//   in parallel. Up to MReuse rows share each loaded B tile. The loop order is
//   M group -> N group -> K block -> row within the M group.
//   M_count_o is the actual row, including the row-within-group offset.
// - MReuse=1 preserves the original controller's sequence.
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
  parameter int unsigned MReuse    = 1
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

  // The ceiling must represent MReuse itself, not only MReuse-1.
  localparam int unsigned RowWidth = (MReuse <= 1) ? 1 : $clog2(MReuse + 1);
  logic [RowWidth-1:0] row_step_count, row_ceiling;
  logic [AddrWidth-1:0] m_group_base, rows_remaining;

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

  //-----------------------
  // Ceilings for 16 cores x 32 inputs
  //-----------------------
  always_comb begin
    m_group_base = m_step_count * AddrWidth'(MReuse);
    rows_remaining = M_size_i - m_group_base;
    row_ceiling = RowWidth'(1);
    // K advances by NumInputs (32) per step
    k_ceiling = (K_size_i + AddrWidth'(NumInputs - 1)) / AddrWidth'(NumInputs);
    if (k_ceiling == '0) begin
      k_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};
    end

    if (is_gemv) begin
      // GeMV: 16 cores compute 16 rows of M in parallel; N is 1
      m_ceiling = (M_size_i + AddrWidth'(NumCores - 1)) / AddrWidth'(NumCores);
      if (m_ceiling == '0) begin
        m_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};
      end
      n_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};
    end else begin
      // GeMM: one M group contains up to MReuse independently accumulated rows.
      m_ceiling = (M_size_i + AddrWidth'(MReuse - 1)) / AddrWidth'(MReuse);
      if (m_ceiling == '0) m_ceiling = AddrWidth'(1);
      row_ceiling = (rows_remaining < AddrWidth'(MReuse)) ?
                    RowWidth'(rows_remaining) : RowWidth'(MReuse);
      if (row_ceiling == '0) row_ceiling = RowWidth'(1);
      n_ceiling = (N_size_i + AddrWidth'(NumCores - 1)) / AddrWidth'(NumCores);
      if (n_ceiling == '0) begin
        n_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};
      end
    end
  end

  // Convert step indices to element offsets
  assign K_count_o = k_step_count * AddrWidth'(NumInputs);
  assign M_count_o = is_gemv ? (m_step_count * AddrWidth'(NumCores)) :
                             (m_group_base + AddrWidth'(row_step_count));
  assign N_count_o = is_gemv ? n_step_count : (n_step_count * AddrWidth'(NumCores));

  //-----------------------
  // Counters for M, K, N steps
  //-----------------------

  // Reuse a GeMM B tile for every row in the current M group before K advances.
  // In GeMV row_ceiling is one, so every accepted block still advances K.
  generate
    if (MReuse == 1) begin : gen_one_row
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

  // N Step Counter (each step = 16 columns in GeMM, 1 in GeMV)
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

  // M Step Counter (each step = NumCores rows in GeMV, MReuse rows in GeMM)
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
      // Every row has a result at its last K block. K only wraps after the
      // final row, so move_N_counter alone would miss the other row results.
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
    move_counter   = 1'b0;
    done_o         = 1'b0;

    case (current_state)
      ControllerIdle: begin
        if (start_i) begin
          move_counter = input_valid_i;
          next_state   = last_counter_last_value ? ControllerDrain : ControllerBusy;
        end
      end

      ControllerBusy: begin
        move_counter = input_valid_i;
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
    if (MReuse == 0 || NumCores == 0 || NumInputs == 0)
      $fatal(1, "Controller dimensions must be positive.");
  end
  // synthesis translate_on

endmodule
