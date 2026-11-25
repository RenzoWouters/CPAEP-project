
//---------------------------
// The GeMM / GeMV Controller Module for MAC Array
// File: gemm_controller_mac_array.sv
//
// Description:
// Parameterized controller for a MAC array:
//
// - Each K step advances by NumInputs elements along K.
// - When N_size_i == 1 (GeMV), NumCores cores process
//   NumCores rows of M in parallel.
//   M_count_o advances by NumCores.
// - When N_size_i > 1 (GeMM), NumCores cores process
//   NumCores columns of N in parallel.
//   N_count_o advances by NumCores.
//
// For the 16x8 configuration:
// - 16 cores x 8 inputs = 128 MACs.
// - Each K step advances by 8 elements.
//
// Pipeline timing (assuming 1-cycle synchronous SRAM read + 1-cycle MAC):
//   cycle t   : counters issue step indices for (m, n, k) as A/B addresses
//   cycle t+1 : A/B data reaches the MAC array
//               data_valid_o = 1, init_save_o = 1 if first K step
//   cycle t+2 : if last K step, MAC outputs hold finished results
//               result_valid_o = 1 (write NumCores outputs to SRAM C)
//---------------------------

module gemm_controller_mac_array #(
  parameter int unsigned AddrWidth = 16,
  parameter int unsigned NumCores  = 16,
  parameter int unsigned NumInputs = 8
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

  assign move_K_counter = move_counter;

  logic clear_counters;
  logic last_counter_last_value;

  // Raw step counters (0 .. ceiling - 1)
  logic [AddrWidth-1:0] k_step_count;
  logic [AddrWidth-1:0] n_step_count;
  logic [AddrWidth-1:0] m_step_count;

  // Ceilings (number of steps per dimension)
  logic [AddrWidth-1:0] k_ceiling;
  logic [AddrWidth-1:0] n_ceiling;
  logic [AddrWidth-1:0] m_ceiling;

  // Mode detection: GeMV when N_size_i == 1, otherwise GeMM
  logic is_gemv;
  assign is_gemv   = (N_size_i == {{(AddrWidth-1){1'b0}}, 1'b1});
  assign is_gemv_o = is_gemv;

  // Pipeline registers that follow the read latency of the SRAMs
  logic data_valid_q;
  logic first_k_q;
  logic last_k_q;
  logic result_valid_q;

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
  // Calculate dimension ceilings
  //-----------------------
  always_comb begin
    // K advances by NumInputs elements per step
    k_ceiling = (K_size_i + AddrWidth'(NumInputs - 1)) / AddrWidth'(NumInputs);

    if (k_ceiling == '0) begin
      k_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};
    end

    if (is_gemv) begin
      // GeMV: NumCores cores compute NumCores rows of M in parallel
      m_ceiling = (M_size_i + AddrWidth'(NumCores - 1)) / AddrWidth'(NumCores);

      if (m_ceiling == '0) begin
        m_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};
      end

      n_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};

    end else begin
      // GeMM: NumCores cores compute NumCores columns of N in parallel
      // M is processed one row at a time
      m_ceiling = (M_size_i == '0) ? {{(AddrWidth-1){1'b0}}, 1'b1} : M_size_i;

      n_ceiling = (N_size_i + AddrWidth'(NumCores - 1)) / AddrWidth'(NumCores);

      if (n_ceiling == '0) begin
        n_ceiling = {{(AddrWidth-1){1'b0}}, 1'b1};
      end
    end
  end

  //-----------------------
  // Convert step indices to element offsets
  //-----------------------
  assign K_count_o = k_step_count * AddrWidth'(NumInputs);

  assign M_count_o = is_gemv
                   ? (m_step_count * AddrWidth'(NumCores))
                   : m_step_count;

  assign N_count_o = is_gemv
                   ? n_step_count
                   : (n_step_count * AddrWidth'(NumCores));

  //-----------------------
  // Counters for M, K, N steps
  //-----------------------

  // K Step Counter: each step = NumInputs elements along K
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

  // N Step Counter:
  // GeMM: each step = NumCores columns
  // GeMV: only one N step is required
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

  // M Step Counter:
  // GeMM: each step = 1 row
  // GeMV: each step = NumCores rows
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

      // move_N_counter is asserted when k_step_count is at its last step
      last_k_q       <= move_counter && move_N_counter;

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

          // Only finish issuing transactions when the final transaction
          // is actually accepted.
          if (input_valid_i && last_counter_last_value) begin
            next_state = ControllerDrain;
          end else begin
            next_state = ControllerBusy;
          end
        end
      end

      ControllerBusy: begin
        move_counter = input_valid_i;

        // Do not enter the drain state until the final transaction
        // has actually been accepted.
        if (input_valid_i && last_counter_last_value) begin
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

endmodule
