//-----------------------------------------------------------------------------
// GeMM / GeMV top: 16 dot-product cores x 32 K lanes = 512 MAC lanes.
//
// Default external SRAM words: A = 512 bits, B = 480 bits, C = 32 bits.
// The supplied SRAMs must retain their one-cycle synchronous read behavior.
// SRAM instances and their initialization remain outside this module.
// Suggested depths for the six required workloads: A=131072, B=512, C=4096.
//
// Layout (byte lane 0 is the least-significant byte):
//   GeMM A: row-major, each row padded to 64-element words. Retain one word
//           for two consecutive 32-element K blocks.
//   GeMV A: [M group][K block][core][K lane], 8 words per 16x32 tile.
//   GeMM B: [N group][K block][core][K lane], 9 words per 16x32 tile;
//           the final 60-byte word has 32 data bytes and 28 padding bytes.
//   GeMV B: one 60-byte word per K block, first 32 bytes contain the vector.
//   C:      scalar int32, row-major, one result per SRAM word.
//
// Uses the user's existing gemm_controller_mac_array, including the Idle
// single-step fix. Its input_valid_i now advances ONE COMPLETE ARRAY BLOCK.
// Its mem_re_o is deliberately not used as a physical SRAM read enable.
// No change to mac_array, general_mac_pe, or the SRAM module is required.
//-----------------------------------------------------------------------------

module gemm_accelerator_top_mac_array #(
  parameter int unsigned InDataWidth    = 8,
  parameter int unsigned OutDataWidth   = 32,
  parameter int unsigned AddrWidth      = 17,
  parameter int unsigned SizeAddrWidth  = 12,
  parameter int unsigned NumCores       = 16,
  parameter int unsigned NumInputs      = 32,
  parameter int unsigned SramADataWidth = 512,
  parameter int unsigned SramBDataWidth = 480
) (
  input  logic                            clk_i,
  input  logic                            rst_ni,
  input  logic                            start_i,
  input  logic [SizeAddrWidth-1:0]        M_size_i,
  input  logic [SizeAddrWidth-1:0]        K_size_i,
  input  logic [SizeAddrWidth-1:0]        N_size_i,
  output logic [AddrWidth-1:0]            sram_a_addr_o,
  output logic [AddrWidth-1:0]            sram_b_addr_o,
  output logic [AddrWidth-1:0]            sram_c_addr_o,
  output logic                            sram_a_re_o,
  output logic                            sram_b_re_o,
  input  logic [SramADataWidth-1:0]       sram_a_rdata_i,
  input  logic [SramBDataWidth-1:0]       sram_b_rdata_i,
  output logic signed [OutDataWidth-1:0] sram_c_wdata_o,
  output logic                            sram_c_we_o,
  output logic                            done_o
);

  // Word widths and the number of physical reads for one array block.
  localparam int unsigned AElements   = SramADataWidth / InDataWidth;
  localparam int unsigned BElements   = SramBDataWidth / InDataWidth;
  localparam int unsigned TileElements = NumCores * NumInputs;
  localparam int unsigned TileBits     = TileElements * InDataWidth;
  localparam int unsigned AReadWords   = (TileElements + AElements - 1) / AElements;
  localparam int unsigned BReadWords   = (TileElements + BElements - 1) / BElements;
  localparam int unsigned MaxReadWords = (AReadWords > BReadWords) ? AReadWords : BReadWords;
  localparam int unsigned BeatWidth    = (MaxReadWords <= 1) ? 1 : $clog2(MaxReadWords);
  localparam int unsigned CoreWidth    = (NumCores <= 1) ? 1 : $clog2(NumCores);
  localparam int unsigned CoreCountWidth = $clog2(NumCores + 1);
  localparam int unsigned InputCountWidth = $clog2(NumInputs + 1);
  localparam int unsigned AOffsetWidth = (AElements <= 1) ? 1 : $clog2(AElements);

  // The extra counter bit also prevents overflow in the controller's ceil
  // additions for sizes near the top of the external size range.
  localparam int unsigned MaxStep = (NumCores > NumInputs) ? NumCores : NumInputs;
  localparam int unsigned StepWidth = (MaxStep <= 1) ? 1 : $clog2(MaxStep);
  localparam int unsigned CountWidth = ((SizeAddrWidth > StepWidth) ?
                                       SizeAddrWidth : StepWidth) + 1;
  localparam int unsigned CalcWidth  = 2 * CountWidth + 1;

  typedef enum logic [2:0] {
    TopIdle, TopStart, TopLoad, TopFlush, TopIssue, TopDrain
  } top_state_t;
  top_state_t state_q;

  logic [SizeAddrWidth-1:0] M_size_q, K_size_q, N_size_q;
  logic [CountWidth-1:0] M_count, K_count, N_count;
  logic is_gemv, controller_busy, controller_done;
  logic valid_data, init_save, result_valid;
  logic start_controller, issue_block, accept_start;
  logic controller_done_seen_q, done_q;

  logic [BeatWidth-1:0] read_beat_q;
  logic last_read, last_k_block, last_job_block;

  // Tags follow the synchronous SRAM responses, not the current read counter.
  logic a_response_valid_q, b_response_valid_q;
  logic [BeatWidth-1:0] a_response_beat_q, b_response_beat_q;
  logic [AddrWidth-1:0] a_response_addr_q;

  // One large operand tile and one shared operand word, reused across modes.
  logic [TileBits-1:0] tile_q;
  logic [SramADataWidth-1:0] shared_q;
  logic a_cache_valid_q;
  logic [AddrWidth-1:0] a_cache_addr_q;
  logic a_cache_hit;

  // The controller advances at ISSUE. These registers retain the metadata
  // needed by the MAC on the following edge.
  logic [AOffsetWidth-1:0] mac_a_offset_q;
  logic [CoreCountWidth-1:0] mac_core_count_q;
  logic [InputCountWidth-1:0] mac_input_count_q;

  // Reserve the output slot at last-K ISSUE, including results still in flight.
  logic output_slot_busy_q, c_valid_q;
  logic [AddrWidth-1:0] pending_c_base_q, c_base_q;
  logic [CoreCountWidth-1:0] pending_c_count_q, c_count_q;
  logic [CoreWidth-1:0] c_core_q;
  logic [NumCores*OutDataWidth-1:0] c_buffer_q;
  logic last_c_write;

  logic [CalcWidth-1:0] k_blocks, a_words_per_row;
  logic [CalcWidth-1:0] a_word_address, b_word_address, c_base_address;
  logic [CalcWidth-1:0] remaining_cores, remaining_inputs;
  logic [CoreCountWidth-1:0] active_cores;
  logic [InputCountWidth-1:0] active_inputs;

  logic signed [NumCores-1:0][NumInputs-1:0][InDataWidth-1:0] array_a_in;
  logic signed [NumCores-1:0][NumInputs-1:0][InDataWidth-1:0] array_b_in;
  logic signed [NumCores-1:0][OutDataWidth-1:0] array_c_out;

  assign accept_start     = (state_q == TopIdle) && start_i;
  assign start_controller = (state_q == TopStart);
  assign issue_block      = (state_q == TopIssue) &&
                            (!last_k_block || !output_slot_busy_q);
  assign done_o           = done_q;

  gemm_controller_mac_array #(
    .AddrWidth (CountWidth),
    .NumCores  (NumCores),
    .NumInputs (NumInputs)
  ) i_gemm_controller_mac_array (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .start_i        (start_controller),
    .input_valid_i  (issue_block),
    .result_valid_o (result_valid),
    .mem_re_o       (), // Physical read enables are generated below.
    .data_valid_o   (valid_data),
    .init_save_o    (init_save),
    .busy_o         (controller_busy),
    .done_o         (controller_done),
    .is_gemv_o      (is_gemv),
    .M_size_i       (CountWidth'(M_size_q)),
    .K_size_i       (CountWidth'(K_size_q)),
    .N_size_i       (CountWidth'(N_size_q)),
    .M_count_o      (M_count),
    .K_count_o      (K_count),
    .N_count_o      (N_count)
  );

  //---------------------------
  // SRAM WORD address generation
  //---------------------------
  always_comb begin
    k_blocks = (CalcWidth'(K_size_q) + CalcWidth'(NumInputs - 1)) /
               CalcWidth'(NumInputs);
    a_words_per_row = (CalcWidth'(K_size_q) + CalcWidth'(AElements - 1)) /
                      CalcWidth'(AElements);

    if (is_gemv) begin
      // A: [M group][K block], followed by the words of this tile.
      a_word_address = ((CalcWidth'(M_count) / CalcWidth'(NumCores)) * k_blocks +
                        CalcWidth'(K_count) / CalcWidth'(NumInputs)) *
                        CalcWidth'(AReadWords) + CalcWidth'(read_beat_q);
      b_word_address = CalcWidth'(K_count) / CalcWidth'(NumInputs);
      c_base_address = CalcWidth'(M_count);
      remaining_cores = CalcWidth'(M_size_q) - CalcWidth'(M_count);
    end else begin
      // A: one row-major word can supply multiple consecutive K blocks.
      a_word_address = CalcWidth'(M_count) * a_words_per_row +
                       CalcWidth'(K_count) / CalcWidth'(AElements);
      // B: [N group][K block], followed by the words of this tile.
      b_word_address = ((CalcWidth'(N_count) / CalcWidth'(NumCores)) * k_blocks +
                        CalcWidth'(K_count) / CalcWidth'(NumInputs)) *
                        CalcWidth'(BReadWords) + CalcWidth'(read_beat_q);
      c_base_address = CalcWidth'(M_count) * CalcWidth'(N_size_q) +
                       CalcWidth'(N_count);
      remaining_cores = CalcWidth'(N_size_q) - CalcWidth'(N_count);
    end

    remaining_inputs = CalcWidth'(K_size_q) - CalcWidth'(K_count);
    active_cores = (remaining_cores < CalcWidth'(NumCores)) ?
                   CoreCountWidth'(remaining_cores) : CoreCountWidth'(NumCores);
    active_inputs = (remaining_inputs < CalcWidth'(NumInputs)) ?
                    InputCountWidth'(remaining_inputs) : InputCountWidth'(NumInputs);

    last_k_block = (CalcWidth'(K_count) + CalcWidth'(NumInputs) >= CalcWidth'(K_size_q));
    if (is_gemv) begin
      last_job_block = last_k_block &&
                      (CalcWidth'(M_count) + CalcWidth'(NumCores) >= CalcWidth'(M_size_q));
    end else begin
      last_job_block = last_k_block &&
                      (CalcWidth'(M_count) + CalcWidth'(1) >= CalcWidth'(M_size_q)) &&
                      (CalcWidth'(N_count) + CalcWidth'(NumCores) >= CalcWidth'(N_size_q));
    end
  end

  assign sram_a_addr_o = AddrWidth'(a_word_address);
  assign sram_b_addr_o = AddrWidth'(b_word_address);
  assign a_cache_hit = a_cache_valid_q && (a_cache_addr_q == sram_a_addr_o);

  assign sram_a_re_o = (state_q == TopLoad) &&
                      (is_gemv || ((read_beat_q == '0) && !a_cache_hit));
  assign sram_b_re_o = (state_q == TopLoad) &&
                      (!is_gemv || (read_beat_q == '0));
  assign last_read = is_gemv ? (read_beat_q == BeatWidth'(AReadWords - 1)) :
                              (read_beat_q == BeatWidth'(BReadWords - 1));

  //---------------------------
  // Load one complete block, then advance the existing block controller.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= TopIdle;
      M_size_q <= '0;
      K_size_q <= '0;
      N_size_q <= '0;
      read_beat_q <= '0;
      controller_done_seen_q <= 1'b0;
      done_q <= 1'b0;
    end else begin
      done_q <= 1'b0;
      if (controller_done) controller_done_seen_q <= 1'b1;

      case (state_q)
        TopIdle: begin
          if (start_i) begin
            M_size_q <= M_size_i;
            K_size_q <= K_size_i;
            N_size_q <= N_size_i;
            read_beat_q <= '0;
            controller_done_seen_q <= 1'b0;
            state_q <= TopStart;
          end
        end
        TopStart: state_q <= TopLoad;
        TopLoad: begin
          if (last_read) begin
            read_beat_q <= '0;
            state_q <= TopFlush;
          end else begin
            read_beat_q <= read_beat_q + BeatWidth'(1);
          end
        end
        // The last SRAM word is captured into the operand buffer on this edge.
        TopFlush: state_q <= TopIssue;
        TopIssue: begin
          if (issue_block) begin
            state_q <= last_job_block ? TopDrain : TopLoad;
          end
        end
        TopDrain: begin
          // Includes the final SRAM C write, not just the final MAC update.
          if (controller_done_seen_q && !output_slot_busy_q) begin
            done_q <= 1'b1;
            state_q <= TopIdle;
          end
        end
        default: state_q <= TopIdle;
      endcase
    end
  end

  //---------------------------
  // Registered read-response tags and input buffers
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      a_response_valid_q <= 1'b0;
      b_response_valid_q <= 1'b0;
      a_response_beat_q <= '0;
      b_response_beat_q <= '0;
      a_response_addr_q <= '0;
      tile_q <= '0;
      shared_q <= '0;
      a_cache_valid_q <= 1'b0;
      a_cache_addr_q <= '0;
    end else begin
      a_response_valid_q <= sram_a_re_o;
      b_response_valid_q <= sram_b_re_o;
      if (sram_a_re_o) begin
        a_response_beat_q <= read_beat_q;
        a_response_addr_q <= sram_a_addr_o;
      end
      if (sram_b_re_o) b_response_beat_q <= read_beat_q;

      // SRAM contents can change between jobs, even at the same word address.
      if (accept_start) a_cache_valid_q <= 1'b0;

      if (a_response_valid_q) begin
        if (is_gemv) begin
          for (int unsigned lane = 0; lane < AElements; lane++) begin
            if ((int'(a_response_beat_q) * AElements + lane) < TileElements)
              tile_q[(int'(a_response_beat_q) * AElements + lane) * InDataWidth +: InDataWidth]
                <= sram_a_rdata_i[lane * InDataWidth +: InDataWidth];
          end
        end else begin
          shared_q <= sram_a_rdata_i;
          a_cache_valid_q <= 1'b1;
          a_cache_addr_q <= a_response_addr_q;
        end
      end

      if (b_response_valid_q) begin
        if (is_gemv) begin
          shared_q <= SramADataWidth'(sram_b_rdata_i[NumInputs*InDataWidth-1:0]);
        end else begin
          for (int unsigned lane = 0; lane < BElements; lane++) begin
            if ((int'(b_response_beat_q) * BElements + lane) < TileElements)
              tile_q[(int'(b_response_beat_q) * BElements + lane) * InDataWidth +: InDataWidth]
                <= sram_b_rdata_i[lane * InDataWidth +: InDataWidth];
          end
        end
      end
    end
  end

  //---------------------------
  // Metadata for the MAC and serialized output writes
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      mac_a_offset_q <= '0;
      mac_core_count_q <= '0;
      mac_input_count_q <= '0;
      output_slot_busy_q <= 1'b0;
      pending_c_base_q <= '0;
      pending_c_count_q <= '0;
      c_buffer_q <= '0;
      c_base_q <= '0;
      c_count_q <= '0;
      c_core_q <= '0;
      c_valid_q <= 1'b0;
    end else begin
      if (issue_block) begin
        mac_a_offset_q <= AOffsetWidth'(CalcWidth'(K_count) % CalcWidth'(AElements));
        mac_core_count_q <= active_cores;
        mac_input_count_q <= active_inputs;
        if (last_k_block) begin
          output_slot_busy_q <= 1'b1;
          pending_c_base_q <= AddrWidth'(c_base_address);
          pending_c_count_q <= active_cores;
        end
      end

      // With the existing controller this is sampled two edges after ISSUE.
      // The registered MAC result is already stable before this edge.
      if (result_valid) begin
        c_buffer_q <= array_c_out;
        c_base_q <= pending_c_base_q;
        c_count_q <= pending_c_count_q;
        c_core_q <= '0;
        c_valid_q <= 1'b1;
      end else if (c_valid_q) begin
        if (last_c_write) begin
          c_valid_q <= 1'b0;
          output_slot_busy_q <= 1'b0;
        end else begin
          c_core_q <= c_core_q + CoreWidth'(1);
        end
      end
    end
  end

  assign sram_c_we_o = c_valid_q;
  assign sram_c_addr_o = c_base_q + AddrWidth'(c_core_q);
  assign sram_c_wdata_o = $signed(c_buffer_q[int'(c_core_q)*OutDataWidth +: OutDataWidth]);
  assign last_c_write = c_valid_q &&
                       (CoreCountWidth'(c_core_q) + CoreCountWidth'(1) == c_count_q);

  //---------------------------
  // Array input selection and partial-tile masking
  //---------------------------
  always_comb begin
    array_a_in = '0;
    array_b_in = '0;
    for (int unsigned c = 0; c < NumCores; c++) begin
      for (int unsigned j = 0; j < NumInputs; j++) begin
        if ((c < mac_core_count_q) && (j < mac_input_count_q)) begin
          if (is_gemv) begin
            array_a_in[c][j] = $signed(tile_q[(c*NumInputs+j)*InDataWidth +: InDataWidth]);
            array_b_in[c][j] = $signed(shared_q[j*InDataWidth +: InDataWidth]);
          end else begin
            array_a_in[c][j] = $signed(shared_q[(int'(mac_a_offset_q)+j)*InDataWidth +: InDataWidth]);
            array_b_in[c][j] = $signed(tile_q[(c*NumInputs+j)*InDataWidth +: InDataWidth]);
          end
        end
      end
    end
  end

  mac_array #(
    .InDataWidth (InDataWidth),
    .NumCores    (NumCores),
    .NumInputs   (NumInputs),
    .OutDataWidth(OutDataWidth)
  ) i_mac_array (
    .clk_i       (clk_i),
    .rst_ni      (rst_ni),
    .a_i         (array_a_in),
    .b_i         (array_b_in),
    .a_valid_i   (valid_data),
    .b_valid_i   (valid_data),
    .init_save_i (init_save),
    .acc_clr_i   (!controller_busy),
    .c_o         (array_c_out)
  );

  // synthesis translate_off
  // Simulation checks, not a testbench. Positive matrix sizes are required.
  initial begin
    if (InDataWidth != 8 || OutDataWidth != 32)
      $fatal(1, "This project interface requires int8 inputs and int32 outputs.");
    if (NumCores == 0 || NumInputs == 0 || TileElements > 512)
      $fatal(1, "Invalid array dimensions or more than 512 MAC lanes.");
    if ((SramADataWidth % InDataWidth) != 0 || (SramBDataWidth % InDataWidth) != 0 ||
        AElements < NumInputs || BElements < NumInputs || (AElements % NumInputs) != 0)
      $fatal(1, "SRAM words must contain whole elements; A must contain whole K blocks.");
    if ((SramADataWidth + SramBDataWidth + OutDataWidth) > 1024)
      $fatal(1, "Total physical SRAM port width exceeds 1024 bits.");
  end

  always @(posedge clk_i) begin
    if (rst_ni) begin
      if (accept_start && (M_size_i == '0 || K_size_i == '0 || N_size_i == '0))
        $fatal(1, "M, K and N must be strictly positive.");
      if (state_q == TopLoad &&
          ((a_word_address >> AddrWidth) != '0 || (b_word_address >> AddrWidth) != '0))
        $fatal(1, "SRAM word address does not fit AddrWidth.");
      if (issue_block && last_k_block &&
          ((c_base_address + CalcWidth'(active_cores) - CalcWidth'(1)) >> AddrWidth) != '0)
        $fatal(1, "Output word address does not fit AddrWidth.");
      if (result_valid && (!output_slot_busy_q || c_valid_q))
        $fatal(1, "Result arrived without a free, reserved output slot.");
    end
  end

  // Account for registers added by this top. The PE already accounts for its
  // own registers/arithmetic. Value-change activity follows the supplied PE's
  // register-energy convention; combinational address/control costs are not
  // estimated by this register-only contribution.
  localparam int unsigned TopRegisterBits = $bits({
    state_q, M_size_q, K_size_q, N_size_q, read_beat_q,
    controller_done_seen_q, done_q, a_response_valid_q, b_response_valid_q,
    a_response_beat_q, b_response_beat_q, a_response_addr_q,
    tile_q, shared_q, a_cache_valid_q, a_cache_addr_q,
    mac_a_offset_q, mac_core_count_q, mac_input_count_q,
    output_slot_busy_q, pending_c_base_q, pending_c_count_q,
    c_buffer_q, c_base_q, c_count_q, c_core_q, c_valid_q
  });
  initial begin
    #0;
    metrics_pkg::area += real'(TopRegisterBits) / 32.0;
  end

  // Buffers are counted in their own 32-bit chunks, so a partial tile load
  // does not charge an update of the whole 4096-bit buffer.
  for (genvar g = 0; g < (TileBits + 31) / 32; g++) begin : gen_tile_energy
    localparam int unsigned ChunkWidth = ((TileBits - 32*g) >= 32) ? 32 : (TileBits - 32*g);
    always @(tile_q[32*g +: ChunkWidth]) begin
      if (rst_ni) metrics_pkg::energy += real'(ChunkWidth) / 32.0;
    end
  end
  for (genvar g = 0; g < (SramADataWidth + 31) / 32; g++) begin : gen_shared_energy
    localparam int unsigned ChunkWidth = ((SramADataWidth - 32*g) >= 32) ?
                                         32 : (SramADataWidth - 32*g);
    always @(shared_q[32*g +: ChunkWidth]) begin
      if (rst_ni) metrics_pkg::energy += real'(ChunkWidth) / 32.0;
    end
  end
  for (genvar g = 0; g < NumCores; g++) begin : gen_output_energy
    always @(c_buffer_q[g*OutDataWidth +: OutDataWidth]) begin
      if (rst_ni) metrics_pkg::energy += real'(OutDataWidth) / 32.0;
    end
  end

  `define CPAEP_TOP_REGISTER_ENERGY(REG) \
    always @(REG) begin \
      if (rst_ni) metrics_pkg::energy += real'($bits(REG)) / 32.0; \
    end
  `CPAEP_TOP_REGISTER_ENERGY(state_q)
  `CPAEP_TOP_REGISTER_ENERGY(M_size_q)
  `CPAEP_TOP_REGISTER_ENERGY(K_size_q)
  `CPAEP_TOP_REGISTER_ENERGY(N_size_q)
  `CPAEP_TOP_REGISTER_ENERGY(read_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(controller_done_seen_q)
  `CPAEP_TOP_REGISTER_ENERGY(done_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_response_valid_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_response_valid_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_response_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_response_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_response_addr_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_cache_valid_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_cache_addr_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_a_offset_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_core_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_input_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(output_slot_busy_q)
  `CPAEP_TOP_REGISTER_ENERGY(pending_c_base_q)
  `CPAEP_TOP_REGISTER_ENERGY(pending_c_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_base_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_core_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_valid_q)
  `undef CPAEP_TOP_REGISTER_ENERGY
  // synthesis translate_on

endmodule