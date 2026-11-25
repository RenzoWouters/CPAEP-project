//-----------------------------------------------------------------------------
// GeMM / GeMV top: 16 dot-product cores x 32 K lanes = 512 MAC lanes.
//
// Fixed external SRAM words: A=512 bits, B=448 bits, C=64 bits by default.
// The supplied one-cycle synchronous SRAM behavior is kept.
// GeMM loop order: [M group][N group][K block][row within M group].
// One B tile is reused by up to MReuse rows. Each row has its own A-word cache
// and accumulator context. A bounded output FIFO includes reservations for
// results still in the MAC pipeline; C writes overlap following computation.
// GeMV retains the original 16 M rows x 32 K lanes mapping and burst schedule.
//
// Layout (byte lane 0 is the least-significant byte):
//   GeMM A: row-major, each row padded to 64-element words.
//   GeMV A: [M group][K block][core][K lane], 8 words per 16x32 tile.
//   GeMM B: [N group][K block][core][K lane], 10 words per 16x32 tile;
//           the last 56-byte word contains 8 useful bytes and 48 padding bytes.
//   GeMV B: one 56-byte word per K block; first 32 bytes contain the vector.
//   GeMM C: row-major int32, each row padded to a whole SRAM C word.
//   GeMV C: consecutive int32 results, padded only at the end of the vector.
//   C has two int32 lanes per 64-bit word by default; unused lanes are zero.
// SRAM instances and initialization remain outside this module.
//-----------------------------------------------------------------------------
module gemm_accelerator_top_mac_array #(
  parameter int unsigned InDataWidth    = 8,
  parameter int unsigned OutDataWidth   = 32,
  parameter int unsigned AddrWidth      = 17,
  parameter int unsigned SizeAddrWidth  = 12,
  parameter int unsigned NumCores       = 16,
  parameter int unsigned NumInputs      = 32,
  parameter int unsigned SramADataWidth = 512,
  parameter int unsigned SramBDataWidth = 448,
  parameter int unsigned SramCDataWidth = 64,
  parameter int unsigned MReuse         = 4
) (
  input  logic                            clk_i,
  input  logic                            rst_ni,
  input  logic                            start_i,
  input  logic [SizeAddrWidth-1:0]         M_size_i,
  input  logic [SizeAddrWidth-1:0]         K_size_i,
  input  logic [SizeAddrWidth-1:0]         N_size_i,
  output logic [AddrWidth-1:0]             sram_a_addr_o,
  output logic [AddrWidth-1:0]             sram_b_addr_o,
  output logic [AddrWidth-1:0]             sram_c_addr_o,
  output logic                            sram_a_re_o,
  output logic                            sram_b_re_o,
  input  logic [SramADataWidth-1:0]        sram_a_rdata_i,
  input  logic [SramBDataWidth-1:0]        sram_b_rdata_i,
  output logic [SramCDataWidth-1:0]       sram_c_wdata_o,
  output logic                            sram_c_we_o,
  output logic                            done_o
);
  localparam int unsigned AElements = SramADataWidth / InDataWidth;
  localparam int unsigned BElements = SramBDataWidth / InDataWidth;
  localparam int unsigned CElements = SramCDataWidth / OutDataWidth;
  localparam int unsigned TileElements = NumCores * NumInputs;
  localparam int unsigned TileBits = TileElements * InDataWidth;
  localparam int unsigned AReadWords = (TileElements + AElements - 1) / AElements;
  localparam int unsigned BReadWords = (TileElements + BElements - 1) / BElements;
  localparam int unsigned GemmLoadWords = (BReadWords > MReuse) ? BReadWords : MReuse;
  localparam int unsigned MaxReadWords = (AReadWords > GemmLoadWords) ?
                                         AReadWords : GemmLoadWords;
  localparam int unsigned BeatWidth = (MaxReadWords <= 1) ? 1 : $clog2(MaxReadWords);
  localparam int unsigned CoreWidth = (NumCores <= 1) ? 1 : $clog2(NumCores);
  localparam int unsigned CoreCountWidth = $clog2(NumCores + 1);
  localparam int unsigned InputCountWidth = $clog2(NumInputs + 1);
  localparam int unsigned AOffsetWidth = (AElements <= 1) ? 1 : $clog2(AElements);
  localparam int unsigned ReuseWidth = (MReuse <= 1) ? 1 : $clog2(MReuse);
  localparam int unsigned ReuseCountWidth = $clog2(MReuse + 1);
  localparam int unsigned QueueDepth = MReuse;
  localparam int unsigned QueuePtrWidth = (QueueDepth <= 1) ? 1 : $clog2(QueueDepth);
  localparam int unsigned QueueCountWidth = $clog2(QueueDepth + 1);
  localparam int unsigned MaxStep = (NumCores > NumInputs) ? NumCores : NumInputs;
  localparam int unsigned StepWidth = (MaxStep <= 1) ? 1 : $clog2(MaxStep);
  localparam int unsigned CountWidth = ((SizeAddrWidth > StepWidth) ?
                                       SizeAddrWidth : StepWidth) + 1;
  localparam int unsigned CalcWidth = 2 * CountWidth + 1;

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
  logic last_read, last_k_block, last_job_block, last_reuse_row;

  // Tags follow SRAM responses, including the row that issued an A read.
  logic a_response_valid_q, b_response_valid_q;
  logic [BeatWidth-1:0] a_response_beat_q, b_response_beat_q;
  logic [ReuseWidth-1:0] a_response_row_q;
  logic [AddrWidth-1:0] a_response_addr_q;
  logic [TileBits-1:0] tile_q;
  logic [MReuse-1:0][SramADataWidth-1:0] shared_q;
  logic [MReuse-1:0] a_cache_valid_q;
  logic [MReuse-1:0][AddrWidth-1:0] a_cache_addr_q;
  logic a_cache_hit;

  // Issued metadata is consumed by the MAC on the following clock edge.
  logic [AOffsetWidth-1:0] mac_a_offset_q;
  logic [CoreCountWidth-1:0] mac_core_count_q;
  logic [InputCountWidth-1:0] mac_input_count_q;
  logic [ReuseWidth-1:0] mac_context_q;

  // One entry holds up to NumCores final outputs. Reservations count from
  // last-K ISSUE, before the corresponding data arrives two edges later.
  logic [QueuePtrWidth-1:0] c_head_q, c_tail_q;
  logic [QueueCountWidth-1:0] output_queue_used_q;
  logic [QueueDepth-1:0] c_ready_q;
  logic [QueueDepth-1:0][NumCores*OutDataWidth-1:0] c_data_q;
  logic [QueueDepth-1:0][AddrWidth-1:0] c_addresses_q;
  logic [QueueDepth-1:0][CoreCountWidth-1:0] c_counts_q;
  logic [1:0] fifo_pending_valid_q;
  logic [1:0][QueuePtrWidth-1:0] fifo_pending_slot_q;
  logic [CoreWidth-1:0] c_core_q;
  logic reserve_output, last_c_write, output_stall;
  // Named probes retained for the integration TB; these are FIFO-head/status
  // views of registered storage, not additional copies of the registers.
  logic output_slot_busy_q, c_valid_q;
  logic [AddrWidth-1:0] c_base_q;
  logic [CoreCountWidth-1:0] c_count_q;
  logic [NumCores*OutDataWidth-1:0] c_buffer_q;

  logic [CalcWidth-1:0] k_blocks, a_words_per_row, c_words_per_row, m_group_base;
  logic [CalcWidth-1:0] a_word_address, b_word_address, c_base_address;
  logic [CalcWidth-1:0] remaining_cores, remaining_inputs, remaining_rows;
  logic [CoreCountWidth-1:0] active_cores;
  logic [InputCountWidth-1:0] active_inputs;
  logic [ReuseCountWidth-1:0] active_rows;
  logic [ReuseWidth-1:0] row_in_group, load_row;

  logic signed [NumCores-1:0][NumInputs-1:0][InDataWidth-1:0] array_a_in;
  logic signed [NumCores-1:0][NumInputs-1:0][InDataWidth-1:0] array_b_in;
  logic signed [NumCores-1:0][OutDataWidth-1:0] array_c_out;

  function automatic logic [QueuePtrWidth-1:0] next_slot(
    input logic [QueuePtrWidth-1:0] slot
  );
    return (slot == QueuePtrWidth'(QueueDepth-1)) ? '0 : slot + QueuePtrWidth'(1);
  endfunction

  assign accept_start = (state_q == TopIdle) && start_i;
  assign start_controller = (state_q == TopStart);
  assign output_slot_busy_q = (output_queue_used_q == QueueCountWidth'(QueueDepth));
  assign output_stall = (state_q == TopIssue) && last_k_block &&
                        output_slot_busy_q && !last_c_write;
  assign issue_block = (state_q == TopIssue) && !output_stall;
  assign reserve_output = issue_block && last_k_block;
  assign done_o = done_q;

  gemm_controller_mac_array #(
    .AddrWidth(CountWidth), .NumCores(NumCores), .NumInputs(NumInputs),
    .MReuse(MReuse)
  ) i_gemm_controller_mac_array (
    .clk_i(clk_i), .rst_ni(rst_ni), .start_i(start_controller),
    .input_valid_i(issue_block), .result_valid_o(result_valid), .mem_re_o(),
    .data_valid_o(valid_data), .init_save_o(init_save),
    .busy_o(controller_busy), .done_o(controller_done), .is_gemv_o(is_gemv),
    .M_size_i(CountWidth'(M_size_q)), .K_size_i(CountWidth'(K_size_q)),
    .N_size_i(CountWidth'(N_size_q)),
    .M_count_o(M_count), .K_count_o(K_count), .N_count_o(N_count)
  );

  //---------------------------
  // SRAM word addresses. C rows are padded so partial writes cannot overwrite
  // a result from the next row; the SRAM always writes a complete word.
  //---------------------------
  always_comb begin
    k_blocks = (CalcWidth'(K_size_q) + CalcWidth'(NumInputs-1)) / CalcWidth'(NumInputs);
    a_words_per_row = (CalcWidth'(K_size_q) + CalcWidth'(AElements-1)) /
                      CalcWidth'(AElements);
    c_words_per_row = (CalcWidth'(N_size_q) + CalcWidth'(CElements-1)) /
                      CalcWidth'(CElements);
    row_in_group = is_gemv ? '0 : ReuseWidth'(CalcWidth'(M_count) % CalcWidth'(MReuse));
    m_group_base = is_gemv ? CalcWidth'(M_count) :
                   CalcWidth'(M_count) - CalcWidth'(row_in_group);
    load_row = ReuseWidth'(read_beat_q);
    remaining_rows = CalcWidth'(M_size_q) - m_group_base;
    active_rows = is_gemv ? ReuseCountWidth'(1) :
                  ((remaining_rows < CalcWidth'(MReuse)) ?
                    ReuseCountWidth'(remaining_rows) : ReuseCountWidth'(MReuse));

    if (is_gemv) begin
      a_word_address = ((CalcWidth'(M_count) / CalcWidth'(NumCores)) * k_blocks +
                         CalcWidth'(K_count) / CalcWidth'(NumInputs)) *
                         CalcWidth'(AReadWords) + CalcWidth'(read_beat_q);
      b_word_address = CalcWidth'(K_count) / CalcWidth'(NumInputs);
      c_base_address = CalcWidth'(M_count) / CalcWidth'(CElements);
      remaining_cores = CalcWidth'(M_size_q) - CalcWidth'(M_count);
    end else begin
      // During the B burst, fetch/cache one A word for each active M row.
      a_word_address = (m_group_base + CalcWidth'(read_beat_q)) * a_words_per_row +
                        CalcWidth'(K_count) / CalcWidth'(AElements);
      b_word_address = ((CalcWidth'(N_count) / CalcWidth'(NumCores)) * k_blocks +
                         CalcWidth'(K_count) / CalcWidth'(NumInputs)) *
                         CalcWidth'(BReadWords) + CalcWidth'(read_beat_q);
      c_base_address = CalcWidth'(M_count) * c_words_per_row +
                       CalcWidth'(N_count) / CalcWidth'(CElements);
      remaining_cores = CalcWidth'(N_size_q) - CalcWidth'(N_count);
    end
    remaining_inputs = CalcWidth'(K_size_q) - CalcWidth'(K_count);
    active_cores = (remaining_cores < CalcWidth'(NumCores)) ?
                   CoreCountWidth'(remaining_cores) : CoreCountWidth'(NumCores);
    active_inputs = (remaining_inputs < CalcWidth'(NumInputs)) ?
                    InputCountWidth'(remaining_inputs) : InputCountWidth'(NumInputs);
    last_k_block = (CalcWidth'(K_count) + CalcWidth'(NumInputs) >= CalcWidth'(K_size_q));
    last_reuse_row = is_gemv ||
                     (ReuseCountWidth'(row_in_group) + ReuseCountWidth'(1) == active_rows);
    last_job_block = last_k_block &&
      (is_gemv ? (CalcWidth'(M_count) + CalcWidth'(NumCores) >= CalcWidth'(M_size_q)) :
       ((CalcWidth'(M_count) + CalcWidth'(1) >= CalcWidth'(M_size_q)) &&
        (CalcWidth'(N_count) + CalcWidth'(NumCores) >= CalcWidth'(N_size_q))));
  end

  assign sram_a_addr_o = AddrWidth'(a_word_address);
  assign sram_b_addr_o = AddrWidth'(b_word_address);
  always_comb begin
    a_cache_hit = 1'b0;
    if (!is_gemv && (int'(read_beat_q) < int'(active_rows)))
      a_cache_hit = a_cache_valid_q[load_row] &&
                    (a_cache_addr_q[load_row] == sram_a_addr_o);
  end
  assign sram_a_re_o = (state_q == TopLoad) &&
    (is_gemv || ((int'(read_beat_q) < int'(active_rows)) && !a_cache_hit));
  assign sram_b_re_o = (state_q == TopLoad) &&
    (is_gemv ? (read_beat_q == '0) : (int'(read_beat_q) < BReadWords));
  assign last_read = is_gemv ? (read_beat_q == BeatWidth'(AReadWords-1)) :
                              (read_beat_q == BeatWidth'(GemmLoadWords-1));

  //---------------------------
  // Load once, issue up to MReuse rows, then load the next K/N/M tile.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= TopIdle;
      M_size_q <= '0; K_size_q <= '0; N_size_q <= '0;
      read_beat_q <= '0;
      controller_done_seen_q <= 1'b0;
      done_q <= 1'b0;
    end else begin
      done_q <= 1'b0;
      if (controller_done) controller_done_seen_q <= 1'b1;
      case (state_q)
        TopIdle: if (start_i) begin
          M_size_q <= M_size_i; K_size_q <= K_size_i; N_size_q <= N_size_i;
          read_beat_q <= '0;
          controller_done_seen_q <= 1'b0;
          state_q <= TopStart;
        end
        TopStart: state_q <= TopLoad;
        TopLoad: if (last_read) begin
          read_beat_q <= '0;
          state_q <= TopFlush;
        end else read_beat_q <= read_beat_q + BeatWidth'(1);
        // Capture the final synchronous SRAM response before any row is issued.
        TopFlush: state_q <= TopIssue;
        TopIssue: if (issue_block) begin
          if (last_job_block) state_q <= TopDrain;
          else if (last_reuse_row) state_q <= TopLoad;
          // Other rows use the same B tile and their already-cached A word.
        end
        TopDrain: if (controller_done_seen_q && (output_queue_used_q == '0)) begin
          done_q <= 1'b1;
          state_q <= TopIdle;
        end
        default: state_q <= TopIdle;
      endcase
    end
  end

  //---------------------------
  // Registered SRAM-response tags and operand buffers.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      a_response_valid_q <= 1'b0; b_response_valid_q <= 1'b0;
      a_response_beat_q <= '0; b_response_beat_q <= '0;
      a_response_addr_q <= '0; a_response_row_q <= '0;
      tile_q <= '0; shared_q <= '0;
      a_cache_valid_q <= '0; a_cache_addr_q <= '0;
    end else begin
      a_response_valid_q <= sram_a_re_o;
      b_response_valid_q <= sram_b_re_o;
      if (sram_a_re_o) begin
        a_response_beat_q <= read_beat_q;
        a_response_addr_q <= sram_a_addr_o;
        a_response_row_q <= load_row;
      end
      if (sram_b_re_o) b_response_beat_q <= read_beat_q;
      // New jobs may replace SRAM contents at unchanged addresses.
      if (accept_start) a_cache_valid_q <= '0;
      if (a_response_valid_q) begin
        if (is_gemv) begin
          for (int unsigned lane=0; lane<AElements; lane++)
            if ((int'(a_response_beat_q)*AElements + lane) < TileElements)
              tile_q[(int'(a_response_beat_q)*AElements+lane)*InDataWidth +: InDataWidth]
                <= sram_a_rdata_i[lane*InDataWidth +: InDataWidth];
        end else begin
          shared_q[a_response_row_q] <= sram_a_rdata_i;
          a_cache_valid_q[a_response_row_q] <= 1'b1;
          a_cache_addr_q[a_response_row_q] <= a_response_addr_q;
        end
      end
      if (b_response_valid_q) begin
        if (is_gemv) shared_q[0] <= SramADataWidth'(sram_b_rdata_i[NumInputs*InDataWidth-1:0]);
        else begin
          for (int unsigned lane=0; lane<BElements; lane++)
            if ((int'(b_response_beat_q)*BElements + lane) < TileElements)
              tile_q[(int'(b_response_beat_q)*BElements+lane)*InDataWidth +: InDataWidth]
                <= sram_b_rdata_i[lane*InDataWidth +: InDataWidth];
        end
      end
    end
  end

  //---------------------------
  // MAC metadata, output reservations, and C serialization.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      mac_a_offset_q <= '0; mac_core_count_q <= '0;
      mac_input_count_q <= '0; mac_context_q <= '0;
      c_head_q <= '0; c_tail_q <= '0; output_queue_used_q <= '0;
      c_ready_q <= '0; c_data_q <= '0; c_addresses_q <= '0; c_counts_q <= '0;
      fifo_pending_valid_q <= '0; fifo_pending_slot_q <= '0;
      c_core_q <= '0;
    end else begin
      if (issue_block) begin
        mac_a_offset_q <= AOffsetWidth'(CalcWidth'(K_count) % CalcWidth'(AElements));
        mac_core_count_q <= active_cores;
        mac_input_count_q <= active_inputs;
        mac_context_q <= row_in_group;
      end
      fifo_pending_valid_q <= {fifo_pending_valid_q[0], reserve_output};
      if (reserve_output) begin
        fifo_pending_slot_q[0] <= c_tail_q;
        c_addresses_q[c_tail_q] <= AddrWidth'(c_base_address);
        c_counts_q[c_tail_q] <= active_cores;
        c_tail_q <= next_slot(c_tail_q);
      end
      if (fifo_pending_valid_q[0]) fifo_pending_slot_q[1] <= fifo_pending_slot_q[0];
      case ({reserve_output, last_c_write})
        2'b10: output_queue_used_q <= output_queue_used_q + QueueCountWidth'(1);
        2'b01: output_queue_used_q <= output_queue_used_q - QueueCountWidth'(1);
        default: ;
      endcase
      if (c_valid_q) begin
        if (last_c_write) begin
          c_ready_q[c_head_q] <= 1'b0;
          c_head_q <= next_slot(c_head_q);
          c_core_q <= '0;
        end else c_core_q <= c_core_q + CoreWidth'(CElements);
      end
      // c_o identifies the last completed context, independent of a later ISSUE.
      // Capture before this edge's next MAC result updates via NBA.
      if (result_valid) begin
        c_data_q[fifo_pending_slot_q[1]] <= array_c_out;
        c_ready_q[fifo_pending_slot_q[1]] <= 1'b1;
      end
    end
  end
  assign c_valid_q = c_ready_q[c_head_q];
  assign c_base_q = c_addresses_q[c_head_q];
  assign c_count_q = c_counts_q[c_head_q];
  assign c_buffer_q = c_data_q[c_head_q];
  assign sram_c_we_o = c_valid_q;
  assign sram_c_addr_o = c_base_q + AddrWidth'(int'(c_core_q) / CElements);
  // Each lane remains an int32 result. Only the physical SRAM word is wider.
  // Zero-padding the final word also keeps every written bit defined.
  always_comb begin
    sram_c_wdata_o = '0;
    for (int unsigned lane=0; lane<CElements; lane++) begin
      if ((int'(c_core_q) + lane) < int'(c_count_q))
        sram_c_wdata_o[lane*OutDataWidth +: OutDataWidth] =
          c_buffer_q[(int'(c_core_q)+lane)*OutDataWidth +: OutDataWidth];
    end
  end
  assign last_c_write = c_valid_q &&
    (CoreCountWidth'(c_core_q) + CoreCountWidth'(CElements) >= c_count_q);

  //---------------------------
  // One shared 512-MAC array; context zero is used for GeMV.
  //---------------------------
  always_comb begin
    array_a_in = '0;
    array_b_in = '0;
    for (int unsigned c=0; c<NumCores; c++) begin
      for (int unsigned j=0; j<NumInputs; j++) begin
        if ((c < mac_core_count_q) && (j < mac_input_count_q)) begin
          if (is_gemv) begin
            array_a_in[c][j] = $signed(tile_q[(c*NumInputs+j)*InDataWidth +: InDataWidth]);
            array_b_in[c][j] = $signed(shared_q[0][j*InDataWidth +: InDataWidth]);
          end else begin
            array_a_in[c][j] = $signed(shared_q[mac_context_q]
                                      [(int'(mac_a_offset_q)+j)*InDataWidth +: InDataWidth]);
            array_b_in[c][j] = $signed(tile_q[(c*NumInputs+j)*InDataWidth +: InDataWidth]);
          end
        end
      end
    end
  end
  mac_array #(
    .InDataWidth(InDataWidth), .NumCores(NumCores), .NumInputs(NumInputs),
    .OutDataWidth(OutDataWidth), .NumContexts(MReuse)
  ) i_mac_array (
    .clk_i(clk_i), .rst_ni(rst_ni), .a_i(array_a_in), .b_i(array_b_in),
    .a_valid_i(valid_data), .b_valid_i(valid_data), .init_save_i(init_save),
    .acc_clr_i(!controller_busy), .acc_context_i(mac_context_q), .c_o(array_c_out)
  );

  // synthesis translate_off
  initial begin
    if (InDataWidth != 8 || OutDataWidth != 32)
      $fatal(1, "This project requires int8 inputs and int32 outputs.");
    if (NumCores == 0 || NumInputs == 0 || TileElements > 512 || MReuse == 0)
      $fatal(1, "Invalid dimensions, zero MReuse, or more than 512 MAC lanes.");
    if ((SramADataWidth % InDataWidth) != 0 || (SramBDataWidth % InDataWidth) != 0 ||
        AElements < NumInputs || BElements < NumInputs || (AElements % NumInputs) != 0)
      $fatal(1, "SRAM words require whole elements; A requires whole K blocks.");
    if ((SramCDataWidth % OutDataWidth) != 0 || CElements == 0 ||
        CElements > NumCores || (NumCores % CElements) != 0)
      $fatal(1, "SRAM C requires whole int32 lanes and whole words per core tile.");
    if (SramADataWidth + SramBDataWidth + SramCDataWidth > 1024)
      $fatal(1, "Total physical SRAM port width exceeds 1024 bits.");
  end
  always @(posedge clk_i) begin
    if (rst_ni) begin
      if (accept_start && (M_size_i == '0 || K_size_i == '0 || N_size_i == '0))
        $fatal(1, "M, K and N must be strictly positive.");
      if ((sram_a_re_o && ((a_word_address >> AddrWidth) != '0)) ||
          (sram_b_re_o && ((b_word_address >> AddrWidth) != '0)))
        $fatal(1, "Active SRAM word address does not fit AddrWidth.");
      if (reserve_output &&
          ((c_base_address +
            (CalcWidth'(active_cores)+CalcWidth'(CElements-1))/CalcWidth'(CElements) -
            CalcWidth'(1)) >> AddrWidth) != '0)
        $fatal(1, "C word address does not fit AddrWidth.");
      if (output_queue_used_q > QueueCountWidth'(QueueDepth))
        $fatal(1, "Output FIFO overflow.");
      if (last_c_write && (output_queue_used_q == '0))
        $fatal(1, "Output FIFO underflow.");
      if (result_valid !== fifo_pending_valid_q[1])
        $fatal(1, "Result-valid and reserved FIFO metadata are misaligned.");
      if (result_valid && c_ready_q[fifo_pending_slot_q[1]])
        $fatal(1, "A MAC result would overwrite a pending output entry.");
      if (state_q == TopLoad && !is_gemv && row_in_group != '0)
        $fatal(1, "B must be loaded only for the first row of an M group.");
      if (issue_block && !is_gemv && !a_cache_valid_q[row_in_group])
        $fatal(1, "An issued GeMM row has no cached A operand.");
    end
  end

  // Register accounting includes the actual buffers and FIFO storage once.
  // The context-bank PE accounts for its own arithmetic and accumulator bank.
  // The inherited value-change energy convention is retained.
  localparam int unsigned TopRegisterBits = $bits({
    state_q, M_size_q, K_size_q, N_size_q, read_beat_q,
    controller_done_seen_q, done_q, a_response_valid_q, b_response_valid_q,
    a_response_beat_q, b_response_beat_q, a_response_row_q, a_response_addr_q,
    tile_q, shared_q, a_cache_valid_q, a_cache_addr_q,
    mac_a_offset_q, mac_core_count_q, mac_input_count_q, mac_context_q,
    c_head_q, c_tail_q, output_queue_used_q, c_ready_q, c_data_q,
    c_addresses_q, c_counts_q, fifo_pending_valid_q, fifo_pending_slot_q, c_core_q
  });
  initial begin
    #0;
    metrics_pkg::area += real'(TopRegisterBits)/32.0;
  end
  for (genvar g=0; g<(TileBits+31)/32; g++) begin : gen_tile_energy
    localparam int unsigned ChunkWidth = ((TileBits-32*g) >= 32) ? 32 : TileBits-32*g;
    always @(tile_q[32*g +: ChunkWidth]) begin
      if (rst_ni) metrics_pkg::energy += real'(ChunkWidth)/32.0;
    end
  end
  for (genvar r=0; r<MReuse; r++) begin : gen_shared_energy
    for (genvar g=0; g<(SramADataWidth+31)/32; g++) begin : gen_word
      localparam int unsigned ChunkWidth = ((SramADataWidth-32*g) >= 32) ?
                                           32 : SramADataWidth-32*g;
      always @(shared_q[r][32*g +: ChunkWidth]) begin
        if (rst_ni) metrics_pkg::energy += real'(ChunkWidth)/32.0;
      end
    end
  end
  for (genvar q=0; q<QueueDepth; q++) begin : gen_output_energy
    for (genvar c=0; c<NumCores; c++) begin : gen_word
      always @(c_data_q[q][c*OutDataWidth +: OutDataWidth]) begin
        if (rst_ni) metrics_pkg::energy += real'(OutDataWidth)/32.0;
      end
    end
  end
  `define CPAEP_TOP_REGISTER_ENERGY(REG) \
    always @(REG) begin \
      if (rst_ni) metrics_pkg::energy += real'($bits(REG))/32.0; \
    end
  // Metadata banks are charged per changed entry, not as a whole bank when
  // only one row or FIFO slot is updated.
  for (genvar r=0; r<MReuse; r++) begin : gen_cache_tag_energy
    `CPAEP_TOP_REGISTER_ENERGY(a_cache_valid_q[r])
    `CPAEP_TOP_REGISTER_ENERGY(a_cache_addr_q[r])
  end
  for (genvar q=0; q<QueueDepth; q++) begin : gen_queue_tag_energy
    `CPAEP_TOP_REGISTER_ENERGY(c_ready_q[q])
    `CPAEP_TOP_REGISTER_ENERGY(c_addresses_q[q])
    `CPAEP_TOP_REGISTER_ENERGY(c_counts_q[q])
  end
  for (genvar p=0; p<2; p++) begin : gen_pending_tag_energy
    `CPAEP_TOP_REGISTER_ENERGY(fifo_pending_valid_q[p])
    `CPAEP_TOP_REGISTER_ENERGY(fifo_pending_slot_q[p])
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
  `CPAEP_TOP_REGISTER_ENERGY(a_response_row_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_response_addr_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_a_offset_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_core_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_input_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_context_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_head_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_tail_q)
  `CPAEP_TOP_REGISTER_ENERGY(output_queue_used_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_core_q)
  `undef CPAEP_TOP_REGISTER_ENERGY
  // synthesis translate_on
endmodule
