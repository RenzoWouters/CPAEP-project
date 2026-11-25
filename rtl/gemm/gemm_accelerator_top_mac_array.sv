//-----------------------------------------------------------------------------
// 512-MAC GeMM / GeMV accelerator, with fixed A/B/C = 832/64/128-bit SRAM ports.
// The three supplied synchronous SRAMs remain separate and unchanged.
//
// GeMM uses the SAME 16 dot-product cores: 2 M rows x 8 N columns x 32 K lanes.
// MReuse is the number of M ROWS sharing a B tile (64 by default), hence there
// are MReuse/GemmRows = 32 accumulator contexts. Loop order:
//   [M group][N tile][K block][row-pair context].
// Two 256-byte B banks overlap the next 32 reads with the current 32 issues.
// A streams through a two-word FIFO; its occupancy includes pending responses.
// A slot/B bank may be requested again on its last accepted issue: the old
// operand is consumed at the next edge BEFORE the replacement response's NBA.
// Explicit MAC bank/slot registers preserve that one-edge ownership extension.
//
// GeMV retains 16 M rows x 32 K lanes. Five A reads and four B reads run in
// parallel; the last response and issue share TopFlush. That accepted issue
// also prefetches word zero of the next tile. No SRAM-data-to-MAC bypass exists.
//
// Packing, low byte/element first (all padding is stored and read as full words):
//   GeMM A: [M row pair][K block][row within pair][K lane]. One A832 word
//           contains 64 payload bytes and 40 padding bytes.
//   GeMM B: [N tile of 8][K block][column within tile][K lane].
//           A 256-byte tile occupies 32 B64 words.
//   GeMV A: [M tile of 16][K block][core][K lane], 5 A832 words/tile.
//   GeMV B: [K block][K lane], 4 B64 words/32-element vector fragment.
//   GeMM C: row-major int32, each row padded to a whole C128 word.
//   GeMV C: consecutive int32 results, padded only at the end of the vector.
//-----------------------------------------------------------------------------
module gemm_accelerator_top_mac_array #(
  parameter int unsigned InDataWidth    = 8,
  parameter int unsigned OutDataWidth   = 32,
  parameter int unsigned AddrWidth      = 17,
  parameter int unsigned SizeAddrWidth  = 12,
  parameter int unsigned NumCores       = 16,
  parameter int unsigned NumInputs      = 32,
  parameter int unsigned SramADataWidth = 832,
  parameter int unsigned SramBDataWidth = 64,
  parameter int unsigned SramCDataWidth = 128,
  parameter int unsigned GemmRows       = 2,
  parameter int unsigned MReuse         = 64
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
  localparam int unsigned GemmCols = NumCores / GemmRows;
  localparam int unsigned NumContexts = MReuse / GemmRows;
  localparam int unsigned AElements = SramADataWidth / InDataWidth;
  localparam int unsigned BElements = SramBDataWidth / InDataWidth;
  localparam int unsigned CElements = SramCDataWidth / OutDataWidth;
  localparam int unsigned TileElements = NumCores * NumInputs;
  localparam int unsigned TileBits = TileElements * InDataWidth;
  localparam int unsigned GemmTileElements = GemmCols * NumInputs;
  localparam int unsigned GemmTileBits = GemmTileElements * InDataWidth;
  localparam int unsigned BufferBits = (2*GemmTileBits > TileBits) ? 2*GemmTileBits : TileBits;
  localparam int unsigned AReadWords = (TileElements + AElements-1) / AElements;
  localparam int unsigned BReadWords = (GemmTileElements + BElements-1) / BElements;
  localparam int unsigned VectorReadWords = (NumInputs + BElements-1) / BElements;
  localparam int unsigned GemvLoadWords = (AReadWords > VectorReadWords) ? AReadWords : VectorReadWords;
  localparam int unsigned MaxReadWords = (BReadWords > GemvLoadWords) ? BReadWords : GemvLoadWords;
  localparam int unsigned BeatWidth = (MaxReadWords <= 1) ? 1 : $clog2(MaxReadWords);
  localparam int unsigned CoreWidth = (NumCores <= 1) ? 1 : $clog2(NumCores);
  localparam int unsigned CoreCountWidth = $clog2(NumCores+1);
  localparam int unsigned RowCountWidth = $clog2(GemmRows+1);
  localparam int unsigned InputCountWidth = $clog2(NumInputs+1);
  localparam int unsigned ContextWidth = (NumContexts <= 1) ? 1 : $clog2(NumContexts);
  localparam int unsigned QueueDepth = NumContexts;
  localparam int unsigned QueuePtrWidth = (QueueDepth <= 1) ? 1 : $clog2(QueueDepth);
  localparam int unsigned QueueCountWidth = $clog2(QueueDepth+1);
  localparam int unsigned MaxStep0 = (NumCores > NumInputs) ? NumCores : NumInputs;
  localparam int unsigned MaxStep = (MReuse > MaxStep0) ? MReuse : MaxStep0;
  localparam int unsigned StepWidth = (MaxStep <= 1) ? 1 : $clog2(MaxStep);
  localparam int unsigned CountWidth = ((SizeAddrWidth > StepWidth) ? SizeAddrWidth : StepWidth) + 1;
  localparam int unsigned CalcWidth = 2*CountWidth + 2;

  // Encodings retained for the integration testbench's reset phase probes.
  typedef enum logic [2:0] {
    TopIdle, TopStart, TopLoad, TopFlush, TopIssue, TopDrain
  } top_state_t;
  top_state_t state_q;
  logic [SizeAddrWidth-1:0] M_size_q, K_size_q, N_size_q;
  logic [CountWidth-1:0] M_count, K_count, N_count;
  logic is_gemv, controller_busy, controller_done;
  logic valid_data, init_save, result_valid;
  logic accept_start, start_controller, active_job, issue_phase, issue_block;
  logic last_k_block, last_reuse_row, last_job_block;
  logic gemv_prefetch, gemm_release_bank, controller_done_seen_q, done_q;
  logic [BeatWidth-1:0] read_beat_q;

  // GeMM B producer. Valid means a COMPLETE tile, not just a reserved bank.
  logic [1:0] b_valid_q;
  logic b_fetch_bank_q, b_compute_bank_q, b_fetch_done_q;
  logic [BeatWidth-1:0] b_fetch_beat_q;
  logic [CountWidth-1:0] b_fetch_m_q, b_fetch_n_q, b_fetch_k_q;
  logic gemm_b_request, b_fetch_last_word, b_fetch_last_tile;

  // Two-entry A FIFO. Used includes a word whose SRAM response is pending.
  // Payload is retained on pop, because the MAC consumes it one edge later.
  logic [1:0][SramADataWidth-1:0] shared_q;
  logic [1:0] a_ready_q;
  logic a_head_q, a_tail_q;
  logic [1:0] a_used_q;
  logic [CountWidth-1:0] a_fetch_m_q, a_fetch_n_q, a_fetch_k_q;
  logic a_fetch_done_q, gemm_a_request, gemm_a_pop;
  logic [CalcWidth-1:0] a_fetch_group_base;
  logic a_fetch_last_pair, a_fetch_last_k, a_fetch_last_n, a_fetch_last_group;

  // Shared physical payload: two GeMM B banks OR one GeMV A tile.
  logic [BufferBits-1:0] tile_q;
  logic a_response_valid_q, b_response_valid_q;
  logic [BeatWidth-1:0] a_response_beat_q, b_response_beat_q;
  logic a_response_slot_q, b_response_bank_q;

  // The controller's data_valid and these tags are registered on the same edge.
  logic mac_a_slot_q, mac_b_bank_q;
  logic [CoreCountWidth-1:0] mac_core_count_q, mac_col_count_q;
  logic [RowCountWidth-1:0] mac_row_count_q;
  logic [InputCountWidth-1:0] mac_input_count_q;
  logic [ContextWidth-1:0] mac_context_q;

  // FIFO entries are RESERVED at last-K issue; payload arrives two edges later.
  logic [QueuePtrWidth-1:0] c_head_q, c_tail_q;
  logic [QueueCountWidth-1:0] output_queue_used_q;
  logic [QueueDepth-1:0] c_ready_q;
  logic [QueueDepth-1:0][NumCores*OutDataWidth-1:0] c_data_q;
  logic [QueueDepth-1:0][AddrWidth-1:0] c_addresses_q;
  logic [QueueDepth-1:0][CoreCountWidth-1:0] c_counts_q;
  logic [QueueDepth-1:0][RowCountWidth-1:0] c_row_counts_q;
  logic [1:0] fifo_pending_valid_q;
  logic [1:0][QueuePtrWidth-1:0] fifo_pending_slot_q;
  logic [CoreWidth-1:0] c_core_q;
  logic reserve_output, last_c_write, output_stall, output_slot_busy_q, c_valid_q;
  logic [AddrWidth-1:0] c_base_q;
  logic [CoreCountWidth-1:0] c_count_q;
  logic [NumCores*OutDataWidth-1:0] c_buffer_q;
  logic [CalcWidth-1:0] c_row_offset, c_column_offset, c_access_address;

  logic [CalcWidth-1:0] k_blocks, c_words_per_row, m_group_base, row_in_group;
  logic [CalcWidth-1:0] a_access_address, b_access_address, c_base_address;
  logic [CalcWidth-1:0] remaining_rows, remaining_cols, remaining_inputs, group_rows;
  logic [CoreCountWidth-1:0] active_cores, active_cols;
  logic [RowCountWidth-1:0] active_pair_rows;
  logic [InputCountWidth-1:0] active_inputs;
  logic [ContextWidth-1:0] row_context;

  logic signed [NumCores-1:0][NumInputs-1:0][InDataWidth-1:0] array_a_in, array_b_in;
  logic signed [NumCores-1:0][OutDataWidth-1:0] array_c_out;

  function automatic logic [QueuePtrWidth-1:0] next_slot(input logic [QueuePtrWidth-1:0] slot);
    return (slot == QueuePtrWidth'(QueueDepth-1)) ? '0 : slot + QueuePtrWidth'(1);
  endfunction

  assign accept_start = (state_q == TopIdle) && start_i;
  assign start_controller = (state_q == TopStart);
  assign active_job = (state_q == TopLoad) || (state_q == TopFlush) || (state_q == TopIssue);
  assign issue_phase = (state_q == TopFlush) || (!is_gemv && (state_q == TopIssue));
  assign output_slot_busy_q = (output_queue_used_q == QueueCountWidth'(QueueDepth));
  // Count a C stall only when an otherwise-ready block is blocked by C.
  assign output_stall = issue_phase && last_k_block && output_slot_busy_q && !last_c_write &&
                        (is_gemv || (b_valid_q[b_compute_bank_q] && a_ready_q[a_head_q]));
  assign issue_block = issue_phase && !output_stall &&
                       (is_gemv || (b_valid_q[b_compute_bank_q] && a_ready_q[a_head_q]));
  assign reserve_output = issue_block && last_k_block;
  assign gemm_release_bank = !is_gemv && issue_block && last_reuse_row;
  assign gemm_a_pop = !is_gemv && issue_block;
  assign gemv_prefetch = is_gemv && (state_q == TopFlush) && issue_block && !last_job_block;
  assign done_o = done_q;

  gemm_controller_mac_array #(
    .AddrWidth(CountWidth), .NumCores(NumCores), .NumInputs(NumInputs),
    .GemmRows(GemmRows), .MReuse(MReuse)
  ) i_gemm_controller_mac_array (
    .clk_i(clk_i), .rst_ni(rst_ni), .start_i(start_controller),
    .input_valid_i(issue_block), .result_valid_o(result_valid), .mem_re_o(),
    .data_valid_o(valid_data), .init_save_o(init_save), .busy_o(controller_busy),
    .done_o(controller_done), .is_gemv_o(is_gemv),
    .M_size_i(CountWidth'(M_size_q)), .K_size_i(CountWidth'(K_size_q)),
    .N_size_i(CountWidth'(N_size_q)),
    .M_count_o(M_count), .K_count_o(K_count), .N_count_o(N_count)
  );

  //---------------------------
  // Sizes and metadata for the next accepted compute block.
  //---------------------------
  always_comb begin
    k_blocks = (CalcWidth'(K_size_q) + CalcWidth'(NumInputs-1)) / CalcWidth'(NumInputs);
    c_words_per_row = (CalcWidth'(N_size_q) + CalcWidth'(CElements-1)) / CalcWidth'(CElements);
    row_in_group = CalcWidth'(M_count) % CalcWidth'(MReuse);
    m_group_base = CalcWidth'(M_count) - row_in_group;
    group_rows = CalcWidth'(M_size_q) - m_group_base;
    if (group_rows > CalcWidth'(MReuse)) group_rows = CalcWidth'(MReuse);
    remaining_rows = CalcWidth'(M_size_q) - CalcWidth'(M_count);
    remaining_cols = CalcWidth'(N_size_q) - CalcWidth'(N_count);
    remaining_inputs = CalcWidth'(K_size_q) - CalcWidth'(K_count);
    active_pair_rows = (remaining_rows < CalcWidth'(GemmRows)) ?
                       RowCountWidth'(remaining_rows) : RowCountWidth'(GemmRows);
    active_cols = (remaining_cols < CalcWidth'(GemmCols)) ?
                  CoreCountWidth'(remaining_cols) : CoreCountWidth'(GemmCols);
    active_cores = is_gemv ?
      ((remaining_rows < CalcWidth'(NumCores)) ? CoreCountWidth'(remaining_rows) : CoreCountWidth'(NumCores)) :
      CoreCountWidth'(active_pair_rows) * active_cols;
    active_inputs = (remaining_inputs < CalcWidth'(NumInputs)) ?
                    InputCountWidth'(remaining_inputs) : InputCountWidth'(NumInputs);
    row_context = is_gemv ? '0 : ContextWidth'(row_in_group / CalcWidth'(GemmRows));
    c_base_address = is_gemv ? CalcWidth'(M_count)/CalcWidth'(CElements) :
                     CalcWidth'(M_count)*c_words_per_row + CalcWidth'(N_count)/CalcWidth'(CElements);
    last_k_block = (CalcWidth'(K_count)+CalcWidth'(NumInputs) >= CalcWidth'(K_size_q));
    last_reuse_row = is_gemv || (row_in_group+CalcWidth'(GemmRows) >= group_rows);
    last_job_block = last_k_block && (is_gemv ?
      (CalcWidth'(M_count)+CalcWidth'(NumCores) >= CalcWidth'(M_size_q)) :
      ((CalcWidth'(M_count)+CalcWidth'(GemmRows) >= CalcWidth'(M_size_q)) &&
       (CalcWidth'(N_count)+CalcWidth'(GemmCols) >= CalcWidth'(N_size_q))));

    a_fetch_group_base = (CalcWidth'(a_fetch_m_q)/CalcWidth'(MReuse))*CalcWidth'(MReuse);
    a_fetch_last_pair = (CalcWidth'(a_fetch_m_q)+CalcWidth'(GemmRows) >=
                         a_fetch_group_base+CalcWidth'(MReuse)) ||
                       (CalcWidth'(a_fetch_m_q)+CalcWidth'(GemmRows) >= CalcWidth'(M_size_q));
    a_fetch_last_k = (CalcWidth'(a_fetch_k_q)+CalcWidth'(NumInputs) >= CalcWidth'(K_size_q));
    a_fetch_last_n = (CalcWidth'(a_fetch_n_q)+CalcWidth'(GemmCols) >= CalcWidth'(N_size_q));
    a_fetch_last_group = (a_fetch_group_base+CalcWidth'(MReuse) >= CalcWidth'(M_size_q));
    b_fetch_last_word = (b_fetch_beat_q == BeatWidth'(BReadWords-1));
    b_fetch_last_tile = (CalcWidth'(b_fetch_k_q)+CalcWidth'(NumInputs) >= CalcWidth'(K_size_q)) &&
      (CalcWidth'(b_fetch_n_q)+CalcWidth'(GemmCols) >= CalcWidth'(N_size_q)) &&
      (CalcWidth'(b_fetch_m_q)+CalcWidth'(MReuse) >= CalcWidth'(M_size_q));
  end

  // Reuse on the last issue edge is safe: response capture occurs via NBA on
  // the following edge, after that edge's MAC has consumed the old contents.
  assign gemm_a_request = active_job && !is_gemv && !a_fetch_done_q &&
                           ((a_used_q < 2) || gemm_a_pop);
  assign gemm_b_request = active_job && !is_gemv && !b_fetch_done_q &&
    (!b_valid_q[b_fetch_bank_q] || (gemm_release_bank && (b_fetch_bank_q == b_compute_bank_q)));

  always_comb begin
    sram_a_re_o = 1'b0;
    sram_b_re_o = 1'b0;
    a_access_address = '0;
    b_access_address = '0;
    if (is_gemv) begin
      a_access_address = ((CalcWidth'(M_count)/CalcWidth'(NumCores))*k_blocks +
                           CalcWidth'(K_count)/CalcWidth'(NumInputs))*CalcWidth'(AReadWords);
      b_access_address = (CalcWidth'(K_count)/CalcWidth'(NumInputs))*CalcWidth'(VectorReadWords);
      if (gemv_prefetch) begin
        a_access_address += CalcWidth'(AReadWords);
        b_access_address = last_k_block ? '0 : b_access_address+CalcWidth'(VectorReadWords);
        sram_a_re_o = 1'b1;
        sram_b_re_o = 1'b1;
      end else if (state_q == TopLoad) begin
        a_access_address += CalcWidth'(read_beat_q);
        b_access_address += CalcWidth'(read_beat_q);
        sram_a_re_o = (int'(read_beat_q) < AReadWords);
        sram_b_re_o = (int'(read_beat_q) < VectorReadWords);
      end
    end else begin
      a_access_address = (CalcWidth'(a_fetch_m_q)/CalcWidth'(GemmRows))*k_blocks +
                           CalcWidth'(a_fetch_k_q)/CalcWidth'(NumInputs);
      b_access_address = ((CalcWidth'(b_fetch_n_q)/CalcWidth'(GemmCols))*k_blocks +
                           CalcWidth'(b_fetch_k_q)/CalcWidth'(NumInputs))*CalcWidth'(BReadWords) +
                           CalcWidth'(b_fetch_beat_q);
      sram_a_re_o = gemm_a_request;
      sram_b_re_o = gemm_b_request;
    end
  end
  assign sram_a_addr_o = AddrWidth'(a_access_address);
  assign sram_b_addr_o = AddrWidth'(b_access_address);

  //---------------------------
  // Job state. GeMM stays in TopIssue while the independent producers stream.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= TopIdle;
      M_size_q <= '0; K_size_q <= '0; N_size_q <= '0;
      read_beat_q <= '0;
      controller_done_seen_q <= 1'b0; done_q <= 1'b0;
    end else begin
      done_q <= 1'b0;
      if (controller_done) controller_done_seen_q <= 1'b1;
      case (state_q)
        TopIdle: if (start_i) begin
          M_size_q <= M_size_i; K_size_q <= K_size_i; N_size_q <= N_size_i;
          read_beat_q <= '0; controller_done_seen_q <= 1'b0;
          state_q <= TopStart;
        end
        TopStart: state_q <= TopLoad;
        TopLoad: begin
          if (is_gemv) begin
            if (read_beat_q == BeatWidth'(GemvLoadWords-1)) begin
              read_beat_q <= '0; state_q <= TopFlush;
            end else read_beat_q <= read_beat_q + BeatWidth'(1);
          end else if (gemm_b_request && b_fetch_last_word) state_q <= TopFlush;
        end
        TopFlush: if (issue_block) begin
          if (last_job_block) state_q <= TopDrain;
          else if (!is_gemv) state_q <= TopIssue;
          else if (GemvLoadWords == 1) begin
            read_beat_q <= '0; state_q <= TopFlush;
          end else begin
            read_beat_q <= BeatWidth'(1); state_q <= TopLoad;
          end
        end
        TopIssue: if (issue_block && last_job_block) state_q <= TopDrain;
        TopDrain: if (controller_done_seen_q && (output_queue_used_q == '0) &&
                     !a_response_valid_q && !b_response_valid_q) begin
          done_q <= 1'b1; state_q <= TopIdle;
        end
        default: state_q <= TopIdle;
      endcase
    end
  end

  //---------------------------
  // A stream producer and two-slot reservation FIFO.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      a_ready_q <= '0; a_head_q <= 1'b0; a_tail_q <= 1'b0; a_used_q <= '0;
      a_fetch_m_q <= '0; a_fetch_n_q <= '0; a_fetch_k_q <= '0; a_fetch_done_q <= 1'b0;
    end else if (accept_start) begin
      a_ready_q <= '0; a_head_q <= 1'b0; a_tail_q <= 1'b0; a_used_q <= '0;
      a_fetch_m_q <= '0; a_fetch_n_q <= '0; a_fetch_k_q <= '0; a_fetch_done_q <= 1'b0;
    end else if (!is_gemv) begin
      case ({gemm_a_request, gemm_a_pop})
        2'b10: a_used_q <= a_used_q + 2'd1;
        2'b01: a_used_q <= a_used_q - 2'd1;
        default: ;
      endcase
      if (gemm_a_pop) begin
        a_ready_q[a_head_q] <= 1'b0;
        a_head_q <= !a_head_q;
      end
      if (a_response_valid_q) a_ready_q[a_response_slot_q] <= 1'b1;
      if (gemm_a_request) begin
        a_tail_q <= !a_tail_q;
        if (a_fetch_last_pair) begin
          a_fetch_m_q <= CountWidth'(a_fetch_group_base);
          if (a_fetch_last_k) begin
            a_fetch_k_q <= '0;
            if (a_fetch_last_n) begin
              a_fetch_n_q <= '0;
              a_fetch_m_q <= CountWidth'(a_fetch_group_base+CalcWidth'(MReuse));
              if (a_fetch_last_group) a_fetch_done_q <= 1'b1;
            end else a_fetch_n_q <= a_fetch_n_q + CountWidth'(GemmCols);
          end else a_fetch_k_q <= a_fetch_k_q + CountWidth'(NumInputs);
        end else a_fetch_m_q <= a_fetch_m_q + CountWidth'(GemmRows);
      end
    end
  end

  //---------------------------
  // B tile producer and bank ownership.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      b_valid_q <= '0; b_fetch_bank_q <= 1'b0; b_compute_bank_q <= 1'b0;
      b_fetch_done_q <= 1'b0; b_fetch_beat_q <= '0;
      b_fetch_m_q <= '0; b_fetch_n_q <= '0; b_fetch_k_q <= '0;
    end else if (accept_start) begin
      b_valid_q <= '0; b_fetch_bank_q <= 1'b0; b_compute_bank_q <= 1'b0;
      b_fetch_done_q <= 1'b0; b_fetch_beat_q <= '0;
      b_fetch_m_q <= '0; b_fetch_n_q <= '0; b_fetch_k_q <= '0;
    end else if (!is_gemv) begin
      if (gemm_release_bank) begin
        b_valid_q[b_compute_bank_q] <= 1'b0;
        b_compute_bank_q <= !b_compute_bank_q;
      end
      if (b_response_valid_q && (b_response_beat_q == BeatWidth'(BReadWords-1)))
        b_valid_q[b_response_bank_q] <= 1'b1;
      if (gemm_b_request) begin
        if (b_fetch_last_word) begin
          b_fetch_beat_q <= '0;
          b_fetch_bank_q <= !b_fetch_bank_q;
          if (b_fetch_last_tile) b_fetch_done_q <= 1'b1;
          if (CalcWidth'(b_fetch_k_q)+CalcWidth'(NumInputs) >= CalcWidth'(K_size_q)) begin
            b_fetch_k_q <= '0;
            if (CalcWidth'(b_fetch_n_q)+CalcWidth'(GemmCols) >= CalcWidth'(N_size_q)) begin
              b_fetch_n_q <= '0;
              b_fetch_m_q <= b_fetch_m_q + CountWidth'(MReuse);
            end else b_fetch_n_q <= b_fetch_n_q + CountWidth'(GemmCols);
          end else b_fetch_k_q <= b_fetch_k_q + CountWidth'(NumInputs);
        end else b_fetch_beat_q <= b_fetch_beat_q + BeatWidth'(1);
      end
    end
  end

  //---------------------------
  // SRAM responses. Inputs from the synchronous memories are captured one
  // edge after their requests; requests and responses may be consecutive.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      a_response_valid_q <= 1'b0; b_response_valid_q <= 1'b0;
      a_response_beat_q <= '0; b_response_beat_q <= '0;
      a_response_slot_q <= 1'b0; b_response_bank_q <= 1'b0;
      tile_q <= '0; shared_q <= '0;
    end else begin
      a_response_valid_q <= sram_a_re_o;
      b_response_valid_q <= sram_b_re_o;
      if (sram_a_re_o) begin
        a_response_beat_q <= gemv_prefetch ? '0 : read_beat_q;
        a_response_slot_q <= a_tail_q;
      end
      if (sram_b_re_o) begin
        b_response_beat_q <= is_gemv ? (gemv_prefetch ? '0 : read_beat_q) : b_fetch_beat_q;
        b_response_bank_q <= b_fetch_bank_q;
      end
      if (a_response_valid_q) begin
        if (is_gemv) begin
          for (int lane=0; lane<AElements; lane++)
            if (int'(a_response_beat_q)*AElements+lane < TileElements)
              tile_q[(int'(a_response_beat_q)*AElements+lane)*InDataWidth +: InDataWidth]
                <= sram_a_rdata_i[lane*InDataWidth +: InDataWidth];
        end else shared_q[a_response_slot_q] <= sram_a_rdata_i;
      end
      if (b_response_valid_q) begin
        for (int lane=0; lane<BElements; lane++) begin
          if (is_gemv) begin
            if (int'(b_response_beat_q)*BElements+lane < NumInputs)
              shared_q[0][(int'(b_response_beat_q)*BElements+lane)*InDataWidth +: InDataWidth]
                <= sram_b_rdata_i[lane*InDataWidth +: InDataWidth];
          end else if (int'(b_response_beat_q)*BElements+lane < GemmTileElements) begin
            tile_q[(int'(b_response_bank_q)*GemmTileElements+
                    int'(b_response_beat_q)*BElements+lane)*InDataWidth +: InDataWidth]
              <= sram_b_rdata_i[lane*InDataWidth +: InDataWidth];
          end
        end
      end
    end
  end

  //---------------------------
  // MAC operand ownership and result FIFO.
  //---------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      mac_a_slot_q <= 1'b0; mac_b_bank_q <= 1'b0;
      mac_core_count_q <= '0; mac_col_count_q <= '0; mac_row_count_q <= '0;
      mac_input_count_q <= '0; mac_context_q <= '0;
      c_head_q <= '0; c_tail_q <= '0; output_queue_used_q <= '0;
      c_ready_q <= '0; c_addresses_q <= '0;
      for (int slot=0; slot<QueueDepth; slot++) c_data_q[slot] <= '0;
      c_counts_q <= '0; c_row_counts_q <= '0;
      fifo_pending_valid_q <= '0; fifo_pending_slot_q <= '0; c_core_q <= '0;
    end else begin
      if (issue_block) begin
        mac_a_slot_q <= a_head_q; mac_b_bank_q <= b_compute_bank_q;
        mac_core_count_q <= active_cores; mac_col_count_q <= active_cols;
        mac_row_count_q <= active_pair_rows; mac_input_count_q <= active_inputs;
        mac_context_q <= row_context;
      end
      fifo_pending_valid_q <= {fifo_pending_valid_q[0], reserve_output};
      if (reserve_output) begin
        fifo_pending_slot_q[0] <= c_tail_q;
        c_addresses_q[c_tail_q] <= AddrWidth'(c_base_address);
        c_counts_q[c_tail_q] <= is_gemv ? active_cores : active_cols;
        c_row_counts_q[c_tail_q] <= is_gemv ? RowCountWidth'(1) : active_pair_rows;
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
          c_head_q <= next_slot(c_head_q); c_core_q <= '0;
        end else if (!is_gemv && (c_column_offset+CalcWidth'(CElements) >= CalcWidth'(c_count_q)))
          c_core_q <= CoreWidth'((c_row_offset+CalcWidth'(1))*CalcWidth'(GemmCols));
        else c_core_q <= c_core_q + CoreWidth'(CElements);
      end
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
  assign c_row_offset = is_gemv ? '0 : CalcWidth'(c_core_q)/CalcWidth'(GemmCols);
  assign c_column_offset = is_gemv ? CalcWidth'(c_core_q) : CalcWidth'(c_core_q)%CalcWidth'(GemmCols);
  assign c_access_address = CalcWidth'(c_base_q) + c_row_offset*c_words_per_row +
                            c_column_offset/CalcWidth'(CElements);
  assign sram_c_addr_o = AddrWidth'(c_access_address);
  assign sram_c_we_o = c_valid_q;
  assign last_c_write = c_valid_q &&
    (c_column_offset+CalcWidth'(CElements) >= CalcWidth'(c_count_q)) &&
    (is_gemv || (c_row_offset+CalcWidth'(1) >= CalcWidth'(c_row_counts_q[c_head_q])));
  always_comb begin
    sram_c_wdata_o = '0;
    for (int lane=0; lane<CElements; lane++)
      if (c_column_offset+CalcWidth'(lane) < CalcWidth'(c_count_q))
        sram_c_wdata_o[lane*OutDataWidth +: OutDataWidth] =
          c_buffer_q[(int'(c_core_q)+lane)*OutDataWidth +: OutDataWidth];
  end

  //---------------------------
  // Original arithmetic blocks: no extra multipliers or extra MAC array.
  //---------------------------
  always_comb begin
    array_a_in = '0; array_b_in = '0;
    for (int c=0; c<NumCores; c++) begin
      for (int j=0; j<NumInputs; j++) begin
        if (j < int'(mac_input_count_q)) begin
          if (is_gemv) begin
            if (c < int'(mac_core_count_q)) begin
              array_a_in[c][j] = $signed(tile_q[(c*NumInputs+j)*InDataWidth +: InDataWidth]);
              array_b_in[c][j] = $signed(shared_q[0][j*InDataWidth +: InDataWidth]);
            end
          end else if ((c/GemmCols < int'(mac_row_count_q)) &&
                       (c%GemmCols < int'(mac_col_count_q))) begin
            array_a_in[c][j] = $signed(shared_q[mac_a_slot_q]
              [((c/GemmCols)*NumInputs+j)*InDataWidth +: InDataWidth]);
            array_b_in[c][j] = $signed(tile_q
              [(int'(mac_b_bank_q)*GemmTileElements+(c%GemmCols)*NumInputs+j)*InDataWidth +: InDataWidth]);
          end
        end
      end
    end
  end
  mac_array #(
    .InDataWidth(InDataWidth), .NumCores(NumCores), .NumInputs(NumInputs),
    .OutDataWidth(OutDataWidth), .NumContexts(NumContexts)
  ) i_mac_array (
    .clk_i(clk_i), .rst_ni(rst_ni), .a_i(array_a_in), .b_i(array_b_in),
    .a_valid_i(valid_data), .b_valid_i(valid_data), .init_save_i(init_save),
    .acc_clr_i(!controller_busy), .acc_context_i(mac_context_q), .c_o(array_c_out)
  );

  // synthesis translate_off
  initial begin
    if (InDataWidth != 8 || OutDataWidth != 32)
      $fatal(1, "This project requires int8 inputs and int32 outputs.");
    if (NumCores == 0 || NumInputs == 0 || TileElements > 512 ||
        GemmRows == 0 || (NumCores % GemmRows) != 0 ||
        MReuse == 0 || (MReuse % GemmRows) != 0)
      $fatal(1, "Invalid array/mapping dimensions or more than 512 MAC lanes.");
    if ((SramADataWidth % InDataWidth) != 0 || (SramBDataWidth % InDataWidth) != 0 ||
        AElements < GemmRows*NumInputs || BElements == 0)
      $fatal(1, "A must hold one row group; A/B ports must hold whole int8 elements.");
    if ((SramCDataWidth % OutDataWidth) != 0 || CElements == 0 ||
        CElements > GemmCols || (GemmCols % CElements) != 0)
      $fatal(1, "Each GeMM column tile must contain whole C words.");
    if (SramADataWidth+SramBDataWidth+SramCDataWidth > 1024)
      $fatal(1, "Total physical SRAM port width exceeds 1024 bits.");
  end
  always @(posedge clk_i) begin
    if (rst_ni) begin
      if (accept_start && (M_size_i == '0 || K_size_i == '0 || N_size_i == '0))
        $fatal(1, "M, K and N must be strictly positive.");
      if ((sram_a_re_o && ((a_access_address >> AddrWidth) != '0)) ||
          (sram_b_re_o && ((b_access_address >> AddrWidth) != '0)) ||
          (sram_c_we_o && ((c_access_address >> AddrWidth) != '0)))
        $fatal(1, "Active SRAM word address does not fit AddrWidth.");
      // Check before narrowing the descriptor to AddrWidth. Otherwise an
      // overflowing base could already be truncated before serialization.
      if (reserve_output &&
          (((c_base_address + (is_gemv ? '0 :
              (CalcWidth'(active_pair_rows)-CalcWidth'(1))*c_words_per_row) +
             ((is_gemv ? CalcWidth'(active_cores) : CalcWidth'(active_cols)) +
               CalcWidth'(CElements-1))/CalcWidth'(CElements)-CalcWidth'(1))
            >> AddrWidth) != '0))
        $fatal(1, "Reserved C result would exceed the word-address range.");
      if (a_used_q > 2 || (gemm_a_pop && (a_used_q == 0)))
        $fatal(1, "A stream FIFO overflow/underflow.");
      if (gemm_a_request && a_ready_q[a_tail_q] &&
          !(gemm_a_pop && (a_head_q == a_tail_q)))
        $fatal(1, "A producer would overwrite an unconsumed operand.");
      if (output_queue_used_q > QueueCountWidth'(QueueDepth) ||
          (last_c_write && (output_queue_used_q == '0)))
        $fatal(1, "Output FIFO overflow/underflow.");
      if (result_valid !== fifo_pending_valid_q[1])
        $fatal(1, "MAC result-valid and output reservation are misaligned.");
      if (result_valid && c_ready_q[fifo_pending_slot_q[1]])
        $fatal(1, "A MAC result would overwrite a pending output entry.");
      if (gemv_prefetch && read_beat_q != '0)
        $fatal(1, "GeMV prefetch must start at the next tile's first word.");
      if (done_q && (sram_a_re_o || sram_b_re_o || sram_c_we_o ||
                     valid_data || result_valid || output_queue_used_q != '0))
        $fatal(1, "done asserted before all memory/MAC activity drained.");
      if (done_q && !is_gemv &&
          ((a_used_q != '0) || !a_fetch_done_q || !b_fetch_done_q))
        $fatal(1, "GeMM operand producers did not finish with the controller.");
    end
  end

  // Register accounting includes the actual buffers and tags once. Arithmetic
  // and accumulator contexts are accounted for by the unchanged PE modules.
  localparam int unsigned TopRegisterBits = $bits({
    state_q, M_size_q, K_size_q, N_size_q, read_beat_q, controller_done_seen_q, done_q,
    b_valid_q, b_fetch_bank_q, b_compute_bank_q, b_fetch_done_q, b_fetch_beat_q,
    b_fetch_m_q, b_fetch_n_q, b_fetch_k_q,
    shared_q, a_ready_q, a_head_q, a_tail_q, a_used_q,
    a_fetch_m_q, a_fetch_n_q, a_fetch_k_q, a_fetch_done_q, tile_q,
    a_response_valid_q, b_response_valid_q, a_response_beat_q, b_response_beat_q,
    a_response_slot_q, b_response_bank_q,
    mac_a_slot_q, mac_b_bank_q, mac_core_count_q, mac_col_count_q, mac_row_count_q,
    mac_input_count_q, mac_context_q, c_head_q, c_tail_q, output_queue_used_q,
    c_ready_q, c_data_q, c_addresses_q, c_counts_q, c_row_counts_q,
    fifo_pending_valid_q, fifo_pending_slot_q, c_core_q
  });
  initial begin
    #0;
    metrics_pkg::area += real'(TopRegisterBits)/32.0;
  end
  for (genvar g=0; g<(BufferBits+31)/32; g++) begin : gen_tile_energy
    localparam int unsigned Bits = ((BufferBits-32*g) >= 32) ? 32 : BufferBits-32*g;
    always @(tile_q[32*g +: Bits]) if (rst_ni) metrics_pkg::energy += real'(Bits)/32.0;
  end
  for (genvar r=0; r<2; r++) begin : gen_shared_energy
    for (genvar g=0; g<(SramADataWidth+31)/32; g++) begin : gen_word
      localparam int unsigned Bits = ((SramADataWidth-32*g) >= 32) ? 32 : SramADataWidth-32*g;
      always @(shared_q[r][32*g +: Bits]) if (rst_ni) metrics_pkg::energy += real'(Bits)/32.0;
    end
  end
  for (genvar q=0; q<QueueDepth; q++) begin : gen_output_energy
    for (genvar c=0; c<NumCores; c++) begin : gen_word
      always @(c_data_q[q][c*OutDataWidth +: OutDataWidth])
        if (rst_ni) metrics_pkg::energy += real'(OutDataWidth)/32.0;
    end
  end
  `define CPAEP_TOP_REGISTER_ENERGY(REG) \
    always @(REG) begin \
      if (rst_ni) metrics_pkg::energy += real'($bits(REG))/32.0; \
    end
  for (genvar q=0; q<QueueDepth; q++) begin : gen_queue_tag_energy
    `CPAEP_TOP_REGISTER_ENERGY(c_ready_q[q])
    `CPAEP_TOP_REGISTER_ENERGY(c_addresses_q[q])
    `CPAEP_TOP_REGISTER_ENERGY(c_counts_q[q])
    `CPAEP_TOP_REGISTER_ENERGY(c_row_counts_q[q])
  end
  for (genvar p=0; p<2; p++) begin : gen_pending_tag_energy
    `CPAEP_TOP_REGISTER_ENERGY(fifo_pending_valid_q[p])
    `CPAEP_TOP_REGISTER_ENERGY(fifo_pending_slot_q[p])
    `CPAEP_TOP_REGISTER_ENERGY(a_ready_q[p])
    `CPAEP_TOP_REGISTER_ENERGY(b_valid_q[p])
  end
  `CPAEP_TOP_REGISTER_ENERGY(state_q)
  `CPAEP_TOP_REGISTER_ENERGY(M_size_q)
  `CPAEP_TOP_REGISTER_ENERGY(K_size_q)
  `CPAEP_TOP_REGISTER_ENERGY(N_size_q)
  `CPAEP_TOP_REGISTER_ENERGY(read_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(controller_done_seen_q)
  `CPAEP_TOP_REGISTER_ENERGY(done_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_fetch_bank_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_compute_bank_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_fetch_done_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_fetch_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_fetch_m_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_fetch_n_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_fetch_k_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_head_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_tail_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_used_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_fetch_m_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_fetch_n_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_fetch_k_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_fetch_done_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_response_valid_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_response_valid_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_response_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_response_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(a_response_slot_q)
  `CPAEP_TOP_REGISTER_ENERGY(b_response_bank_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_a_slot_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_b_bank_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_core_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_col_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_row_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_input_count_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_context_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_head_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_tail_q)
  `CPAEP_TOP_REGISTER_ENERGY(output_queue_used_q)
  `CPAEP_TOP_REGISTER_ENERGY(c_core_q)
  `undef CPAEP_TOP_REGISTER_ENERGY
  // synthesis translate_on
endmodule
