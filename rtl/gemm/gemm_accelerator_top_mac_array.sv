//-----------------------------------------------------------------------------
// 512-MAC GeMM / GeMV accelerator, eight fixed 128-bit half-duplex SRAM banks.
// Sum of active read/write bank words never exceeds 1024 bits per cycle.
// The supplied SRAM internals and their synchronous read latency are unchanged.
//
// GeMM: 2 M x8 N x32 K, MReuse rows per B tile, unchanged accumulator contexts.
// Banks0..3 deliver a64-byte A row pair; banks4..5 deliver32 B bytes; banks6..7
// write eight int32 outputs. Two B tile buffers and two A slots hide input reads.
//   A: addr=(m/2)*ceil(K/32)+k/32; byte=(m%2)*32+k%32, across banks0..3.
//   B: addr=((n/8)*ceil(K/32)+k/32)*8+((n%8)*32+k%32)/32,
//      byte modulo32 across banks4..5.
//   C: addr=m*ceil(N/8)+n/8, eight elements across banks6..7; pad each row.
//
// GeMV: A tiles remain [Mtile16][Kblock32][core][K lane], now exactly four
// 128-byte bursts across all banks. Logical tail lanes remain present/masked.
// A occupies rows0..Arows-1, Arows=ceil(M/16)*ceil(K/32)*4.
// B starts at Arows; C starts at Arows+ceil(K/128). Both are striped in16-byte
// words across all banks. B is cached once per job for K<=VectorCacheElements.
// Larger K uses one32-byte B fragment per tile, preserving the size interface.
// C pairs two16-result groups into one bank cycle; a final lone group drains.
// C writes pause A/cache requests, but never discard pending SRAM responses.
// Each accessed bank word is charged in full, including final word padding.
//-----------------------------------------------------------------------------
module gemm_accelerator_top_mac_array #(
  parameter int unsigned InDataWidth    = 8,
  parameter int unsigned OutDataWidth   = 32,
  parameter int unsigned AddrWidth      = 17,
  parameter int unsigned SizeAddrWidth  = 12,
  parameter int unsigned NumCores       = 16,
  parameter int unsigned NumInputs      = 32,
  parameter int unsigned NumBanks       = 8,
  parameter int unsigned BankDataWidth  = 128,
  parameter int unsigned VectorCacheElements = 2048,
  parameter int unsigned GemmRows       = 2,
  parameter int unsigned MReuse         = 64,
  parameter int unsigned OutputQueueDepth = 25
) (
  input  logic                            clk_i,
  input  logic                            rst_ni,
  input  logic                            start_i,
  input  logic [SizeAddrWidth-1:0]         M_size_i,
  input  logic [SizeAddrWidth-1:0]         K_size_i,
  input  logic [SizeAddrWidth-1:0]         N_size_i,
  output logic [NumBanks-1:0][AddrWidth-1:0]     bank_addr_o,
  output logic [NumBanks-1:0]                    bank_re_o,
  output logic [NumBanks-1:0]                    bank_we_o,
  input  logic [NumBanks-1:0][BankDataWidth-1:0]  bank_rdata_i,
  output logic [NumBanks-1:0][BankDataWidth-1:0]  bank_wdata_o,
  output logic                            done_o
);
  localparam int unsigned GemmCols = NumCores / GemmRows;
  localparam int unsigned NumContexts = MReuse / GemmRows;
  localparam int unsigned BankBytes = BankDataWidth/InDataWidth;
  localparam int unsigned BurstBytes = NumBanks*BankBytes;
  localparam int unsigned SramADataWidth = 4*BankDataWidth;
  localparam int unsigned SramBDataWidth = 2*BankDataWidth;
  localparam int unsigned SramCDataWidth = 2*BankDataWidth;
  localparam int unsigned AElements = SramADataWidth / InDataWidth;
  localparam int unsigned BElements = SramBDataWidth / InDataWidth;
  localparam int unsigned CElements = SramCDataWidth / OutDataWidth;
  localparam int unsigned TileElements = NumCores * NumInputs;
  localparam int unsigned TileBits = TileElements * InDataWidth;
  localparam int unsigned GemmTileElements = GemmCols * NumInputs;
  localparam int unsigned GemmTileBits = GemmTileElements * InDataWidth;
  localparam int unsigned BufferBits = (2*GemmTileBits > TileBits) ? 2*GemmTileBits : TileBits;
  localparam int unsigned AReadWords = (TileElements + BurstBytes-1) / BurstBytes;
  localparam int unsigned BReadWords = (GemmTileElements + BElements-1) / BElements;
  localparam int unsigned VectorReadWords = (NumInputs + BElements-1) / BElements;
  localparam int unsigned GemvLoadWords = AReadWords;
  localparam int unsigned MaxReadWords = (BReadWords > GemvLoadWords) ? BReadWords : GemvLoadWords;
  localparam int unsigned BeatWidth = (MaxReadWords <= 1) ? 1 : $clog2(MaxReadWords);
  localparam int unsigned CoreWidth = (NumCores <= 1) ? 1 : $clog2(NumCores);
  localparam int unsigned CoreCountWidth = $clog2(NumCores+1);
  localparam int unsigned RowCountWidth = $clog2(GemmRows+1);
  localparam int unsigned InputCountWidth = $clog2(NumInputs+1);
  localparam int unsigned ContextWidth = (NumContexts <= 1) ? 1 : $clog2(NumContexts);
  localparam int unsigned QueueDepth = OutputQueueDepth;
  localparam int unsigned QueuePtrWidth = (QueueDepth <= 1) ? 1 : $clog2(QueueDepth);
  localparam int unsigned QueueCountWidth = (QueueDepth <= 1) ? 1 : $clog2(QueueDepth+1);
  localparam int unsigned MaxStep0 = (NumCores > NumInputs) ? NumCores : NumInputs;
  localparam int unsigned MaxStep = (MReuse > MaxStep0) ? MReuse : MaxStep0;
  localparam int unsigned StepWidth = (MaxStep <= 1) ? 1 : $clog2(MaxStep);
  localparam int unsigned CountWidth = ((SizeAddrWidth > StepWidth) ? SizeAddrWidth : StepWidth) + 1;
  localparam int unsigned CalcWidth = 2*CountWidth + 2;
  // Encodings retained for the integration testbench's reset phase probes.
  typedef enum logic [2:0] {
    TopIdle, TopStart, TopLoad, TopFlush, TopIssue, TopDrain, TopCache, TopCacheFlush
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

  // Shared-port GeMV stream and local vector cache. Cache data may be stale
  // between jobs, but cache_ready_q is cleared and every required byte reloaded.
  logic [CalcWidth-1:0] gemv_a_word_q, gemv_b_base, gemv_c_base;
  logic [CalcWidth-1:0] cache_beat_q, cache_response_beat_q;
  logic [NumBanks-1:0] cache_bank_mask, cache_response_mask_q;
  logic [2:0] cache_first_bank, cache_response_first_bank_q;
  logic cache_request, cache_response_valid_q, cache_last_request;
  logic cache_fallback, cache_ready_q, gemv_a_request;
  logic [VectorCacheElements*InDataWidth-1:0] vector_cache_q;
  logic [CountWidth-1:0] mac_k_q;
  logic [1:0] c_pop_count;
  logic gemv_c_write, gemv_pair_write;
  logic [NumBanks-1:0][CalcWidth-1:0] bank_access_address;
  logic [SramADataWidth-1:0] gemm_a_rdata;
  logic [SramBDataWidth-1:0] gemm_b_rdata;
  logic [SramCDataWidth-1:0] gemm_c_wdata;
  logic sram_a_re_o, sram_b_re_o, sram_c_we_o;

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
  assign gemv_prefetch = is_gemv && (state_q == TopFlush) && issue_block &&
                         !last_job_block && !cache_fallback && !gemv_c_write;
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
    c_base_address = is_gemv ? CalcWidth'(M_count) :
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

  assign gemv_b_base = ((CalcWidth'(M_size_q)+CalcWidth'(NumCores-1))/CalcWidth'(NumCores)) *
                       k_blocks * CalcWidth'(AReadWords);
  assign gemv_c_base = gemv_b_base +
                      (CalcWidth'(K_size_q)+CalcWidth'(BurstBytes-1))/CalcWidth'(BurstBytes);
  assign cache_fallback = (CalcWidth'(K_size_q) > CalcWidth'(VectorCacheElements));
  assign cache_request = is_gemv && (state_q == TopCache) && !gemv_c_write;
  assign gemv_a_request = is_gemv && !gemv_c_write &&
                         ((state_q == TopLoad) || gemv_prefetch);
  assign sram_a_re_o = is_gemv ? gemv_a_request : gemm_a_request;
  assign sram_b_re_o = is_gemv ? cache_request : gemm_b_request;
  assign sram_c_we_o = is_gemv ? gemv_c_write : c_valid_q;

  always_comb begin
    a_access_address = is_gemv ? gemv_a_word_q :
      (CalcWidth'(a_fetch_m_q)/CalcWidth'(GemmRows))*k_blocks +
      CalcWidth'(a_fetch_k_q)/CalcWidth'(NumInputs);
    b_access_address = is_gemv ?
      gemv_b_base + (cache_fallback ? CalcWidth'(K_count)/CalcWidth'(BurstBytes) : cache_beat_q) :
      ((CalcWidth'(b_fetch_n_q)/CalcWidth'(GemmCols))*k_blocks +
       CalcWidth'(b_fetch_k_q)/CalcWidth'(NumInputs))*CalcWidth'(BReadWords)+CalcWidth'(b_fetch_beat_q);
    cache_first_bank = cache_fallback ? 3'((CalcWidth'(K_count)%CalcWidth'(BurstBytes))/CalcWidth'(BankBytes)) : '0;
    cache_bank_mask = '0;
    for (int bank=0; bank<NumBanks; bank++) begin
      if (cache_fallback) begin
        if ((bank >= int'(cache_first_bank)) &&
            (bank < int'(cache_first_bank)+(int'(active_inputs)+BankBytes-1)/BankBytes))
          cache_bank_mask[bank] = 1'b1;
      end else if (cache_beat_q*CalcWidth'(BurstBytes)+CalcWidth'(bank*BankBytes) < CalcWidth'(K_size_q))
        cache_bank_mask[bank] = 1'b1;
    end
    cache_last_request = cache_fallback ||
      ((cache_beat_q+CalcWidth'(1))*CalcWidth'(BurstBytes) >= CalcWidth'(K_size_q));
  end

  // A cursor and cache request cursor advance only on granted accesses.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      gemv_a_word_q <= '0; cache_beat_q <= '0; cache_ready_q <= 1'b0;
    end else if (accept_start) begin
      gemv_a_word_q <= '0; cache_beat_q <= '0; cache_ready_q <= 1'b0;
    end else begin
      if (gemv_a_request) gemv_a_word_q <= gemv_a_word_q + CalcWidth'(1);
      if (cache_request) cache_beat_q <= cache_beat_q + CalcWidth'(1);
      if (state_q == TopCacheFlush) cache_ready_q <= 1'b1;
      if (is_gemv && issue_block && cache_fallback) begin
        cache_beat_q <= '0; cache_ready_q <= 1'b0;
      end
    end
  end

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
        TopStart: state_q <= is_gemv ? TopCache : TopLoad;
        TopCache: if (cache_request && cache_last_request) state_q <= TopCacheFlush;
        TopCacheFlush: begin read_beat_q <= '0; state_q <= TopLoad; end
        TopLoad: begin
          if (is_gemv) begin
            if (gemv_a_request) begin
              if (read_beat_q == BeatWidth'(AReadWords-1)) begin
                read_beat_q <= '0; state_q <= TopFlush;
              end else read_beat_q <= read_beat_q + BeatWidth'(1);
            end
          end else if (gemm_b_request && b_fetch_last_word) state_q <= TopFlush;
        end
        TopFlush: if (issue_block) begin
          if (last_job_block) state_q <= TopDrain;
          else if (!is_gemv) state_q <= TopIssue;
          else if (cache_fallback) begin
            read_beat_q <= '0; state_q <= TopCache;
          end else begin
            // C can steal the issue cycle's speculative first A request.
            read_beat_q <= gemv_prefetch ? BeatWidth'(1) : '0;
            state_q <= TopLoad;
          end
        end
        TopIssue: if (issue_block && last_job_block) state_q <= TopDrain;
        TopDrain: if (controller_done_seen_q && (output_queue_used_q == '0) &&
                     !a_response_valid_q && !b_response_valid_q && !cache_response_valid_q) begin
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
  // Synchronous SRAM responses: always consume pending responses, even while
  // this cycle is assigned to C writes. Data are sampled BEFORE buffer NBAs.
  //---------------------------
  assign gemm_a_rdata = bank_rdata_i[3:0];
  assign gemm_b_rdata = bank_rdata_i[5:4];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      a_response_valid_q <= 1'b0; b_response_valid_q <= 1'b0;
      a_response_beat_q <= '0; b_response_beat_q <= '0;
      a_response_slot_q <= 1'b0; b_response_bank_q <= 1'b0;
      cache_response_valid_q <= 1'b0; cache_response_beat_q <= '0;
      cache_response_mask_q <= '0; cache_response_first_bank_q <= '0;
      tile_q <= '0; shared_q <= '0;
      for (int word=0; word<VectorCacheElements/BankBytes; word++)
        vector_cache_q[word*BankDataWidth +: BankDataWidth] <= '0;
    end else begin
      a_response_valid_q <= sram_a_re_o;
      b_response_valid_q <= gemm_b_request;
      cache_response_valid_q <= cache_request;
      if (sram_a_re_o) begin
        a_response_beat_q <= gemv_prefetch ? '0 : read_beat_q;
        a_response_slot_q <= a_tail_q;
      end
      if (gemm_b_request) begin
        b_response_beat_q <= b_fetch_beat_q;
        b_response_bank_q <= b_fetch_bank_q;
      end
      if (cache_request) begin
        cache_response_beat_q <= cache_beat_q;
        cache_response_mask_q <= cache_bank_mask;
        cache_response_first_bank_q <= cache_first_bank;
      end
      if (a_response_valid_q) begin
        if (is_gemv)
          tile_q[int'(a_response_beat_q)*NumBanks*BankDataWidth +: NumBanks*BankDataWidth] <= bank_rdata_i;
        else shared_q[a_response_slot_q] <= gemm_a_rdata;
      end
      if (b_response_valid_q)
        tile_q[(int'(b_response_bank_q)*GemmTileElements+int'(b_response_beat_q)*BElements)*InDataWidth
               +: SramBDataWidth] <= gemm_b_rdata;
      if (cache_response_valid_q) begin
        for (int bank=0; bank<NumBanks; bank++) begin
          if (cache_response_mask_q[bank]) begin
            for (int lane=0; lane<BankBytes; lane++) begin
              if (cache_fallback) begin
                if (bank >= int'(cache_response_first_bank_q))
                  vector_cache_q[((bank-int'(cache_response_first_bank_q))*BankBytes+lane)*InDataWidth +: InDataWidth]
                    <= bank_rdata_i[bank][lane*InDataWidth +: InDataWidth];
              end else if (int'(cache_response_beat_q)*BurstBytes+bank*BankBytes+lane < VectorCacheElements)
                vector_cache_q[(int'(cache_response_beat_q)*BurstBytes+bank*BankBytes+lane)*InDataWidth +: InDataWidth]
                  <= bank_rdata_i[bank][lane*InDataWidth +: InDataWidth];
            end
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
      mac_input_count_q <= '0; mac_context_q <= '0; mac_k_q <= '0;
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
        mac_context_q <= row_context; mac_k_q <= K_count;
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
      output_queue_used_q <= output_queue_used_q + QueueCountWidth'(reserve_output) - QueueCountWidth'(c_pop_count);
      if (c_pop_count != 0) begin
        c_ready_q[c_head_q] <= 1'b0;
        if (c_pop_count == 2) begin
          c_ready_q[next_slot(c_head_q)] <= 1'b0;
          c_head_q <= next_slot(next_slot(c_head_q));
        end else c_head_q <= next_slot(c_head_q);
        c_core_q <= '0;
      end else if (!is_gemv && c_valid_q) begin
        if (c_column_offset+CalcWidth'(CElements) >= CalcWidth'(c_count_q))
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
  assign gemv_pair_write = is_gemv && (QueueDepth > 1) &&
                           (output_queue_used_q >= QueueCountWidth'(2)) && c_valid_q &&
                           c_ready_q[next_slot(c_head_q)];
  assign gemv_c_write = is_gemv && c_valid_q &&
    (gemv_pair_write || (QueueDepth == 1) ||
     ((state_q == TopDrain) && (output_queue_used_q == QueueCountWidth'(1)) &&
      (fifo_pending_valid_q == '0)));
  assign last_c_write = is_gemv ? gemv_c_write : c_valid_q &&
    (c_column_offset+CalcWidth'(CElements) >= CalcWidth'(c_count_q)) &&
    (c_row_offset+CalcWidth'(1) >= CalcWidth'(c_row_counts_q[c_head_q]));
  assign c_pop_count = gemv_pair_write ? 2'd2 : (last_c_write ? 2'd1 : 2'd0);
  always_comb begin
    gemm_c_wdata = '0;
    for (int lane=0; lane<CElements; lane++)
      if (c_column_offset+CalcWidth'(lane) < CalcWidth'(c_count_q))
        gemm_c_wdata[lane*OutDataWidth +: OutDataWidth] =
          c_buffer_q[(int'(c_core_q)+lane)*OutDataWidth +: OutDataWidth];
  end

  //---------------------------
  // Physical bank routing. A bank has exactly one owner per cycle. GeMV C
  // writes have priority over A/cache requests; all access counters use grants.
  //---------------------------
  always_comb begin
    bank_re_o = '0; bank_we_o = '0; bank_wdata_o = '0;
    bank_access_address = '0;
    if (is_gemv) begin
      if (gemv_c_write) begin
        for (int group=0; group<2; group++) begin
          for (int core=0; core<NumCores; core++) begin
            if (group < int'(c_pop_count)) begin
              if (core < int'(c_counts_q[(group == 0) ? c_head_q : next_slot(c_head_q)])) begin
                bank_we_o[((int'(c_addresses_q[(group == 0) ? c_head_q : next_slot(c_head_q)])+core)/4)%NumBanks] = 1'b1;
                bank_access_address[((int'(c_addresses_q[(group == 0) ? c_head_q : next_slot(c_head_q)])+core)/4)%NumBanks] =
                  gemv_c_base+(CalcWidth'(c_addresses_q[(group == 0) ? c_head_q : next_slot(c_head_q)])+CalcWidth'(core))/CalcWidth'(32);
                bank_wdata_o[((int'(c_addresses_q[(group == 0) ? c_head_q : next_slot(c_head_q)])+core)/4)%NumBanks]
                  [((int'(c_addresses_q[(group == 0) ? c_head_q : next_slot(c_head_q)])+core)%4)*OutDataWidth +: OutDataWidth] =
                    c_data_q[(group == 0) ? c_head_q : next_slot(c_head_q)][core*OutDataWidth +: OutDataWidth];
              end
            end
          end
        end
      end else if (cache_request) begin
        bank_re_o = cache_bank_mask;
        for (int bank=0; bank<NumBanks; bank++) bank_access_address[bank] = b_access_address;
      end else if (gemv_a_request) begin
        bank_re_o = '1;
        for (int bank=0; bank<NumBanks; bank++) bank_access_address[bank] = a_access_address;
      end
    end else begin
      for (int bank=0; bank<4; bank++) begin
        bank_re_o[bank] = gemm_a_request;
        bank_access_address[bank] = a_access_address;
      end
      for (int bank=4; bank<6; bank++) begin
        bank_re_o[bank] = gemm_b_request;
        bank_access_address[bank] = b_access_address;
      end
      for (int bank=6; bank<8; bank++) begin
        bank_we_o[bank] = c_valid_q;
        bank_access_address[bank] = c_access_address;
        bank_wdata_o[bank] = gemm_c_wdata[(bank-6)*BankDataWidth +: BankDataWidth];
      end
    end
    // Gate physical memory operations throughout asynchronous reset.
    if (!rst_ni) begin bank_re_o = '0; bank_we_o = '0; end
    for (int bank=0; bank<NumBanks; bank++) bank_addr_o[bank] = AddrWidth'(bank_access_address[bank]);
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
              array_b_in[c][j] = $signed(vector_cache_q[((cache_fallback ? 0 : int'(mac_k_q))+j)*InDataWidth +: InDataWidth]);
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
    if (InDataWidth != 8 || OutDataWidth != 32 || NumBanks != 8 || BankDataWidth != 128)
      $fatal(1, "Banked accelerator requires int8/int32 and eight128-bit banks.");
    if (NumCores != 16 || NumInputs != 32 || GemmRows != 2)
      $fatal(1, "This bank layout requires16 cores x32 K lanes, GeMM2 M x8 N.");
    if (NumCores*NumInputs > 512 || NumBanks*BankDataWidth > 1024)
      $fatal(1, "Multiplier or aggregate bank-bandwidth budget exceeded.");
    if (MReuse == 0 || (MReuse % GemmRows) != 0 || OutputQueueDepth == 0)
      $fatal(1, "Invalid M reuse or output queue depth.");
    if (VectorCacheElements < NumInputs || (VectorCacheElements % BurstBytes) != 0)
      $fatal(1, "Vector cache must hold at least one fragment and whole128-byte bursts.");
  end
  always @(posedge clk_i) begin
    if (rst_ni) begin
      if (accept_start && (M_size_i == '0 || K_size_i == '0 || N_size_i == '0))
        $fatal(1, "M, K and N must be strictly positive.");
      for (int bank=0; bank<NumBanks; bank++) begin
        if (bank_re_o[bank] && bank_we_o[bank])
          $fatal(1, "Bank%0d is both read and written.", bank);
        if ((bank_re_o[bank] || bank_we_o[bank]) && ((bank_access_address[bank] >> AddrWidth) != '0))
          $fatal(1, "Bank%0d word address exceeds AddrWidth.", bank);
      end
      if (reserve_output && ((c_base_address >> AddrWidth) != '0))
        $fatal(1, "Output descriptor address overflow.");
      if (a_used_q > 2 || (gemm_a_pop && (a_used_q == 0)))
        $fatal(1, "A stream FIFO overflow/underflow.");
      if (gemm_a_request && a_ready_q[a_tail_q] && !(gemm_a_pop && (a_head_q == a_tail_q)))
        $fatal(1, "A producer would overwrite an unconsumed operand.");
      if (output_queue_used_q > QueueCountWidth'(QueueDepth) || int'(c_pop_count) > int'(output_queue_used_q))
        $fatal(1, "Output FIFO overflow/underflow.");
      if ((int'(c_head_q) >= QueueDepth) || (int'(c_tail_q) >= QueueDepth))
        $fatal(1, "Output FIFO pointer exceeds its configured depth.");
      for (int stage=0; stage<2; stage++)
        if (fifo_pending_valid_q[stage] && (int'(fifo_pending_slot_q[stage]) >= QueueDepth))
          $fatal(1, "Pending output reservation points outside the C FIFO.");
      if (result_valid !== fifo_pending_valid_q[1])
        $fatal(1, "MAC result-valid and output reservation are misaligned.");
      if (result_valid && c_ready_q[fifo_pending_slot_q[1]])
        $fatal(1, "A MAC result would overwrite a pending output entry.");
      if (gemv_pair_write &&
          (CalcWidth'(c_addresses_q[next_slot(c_head_q)]) !=
             CalcWidth'(c_addresses_q[c_head_q])+CalcWidth'(NumCores)))
        $fatal(1, "Paired GeMV groups must refer to consecutive rows.");
      if (gemv_pair_write && c_counts_q[c_head_q] != CoreCountWidth'(NumCores))
        $fatal(1, "Only the final GeMV group may contain a partial row tile.");
      if (gemv_prefetch && read_beat_q != '0)
        $fatal(1, "GeMV prefetch must start at the next tile's first word.");
      if (is_gemv && issue_block && !cache_ready_q)
        $fatal(1, "GeMV issued before its B data were reloaded.");
      if (gemv_a_request && gemv_c_write)
        $fatal(1, "GeMV A/C arbitration failed.");
      if (done_q && ((|bank_re_o) || (|bank_we_o) || valid_data || result_valid ||
                     output_queue_used_q != '0 || cache_response_valid_q))
        $fatal(1, "done asserted before all memory/MAC activity drained.");
      if (done_q && !is_gemv && ((a_used_q != '0) || !a_fetch_done_q || !b_fetch_done_q))
        $fatal(1, "GeMM operand producers did not finish with the controller.");
    end
  end

  // Register accounting includes the actual buffers and tags once. Arithmetic
  // and accumulator contexts are accounted for by the unchanged PE modules.
  localparam int unsigned TopRegisterBits = $bits({
    state_q, M_size_q, K_size_q, N_size_q, read_beat_q, controller_done_seen_q, done_q,
    gemv_a_word_q, cache_beat_q, cache_ready_q, vector_cache_q, mac_k_q,
    cache_response_valid_q, cache_response_beat_q, cache_response_mask_q, cache_response_first_bank_q,
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
  for (genvar g=0; g<(VectorCacheElements*InDataWidth)/32; g++) begin : gen_cache_energy
    always @(vector_cache_q[32*g +: 32]) if (rst_ni) metrics_pkg::energy += 1.0;
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
  `CPAEP_TOP_REGISTER_ENERGY(gemv_a_word_q)
  `CPAEP_TOP_REGISTER_ENERGY(cache_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(cache_ready_q)
  `CPAEP_TOP_REGISTER_ENERGY(cache_response_valid_q)
  `CPAEP_TOP_REGISTER_ENERGY(cache_response_beat_q)
  `CPAEP_TOP_REGISTER_ENERGY(cache_response_mask_q)
  `CPAEP_TOP_REGISTER_ENERGY(cache_response_first_bank_q)
  `CPAEP_TOP_REGISTER_ENERGY(mac_k_q)
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
