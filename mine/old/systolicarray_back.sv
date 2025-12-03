// Top
module systolicarray_back #(
    parameter N = 5
)(
    input   wire clk,
    input   wire rst,
    input   wire start,
    input   wire [31:0] matrix_in [0:N-1][0:N-1],
    input   wire [31:0] b_in [0:N-1],
    // input   reg  mode,

    output  wire [31:0] r_out [0:N-1][0:N-1],
    output  wire [31:0] b_out [0:N-1],
    output  wire [31:0] x_out [0:N-1],
    output  reg         done
);
    // Data flow wires
    wire [31:0] y_flow [0:N][0:N];
    wire [31:0] c_flow [0:N-1][0:N+1];
    wire [31:0] s_flow [0:N-1][0:N+1];

    // PE output wires before routing
    wire [31:0] y_out_from_pe [0:N-1][0:N];
    wire [31:0] c_out_from_pe [0:N-1][0:N];
    wire [31:0] s_out_from_pe [0:N-1][0:N];
    wire [31:0] r1_reg [0:N-1][0:N];
    // wire [31:0] r2_reg [0:N-1][0:N];

    // sqrt to divider communication wires
    wire [31:0] r_new_wire [0:N-1]; 
    wire [31:0] x_orig_wire [0:N-1];
    wire [31:0] y_orig_wire [0:N-1];

    // output from divider PEs
    wire [31:0] r_from_div_bottom [0:N-1][0:N];

    // B vector data flow
    wire [31:0] b_flow [0:N-1];         // b_in
    wire [31:0] b_out_from_pe [0:N-1];                                 
    wire [31:0] b_reg_from_pe [0:N-1]; 

    // Back-sub accumulation flow (right-to-left)
    wire [31:0] back_sum_flow [0:N-1][0:N+1];
    // Back-sub x broadcast flow (bottom-to-top), include bottom row N for boundary
    wire [31:0] back_x_flow   [0:N][0:N];
    // Back-sub enable fuse-network
    wire back_enable_src;

    wire back_en_flow [0:N-1][0:N+1];
    // Divider x outputs
    wire [31:0] x_left_from_div [0:N-1];
    wire [31:0] x_up_from_div   [0:N-1];
    // Back-sub y alignment into rotation
    wire [31:0] back_sum_to_rot [0:N-1][0:N];

    // ------------------------------------------------------------
    // Enable flow signals - parallel to data flow (extended for b_calculator column)
    wire enable_flow [0:N][0:N+1];
    wire en_out_from_pe [0:N-1][0:N];
    // External head pulse for QR: single-point trigger then N-cycle stretch
    wire qr_enable_src;
    reg  [9:0] qr_cnt;

    // Row index for injection
    reg [7:0] inject_row;

    // Output R
    genvar mi, mj;
    generate
        for (mi = 0; mi < N; mi = mi + 1) begin : MAP_OUT_ROW
            for (mj = 0; mj < N; mj = mj + 1) begin : MAP_OUT_COL
                if (mi > mj) begin : LOWER_TRI
                    // Lower triangle: 0
                    assign r_out[mi][mj] = 32'h0;
                end else if (mi == mj) begin : DIAGONAL_POS
                    // Diagonal: use r_out_bottom from divider
                    assign r_out[mi][mj] = r_from_div_bottom[mi][mi+1];
                end else begin : UPPER_TRI
                    // All non-diagonal upper-triangle positions: take from rotation PE at (mi, mj+1)
                    assign r_out[mi][mj] = r1_reg[mi][mj+1];
                    // assign r_out[mi][mj] = r2_reg[mi][mj+1];
                end
            end
        end
    endgenerate                              
    
    // Output b_new
    genvar final_i;
    generate
      for (final_i = 0; final_i < N; final_i = final_i + 1) begin : B_OUT
        assign b_out[final_i] = b_reg_from_pe[final_i]; // get the results from the reg
      end
    endgenerate

    // X_OUT
    genvar xi_idx;
    generate
      for (xi_idx = 0; xi_idx < N; xi_idx = xi_idx + 1) begin : X_OUT
        assign x_out[xi_idx] = pe_row[xi_idx].pe_col[xi_idx].sqrt_pe_inst.u_sqrt_pe.x_reg;
      end
    endgenerate

    // FSM
    reg [2:0] state;
    reg [9:0] cycle_count;

    localparam S_IDLE      = 3'h0,
               S_QR_DECOMP = 3'h1,
               S_BACK_SUB  = 3'h2,
               S_DONE      = 3'h3;

    localparam integer TOTAL_QR_CYCLES   = 4 * N - 2; // 4 * N - 3 is the total number of cycles for a (N,N) matrix, come from 2(N-1) (num of rows excepting the last row) + 1 (num of rows of the last row) + (N-1) (the num between the rows) + N (size of the matrix) - 1 (start from 0) + 1 for b_calculator
    localparam integer TOTAL_BACK_CYCLES = 4 * N;     // TODO

    wire mode;
    assign mode = (state == S_BACK_SUB);
    // QR enable source with built-in N-cycle stretch
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            qr_cnt <= '0;
        end else if ((state == S_QR_DECOMP) && (cycle_count == 0)) begin
            qr_cnt <= N[9:0] - 10'd1; // start from N-1 and count down
        end else if (qr_cnt != 0) begin
            qr_cnt <= qr_cnt - 10'd1;
        end
    end
    assign qr_enable_src = (state == S_QR_DECOMP) && (cycle_count < N);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            done <= 1'b0;
            cycle_count <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    cycle_count <= '0;
                    if (start) begin
                        state <= S_QR_DECOMP;
                    end
                end
                S_QR_DECOMP: begin
                    if (cycle_count >= TOTAL_QR_CYCLES) begin
                        state <= S_BACK_SUB;
                        cycle_count <= '0;
                    end else begin
                        cycle_count <= cycle_count + 1'b1;
                    end
                end
                S_BACK_SUB: begin
                    if (cycle_count >= TOTAL_BACK_CYCLES) begin
                        state <= S_DONE;
                        cycle_count <= '0;
                    end else begin
                        cycle_count <= cycle_count + 1'b1;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // Enable source: only in QR injection; in back-substitution, use global enable
    assign enable_src = (state == S_QR_DECOMP) && (inject_row < N);

    always @(posedge clk) begin
        case (state)
            S_IDLE: begin
                inject_row <= 8'd0;
            end
            S_QR_DECOMP: begin
                if (inject_row < N) begin
                    inject_row <= inject_row + 8'd1;
                end
            end
            default: ;
        endcase
    end

    assign back_enable_src = (state == S_BACK_SUB) && (cycle_count == 0);

    // QR enable self-propagation along the top row from PE(0,0).en_out
    // Match c/s horizontal buffering rule to keep timing aligned.
    genvar ecol;
    generate
      for (ecol = 0; ecol <= N+1; ecol = ecol + 1) begin : enable_inject
        if (ecol == 0) begin : TOP_LEFT_SRC
          assign enable_flow[0][0] = qr_enable_src; // feed sqrt(0,0)
        end else if (ecol == 1) begin : DIV_COL_DISABLE
          assign enable_flow[0][1] = 1'b0; // divider column has no direct top-row enable
        end else if (ecol == N+1) begin : B_CALC_COL
          assign enable_flow[0][N+1] = enable_flow[0][N];
        end else begin : TOP_ROW_PROP
          localparam integer i0 = 0;
          localparam integer j_prev = (ecol - 1);
          // needs_horiz_buffer condition copied from c/s horizontal routing
          localparam is_buffer_input = ((j_prev - i0 - 1) % 2 == 0) && ((j_prev - i0 - 1) >= 2);
          localparam h_val = (j_prev - i0 - 1) / 2;
          localparam needs_horiz_buffer = (i0 < N - 3) && is_buffer_input && (h_val >= 1) && (h_val < (N-i0)/2);
          if (needs_horiz_buffer) begin : EN_HBUF
            en_buffer en_h(
              .clk(clk), .rst(rst),
              .en_in(en_out_from_pe[0][j_prev]),
              .en_out(enable_flow[0][ecol])
            );
          end else begin : EN_HDIR
            assign enable_flow[0][ecol] = en_out_from_pe[0][j_prev];
          end
        end
      end
    endgenerate

    genvar col, buf_idx;
    generate
      for (col = 0; col <= N; col = col + 1) begin : input_col
        if (col == 1) begin : NON_DATA_COL
            // Divider column (col==1): no injection
            assign y_flow[0][col] = 32'h0;
        end else begin : DATA_COL
            // Data injection schedule based on PE column index
            // Map PE column to matrix column: PE[0][0]->matrix[0], PE[0][2]->matrix[1], PE[0][3]->matrix[2]
            localparam integer matrix_col = (col == 0) ? 0 : col - 1;
            localparam integer num_buf = (col == 0) ? 0 : (col % 2 == 0) ? (3 * col / 2 - 1) : (3 * (col - 1) / 2);
            if (num_buf == 0) begin : NO_DELAY
                assign y_flow[0][col] = (state == S_QR_DECOMP && inject_row < N && matrix_col < N) ? matrix_in[inject_row][matrix_col] :
                                       (state == S_QR_DECOMP && inject_row >= N && matrix_col < N) ? matrix_in[N-1][matrix_col] : 32'h0;
            end else begin : BUFFER
                wire [31:0] buffer_chain [0:num_buf];
                assign buffer_chain[0] = (state == S_QR_DECOMP && inject_row < N && matrix_col < N) ? matrix_in[inject_row][matrix_col] :
                                        (state == S_QR_DECOMP && inject_row >= N && matrix_col < N) ? matrix_in[N-1][matrix_col] : 32'h0;
                for (buf_idx = 0; buf_idx < num_buf; buf_idx = buf_idx + 1) begin : BUFFER_INST
                    buffer buf_inst(
                        .clk(clk),
                        .rst(rst),
                        .x_in(buffer_chain[buf_idx]),
                        .x_out(buffer_chain[buf_idx+1])
                    );
                end
                assign y_flow[0][col] = buffer_chain[num_buf];
            end
        end
      end
    endgenerate

    // Back-sub y (back_sum) alignment into rotation inputs:
    genvar bs_i, bs_j, bs_k;
    generate
      for (bs_i = 0; bs_i < N; bs_i = bs_i + 1) begin : BACKSUM_ALIGN_ROW
        for (bs_j = 0; bs_j <= N; bs_j = bs_j + 1) begin : BACKSUM_ALIGN_COL
          if (bs_j == N) begin : RIGHT_EDGE
            localparam integer num_bs_buf = (N - 1 - bs_i);
            if (num_bs_buf == 0) begin : NO_BUF
              assign back_sum_to_rot[bs_i][N] = back_sum_flow[bs_i][N+1];
            end else begin : BUF_CHAIN
              wire [31:0] bs_chain [0:num_bs_buf];
              assign bs_chain[0] = back_sum_flow[bs_i][N+1];
              for (bs_k = 0; bs_k < num_bs_buf; bs_k = bs_k + 1) begin : BSB
                buffer bsum_boundary(
                  .clk(clk), .rst(rst),
                  .x_in(bs_chain[bs_k]),
                  .x_out(bs_chain[bs_k+1])
                );
              end
              assign back_sum_to_rot[bs_i][N] = bs_chain[num_bs_buf];
            end
          end else begin : DIRECT
            assign back_sum_to_rot[bs_i][bs_j] = back_sum_flow[bs_i][bs_j+1];
          end
        end
      end
    endgenerate

    // Step 1: Instantiate main PEs and b_calculator PEs
    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : pe_row
            for (j = 0; j <= N+1; j = j + 1) begin : pe_col
                // Lower triangle has no PEs
                if (i > j) begin : LOWER_EMPTY
                // sqrt PE at diagonal position (i,i)
                end else if (i == j) begin : sqrt_pe_inst
                    sqrt u_sqrt_pe (
                        .clk(clk), .rst(rst),
                        .mode(mode),
                        .enable(mode ? back_en_flow[i][j] : ((i==0 && j==0) ? qr_enable_src : enable_flow[i][j])),
                        .data_in(y_flow[i][j]),
                        .xi_in(x_left_from_div[i]),
                        .r_new(r_new_wire[i]),
                        .x_orig(x_orig_wire[i]),
                        .y_orig(y_orig_wire[i]),
                        .en_out(en_out_from_pe[i][j])
                    );
                // divider PE at position (i, i+1)
                end else if (j == i + 1) begin : divider_pe_inst
                    divider u_divider_pe (
                        .clk(clk), .rst(rst),
                        .mode(mode),
                        .enable(mode ? back_en_flow[i][j] : en_out_from_pe[i][i]),
                        .r_new(r_new_wire[i]),
                        .x_orig(x_orig_wire[i]),
                        .y_orig(y_orig_wire[i]),
                        .back_sum_in(back_sum_flow[i][i+2]),
                        .x_left(x_left_from_div[i]),
                        .x_up(x_up_from_div[i]),
                        .c_out(c_out_from_pe[i][j]),
                        .s_out(s_out_from_pe[i][j]),
                        .r_out_bottom(r_from_div_bottom[i][j]),
                        .en_out(en_out_from_pe[i][j])
                    );
                // b_calculator PE at right column (j == N+1)
                end else if (j == N+1) begin : b_calc_pe_inst
                    b_calculator u_b_calc_pe (
                        .clk(clk), .rst(rst),
                        .mode(mode),
                        .enable(mode ? back_en_flow[i][N] : en_out_from_pe[i][N]),
                        .c_in(c_out_from_pe[i][N]),
                        .s_in(s_out_from_pe[i][N]),
                        .b_in(b_flow[i]),
                        .b_reg(b_reg_from_pe[i]),
                        .b_out(b_out_from_pe[i]),
                        .back_sum_out(back_sum_flow[i][N+1])
                    );
                // else: rotation PE
                end else begin : rotation_pe_inst
                    rotation u_rotation_pe (
                        .clk(clk), .rst(rst),
                        .mode(mode),
                        .enable(mode ? back_en_flow[i][j] : enable_flow[i][j]),
                        .data_in(mode ? back_sum_to_rot[i][j] : y_flow[i][j]),
                        .c_in(c_flow[i][j]),
                        .s_in(s_flow[i][j]),
                        .data_out(y_out_from_pe[i][j]),
                        .c_out(c_out_from_pe[i][j]),
                        .s_out(s_out_from_pe[i][j]),
                        .en_out(en_out_from_pe[i][j]),
                        .back_sum_in(back_sum_to_rot[i][j]),
                        .back_sum_out(back_sum_flow[i][j]),
                        .x_from_down(back_x_flow[i+1][j]),
                        .x_to_up(back_x_flow[i][j])
                    );
                    assign r1_reg[i][j] = pe_row[i].pe_col[j].rotation_pe_inst.u_rotation_pe.r1_reg;
                    // assign r2_reg[i][j] = pe_row[i].pe_col[j].rotation_pe_inst.u_rotation_pe.r2_reg;
                end
            end
        end
    endgenerate



    // Step 2: Implement data flow routing with buffers
    genvar h, v, ei, ej;
    generate
        // Horizontal connections
        for (i = 0; i < N; i = i + 1) begin : horiz_flow_logic
            // Connect divider output to start the c/s
            if (i + 2 <= N) begin : CONNECT_DIVIDER
                assign c_flow[i][i+2] = c_out_from_pe[i][i+1];
                assign s_flow[i][i+2] = s_out_from_pe[i][i+1];
            end
            // Horizontal propagation of c/s
            for (j = i + 2; j < N; j = j + 1) begin : horiz_link
                // Insert horizontal buffers according to rule:
                // Buffer between PE[i][i+2h+1] and PE[i][i+2h+2]
                // where i < N-3 and 1 <= h < (N-i)/2
                localparam is_buffer_input = ((j - i - 1) % 2 == 0) && ((j - i - 1) >= 2);
                localparam h_val = (j - i - 1) / 2;
                localparam needs_horiz_buffer = (i < N - 3) &&
                                              is_buffer_input &&
                                              (h_val >= 1) &&
                                              (h_val < (N-i)/2);
                if (needs_horiz_buffer) begin : HORIZ_BUFFER
                    // Buffer connects PE[i][j] output to PE[i][j+1] input
                    buffer c_buf(
                            .clk(clk),
                            .rst(rst),
                            .x_in(c_out_from_pe[i][j]),
                            .x_out(c_flow[i][j+1])
                        );
                    buffer s_buf(
                            .clk(clk),
                            .rst(rst),
                            .x_in(s_out_from_pe[i][j]),
                            .x_out(s_flow[i][j+1])
                        );
                end else begin : HORIZ_DIRECT
                    // Direct connection: PE[i][j] output to PE[i][j+1] input
                    assign c_flow[i][j+1] = c_out_from_pe[i][j];
                    assign s_flow[i][j+1] = s_out_from_pe[i][j];
                end
            end

            // Right boundary: avoid double-driving c_flow[i][N-1]
            // The horiz_link loop already assigns j=N-2 -> c_flow[i][N-1] when i <= N-3
            if (i >= N - 2) begin : RIGHT_BOUNDARY_DIRECT
                assign c_flow[i][N-1] = c_out_from_pe[i][N-1];
                assign s_flow[i][N-1] = s_out_from_pe[i][N-1];
            end
        end

        // Vertical connections
        for (i = 0; i < N - 1; i = i + 1) begin : vert_flow_logic
            for (j = i; j <= N; j = j + 1) begin : vert_link
                if (i < j) begin : UPPER_TRI_VERT // Only process upper triangle
                    // First y-flow comes from first rotation PE to next row sqrt PE
                    // Skip divider PE at (i, i+1), start from first rotation PE at (i, i+2)
                    if (j == i + 2) begin : FIRST_Y_FLOW
                        assign y_flow[i+1][i+1] = y_out_from_pe[i][j];
                    end
                    // Normal y-flow for other rotation PEs
                    else if (j > i + 2) begin : NORMAL_Y_FLOW
                        // Insert vertical buffers
                        localparam v_val = (j - i - 1) / 2;
                        localparam needs_vert_buffer = (i <= N - 3) &&
                                                     ((j - i - 1) % 2 == 0) &&
                                                     (v_val >= 1) &&
                                                     (v_val <= (N-i-1)/2);

                        if (needs_vert_buffer) begin : VERT_BUFFER
                            buffer y_buf(
                                    .clk(clk),
                                    .rst(rst),
                                    .x_in(y_out_from_pe[i][j]),
                                    .x_out(y_flow[i+1][j])
                                );
                        end else begin : VERT_DIRECT
                            assign y_flow[i+1][j] = y_out_from_pe[i][j];
                        end
                    end
                end
            end

            // Vertical enable flow: mirror y_flow routing so enable aligns with data
            for (ei = 0; ei < N - 1; ei = ei + 1) begin : vert_en_flow_logic
                for (ej = ei; ej <= N+1; ej = ej + 1) begin : vert_en_link
                    if (ei < ej) begin : UPPER_TRI_VERT_EN
                        if (ej == ei + 2) begin : FIRST_EN_FLOW
                            // First rotation in row ei feeds next-row sqrt at (ei+1,ei+1)
                            // Base 1-cycle delay to match upper PE's registered data_out (enable_q)
                            en_buffer en_first_y(
                                .clk(clk), .rst(rst),
                                .en_in(enable_flow[ei][ej]),
                                .en_out(enable_flow[ei+1][ei+1])
                            );
                        end else if (ej > ei + 2 && ej <= N) begin : NORMAL_EN_FLOW
                            localparam v_val_en = (ej - ei - 1) / 2;
                            localparam needs_vert_enbuf = (ei <= N - 3) &&
                                                        ((ej - ei - 1) % 2 == 0) &&
                                                        (v_val_en >= 1) &&
                                                        (v_val_en <= (N - ei - 1)/2);
                            if (needs_vert_enbuf) begin : VERT_EN_BUFFER
                                // Base 1-cycle delay for rotation's registered data_out, plus
                                // 1 more cycle to mirror the explicit vertical data buffer
                                wire en_base;
                                en_buffer en_y0(
                                    .clk(clk), .rst(rst),
                                    .en_in(enable_flow[ei][ej]),
                                    .en_out(en_base)
                                );
                                en_buffer en_y1(
                                    .clk(clk), .rst(rst),
                                    .en_in(en_base),
                                    .en_out(enable_flow[ei+1][ej])
                                );
                            end else begin : VERT_EN_DIRECT
                                // Still need the base 1-cycle delay to match rotation's register
                                en_buffer en_y0(
                                    .clk(clk), .rst(rst),
                                    .en_in(enable_flow[ei][ej]),
                                    .en_out(enable_flow[ei+1][ej])
                                );
                            end
                        end
                    end
                end
            end
        end
    endgenerate

    // B vector vertical flow logic - first row injects original B vector, other rows get from previous row
    genvar b_row;
    generate
        for (b_row = 0; b_row < N; b_row = b_row + 1) begin : b_vertical_flow
            if (b_row == 0) begin : FIRST_ROW_B_INJECT
                // First row: inject original B vector, delay 1 more cycle than column N
                localparam integer col_N_delay = (N % 2 == 0) ? (3 * N / 2 - 1) : (3 * (N - 1) / 2);
                localparam integer b_delay = col_N_delay + 1;
                
                if (b_delay == 0) begin : NO_B_DELAY
                    // Inject b_in sequentially using inject_row, aligned with matrix injection
                    assign b_flow[b_row] = (state == S_QR_DECOMP && inject_row < N) ? b_in[inject_row] : 32'h0;
                end else begin : B_BUFFER
                    wire [31:0] b_buffer_chain [0:b_delay];
                    // Drive with b_in[inject_row] during the N-cycle injection window
                    assign b_buffer_chain[0] = (state == S_QR_DECOMP && inject_row < N) ? b_in[inject_row] : 32'h0;

                    genvar b_buf_idx;
                    for (b_buf_idx = 0; b_buf_idx < b_delay; b_buf_idx = b_buf_idx + 1) begin : B_BUFFER_INST
                        buffer b_buf_inst(
                            .clk(clk),
                            .rst(rst),
                            .x_in(b_buffer_chain[b_buf_idx]),
                            .x_out(b_buffer_chain[b_buf_idx+1])
                        );
                    end
                    assign b_flow[b_row] = b_buffer_chain[b_delay];
                end
            end else begin : OTHER_ROW_B_FLOW
                if (b_row == N-1) begin : LAST_TWO_ROWS_BUF
                    buffer b_last_vert(
                        .clk(clk), .rst(rst),
                        .x_in(b_out_from_pe[b_row-1]),
                        .x_out(b_flow[b_row])
                    );
                end else begin : NORMAL_B_VERT
                    // Mirror vertical data routing on column N to decide if a buffer is required
                    // In y_flow vertical routing, the buffer condition is:
                    // needs_vert_buffer = (i <= N - 3) && (((j - i - 1) % 2) == 0) && (v_val >= 1) && (v_val <= (N-i-1)/2)
                    // Here, for B column we mirror j = N and i = b_row - 1
                    localparam integer delta_col_b = N - (b_row - 1) - 1; // j - i - 1 with j=N
                    localparam integer v_val_b = (delta_col_b) / 2;
                    localparam needs_vert_buffer_b = ((b_row - 1) <= N - 3) &&
                                                     (((delta_col_b) % 2) == 0) &&
                                                     (v_val_b >= 1) &&
                                                     (v_val_b <= (N - (b_row - 1) - 1)/2);

                    if (needs_vert_buffer_b) begin : B_VERT_BUFFER
                        buffer b_y_buf(
                                .clk(clk), .rst(rst),
                                .x_in(b_out_from_pe[b_row-1]),
                                .x_out(b_flow[b_row])
                            );
                    end else begin : B_VERT_DIRECT
                        assign b_flow[b_row] = b_out_from_pe[b_row-1];
                    end
                end
            end
        end
    endgenerate

    // ------------------------------------------------------------
    // back-substitution data flow

    // x broadcast at divider column j=i+1
    genvar xi_seed;
    generate
      for (xi_seed = 0; xi_seed < N; xi_seed = xi_seed + 1) begin : SEED_X_BROADCAST
        assign back_x_flow[xi_seed][xi_seed+1] = x_up_from_div[xi_seed];
      end
    endgenerate

    // Bottom boundary (row N) default 0 for x broadcast
    genvar xb;
    generate
      for (xb = 0; xb <= N; xb = xb + 1) begin : X_BOTTOM_ZERO
        assign back_x_flow[N][xb] = 32'h0;
      end
    endgenerate

    // Fuse network: back-substitution enable propagation (right-to-left, bottom-to-top)
    genvar fi, fj;
    generate
      for (fi = 0; fi < N; fi = fi + 1) begin : FUSE_ROW
        for (fj = 0; fj <= N; fj = fj + 1) begin : FUSE_COL
          if (fi > fj) begin
            // Lower triangle: no enable
            assign back_en_flow[fi][fj] = 1'b0;
          end else begin : UPPER_TRI_FUSE
            // Special-case rightmost rotation column N (neighbor to b_calculator)
            if (fj == N) begin
              if (fi == N-1) begin
                // Seed the chain at bottom-right via fire source
                assign back_en_flow[fi][fj] = back_enable_src;
              end else begin
                // Other rows take enable from the row below, same column
                assign back_en_flow[fi][fj] = en_out_from_pe[fi+1][fj];
              end
            end else if (fj > fi) begin
              // Rotation and divider columns take enable from right neighbor in the same row
              assign back_en_flow[fi][fj] = en_out_from_pe[fi][fj+1];
            end else if (fj == fi) begin
              // Sqrt column takes enable from divider at (i,i+1)
              assign back_en_flow[fi][fj] = en_out_from_pe[fi][fj+1];
            end else begin
              // Should not reach here
              assign back_en_flow[fi][fj] = 1'b0;
            end
          end
        end
      end
    endgenerate
endmodule

// sqrt
module sqrt (
    input   wire clk,
    input   wire rst,
    input   wire mode,
    input   wire enable,
    input   wire [31:0] data_in,
    input   wire [31:0] xi_in,
    output  reg  [31:0] r_new,
    output  reg  [31:0] x_orig,
    output  reg  [31:0] y_orig,
    output  reg en_out
);
    // Store previous r value
    wire [31:0] x_sq, y_sq, sum_sq, r_computed;

    // x²
    wire [7:0] mult_x_status;
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),

        .ieee_compliance(1)
    ) u_mult_x (
        .a(data_in),
        .b(data_in),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(x_sq),
        .status(mult_x_status)
    );

    // y²
    wire [7:0] mult_y_status;
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_mult_y (
        .a(r_new),
        .b(r_new),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(y_sq),
        .status(mult_y_status)
    );

    // x² + y²
    wire [7:0] add_status;
    DW_fp_add #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_add (
        .a(x_sq),
        .b(y_sq),
        .rnd(3'b000),
        .z(sum_sq),
        .status(add_status)
    );

    // sqrt(x² + y²)
    wire [7:0] sqrt_status;
    DW_fp_sqrt #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sqrt (
        .a(sum_sq),
        .rnd(3'b000),  // Round to nearest even
        .z(r_computed),
        .status(sqrt_status)
    );

    // Back-sub result register for xi
    reg [31:0] x_reg;

    // Register enable to align with downstream data registers (one-cycle later)
    always @(posedge clk or posedge rst) begin
        if (rst) en_out <= 1'b0; else en_out <= enable;
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            r_new <= 32'h0;
            x_orig <= 32'h0;
            y_orig <= 32'h0;
            x_reg <= 32'h0;
        end else if (!mode && enable) begin
            r_new <= r_computed;
            x_orig <= data_in;
            y_orig <= r_new;
        end else if (mode && enable) begin
            x_reg <= xi_in;
        end else begin
            r_new <= r_new; // hold previous value until the last PE finishing computation
            x_orig <= 0;
            y_orig <= 0;
        end
    end
endmodule

// divider
module divider (
    input   wire clk,
    input   wire rst,
    input   wire mode,
    input   wire enable,
    input   wire [31:0] r_new,
    input   wire [31:0] x_orig,
    input   wire [31:0] y_orig,
    input   wire [31:0] back_sum_in,
    output  wire [31:0] x_left,
    output  wire [31:0] x_up,
    output  reg  [31:0] c_out,
    output  reg  [31:0] s_out,
    output  reg  [31:0] r_out_bottom,
    output  reg en_out
);
    wire [31:0] c_temp, s_temp;

    // c = y/r
    wire [7:0] div_c_status;
    wire [7:0] div_s_status;

    // Reuse u_div_c for xi in back-sub: a/b are muxed by mode
    wire [31:0] divc_a = mode ? back_sum_in : y_orig;

    DW_fp_div #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_div_c (
        .a(divc_a),
        .b(r_new),
        .rnd(3'b000),
        .z(c_temp),
        .status(div_c_status)
    );

    // s = x/r
    DW_fp_div #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_div_s (
        .a(x_orig),
        .b(r_new),
        .rnd(3'b000),
        .z(s_temp),
        .status(div_s_status)
    );


    // Back-sub xi register (updated once by enable_back)
    reg [31:0] xi_reg;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            xi_reg <= 32'h0;
        end else if (mode && enable) begin
            xi_reg <= (r_out_bottom == 32'h00000000) ? 32'h0 : c_temp;
        end
    end
    assign x_left = mode ? xi_reg : 32'h0;
    assign x_up   = x_left;

    // Register enable one-cycle to align with r_new/x_orig/y_orig
    always @(posedge clk or posedge rst) begin
        if (rst) en_out <= 1'b0; else en_out <= enable;
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c_out <= 32'h3F800000; // Default c = 1.0
            s_out <= 32'h0;        // Default s = 0.0
            r_out_bottom <= 32'h0;
        end else if (!mode && enable) begin
            r_out_bottom <= r_new;
            if (r_new == 32'h00000000) begin
                // TODO
                // Guard: avoid 0/0 -> NaN; for zero vector define c=1, s=0
                c_out <= 32'h3F800000; // 1.0
                s_out <= 32'h00000000; // 0.0
            end else begin
                c_out <= c_temp;
                s_out <= s_temp;
            end
        end else begin
            c_out <= 32'h3F800000;
            s_out <= 32'h0;
            r_out_bottom <= r_out_bottom; // hold previous value until the last PE finishing computation
        end
    end
endmodule

// rotation
module rotation (
    input   wire clk,
    input   wire rst,
    input   wire mode,
    input   wire enable,
    input   wire [31:0] data_in,
    input   wire [31:0] c_in,
    input   wire [31:0] s_in,
    input   wire [31:0] back_sum_in,
    input   wire [31:0] x_from_down,
    output  reg [31:0] back_sum_out,
    output  reg [31:0] x_to_up,
    output  reg  [31:0] data_out,
    output  reg  [31:0] c_out,
    output  reg  [31:0] s_out,
    output  reg en_out
);
    // Store previous r2 value
    reg [31:0] r1_reg = 32'h0;
    // reg [31:0] r2_reg  = 32'h0;
    wire [7:0] cr1_status, sr2_status, sr1_status, cr2_status;


    wire [31:0] r1_prime, r2_prime;
    wire [31:0] cr1, sr2, sr1, cr2;

    wire [31:0] mult_a = mode ? r1_reg : s_in;
    wire [31:0] mult_b = mode ? x_from_down : r1_reg;

    wire [31:0] sub_a = mode ? back_sum_in : cr2;
    
    // c*r1
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_cr1 (
        .a(c_in),
        .b(r1_reg),
        // .b(data_in),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(cr1),
        .status(cr1_status)
    );

    // s*r2
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sr2 (
        .a(s_in),
        .b(data_in),
        // .b(r2_reg),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(sr2),
        .status(sr2_status)
    );

    // QR: s*r1
    // BackSub: r_from_down * r1_reg
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sr1 (
        .a(mult_a),
        .b(mult_b),
        // .b(data_in),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(sr1),
        .status(sr1_status)
    );

    // c*r2
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_cr2 (
        .a(c_in),
        .b(data_in),
        // .b(r2_reg),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(cr2),
        .status(cr2_status)
    );

    // r1' = c*r1 + s*r2
    wire [7:0] add1_status;
    DW_fp_add #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_add1 (
        .a(cr1),
        .b(sr2),
        .rnd(3'b000),
        .z(r1_prime),
        .status(add1_status)
    );

    // QR: r2' = c*r2 - s*r1
    // BackSub: back_sum_in - sr1
    wire [7:0] sub1_status;
    DW_fp_sub #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sub1 (
        .a(sub_a),
        .b(sr1),
        .rnd(3'b000),
        .z(r2_prime),
        .status(sub1_status)
    );


    // Register enable to align with internal mult/add regs
    always @(posedge clk or posedge rst) begin
        if (rst) en_out <= 1'b0; else en_out <= enable;
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            r1_reg <= 32'h0;
            // r2_reg <= 32'h0;
            data_out <= 32'h0;
            c_out <= 32'h0;
            s_out <= 32'h0;
        end else if (!mode && enable) begin
            c_out <= c_in;
            s_out <= s_in;
            data_out <= r2_prime;
            r1_reg <= r1_prime;
            // r2_reg <= r2_prime;
        end else begin
            data_out <= 32'h0;
            c_out <= 32'h0;
            s_out <= 32'h0;
            r1_reg <= r1_reg; // hold previous value until the last PE finishing computation
        end
    end
    
    // ----- Back -----
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            back_sum_out <= 32'h0;
            x_to_up <= 32'h0;
        end else if (mode && enable) begin
            back_sum_out <= r2_prime;
            x_to_up <= x_from_down;
        end
    end
endmodule

// buffer
module buffer (
    input   wire clk,
    input   wire rst,
    input   wire [31:0] x_in,
    output  reg  [31:0] x_out
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            x_out <= 32'h0;
        end else begin
            x_out <= x_in;
        end
    end
endmodule

// 1-bit enable buffer
module en_buffer (
    input  wire clk,
    input  wire rst,
    input  wire en_in,
    output reg  en_out
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            en_out <= 1'b0;
        end else begin
            en_out <= en_in;
        end
    end
endmodule

module b_calculator (
    input   wire clk,
    input   wire rst,
    input   wire mode,
    input   wire enable,
    input   wire [31:0] c_in,
    input   wire [31:0] s_in,
    input   wire [31:0] b_in,  // b2
    output  reg  [31:0] b_reg, // b1
    output  reg  [31:0] b_out,
    output  wire [31:0] back_sum_out
);

    wire [31:0] cb1, sb1, cb2, sb2;
    wire [7:0] st_cb1, st_sb2, st_cb2, st_sb1;

    // cb1
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_cb1 (
        .a(c_in),
        .b(b_reg),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(cb1),
        .status(st_cb1)
    );

    // sb2
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sb2 (
        .a(s_in),
        .b(b_in),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(sb2),
        .status(st_sb2)
    );

    // cb1 + sb2
    wire [31:0] b1_new;
    wire [7:0] st_b1_new;
    DW_fp_add #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_b1_new (
        .a(cb1), 
        .b(sb2), 
        .rnd(3'b000), 
        .z(b1_new), 
        .status(st_b1_new)
    );

    // cb2
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_cb2 (
        .a(c_in),
        .b(b_in),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(cb2),
        .status(st_cb2)
    );

    // sb1
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sb1 (
        .a(s_in),
        .b(b_reg),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(sb1),
        .status(st_sb1)
    );

    // cb2 - sb1
    wire [31:0] b2_new;
    wire [7:0] st_b2_new;
    DW_fp_sub #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_b2_new (
        .a(cb2), 
        .b(sb1), 
        .rnd(3'b000), 
        .z(b2_new), 
        .status(st_b2_new)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            b_out <= 32'h0;
            b_reg <= 32'h0;
        end else if (!mode && enable) begin
            b_out <= b2_new;
            b_reg <= b1_new;
        end else begin
            // Back-sub: hold b_reg (stores y_i), b_out not used
            b_out <= 32'h0;
            b_reg <= b_reg;
        end
    end

    assign back_sum_out = mode ? b_reg : 32'h0;
endmodule