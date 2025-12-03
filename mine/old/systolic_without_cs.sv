// Top
module systolicarray #(
    parameter N = 9
)(
    input   wire clk,
    input   wire rst,
    input   wire start,
    input   wire [31:0] matrix_in [0:N-1][0:N-1],

    output  wire [31:0] r_out [0:N-1][0:N-1],
    output  wire [31:0] c_out [0:N-1],
    output  wire [31:0] s_out [0:N-1],
    output  reg         done
);
    // Data flow wires - inputs to PEs
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


    // Map outputs
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

    // Assign final c/s outputs from the right edge
    genvar final_i;
    generate
      for (final_i = 0; final_i < N; final_i = final_i + 1) begin : C_OUT
        if (final_i == N-1) begin : LAST_ROW_CS
          // TODO
          // Last row has sqrt PE at [N-1][N-1], no c/s output?
          assign c_out[final_i] = 32'h0;
          assign s_out[final_i] = 32'h0;
        end else begin : NORMAL_ROW_CS
          assign c_out[final_i] = c_flow[final_i][N-1];
          assign s_out[final_i] = s_flow[final_i][N-1];
        end
      end
    endgenerate

    // FSM
    reg [2:0] state;
    reg [7:0] cycle_count;

    // Enable flow signals - parallel to data flow (match y_flow shape)
    wire enable_flow [0:N][0:N];  // 1-bit enable flows along y_flow routing
    wire enable_src;              // Source enable (N-cycle high during injection)

    // Explicit enable chain: en_in to each PE, en_out from each PE
    wire en_in  [0:N-1][0:N];
    wire en_out_from_pe [0:N-1][0:N];


    localparam IDLE = 3'h0,
               PROCESSING = 3'h1,
               DONE_STATE = 3'h2;

    // TODO
    // 加入cs和b的直接计算
    localparam integer TOTAL_CYCLES = 4 * N - 3; // 4 * N - 3 is the total number of cycles for a (N,N) matrix, come from 2(N-1) (num of rows excepting the last row) + 1 (num of rows of the last row) + (N-1) (the num between the rows) + N (size of the matrix) - 1 (start from 0)

    always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        cycle_count <= '0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        case (state)
        IDLE: if (start) begin
            state <= PROCESSING;
            cycle_count <= '0;
        end
        PROCESSING: begin
            if (cycle_count >= TOTAL_CYCLES) state <= DONE_STATE;
            cycle_count <= cycle_count + 1'b1;
        end
        DONE_STATE: begin
            done <= 1'b1;
            cycle_count <= '0;
            state <= IDLE;
        end
        endcase
    end
    end

    // Row index for injection
    reg [7:0] inject_row;

    // Enable source: high for exactly N+1 cycles during matrix injection
    assign enable_src = (state == PROCESSING) && (inject_row < N);

    // Divider enable: 1-cycle after sqrt column enable (align with r_new/x/y regs)
    wire enable_div [0:N-1];
    genvar di;
    generate
      for (di = 0; di < N; di = di + 1) begin : GEN_EN_DIV
        en_buffer enb_div(
          .clk(clk), .rst(rst),
          .en_in(enable_flow[di][di]),
          .en_out(enable_div[di])
        );
      end
    endgenerate
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            inject_row <= 8'd0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        inject_row <= 8'd0;
                    end
                end
                PROCESSING: begin
                    if (inject_row < N) begin
                        inject_row <= inject_row + 8'd1;
                    end
                end
                default: ;
            endcase
        end
    end

    // Input data injection with buffer
    // Enable injection along top row with same buffering as y_flow
    genvar ecol, ebuf;
    generate
      for (ecol = 0; ecol <= N; ecol = ecol + 1) begin : enable_inject
        if (ecol == 1) begin
          assign enable_flow[0][ecol] = 1'b0; // divider col has no data injection
        end else begin
          localparam integer enum_buf = (ecol == 0) ? 0 : (ecol % 2 == 0) ? (3 * ecol / 2 - 1) : (3 * (ecol - 1) / 2);
          if (enum_buf == 0) begin
            assign enable_flow[0][ecol] = enable_src;
          end else begin
            wire en_chain [0:enum_buf];
            assign en_chain[0] = enable_src;
            for (ebuf = 0; ebuf < enum_buf; ebuf = ebuf + 1) begin : EN_BUF
              en_buffer en_buf(
                .clk(clk), .rst(rst),
                .en_in(en_chain[ebuf]),
                .en_out(en_chain[ebuf+1])
              );
            end
            assign enable_flow[0][ecol] = en_chain[enum_buf];
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
            // Map PE column to matrix column: PE[0][0]->matrix[0], PE[0][2]->matrix[1], PE[0][3]->matrix[2], PE[0][4]->matrix[3], PE[0][5]->matrix[4]
            localparam integer matrix_col = (col == 0) ? 0 : col - 1;
            localparam integer num_buf = (col == 0) ? 0 : (col % 2 == 0) ? (3 * col / 2 - 1) : (3 * (col - 1) / 2);
            if (num_buf == 0) begin : NO_DELAY
                assign y_flow[0][col] = (state == PROCESSING && inject_row < N && matrix_col < N) ? matrix_in[inject_row][matrix_col] :
                                       (state == PROCESSING && inject_row >= N && matrix_col < N) ? matrix_in[N-1][matrix_col] : 32'h0;
            end else begin : BUFFER
                wire [31:0] buffer_chain [0:num_buf];
                assign buffer_chain[0] = (state == PROCESSING && inject_row < N && matrix_col < N) ? matrix_in[inject_row][matrix_col] :
                                        (state == PROCESSING && inject_row >= N && matrix_col < N) ? matrix_in[N-1][matrix_col] : 32'h0;
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

    // Step 1: Instantiate main PEs
    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : pe_row
            for (j = 0; j <= N; j = j + 1) begin : pe_col
                // Lower triangle has no PEs
                if (i > j) begin : LOWER_EMPTY
                // sqrt PE at diagonal position (i,i)
                end else if (i == j) begin : sqrt_pe_inst
                    sqrt u_sqrt_pe (
                        .clk(clk), .rst(rst),
                        .enable(enable_flow[i][j]),
                        .data_in(y_flow[i][j]),
                        .r_new(r_new_wire[i]),
                        .x_orig(x_orig_wire[i]),
                        .y_orig(y_orig_wire[i]),
                        .en_out()
                    );
                // divider PE at position (i, i+1)
                end else if (j == i + 1) begin : divider_pe_inst
                    divider u_divider_pe (
                        .clk(clk), .rst(rst),
                        .enable(enable_div[i]),
                        .r_new(r_new_wire[i]),
                        .x_orig(x_orig_wire[i]),
                        .y_orig(y_orig_wire[i]),
                        .c_out(c_out_from_pe[i][j]),
                        .s_out(s_out_from_pe[i][j]),
                        .r_out_bottom(r_from_div_bottom[i][j]),
                        .en_out()
                    );
                // else: rotation PE
                end else begin : rotation_pe_inst
                    rotation u_rotation_pe (
                        .clk(clk), .rst(rst),
                        .enable(enable_flow[i][j]),
                        .data_in(y_flow[i][j]),
                        .c_in(c_flow[i][j]),
                        .s_in(s_flow[i][j]),
                        .data_out(y_out_from_pe[i][j]),
                        .c_out(c_out_from_pe[i][j]),
                        .s_out(s_out_from_pe[i][j]),
                        .en_out()
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
                for (ej = ei; ej <= N; ej = ej + 1) begin : vert_en_link
                    if (ei < ej) begin : UPPER_TRI_VERT_EN
                        if (ej == ei + 2) begin : FIRST_EN_FLOW
                            // First rotation in row ei feeds next-row sqrt at (ei+1,ei+1)
                            // Base 1-cycle delay to match upper PE's registered data_out (enable_q)
                            en_buffer en_first_y(
                                .clk(clk), .rst(rst),
                                .en_in(enable_flow[ei][ej]),
                                .en_out(enable_flow[ei+1][ei+1])
                            );
                        end else if (ej > ei + 2) begin : NORMAL_EN_FLOW
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

endmodule

// sqrt
module sqrt (
    input   wire clk,
    input   wire rst,
    input   wire enable,
    input   wire [31:0] data_in,
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

    // Register enable to align with downstream data registers (one-cycle later)
    always @(posedge clk or posedge rst) begin
        if (rst) en_out <= 1'b0; else en_out <= enable;
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            r_new <= 32'h0;
            x_orig <= 32'h0;
            y_orig <= 32'h0;
        end else if (enable) begin
            r_new <= r_computed;
            x_orig <= data_in;
            y_orig <= r_new;
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
    input   wire enable,
    input   wire [31:0] r_new,
    input   wire [31:0] x_orig,
    input   wire [31:0] y_orig,
    output  reg  [31:0] c_out,
    output  reg  [31:0] s_out,
    output  reg  [31:0] r_out_bottom,
    output  reg en_out
);
    wire [31:0] c_temp, s_temp;


    // c = y/r
    wire [7:0] div_c_status;
    wire [7:0] div_s_status;

    DW_fp_div #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_div_c (
        .a(y_orig),
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


    // Register enable one-cycle to align with r_new/x_orig/y_orig
    always @(posedge clk or posedge rst) begin
        if (rst) en_out <= 1'b0; else en_out <= enable;
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c_out <= 32'h3F800000; // Default c = 1.0
            s_out <= 32'h0;        // Default s = 0.0
            r_out_bottom <= 32'h0;
        end else if (enable) begin
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
    input   wire enable,
    input   wire [31:0] data_in,
    input   wire [31:0] c_in,
    input   wire [31:0] s_in,
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

    // s*r1
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sr1 (
        .a(s_in),
        .b(r1_reg),
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

    // r2' = c*r2 - s*r1
    wire [7:0] sub1_status;
    DW_fp_sub #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sub1 (
        .a(cr2),
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
        end else if (enable) begin
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
