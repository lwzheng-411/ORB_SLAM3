module baseline #(
    parameter M = 5,  // Number of rows
    parameter N = 5   // Number of columns  
)(
    input   wire clk,
    input   wire rst,
    input   wire start,
    input   wire [31:0] matrix_in [0:M-1][0:N-1],

    output  wire [31:0] r_out [0:M-1][0:N-1],
    output  reg         done
);

    // Vertical data flow (row data from top to bottom)
    wire [31:0] y_flow [0:M][0:N-1];      // Vertical data flow between rows
    
    // Horizontal parameter flow (c, s broadcast from diagonal PE to row)
    wire [31:0] c_flow [0:M-1][0:N];       // Horizontal c parameter flow
    wire [31:0] s_flow [0:M-1][0:N];       // Horizontal s parameter flow
    
    // PE output wires
    wire [31:0] y_out_from_pe [0:M-1][0:N-1];
    wire [31:0] c_out_from_pe [0:M-1][0:N-1];
    wire [31:0] s_out_from_pe [0:M-1][0:N-1];
    
    // R matrix storage in PEs
    wire [31:0] r_stored [0:M-1][0:N-1];
    
    // FSM states
    reg [2:0] state;
    reg [7:0] cycle_count;    // Total cycle counter
    
    parameter IDLE = 3'h0,
              PROCESSING = 3'h1,
              DONE_STATE = 3'h2;
   
    genvar ri, rj;
    generate
        for (ri = 0; ri < M; ri = ri + 1) begin : R_OUT_ROW
            for (rj = 0; rj < N; rj = rj + 1) begin : R_OUT_COL
                if (ri > rj) begin
                    // Lower triangle: always zero for R matrix
                    assign r_out[ri][rj] = 32'h0;
                end else begin
                    // Upper triangle and diagonal: from PE storage
                    assign r_out[ri][rj] = r_stored[ri][rj];
                end
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Main Control FSM
    //--------------------------------------------------------------------------
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            cycle_count <= 8'h0;
            done <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= PROCESSING;
                        cycle_count <= 8'h0;
                    end
                    done <= 1'b0;
                end
                
                PROCESSING: begin
                    // Input matrix rows sequentially for 2D systolic processing
                    if (cycle_count < 40) begin
                        cycle_count <= cycle_count + 1'b1;
                    end else begin
                        state <= DONE_STATE;
                    end
                end
                
                DONE_STATE: begin
                    if (!start) begin
                        state <= IDLE;
                        done <= 1'b0;
                    end
                end
            endcase
        end
    end

    //--------------------------------------------------------------------------
    // Input Data Injection (Row-wise from Top)
    //--------------------------------------------------------------------------
    
    genvar ic;
    generate
        for (ic = 0; ic < N; ic = ic + 1) begin : PROCESSING_COL
            // Top row input: inject matrix data row by row
            assign y_flow[0][ic] = (state == PROCESSING && cycle_count < M) ? 
                                   matrix_in[cycle_count][ic] : 32'h0;
        end
    endgenerate

    //--------------------------------------------------------------------------
    // PE Instantiation
    //--------------------------------------------------------------------------
    
    genvar i, j;
    generate
        for (i = 0; i < M; i = i + 1) begin : PE_ROW
            for (j = 0; j < N; j = j + 1) begin : PE_COL
                
                if (i > j) begin : LOWER_TRIANGLE
                    // Lower triangle: no PEs needed for QR decomposition
                    assign y_out_from_pe[i][j] = 32'h0;
                    assign c_out_from_pe[i][j] = 32'h0;
                    assign s_out_from_pe[i][j] = 32'h0;
                    assign r_stored[i][j] = 32'h0;
                    
                end else if (i == j) begin : DIAGONAL_PE
                    // Diagonal PE: Compute Givens rotation parameters
                    diagonal_pe u_diagonal_pe (
                        .clk(clk),
                        .rst(rst),
                        .x_in(y_flow[i][j]),        // From previous row
                        .c_out(c_out_from_pe[i][j]),
                        .s_out(s_out_from_pe[i][j]),
                        .y_out(y_out_from_pe[i][j]),
                        .r_stored(r_stored[i][j])
                    );
                    
                end else begin : OFF_DIAGONAL_PE
                    // Off-diagonal PE: Apply Givens rotation
                    off_diagonal_pe u_off_diagonal_pe (
                        .clk(clk),
                        .rst(rst),
                        .c_in(c_flow[i][j]),
                        .s_in(s_flow[i][j]),
                        .r1_in(y_flow[i][j]),       // From previous row
                        .r1_out(y_out_from_pe[i][j]),
                        .c_out(c_out_from_pe[i][j]),
                        .s_out(s_out_from_pe[i][j]),
                        .r_stored(r_stored[i][j])
                    );
                end
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Data Flow Routing
    //--------------------------------------------------------------------------
    
    generate
        // Vertical connections: data flows from top to bottom
        for (i = 0; i < M-1; i = i + 1) begin : VERT_FLOW
            for (j = 0; j < N; j = j + 1) begin : VERT_COL
                if (i < j) begin  // Only in upper triangle
                    assign y_flow[i+1][j] = y_out_from_pe[i][j];
                end else begin
                    assign y_flow[i+1][j] = 32'h0;
                end
            end
        end
        
        // Horizontal connections: c,s parameters broadcast along rows
        for (i = 0; i < M; i = i + 1) begin : HORIZ_FLOW
            // Connect diagonal PE output to start horizontal broadcast
            if (i < N) begin
                assign c_flow[i][i] = c_out_from_pe[i][i];
                assign s_flow[i][i] = s_out_from_pe[i][i];
            end
            
            // Propagate c,s horizontally across the row
            for (j = i+1; j < N; j = j + 1) begin : HORIZ_COL
                assign c_flow[i][j] = c_out_from_pe[i][j-1];
                assign s_flow[i][j] = s_out_from_pe[i][j-1];
            end
        end
    endgenerate

endmodule

//==============================================================================
// Diagonal PE: Computes Givens rotation parameters (c, s)
// Implements: c = x/sqrt(x²+y²), s = -y/sqrt(x²+y²)
//==============================================================================
module diagonal_pe (
    input   wire clk,
    input   wire rst,
    input   wire [31:0] x_in,           // Input from previous row
    output  reg  [31:0] c_out,
    output  reg  [31:0] s_out,
    output  reg  [31:0] y_out,
    output  reg  [31:0] r_stored
);

    // Internal storage for y value
    reg [31:0] y_mem;
    
    // Intermediate computation wires
    wire [31:0] x_squared, y_squared, sum_squares;
    wire [31:0] sqrt_result, c_temp, s_temp_pos;
    wire [7:0] mult_x_status, mult_y_status, add_status, sqrt_status;
    wire [7:0] div_c_status, div_s_status;

    // IEEE 754 Single Precision Floating Point Units
    
    // x² computation
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_mult_x (
        .a(x_in),
        .b(x_in),
        .rnd(3'b000),        // Round to nearest even
        .DG_ctrl(1'b1),      // Data gating control
        .z(x_squared),
        .status(mult_x_status)
    );
    
    // y² computation  
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_mult_y (
        .a(y_mem),
        .b(y_mem),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(y_squared),
        .status(mult_y_status)
    );
    
    // x² + y² computation
    DW_fp_add #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_add (
        .a(x_squared),
        .b(y_squared),
        .rnd(3'b000),
        .z(sum_squares),
        .status(add_status)
    );
    
    // sqrt(x² + y²) computation
    DW_fp_sqrt #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sqrt (
        .a(sum_squares),
        .rnd(3'b000),
        .z(sqrt_result),
        .status(sqrt_status)
    );
    
    // c = x / sqrt(x² + y²)
    DW_fp_div #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_div_c (
        .a(x_in),
        .b(sqrt_result),
        .rnd(3'b000),
        .z(c_temp),
        .status(div_c_status)
    );
    
    // s = y / sqrt(x² + y²) (will negate later)
    DW_fp_div #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_div_s (
        .a(y_mem),
        .b(sqrt_result),
        .rnd(3'b000),
        .z(s_temp_pos),
        .status(div_s_status)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            y_mem <= 32'h0;
            c_out <= 32'h3F800000;  // 1.0 (default)
            s_out <= 32'h0;         // 0.0 (default)
            y_out <= 32'h0;
            r_stored <= 32'h0;
        end else if (enable) begin
            // Update y_mem with input x for next iteration
            y_mem <= x_in;
            
            // Output computed Givens parameters
            c_out <= c_temp;
            s_out <= {~s_temp_pos[31], s_temp_pos[30:0]};  // Negate s
            
            // Output sqrt result (this becomes the R matrix element)
            y_out <= sqrt_result;
            r_stored <= sqrt_result;
        end
    end

endmodule

//==============================================================================
// Off-Diagonal PE: Applies Givens rotation
// Implements: r1' = c*r1 + s*r2, r2' = c*r2 - s*r1
//==============================================================================
module off_diagonal_pe (
    input   wire clk,
    input   wire rst,
    input   wire enable,
    input   wire [31:0] c_in,
    input   wire [31:0] s_in,
    input   wire [31:0] r1_in,          // From previous row
    input   wire [31:0] r2_stored_in,   // For external init (unused)
    output  reg  [31:0] r1_out,
    output  wire [31:0] r2_out,         // Internal, not used externally
    output  reg  [31:0] c_out,          // Pass-through
    output  reg  [31:0] s_out,          // Pass-through
    output  reg  [31:0] r_stored
);

    // Internal storage for r2 value
    reg [31:0] r2_mem;
    
    // Intermediate computation wires
    wire [31:0] c_r1, s_r2, c_r2, s_r1;
    wire [31:0] r1_prime, r2_prime;
    wire [7:0] mult_status[3:0], add_status, sub_status;

    // c * r1
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_mult_cr1 (
        .a(c_in),
        .b(r1_in),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(c_r1),
        .status(mult_status[0])
    );
    
    // s * r2
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_mult_sr2 (
        .a(s_in),
        .b(r2_mem),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(s_r2),
        .status(mult_status[1])
    );
    
    // c * r2
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_mult_cr2 (
        .a(c_in),
        .b(r2_mem),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(c_r2),
        .status(mult_status[2])
    );
    
    // s * r1
    DW_fp_mult_DG #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_mult_sr1 (
        .a(s_in),
        .b(r1_in),
        .rnd(3'b000),
        .DG_ctrl(1'b1),
        .z(s_r1),
        .status(mult_status[3])
    );
    
    // r1' = c*r1 + s*r2
    DW_fp_add #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_add (
        .a(c_r1),
        .b(s_r2),
        .rnd(3'b000),
        .z(r1_prime),
        .status(add_status)
    );
    
    // r2' = c*r2 - s*r1
    DW_fp_sub #(
        .sig_width(23),
        .exp_width(8),
        .ieee_compliance(1)
    ) u_sub (
        .a(c_r2),
        .b(s_r1),
        .rnd(3'b000),
        .z(r2_prime),
        .status(sub_status)
    );
    
    assign r2_out = r2_prime;  // For internal routing

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            r2_mem <= 32'h0;
            r1_out <= 32'h0;
            c_out <= 32'h0;
            s_out <= 32'h0;
            r_stored <= 32'h0;
        end else if (enable) begin
            // Update internal storage
            r2_mem <= r2_prime;
            
            // Output transformed r1
            r1_out <= r1_prime;
            
            // Pass through c,s parameters
            c_out <= c_in;
            s_out <= s_in;
            
            // Store final R matrix element
            r_stored <= r1_prime;
        end else begin
            // Pass through when not enabled
            r1_out <= r1_in;
            c_out <= c_in;
            s_out <= s_in;
        end
    end

endmodule