// 3x3 systolic array matrix multiply: C = R1 * R2
// Stream k=0..2: inject R1[:,k] from left, R2[k,:] from top; each PE does MAC

module RR (
    input  wire        clk,
    input  wire        rstn,
    input  wire        start,
    input  wire [31:0] R1 [0:2][0:2],
    input  wire [31:0] R2 [0:2][0:2],
    output reg         done,
    output wire [31:0] C  [0:2][0:2]
);
    // PE with FP32 MAC
    module PE_FPMAC(
        input  wire        clk,
        input  wire        rstn,
        input  wire [31:0] a_in,
        input  wire [31:0] b_in,
        output wire [31:0] a_out,
        output wire [31:0] b_out,
        output wire [31:0] c_val,
        input  wire        clr
    );
        wire [31:0] prod, sum;
        wire [7:0]  stp, sta;
        DW_fp_mult_DG #(
          .sig_width(23),
          .exp_width(8),
          .ieee_compliance(1)
          ) mul (
            .a(a_in),
            .b(b_in),
            .rnd(3'b000),
            .DG_ctrl(1'b1), 
            .z(prod), 
            .status(stp)
            );
        DW_fp_add     #(
          .sig_width(23),
          .exp_width(8),
          .ieee_compliance(1)
          ) add (
            .a(c_val), 
            .b(prod), 
            .rnd(3'b000), 
            .z(sum), 
            .status(sta)
            );
        // simple shift registers for a/b passthrough
        reg [31:0] a_r, b_r, c_r;
        always @(posedge clk or negedge rstn) begin
            if (!rstn) begin a_r<=0; b_r<=0; c_r<=0; end
            else begin
                a_r <= a_in;
                b_r <= b_in;
                c_r <= clr ? 32'h00000000 : sum; // accumulate
            end
        end
        assign a_out = a_r;
        assign b_out = b_r;
        assign c_val = c_r;
    endmodule

    // 3x3 grid
    wire [31:0] a_w [0:2][0:3]; // 4 vertical taps per row (left port a_w[*][0])
    wire [31:0] b_w [0:3][0:2]; // 4 horizontal taps per col (top port b_w[0][*])
    wire [31:0] c_w [0:2][0:2];

    genvar i,j;
    generate
      for (i=0;i<3;i=i+1) begin: ROW
        for (j=0;j<3;j=j+1) begin: COL
          PE_FPMAC upe(
            .clk(clk), .rstn(rstn),
            .a_in(a_w[i][j]), .b_in(b_w[j][i]),
            .a_out(a_w[i][j+1]), .b_out(b_w[j+1][i]),
            .c_val(c_w[i][j]), .clr(start)
          );
        end
      end
    endgenerate

    // Injection FSM: 3 steps + 2 flush
    reg [2:0] t;
    integer r,c;
    always @(posedge clk or negedge rstn) begin
      if (!rstn) begin
        t <= 0; done <= 1'b0;
      end else begin
        if (start) begin t <= 0; done <= 1'b0; end
        else if (t < 5) begin t <= t + 1'b1; if (t==4) done<=1'b1; end
        else done <= 1'b0;
      end
    end

    // Provide inputs for each cycle t: a_w[i][0]=R1[i][t] (t<3 else 0); b_w[0][j]=R2[t][j]
    generate
      for (i=0;i<3;i=i+1) begin: AIN
        for (j=0;j<4;j=j+1) begin: AINIT
          if (j==0) begin: LEFT
            assign a_w[i][0] = (t<3) ? R1[i][t] : 32'h00000000;
          end
        end
      end
      for (j=0;j<3;j=j+1) begin: BIN
        for (i=0;i<4;i=i+1) begin: BINIT
          if (i==0) begin: TOP
            assign b_w[0][j] = (t<3) ? R2[t][j] : 32'h00000000;
          end
        end
      end
    endgenerate

    // Map C outputs
    generate
      for (i=0;i<3;i=i+1) begin: CROW
        for (j=0;j<3;j=j+1) begin: CCOL
          assign C[i][j] = c_w[i][j];
        end
      end
    endgenerate
endmodule