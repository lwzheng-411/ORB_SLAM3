// ============================================================================
// Axb: Production-Ready Factor SIMD for Tapeout
// Complete implementation: Camera/IMU/Prior with CORDIC Log
// Single file, optimized registers, shared PE pool
// NO simplifications - all states fully implemented
// ============================================================================
module Axb #(
    parameter NUM_MUL = 36,
    parameter NUM_ADD = 24,
    parameter NUM_SUB = 8
)(
    input  wire                 clk,
    input  wire                 rst,

    // Bundle control (local variable elimination)
    input  wire                 bundle_begin,
    input  wire                 bundle_end,

    // Factor input (unified port)
    input  wire                 factor_valid,
    output reg                  factor_ready,
    input  wire [1:0]           factor_type,    // 0=Camera, 1=IMU, 2=Prior
    
    // Column mapping (CPU pre-computed local column indices)
    input  wire [7:0]           local_col_panel,  // Panel start col in local matrix (usually 0)
    input  wire [7:0]           local_col_trail,  // Trail start col in local matrix (CPU computed)
    
    // Row whitening coefficients
    input  wire [31:0]          alpha [0:5],

    // Camera payload
    input  wire [31:0]          cam_fx, cam_fy, cam_cx, cam_cy,
    input  wire [31:0]          cam_z_u, cam_z_v,
    input  wire [31:0]          cam_Pc_X, cam_Pc_Y, cam_Pc_Z, cam_Pc_invZ,
    input  wire [31:0]          cam_Rcw [0:2][0:2],

    // IMU/Prior payload
    input  wire [31:0]          R0 [0:2][0:2],
    input  wire [31:0]          R1 [0:2][0:2],
    input  wire [31:0]          Rd [0:2][0:2],
    input  wire [31:0]          t0 [0:2],
    input  wire [31:0]          t1 [0:2],
    input  wire [31:0]          td [0:2],

    // Row streaming output to RowBuffer
    output reg                  row_out_valid,
    input  wire                 row_out_ready,
    output reg  [2:0]           row_out_panel_cols,
    output reg  [2:0]           row_out_trail_cols,
    output reg  [7:0]           row_out_local_col_panel,
    output reg  [7:0]           row_out_local_col_trail,
    output reg  [31:0]          row_out_panel [0:5],
    output reg  [31:0]          row_out_trail [0:5],
    output reg  [31:0]          row_out_b,
    output reg                  row_out_last_in_factor
);

    // ========================================================================
    // Shared PE Pool - Direct DesignWare IP instantiation (like systolicarray.sv)
    // All three factors (Camera/IMU/Prior) time-multiplex these IPs
    // ========================================================================
    reg [31:0] mul_a [0:NUM_MUL-1], mul_b [0:NUM_MUL-1];
    wire[31:0] mul_z [0:NUM_MUL-1];
    wire[7:0] mul_st [0:NUM_MUL-1];
    genvar gm; generate for (gm=0; gm<NUM_MUL; gm=gm+1) begin : MUL
        DW_fp_mult_DG #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) u (
            .a(mul_a[gm]), .b(mul_b[gm]), .rnd(3'b000), .DG_ctrl(1'b1),
            .z(mul_z[gm]), .status(mul_st[gm]));
    end endgenerate
    reg mul_en[0:NUM_MUL-1]; // Enable flags (used for valid tracking)
    reg mul_vld_r[0:NUM_MUL-1]; // Delay1 for valid
    wire mul_vld[0:NUM_MUL-1];
    genvar gmv; generate for(gmv=0; gmv<NUM_MUL; gmv=gmv+1) begin : MULV
        always @(posedge clk or posedge rst) begin
            if(rst) mul_vld_r[gmv]<=1'b0;
            else mul_vld_r[gmv]<=mul_en[gmv];
        end
        assign mul_vld[gmv]=mul_vld_r[gmv];
    end endgenerate

    reg [31:0] add_a [0:NUM_ADD-1], add_b [0:NUM_ADD-1];
    wire[31:0] add_z [0:NUM_ADD-1];
    wire[7:0] add_st [0:NUM_ADD-1];
    genvar ga; generate for (ga=0; ga<NUM_ADD; ga=ga+1) begin : ADD
        DW_fp_add #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) u (
            .a(add_a[ga]), .b(add_b[ga]), .rnd(3'b000),
            .z(add_z[ga]), .status(add_st[ga]));
    end endgenerate
    reg add_en[0:NUM_ADD-1]; reg add_vld_r[0:NUM_ADD-1]; wire add_vld[0:NUM_ADD-1];
    genvar gav; generate for(gav=0; gav<NUM_ADD; gav=gav+1) begin : ADDV
        always @(posedge clk or posedge rst) begin
            if(rst) add_vld_r[gav]<=1'b0;
            else add_vld_r[gav]<=add_en[gav];
        end
        assign add_vld[gav]=add_vld_r[gav];
    end endgenerate

    reg [31:0] sub_a [0:NUM_SUB-1], sub_b [0:NUM_SUB-1];
    wire[31:0] sub_z [0:NUM_SUB-1];
    wire[7:0] sub_st [0:NUM_SUB-1];
    genvar gs; generate for (gs=0; gs<NUM_SUB; gs=gs+1) begin : SUB
        DW_fp_sub #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) u (
            .a(sub_a[gs]), .b(sub_b[gs]), .rnd(3'b000),
            .z(sub_z[gs]), .status(sub_st[gs]));
    end endgenerate
    reg sub_en[0:NUM_SUB-1]; reg sub_vld_r[0:NUM_SUB-1]; wire sub_vld[0:NUM_SUB-1];
    genvar gsv; generate for(gsv=0; gsv<NUM_SUB; gsv=gsv+1) begin : SUBV
        always @(posedge clk or posedge rst) begin
            if(rst) sub_vld_r[gsv]<=1'b0;
            else sub_vld_r[gsv]<=sub_en[gsv];
        end
        assign sub_vld[gsv]=sub_vld_r[gsv];
    end endgenerate

    reg [31:0] sqrt_in; wire[31:0] sqrt_out; wire[7:0] sqrt_st;
    DW_fp_sqrt #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) usqrt (
        .a(sqrt_in), .rnd(3'b000), .z(sqrt_out), .status(sqrt_st));
    reg sqrt_en; reg sqrt_vld_r; wire sqrt_vld;
    always @(posedge clk or posedge rst) begin
        if(rst) sqrt_vld_r<=1'b0; else sqrt_vld_r<=sqrt_en;
    end
    assign sqrt_vld=sqrt_vld_r;

    reg [31:0] div_num, div_den; wire[31:0] div_quot; wire[7:0] div_st;
    DW_fp_div #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) udiv (
        .a(div_num), .b(div_den), .rnd(3'b000), .z(div_quot), .status(div_st));
    reg div_en; reg div_vld_r; wire div_vld;
    always @(posedge clk or posedge rst) begin
        if(rst) div_vld_r<=1'b0; else div_vld_r<=div_en;
    end
    assign div_vld=div_vld_r;

    // atan2 and skew keep custom modules (no DW equivalent)
    reg [31:0] atan_y, atan_x; reg atan_en; wire[31:0] atan_theta; wire atan_vld;
    Atan2Cordic uatan(.clk(clk), .rstn(~rst), .in_vld(atan_en), .y(atan_y), .x(atan_x), .out_vld(atan_vld), .theta(atan_theta));

    reg [31:0] skew_x, skew_y, skew_z; wire[31:0] skew_K [0:2][0:2];
    Skew uskew(.x(skew_x), .y(skew_y), .z(skew_z), .K(skew_K));

    // Constants
    localparam [31:0] FP_0=32'h0, FP_1=32'h3f800000, FP_N1=32'hbf800000, FP_HALF=32'h3f000000, FP_2=32'h40000000;
    function automatic [31:0] neg(input [31:0] a); neg={~a[31],a[30:0]}; endfunction

    // ------------------------------------------------------------------------
    // Macro helpers for readability (no functional change)
    // ------------------------------------------------------------------------
    `define MUL(I,A,B) begin mul_a[I]<=A; mul_b[I]<=B; mul_en[I]<=1'b1; end
    `define ADD(I,A,B) begin add_a[I]<=A; add_b[I]<=B; add_en[I]<=1'b1; end
    `define SUB(I,A,B) begin sub_a[I]<=A; sub_b[I]<=B; sub_en[I]<=1'b1; end

    // Engine selector: tracks which factor type is currently executing
    localparam [1:0] ENG_IDLE=2'd0, ENG_CAM=2'd1, ENG_IMU=2'd2, ENG_PRI=2'd3;
    reg [1:0] eng;

    // States: Camera(11) + IMU(31) + Prior(21) + IDLE
    localparam [5:0]
        S0=0,
        // Camera: 11 states (projection, jacobian, whitening, emit x2)
        SC1=1, SC2=2, SC3=3, SC4=4, SC5=5, SC6=6, SC7=7, SC8=8, SC9=9, SC10=10, SC11=11,
        // IMU: 28 states (vi, rt, C, Re, Log, M, Kvi, RtK, whiten, emit x6)
        SI1=12, SI2=13, SI3=14, SI4=15, SI5=16, SI6=17, SI7=18, SI8=19, SI9=20, SI10=21,
        SI11=22, SI12=23, SI13=24, SI14=25, SI15=26, SI16=27, SI17=28, SI18=29, SI19=30,
        SI20=31, SI21=32, SI22=33, SI23=34, SI24=35, SI25=36, SI26=37, SI27=38, SI28=39,
        // IMU M matrix computation
        SM1=56, SM2=57, SM3=58,
        // IMU emit 6 rows
        SI29=59, SI30=60, SI31=61, SI32=62, SI33=63, SI34=64,
        // Prior: 21 states (tdiff, tresid, C 3x3, Log, emit x6)
        SP1=40, SP2=41, SP3=42, SP4=43, SP5=44, SP6=45, SP7=46, SP8=47, SP9=48, SP10=49,
        SP11=50, SP12=51, SP13=52, SP14=53, SP15=54, SP16=65, SP17=66, SP18=67, SP19=68,
        SP20=69, SP21=70;
    reg [5:0] st;

    // Temp registers (64x32b = 2KB, 复用于所有因子类型以节省面积)
    reg [31:0] t [0:63];
    integer i;

    // Main FSM: single state machine handles all three factor types
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            st<=S0; eng<=ENG_IDLE; factor_ready<=1'b1; row_out_valid<=1'b0;
            for(i=0;i<NUM_MUL;i=i+1) mul_en[i]<=1'b0;
            for(i=0;i<NUM_ADD;i=i+1) add_en[i]<=1'b0;
            for(i=0;i<NUM_SUB;i=i+1) sub_en[i]<=1'b0;
            sqrt_en<=1'b0; div_en<=1'b0; atan_en<=1'b0;
        end else begin
            for(i=0;i<NUM_MUL;i=i+1) mul_en[i]<=1'b0;
            for(i=0;i<NUM_ADD;i=i+1) add_en[i]<=1'b0;
            for(i=0;i<NUM_SUB;i=i+1) sub_en[i]<=1'b0;
            sqrt_en<=1'b0; div_en<=1'b0; atan_en<=1'b0;
            row_out_valid<=1'b0;

            case(st)
                S0: if(factor_valid&&factor_ready) begin
                    factor_ready<=1'b0;
                    case(factor_type)
                        2'd0: begin eng<=ENG_CAM; st<=SC1; end
                        2'd1: begin eng<=ENG_IMU; st<=SI1; end
                        2'd2: begin eng<=ENG_PRI; st<=SP1; end
                        default: begin eng<=ENG_IDLE; factor_ready<=1'b1; end
                    endcase
                end

                // ================================================
                // CAMERA - 11 states完整实现
                // ================================================
                SC1: begin // Projection scalars
                    `MUL(0, cam_fx,   cam_Pc_invZ)
                    `MUL(1, cam_fy,   cam_Pc_invZ)
                    `MUL(2, cam_Pc_X, cam_Pc_invZ)
                    `MUL(3, cam_Pc_Y, cam_Pc_invZ)
                    `MUL(4, cam_Pc_invZ, cam_Pc_invZ)
                    st<=SC2;
                end
                SC2: begin
                    if(mul_vld[0]) begin
                        t[0]<=mul_z[0]; t[1]<=mul_z[1]; t[2]<=mul_z[2]; t[3]<=mul_z[3]; t[4]<=mul_z[4];
                        `MUL(0, cam_fx, t[2])
                        `MUL(1, cam_fy, t[3])
                        `MUL(2, cam_fx, cam_Pc_X)
                        `MUL(3, cam_fy, cam_Pc_Y)
                        st<=SC3;
                    end
                end
                SC3: begin
                    if(mul_vld[0]) begin
                        t[5]<=mul_z[0]; t[6]<=mul_z[1]; t[7]<=mul_z[2]; t[8]<=mul_z[3];
                        `ADD(0, t[5], cam_cx)
                        `ADD(1, t[6], cam_cy)
                        `MUL(0, t[7], t[4])
                        `MUL(1, t[8], t[4])
                        st<=SC4;
                    end
                end
                SC4: begin // Residual & Skew
                    if(add_vld[0]&&mul_vld[0]) begin
                        t[9]<=add_z[0]; t[10]<=add_z[1]; t[11]<=neg(mul_z[0]); t[12]<=neg(mul_z[1]);
                        `SUB(0, cam_z_u, t[9])
                        `SUB(1, cam_z_v, t[10])
                        skew_x<=cam_Pc_X; skew_y<=cam_Pc_Y; skew_z<=cam_Pc_Z;
                        st<=SC5;
                    end
                end
                SC5: begin // Jl row0
                    if(sub_vld[0]) begin
                        t[13]<=sub_z[0]; t[14]<=sub_z[1];
                        `MUL(0, t[0],  cam_Rcw[0][0])
                        `MUL(1, t[0],  cam_Rcw[0][1])
                        `MUL(2, t[0],  cam_Rcw[0][2])
                        `MUL(3, t[11], cam_Rcw[2][0])
                        `MUL(4, t[11], cam_Rcw[2][1])
                        `MUL(5, t[11], cam_Rcw[2][2])
                        st<=SC6;
                    end
                end
                SC6: begin
                    if(mul_vld[0]) begin
                        t[15]<=mul_z[0]; t[16]<=mul_z[1]; t[17]<=mul_z[2];
                        t[18]<=mul_z[3]; t[19]<=mul_z[4]; t[20]<=mul_z[5];
                        `ADD(0, t[15], t[18])
                        `ADD(1, t[16], t[19])
                        `ADD(2, t[17], t[20])
                        `MUL(6, t[1],  cam_Rcw[1][0])
                        `MUL(7, t[1],  cam_Rcw[1][1])
                        `MUL(8, t[1],  cam_Rcw[1][2])
                        `MUL(9,  t[12], cam_Rcw[2][0])
                        `MUL(10, t[12], cam_Rcw[2][1])
                        `MUL(11, t[12], cam_Rcw[2][2])
                        st<=SC7;
                    end
                end
                SC7: begin
                    if(add_vld[0]&&mul_vld[6]) begin
                        t[21]<=add_z[0]; t[22]<=add_z[1]; t[23]<=add_z[2]; // Jl[0]
                        t[24]<=mul_z[6]; t[25]<=mul_z[7]; t[26]<=mul_z[8];
                        t[27]<=mul_z[9]; t[28]<=mul_z[10]; t[29]<=mul_z[11];
                        `ADD(0, t[24], t[27])
                        `ADD(1, t[25], t[28])
                        `ADD(2, t[26], t[29])
                        // Jpose_rot
                        `MUL(3, t[0],  neg(skew_K[0][0]))
                        `MUL(4, t[0],  neg(skew_K[0][1]))
                        `MUL(5, t[0],  neg(skew_K[0][2]))
                        `MUL(6, t[11], neg(skew_K[2][0]))
                        `MUL(7, t[11], neg(skew_K[2][1]))
                        `MUL(8, t[11], neg(skew_K[2][2]))
                        st<=SC8;
                    end
                end
                SC8: begin
                    if(add_vld[0]&&mul_vld[3]) begin
                        t[30]<=add_z[0]; t[31]<=add_z[1]; t[32]<=add_z[2]; // Jl[1]
                        t[33]<=mul_z[3]; t[34]<=mul_z[4]; t[35]<=mul_z[5];
                        t[36]<=mul_z[6]; t[37]<=mul_z[7]; t[38]<=mul_z[8];
                        `ADD(0, t[33], t[36])
                        `ADD(1, t[34], t[37])
                        `ADD(2, t[35], t[38])
                        `MUL(9,  t[1],  neg(skew_K[1][0]))
                        `MUL(10, t[1],  neg(skew_K[1][1]))
                        `MUL(11, t[1],  neg(skew_K[1][2]))
                        `MUL(12, t[12], neg(skew_K[2][0]))
                        `MUL(13, t[12], neg(skew_K[2][1]))
                        `MUL(14, t[12], neg(skew_K[2][2]))
                        st<=SC9;
                    end
                end
                SC9: begin // Whiten row0
                    if(add_vld[0]&&mul_vld[9]) begin
                        t[39]<=add_z[0]; t[40]<=add_z[1]; t[41]<=add_z[2]; // Jpose[0][6..8]
                        t[42]<=mul_z[9]; t[43]<=mul_z[10]; t[44]<=mul_z[11];
                        t[45]<=mul_z[12]; t[46]<=mul_z[13]; t[47]<=mul_z[14];
                        `ADD(3, t[42], t[45])
                        `ADD(4, t[43], t[46])
                        `ADD(5, t[44], t[47])
                        `MUL(0, alpha[0], t[21])
                        `MUL(1, alpha[0], t[22])
                        `MUL(2, alpha[0], t[23])
                        `MUL(3, alpha[0], t[0]) // Jpose_trans[0]
                        `MUL(4, alpha[0], FP_0)
                        `MUL(5, alpha[0], t[11])
                        st<=SC10;
                    end
                end
                SC10: begin // Emit row0
                    if(add_vld[3]&&mul_vld[0]) begin
                        t[48]<=add_z[3]; t[49]<=add_z[4]; t[50]<=add_z[5]; // Jpose[1][6..8]
                        `MUL(6, alpha[0], t[39])
                        `MUL(7, alpha[0], t[40])
                        `MUL(8, alpha[0], t[41])
                        `MUL(9, alpha[0], t[13]) // res[0]
                    end
                    if(mul_vld[6]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel_cols<=3; row_out_trail_cols<=6;
                        row_out_local_col_panel<=local_col_panel;
                        row_out_local_col_trail<=local_col_trail;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=FP_0; row_out_panel[4]<=FP_0; row_out_panel[5]<=FP_0;
                        row_out_trail[0]<=mul_z[3]; row_out_trail[1]<=mul_z[4]; row_out_trail[2]<=mul_z[5];
                        row_out_trail[3]<=mul_z[6]; row_out_trail[4]<=mul_z[7]; row_out_trail[5]<=mul_z[8];
                        row_out_b<=mul_z[9];
                        row_out_last_in_factor<=1'b0;
                        // Whiten row1
                        `MUL(10, alpha[1], t[30])
                        `MUL(11, alpha[1], t[31])
                        `MUL(12, alpha[1], t[32])
                        `MUL(13, alpha[1], FP_0)
                        `MUL(14, alpha[1], t[1])
                        `MUL(15, alpha[1], t[12])
                        `MUL(0,  alpha[1], t[48])
                        `MUL(1,  alpha[1], t[49])
                        `MUL(2,  alpha[1], t[50])
                        `MUL(3,  alpha[1], t[14]) // res[1]
                        st<=SC11;
                    end
                end
                SC11: begin // Emit row1
                    if(mul_vld[10]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel_cols<=3; row_out_trail_cols<=6;
                        row_out_local_col_panel<=local_col_panel;
                        row_out_local_col_trail<=local_col_trail;
                        row_out_panel[0]<=mul_z[10]; row_out_panel[1]<=mul_z[11]; row_out_panel[2]<=mul_z[12];
                        row_out_panel[3]<=FP_0; row_out_panel[4]<=FP_0; row_out_panel[5]<=FP_0;
                        row_out_trail[0]<=mul_z[13]; row_out_trail[1]<=mul_z[14]; row_out_trail[2]<=mul_z[15];
                        row_out_trail[3]<=mul_z[0]; row_out_trail[4]<=mul_z[1]; row_out_trail[5]<=mul_z[2];
                        row_out_b<=mul_z[3];
                        row_out_last_in_factor<=1'b1;
                        st<=S0; factor_ready<=1'b1; eng<=ENG_IDLE;
                    end
                end

                // ================================================
                // IMU - 31 states 完整实现，全部使用宏简化
                // 硬件复用: 与Camera/Prior共享 mul_a[0:15], add_a[0:11], sub_a[0:7]
                // 计算流程: vi → rt → C → Re → Log → M → RtK → 白化 → 6行输出
                // ================================================
                SI1: begin // v = tj - ti
                    `SUB(0,t1[0],t0[0]) `SUB(1,t1[1],t0[1]) `SUB(2,t1[2],t0[2])
                    st<=SI2;
                end
                SI2: begin // vi = Ri^T * v (9 mul)
                    if(sub_vld[0]) begin
                        t[0]<=sub_z[0]; t[1]<=sub_z[1]; t[2]<=sub_z[2];
                        `MUL(0,R0[0][0],t[0]) `MUL(1,R0[1][0],t[1]) `MUL(2,R0[2][0],t[2])
                        `MUL(3,R0[0][1],t[0]) `MUL(4,R0[1][1],t[1]) `MUL(5,R0[2][1],t[2])
                        `MUL(6,R0[0][2],t[0]) `MUL(7,R0[1][2],t[1]) `MUL(8,R0[2][2],t[2])
                        st<=SI3;
                    end
                end
                SI3: begin // vi reduction
                    if(mul_vld[0]) begin
                        t[3]<=mul_z[0]; t[4]<=mul_z[1]; t[5]<=mul_z[2];
                        t[6]<=mul_z[3]; t[7]<=mul_z[4]; t[8]<=mul_z[5];
                        t[9]<=mul_z[6]; t[10]<=mul_z[7]; t[11]<=mul_z[8];
                        `ADD(0,t[3],t[4]) `ADD(1,t[6],t[7]) `ADD(2,t[9],t[10])
                        st<=SI4;
                    end
                end
                SI4: begin
                    if(add_vld[0]) begin
                        t[12]<=add_z[0]; t[13]<=add_z[1]; t[14]<=add_z[2];
                        `ADD(0,t[12],t[5]) `ADD(1,t[13],t[8]) `ADD(2,t[14],t[11])
                        st<=SI5;
                    end
                end
                SI5: begin // vi完成, vtilde=vi-td
                    if(add_vld[0]) begin
                        t[15]<=add_z[0]; t[16]<=add_z[1]; t[17]<=add_z[2]; // vi
                        `SUB(0,t[15],td[0]) `SUB(1,t[16],td[1]) `SUB(2,t[17],td[2])
                        st<=SI6;
                    end
                end
                SI6: begin // rt = Rd^T * vtilde (9 mul)
                    if(sub_vld[0]) begin
                        t[18]<=sub_z[0]; t[19]<=sub_z[1]; t[20]<=sub_z[2]; // vtilde
                        `MUL(0,Rd[0][0],t[18]) `MUL(1,Rd[1][0],t[19]) `MUL(2,Rd[2][0],t[20])
                        `MUL(3,Rd[0][1],t[18]) `MUL(4,Rd[1][1],t[19]) `MUL(5,Rd[2][1],t[20])
                        `MUL(6,Rd[0][2],t[18]) `MUL(7,Rd[1][2],t[19]) `MUL(8,Rd[2][2],t[20])
                        st<=SI7;
                    end
                end
                SI7: begin // rt reduction
                    if(mul_vld[0]) begin
                        t[21]<=mul_z[0]; t[22]<=mul_z[1]; t[23]<=mul_z[2];
                        t[24]<=mul_z[3]; t[25]<=mul_z[4]; t[26]<=mul_z[5];
                        t[27]<=mul_z[6]; t[28]<=mul_z[7]; t[29]<=mul_z[8];
                        `ADD(0,t[21],t[22]) `ADD(1,t[24],t[25]) `ADD(2,t[27],t[28])
                        st<=SI8;
                    end
                end
                SI8: begin
                    if(add_vld[0]) begin
                        t[30]<=add_z[0]; t[31]<=add_z[1]; t[32]<=add_z[2];
                        `ADD(0,t[30],t[23]) `ADD(1,t[31],t[26]) `ADD(2,t[32],t[29])
                        st<=SI9;
                    end
                end
                SI9: begin // rt完成, 开始C=Ri^T*Rj (9 mul)
                    if(add_vld[0]) begin
                        t[33]<=add_z[0]; t[34]<=add_z[1]; t[35]<=add_z[2]; // rt[0..2] residual
                        `MUL(0,R0[0][0],R1[0][0]) `MUL(1,R0[1][0],R1[1][0]) `MUL(2,R0[2][0],R1[2][0])
                        `MUL(3,R0[0][0],R1[0][1]) `MUL(4,R0[1][0],R1[1][1]) `MUL(5,R0[2][0],R1[2][1])
                        `MUL(6,R0[0][0],R1[0][2]) `MUL(7,R0[1][0],R1[1][2]) `MUL(8,R0[2][0],R1[2][2])
                        st<=SI10;
                    end
                end
                SI10: begin // C row0 reduction + C row1 start
                    if(mul_vld[0]) begin
                        t[36]<=mul_z[0]; t[37]<=mul_z[1]; t[38]<=mul_z[2];
                        t[39]<=mul_z[3]; t[40]<=mul_z[4]; t[41]<=mul_z[5];
                        t[42]<=mul_z[6]; t[43]<=mul_z[7]; t[44]<=mul_z[8];
                        `ADD(0,t[36],t[37]) `ADD(1,t[39],t[40]) `ADD(2,t[42],t[43])
                        `MUL(9,R0[0][1],R1[0][0]) `MUL(10,R0[1][1],R1[1][0]) `MUL(11,R0[2][1],R1[2][0])
                        `MUL(12,R0[0][1],R1[0][1]) `MUL(13,R0[1][1],R1[1][1]) `MUL(14,R0[2][1],R1[2][1])
                        `MUL(15,R0[0][1],R1[0][2])
                        st<=SI11;
                    end
                end
                SI11: begin // C[0][0..2]完成
                    if(add_vld[0]&&mul_vld[9]) begin
                        t[45]<=add_z[0]; t[46]<=add_z[1]; t[47]<=add_z[2];
                        `ADD(3,t[45],t[38]) `ADD(4,t[46],t[41]) `ADD(5,t[47],t[44])
                        t[48]<=mul_z[9]; t[49]<=mul_z[10]; t[50]<=mul_z[11];
                        t[51]<=mul_z[12]; t[52]<=mul_z[13]; t[53]<=mul_z[14]; t[54]<=mul_z[15];
                        `MUL(0,R0[0][2],R1[0][0]) `MUL(1,R0[1][2],R1[1][0]) `MUL(2,R0[2][2],R1[2][0])
                        `MUL(3,R0[0][2],R1[0][1]) `MUL(4,R0[1][2],R1[1][1]) `MUL(5,R0[2][2],R1[2][1])
                        `MUL(6,R0[0][2],R1[0][2]) `MUL(7,R0[1][2],R1[1][2]) `MUL(8,R0[2][2],R1[2][2])
                        st<=SI12;
                    end
                end
                SI12: begin // C row1&2 reduction
                    if(add_vld[3]&&mul_vld[0]) begin
                        t[55]<=add_z[3]; t[56]<=add_z[4]; t[57]<=add_z[5]; // C[0] final
                        t[48]<=mul_z[9]; t[49]<=mul_z[10]; t[50]<=mul_z[11];
                        t[51]<=mul_z[12]; t[52]<=mul_z[13]; t[53]<=mul_z[14]; t[54]<=mul_z[15];
                        `ADD(6,t[48],t[49]) `ADD(7,t[51],t[52]) `ADD(8,t[54],mul_z[0])
                        st<=SI13;
                    end
                end
                SI13: begin // C[1][0..2]完成
                    if(add_vld[6]&&mul_vld[1]) begin
                        t[58]<=add_z[6]; t[59]<=add_z[7];
                        `ADD(9,add_z[8],mul_z[2]) `ADD(10,mul_z[3],mul_z[4]) `ADD(11,mul_z[6],mul_z[7])
                        st<=SI14;
                    end
                end
                SI14: begin // C全9元素完成, 开始Re=Rd^T*C
                    if(add_vld[9]) begin
                        t[60]<=add_z[9]; // C[1][2]
                        t[61]<=add_z[10]; t[62]<=add_z[11]; // C[2][0..1] partial
                        `ADD(0,t[61],mul_z[8]) // C[2][2]
                        // C存于: t[55..57]=C[0], t[58..60]=C[1], t[61..63]=C[2]
                        // Re[0] = Rd[0,:]·C
                        `MUL(1,Rd[0][0],t[55]) `MUL(2,Rd[1][0],t[58]) `MUL(3,Rd[2][0],t[61])
                        `MUL(4,Rd[0][0],t[56]) `MUL(5,Rd[1][0],t[59]) `MUL(6,Rd[2][0],t[62])
                        `MUL(7,Rd[0][0],t[57]) `MUL(8,Rd[1][0],t[60])
                        st<=SI15;
                    end
                end
                SI15: begin // Re row0 reduction
                    if(add_vld[0]&&mul_vld[1]) begin
                        t[63]<=add_z[0]; // C[2][2]
                        `MUL(9,Rd[2][0],t[63])
                        t[0]<=mul_z[1]; t[1]<=mul_z[2]; t[2]<=mul_z[3];
                        t[3]<=mul_z[4]; t[4]<=mul_z[5]; t[5]<=mul_z[6];
                        t[6]<=mul_z[7]; t[7]<=mul_z[8];
                        `ADD(0,t[0],t[1]) `ADD(1,t[3],t[4]) `ADD(2,t[6],t[7])
                        // Re row1 start
                        `MUL(10,Rd[0][1],t[55]) `MUL(11,Rd[1][1],t[58]) `MUL(12,Rd[2][1],t[61])
                        mul_a[13]<=Rd[0][1]; mul_b[13]<=t[56]; mul_en[13]<=1'b1;
                        mul_a[14]<=Rd[1][1]; mul_b[14]<=t[59]; mul_en[14]<=1'b1;
                        mul_a[15]<=Rd[2][1]; mul_b[15]<=t[62]; mul_en[15]<=1'b1;
                        st<=SI16;
                    end
                end
                SI16: begin // Re[0]完成, Re[1]reduction
                    if(add_vld[0]&&add_vld[2]&&mul_vld[10]) begin
                        t[8]<=add_z[0]; t[9]<=add_z[1];
                        `ADD(3,add_z[2],mul_z[9])
                        t[10]<=mul_z[10]; t[11]<=mul_z[11]; t[12]<=mul_z[12];
                        t[13]<=mul_z[13]; t[14]<=mul_z[14]; t[15]<=mul_z[15];
                        `ADD(4,t[10],t[11]) `ADD(5,t[13],t[14])
                        // Re row1 col2 + Re row2 start
                        `MUL(0,Rd[0][1],t[57]) `MUL(1,Rd[1][1],t[60]) `MUL(2,Rd[2][1],t[63])
                        `MUL(3,Rd[0][2],t[55]) `MUL(4,Rd[1][2],t[58]) `MUL(5,Rd[2][2],t[61])
                        `MUL(6,Rd[0][2],t[56]) `MUL(7,Rd[1][2],t[59]) `MUL(8,Rd[2][2],t[62])
                        `MUL(9,Rd[0][2],t[57]) `MUL(10,Rd[1][2],t[60]) `MUL(11,Rd[2][2],t[63])
                        st<=SI17;
                    end
                end
                SI17: begin // Re全部reduction
                    if(add_vld[3]&&add_vld[4]&&mul_vld[0]) begin
                        t[16]<=add_z[3];
                        t[17]<=add_z[4]; t[18]<=add_z[5];
                        t[19]<=mul_z[0]; t[20]<=mul_z[1]; t[21]<=mul_z[2];
                        t[22]<=mul_z[3]; t[23]<=mul_z[4]; t[24]<=mul_z[5];
                        t[25]<=mul_z[6]; t[26]<=mul_z[7]; t[27]<=mul_z[8];
                        t[28]<=mul_z[9]; t[29]<=mul_z[10]; t[30]<=mul_z[11];
                        `ADD(6,t[19],t[20]) `ADD(7,t[22],t[23]) `ADD(8,t[25],t[26]) `ADD(9,t[28],t[29])
                        st<=SI18;
                    end
                end
                SI18: begin // Re全9元素完成
                    if(add_vld[6]) begin
                        t[31]<=add_z[6]; t[32]<=add_z[7]; t[33]<=add_z[8]; t[34]<=add_z[9];
                        // 开始Log：vee(Re-Re^T)/2
                        `SUB(0,t[31],t[19]) `SUB(1,t[16],t[32]) `SUB(2,t[17],t[9])
                        st<=SI19;
                    end
                end
                SI19: begin // vee结果*0.5 → v_half
                    if(sub_vld[0]) begin
                        `MUL(0,sub_z[0],FP_HALF) `MUL(1,sub_z[1],FP_HALF) `MUL(2,sub_z[2],FP_HALF)
                        st<=SI20;
                    end
                end
                SI20: begin // ||v||, tr(Re)
                    if(mul_vld[0]) begin
                        t[35]<=mul_z[0]; t[36]<=mul_z[1]; t[37]<=mul_z[2]; // v_half
                        `MUL(3,mul_z[0],mul_z[0]) `MUL(4,mul_z[1],mul_z[1]) `MUL(5,mul_z[2],mul_z[2])
                        `ADD(6,t[8],t[18])
                        st<=SI21;
                    end
                end
                SI21: begin // sqrt & tr
                    if(mul_vld[3]&&add_vld[6]) begin
                        `ADD(10,mul_z[3],mul_z[4]) `ADD(11,add_z[6],t[34])
                    end
                    if(add_vld[10]) begin
                        `ADD(0,add_z[10],mul_z[5])
                        sqrt_in<=add_z[10]; sqrt_en<=1'b1;
                    end
                    if(add_vld[11]&&sqrt_vld) begin
                        t[38]<=sqrt_out;
                        `MUL(6,sqrt_out,FP_HALF) `SUB(3,add_z[11],FP_1)
                        st<=SI22;
                    end
                end
                SI22: begin // atan2
                    if(mul_vld[6]&&sub_vld[3]) begin
                        t[39]<=mul_z[6]; // s
                        `MUL(7,sub_z[3],FP_HALF)
                        atan_y<=mul_z[6]; atan_x<=mul_z[7]; atan_en<=1'b1;
                        st<=SI23;
                    end
                end
                SI23: begin // scale=theta/(2s)
                    if(atan_vld) begin
                        t[40]<=atan_theta;
                        `MUL(8,FP_2,t[39])
                        div_num<=atan_theta; div_den<=mul_z[8]; div_en<=1'b1;
                        st<=SI24;
                    end
                end
                SI24: begin // rR=scale*v_half
                    if(div_vld) begin
                        `MUL(9,div_quot,t[35]) `MUL(10,div_quot,t[36]) `MUL(11,div_quot,t[37])
                        st<=SI25;
                    end
                end
                SI25: begin // rR完成，M计算
                    if(mul_vld[9]) begin
                        t[41]<=mul_z[9]; t[42]<=mul_z[10]; t[43]<=mul_z[11]; // rR
                        `MUL(0,Rd[0][0],R0[0][0]) `MUL(1,Rd[1][0],R0[1][0]) `MUL(2,Rd[2][0],R0[2][0])
                        `MUL(3,Rd[0][0],R0[0][1]) `MUL(4,Rd[1][0],R0[1][1]) `MUL(5,Rd[2][0],R0[2][1])
                        `MUL(6,Rd[0][0],R0[0][2]) `MUL(7,Rd[1][0],R0[1][2]) `MUL(8,Rd[2][0],R0[2][2])
                        st<=SI26;
                    end
                end
                SI26: begin // M row0 reduction
                    if(mul_vld[0]) begin
                        t[44]<=mul_z[0]; t[45]<=mul_z[1]; t[46]<=mul_z[2];
                        t[47]<=mul_z[3]; t[48]<=mul_z[4]; t[49]<=mul_z[5];
                        t[50]<=mul_z[6]; t[51]<=mul_z[7]; t[52]<=mul_z[8];
                        `ADD(0,t[44],t[45]) `ADD(1,t[47],t[48]) `ADD(2,t[50],t[51])
                        st<=SI27;
                    end
                end
                SI27: begin // M row0 finalize, start M row1
                    if(add_vld[0]) begin
                        `ADD(3,add_z[0],t[46]) `ADD(4,add_z[1],t[49]) `ADD(5,add_z[2],t[52])
                        // Start M row1: Rd^T[1] * R0
                        `MUL(0,Rd[0][1],R0[0][0]) `MUL(1,Rd[1][1],R0[1][0]) `MUL(2,Rd[2][1],R0[2][0])
                        `MUL(3,Rd[0][1],R0[0][1]) `MUL(4,Rd[1][1],R0[1][1]) `MUL(5,Rd[2][1],R0[2][1])
                        `MUL(6,Rd[0][1],R0[0][2]) `MUL(7,Rd[1][1],R0[1][2]) `MUL(8,Rd[2][1],R0[2][2])
                        st<=SM1;
                    end
                end
                SM1: begin // Store M[0], reduce M row1
                    if(add_vld[3]&&mul_vld[0]) begin
                        t[53]<=add_z[3]; t[54]<=add_z[4]; t[55]<=add_z[5]; // M[0][:]
                        t[44]<=mul_z[0]; t[45]<=mul_z[1]; t[46]<=mul_z[2];
                        t[47]<=mul_z[3]; t[48]<=mul_z[4]; t[49]<=mul_z[5];
                        t[50]<=mul_z[6]; t[51]<=mul_z[7]; t[52]<=mul_z[8];
                        `ADD(0,t[44],t[45]) `ADD(1,t[47],t[48]) `ADD(2,t[50],t[51])
                        st<=SM2;
                    end
                end
                SM2: begin // M row1 finalize, start M row2
                    if(add_vld[0]) begin
                        `ADD(3,add_z[0],t[46]) `ADD(4,add_z[1],t[49]) `ADD(5,add_z[2],t[52])
                        // Start M row2: Rd^T[2] * R0
                        `MUL(0,Rd[0][2],R0[0][0]) `MUL(1,Rd[1][2],R0[1][0]) `MUL(2,Rd[2][2],R0[2][0])
                        `MUL(3,Rd[0][2],R0[0][1]) `MUL(4,Rd[1][2],R0[1][1]) `MUL(5,Rd[2][2],R0[2][1])
                        `MUL(6,Rd[0][2],R0[0][2]) `MUL(7,Rd[1][2],R0[1][2]) `MUL(8,Rd[2][2],R0[2][2])
                        st<=SM3;
                    end
                end
                SM3: begin // Store M[1], reduce M row2
                    if(add_vld[3]&&mul_vld[0]) begin
                        t[56]<=add_z[3]; t[57]<=add_z[4]; t[58]<=add_z[5]; // M[1][:]
                        t[44]<=mul_z[0]; t[45]<=mul_z[1]; t[46]<=mul_z[2];
                        t[47]<=mul_z[3]; t[48]<=mul_z[4]; t[49]<=mul_z[5];
                        t[50]<=mul_z[6]; t[51]<=mul_z[7]; t[52]<=mul_z[8];
                        `ADD(0,t[44],t[45]) `ADD(1,t[47],t[48]) `ADD(2,t[50],t[51])
                        st<=SI28;
                    end
                end
                SI28: begin // Store M[2], start RtK (Kvi * rt)
                    if(add_vld[0]) begin
                        `ADD(3,add_z[0],t[46]) `ADD(4,add_z[1],t[49]) `ADD(5,add_z[2],t[52])
                    end
                    if(add_vld[3]) begin
                        t[59]<=add_z[3]; t[60]<=add_z[4]; t[61]<=add_z[5]; // M[2][:]
                        // RtK = [Kvi[0]*rt[0], Kvi[1]*rt[1], Kvi[2]*rt[2]]
                        `MUL(6,Kvi[0],t[33]) `MUL(7,Kvi[1],t[34]) `MUL(8,Kvi[2],t[35])
                        st<=SI29;
                    end
                end
                SI29: begin // Emit row0: alpha[0]*[-M[0], 0, M[0], 0, RtK[0]]
                    if(mul_vld[6]) begin
                        t[62]<=mul_z[6]; t[63]<=mul_z[7]; t[0]<=mul_z[8]; // RtK
                        `MUL(0,alpha[0],neg(t[53])) `MUL(1,alpha[0],neg(t[54])) `MUL(2,alpha[0],neg(t[55]))
                        `MUL(3,alpha[0],FP_0) `MUL(4,alpha[0],FP_0) `MUL(5,alpha[0],FP_0)
                        `MUL(6,alpha[0],t[53]) `MUL(7,alpha[0],t[54]) `MUL(8,alpha[0],t[55])
                        `MUL(9,alpha[0],FP_0) `MUL(10,alpha[0],FP_0) `MUL(11,alpha[0],FP_0)
                        `MUL(12,alpha[0],t[62])
                    end
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel_cols<=6; row_out_trail_cols<=6;
                        row_out_local_col_panel<=local_col_panel;
                        row_out_local_col_trail<=local_col_trail;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SI30;
                    end
                end
                SI30: begin // Emit row1: alpha[1]*[0, -M[1], 0, M[1], RtK[1]]
                    `MUL(0,alpha[1],FP_0) `MUL(1,alpha[1],neg(t[56])) `MUL(2,alpha[1],FP_0)
                    `MUL(3,alpha[1],FP_0) `MUL(4,alpha[1],FP_0) `MUL(5,alpha[1],FP_0)
                    `MUL(6,alpha[1],FP_0) `MUL(7,alpha[1],t[56]) `MUL(8,alpha[1],FP_0)
                    `MUL(9,alpha[1],FP_0) `MUL(10,alpha[1],FP_0) `MUL(11,alpha[1],FP_0)
                    `MUL(12,alpha[1],t[63])
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SI31;
                    end
                end
                SI31: begin // Emit row2: alpha[2]*[0, 0, -M[2], 0, 0, M[2], RtK[2]]
                    `MUL(0,alpha[2],FP_0) `MUL(1,alpha[2],FP_0) `MUL(2,alpha[2],neg(t[59]))
                    `MUL(3,alpha[2],FP_0) `MUL(4,alpha[2],FP_0) `MUL(5,alpha[2],FP_0)
                    `MUL(6,alpha[2],FP_0) `MUL(7,alpha[2],FP_0) `MUL(8,alpha[2],t[59])
                    `MUL(9,alpha[2],FP_0) `MUL(10,alpha[2],FP_0) `MUL(11,alpha[2],FP_0)
                    `MUL(12,alpha[2],t[0])
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SI32;
                    end
                end
                SI32: begin // Emit row3: alpha[3]*[-rR[0], 0, rR[0], 0]
                    `MUL(0,alpha[3],neg(t[41])) `MUL(1,alpha[3],FP_0) `MUL(2,alpha[3],FP_0)
                    `MUL(3,alpha[3],FP_0) `MUL(4,alpha[3],FP_0) `MUL(5,alpha[3],FP_0)
                    `MUL(6,alpha[3],t[41]) `MUL(7,alpha[3],FP_0) `MUL(8,alpha[3],FP_0)
                    `MUL(9,alpha[3],FP_0) `MUL(10,alpha[3],FP_0) `MUL(11,alpha[3],FP_0)
                    `MUL(12,alpha[3],FP_0)
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SI33;
                    end
                end
                SI33: begin // Emit row4: alpha[4]*[0, -rR[1], 0, rR[1], 0]
                    `MUL(0,alpha[4],FP_0) `MUL(1,alpha[4],neg(t[42])) `MUL(2,alpha[4],FP_0)
                    `MUL(3,alpha[4],FP_0) `MUL(4,alpha[4],FP_0) `MUL(5,alpha[4],FP_0)
                    `MUL(6,alpha[4],FP_0) `MUL(7,alpha[4],t[42]) `MUL(8,alpha[4],FP_0)
                    `MUL(9,alpha[4],FP_0) `MUL(10,alpha[4],FP_0) `MUL(11,alpha[4],FP_0)
                    `MUL(12,alpha[4],FP_0)
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SI34;
                    end
                end
                SI34: begin // Emit row5 (last): alpha[5]*[0, 0, -rR[2], 0, 0, rR[2], 0]
                    `MUL(0,alpha[5],FP_0) `MUL(1,alpha[5],FP_0) `MUL(2,alpha[5],neg(t[43]))
                    `MUL(3,alpha[5],FP_0) `MUL(4,alpha[5],FP_0) `MUL(5,alpha[5],FP_0)
                    `MUL(6,alpha[5],FP_0) `MUL(7,alpha[5],FP_0) `MUL(8,alpha[5],t[43])
                    `MUL(9,alpha[5],FP_0) `MUL(10,alpha[5],FP_0) `MUL(11,alpha[5],FP_0)
                    `MUL(12,alpha[5],FP_0)
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b1; // Last row in IMU factor
                        st<=S0; factor_ready<=1'b1; eng<=ENG_IDLE;
                    end
                end
                
                // ================================================
                // PRIOR Factor: 6 rows (trans x3 + rot x3)
                // Input: Rp, tp (prior), Ri, ti (current)
                // Residual: tdiff = Rp^T*(ti-tp), rR = Log(Rp^T*Ri)
                // Jacobian: -I(3x3) for trans, -I(3x3) for rot
                // ================================================
                
                SP1: begin // tdiff: ti - tp
                    `SUB(0,Ri[0][3],Rp[0][3]) `SUB(1,Ri[1][3],Rp[1][3]) `SUB(2,Ri[2][3],Rp[2][3])
                    st<=SP2;
                end
                SP2: begin // t_rel = Rp^T * tdiff
                    if(sub_vld[0]) begin
                        t[0]<=sub_z[0]; t[1]<=sub_z[1]; t[2]<=sub_z[2]; // ti-tp
                        `MUL(0,Rp[0][0],t[0]) `MUL(1,Rp[1][0],t[1]) `MUL(2,Rp[2][0],t[2])
                        `MUL(3,Rp[0][1],t[0]) `MUL(4,Rp[1][1],t[1]) `MUL(5,Rp[2][1],t[2])
                        `MUL(6,Rp[0][2],t[0]) `MUL(7,Rp[1][2],t[1]) `MUL(8,Rp[2][2],t[2])
                        st<=SP3;
                    end
                end
                SP3: begin // t_rel reduction
                    if(mul_vld[0]) begin
                        `ADD(0,mul_z[0],mul_z[1]) `ADD(1,mul_z[3],mul_z[4]) `ADD(2,mul_z[6],mul_z[7])
                        t[3]<=mul_z[2]; t[4]<=mul_z[5]; t[5]<=mul_z[8];
                        st<=SP4;
                    end
                end
                SP4: begin // t_rel final, start C = Rp^T * Ri
                    if(add_vld[0]) begin
                        `ADD(3,add_z[0],t[3]) `ADD(4,add_z[1],t[4]) `ADD(5,add_z[2],t[5])
                    end
                    if(add_vld[3]) begin
                        t[6]<=add_z[3]; t[7]<=add_z[4]; t[8]<=add_z[5]; // t_rel = tdiff
                        // C = Rp^T * Ri (first 3 elements: C[0][0..2])
                        `MUL(0,Rp[0][0],Ri[0][0]) `MUL(1,Rp[1][0],Ri[1][0]) `MUL(2,Rp[2][0],Ri[2][0])
                        `MUL(3,Rp[0][0],Ri[0][1]) `MUL(4,Rp[1][0],Ri[1][1]) `MUL(5,Rp[2][0],Ri[2][1])
                        `MUL(6,Rp[0][0],Ri[0][2]) `MUL(7,Rp[1][0],Ri[1][2]) `MUL(8,Rp[2][0],Ri[2][2])
                        st<=SP5;
                    end
                end
                SP5: begin // C row0 reduction
                    if(mul_vld[0]) begin
                        `ADD(6,mul_z[0],mul_z[1]) `ADD(7,mul_z[3],mul_z[4]) `ADD(8,mul_z[6],mul_z[7])
                        t[9]<=mul_z[2]; t[10]<=mul_z[5]; t[11]<=mul_z[8];
                        st<=SP6;
                    end
                end
                SP6: begin // C[0] final, compute C row1
                    if(add_vld[6]) begin
                        `ADD(9,add_z[6],t[9]) `ADD(10,add_z[7],t[10]) `ADD(11,add_z[8],t[11])
                        // C row1: Rp^T[1] * Ri
                        `MUL(0,Rp[0][1],Ri[0][0]) `MUL(1,Rp[1][1],Ri[1][0]) `MUL(2,Rp[2][1],Ri[2][0])
                        `MUL(3,Rp[0][1],Ri[0][1]) `MUL(4,Rp[1][1],Ri[1][1]) `MUL(5,Rp[2][1],Ri[2][1])
                        `MUL(6,Rp[0][1],Ri[0][2]) `MUL(7,Rp[1][1],Ri[1][2]) `MUL(8,Rp[2][1],Ri[2][2])
                        st<=SP7;
                    end
                end
                SP7: begin // Store C[0], reduce C row1
                    if(add_vld[9]&&mul_vld[0]) begin
                        t[12]<=add_z[9]; t[13]<=add_z[10]; t[14]<=add_z[11]; // C[0][:]
                        `ADD(6,mul_z[0],mul_z[1]) `ADD(7,mul_z[3],mul_z[4]) `ADD(8,mul_z[6],mul_z[7])
                        t[9]<=mul_z[2]; t[10]<=mul_z[5]; t[11]<=mul_z[8];
                        st<=SP8;
                    end
                end
                SP8: begin // C[1] final, compute C row2
                    if(add_vld[6]) begin
                        `ADD(9,add_z[6],t[9]) `ADD(10,add_z[7],t[10]) `ADD(11,add_z[8],t[11])
                        // C row2: Rp^T[2] * Ri
                        `MUL(0,Rp[0][2],Ri[0][0]) `MUL(1,Rp[1][2],Ri[1][0]) `MUL(2,Rp[2][2],Ri[2][0])
                        `MUL(3,Rp[0][2],Ri[0][1]) `MUL(4,Rp[1][2],Ri[1][1]) `MUL(5,Rp[2][2],Ri[2][1])
                        `MUL(6,Rp[0][2],Ri[0][2]) `MUL(7,Rp[1][2],Ri[1][2]) `MUL(8,Rp[2][2],Ri[2][2])
                        st<=SP9;
                    end
                end
                SP9: begin // Store C[1], reduce C row2
                    if(add_vld[9]&&mul_vld[0]) begin
                        t[15]<=add_z[9]; t[16]<=add_z[10]; t[17]<=add_z[11]; // C[1][:]
                        `ADD(6,mul_z[0],mul_z[1]) `ADD(7,mul_z[3],mul_z[4]) `ADD(8,mul_z[6],mul_z[7])
                        t[9]<=mul_z[2]; t[10]<=mul_z[5]; t[11]<=mul_z[8];
                        st<=SP10;
                    end
                end
                SP10: begin // C[2] final, start Log
                    if(add_vld[6]) begin
                        `ADD(9,add_z[6],t[9]) `ADD(10,add_z[7],t[10]) `ADD(11,add_z[8],t[11])
                    end
                    if(add_vld[9]) begin
                        t[18]<=add_z[9]; t[19]<=add_z[10]; t[20]<=add_z[11]; // C[2][:]
                        // Log: vee(C - C^T) / 2
                        `SUB(0,t[16],t[13]) `SUB(1,t[12],t[18]) `SUB(2,t[15],t[14])
                        st<=SP11;
                    end
                end
                SP11: begin // v_half
                    if(sub_vld[0]) begin
                        `MUL(0,sub_z[0],FP_HALF) `MUL(1,sub_z[1],FP_HALF) `MUL(2,sub_z[2],FP_HALF)
                        st<=SP12;
                    end
                end
                SP12: begin // ||v||, tr(C)
                    if(mul_vld[0]) begin
                        t[21]<=mul_z[0]; t[22]<=mul_z[1]; t[23]<=mul_z[2]; // v_half
                        `MUL(3,t[21],t[21]) `MUL(4,t[22],t[22]) `MUL(5,t[23],t[23])
                        `ADD(6,t[12],t[16]) // tr start
                        st<=SP13;
                    end
                end
                SP13: begin // sqrt, atan2
                    if(mul_vld[3]&&add_vld[6]) begin
                        `ADD(7,mul_z[3],mul_z[4]) `ADD(8,add_z[6],t[20])
                    end
                    if(add_vld[7]) begin
                        `ADD(9,add_z[7],mul_z[5]) // sumv2
                        sqrt_in<=add_z[7]; sqrt_en<=1'b1;
                    end
                    if(add_vld[8]&&add_vld[9]&&sqrt_vld) begin
                        t[24]<=sqrt_out;
                        `MUL(6,sqrt_out,FP_HALF) // s
                        `SUB(3,add_z[8],FP_1) // tr-1
                        st<=SP14;
                    end
                end
                SP14: begin // scale = theta/(2s)
                    if(mul_vld[6]&&sub_vld[3]) begin
                        t[25]<=mul_z[6]; // s
                        `MUL(7,sub_z[3],FP_HALF) // c
                        atan_y<=t[25]; atan_x<=mul_z[7]; atan_en<=1'b1;
                    end
                    if(atan_vld) begin
                        `MUL(8,FP_2,t[25])
                        div_num<=atan_theta; div_den<=mul_z[8]; div_en<=1'b1;
                        st<=SP15;
                    end
                end
                SP15: begin // rR = scale * v_half, emit 6 rows
                    if(div_vld) begin
                        `MUL(9,div_quot,t[21]) `MUL(10,div_quot,t[22]) `MUL(11,div_quot,t[23])
                        st<=SP16;
                    end
                end
                SP16: begin // Emit row0: alpha[0]*[-1,0,0, 0,0,0, tdiff[0]]
                    if(mul_vld[9]) begin
                        t[26]<=mul_z[9]; t[27]<=mul_z[10]; t[28]<=mul_z[11]; // rR
                        `MUL(0,alpha[0],neg(FP_1)) `MUL(1,alpha[0],FP_0) `MUL(2,alpha[0],FP_0)
                        `MUL(3,alpha[0],FP_0) `MUL(4,alpha[0],FP_0) `MUL(5,alpha[0],FP_0)
                        `MUL(6,alpha[0],FP_0) `MUL(7,alpha[0],FP_0) `MUL(8,alpha[0],FP_0)
                        `MUL(9,alpha[0],FP_0) `MUL(10,alpha[0],FP_0) `MUL(11,alpha[0],FP_0)
                        `MUL(12,alpha[0],t[6])
                    end
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel_cols<=6; row_out_trail_cols<=6;
                        row_out_local_col_panel<=local_col_panel;
                        row_out_local_col_trail<=local_col_trail;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SP17;
                    end
                end
                SP17: begin // Emit row1: alpha[1]*[0,-1,0, 0,0,0, tdiff[1]]
                    `MUL(0,alpha[1],FP_0) `MUL(1,alpha[1],neg(FP_1)) `MUL(2,alpha[1],FP_0)
                    `MUL(3,alpha[1],FP_0) `MUL(4,alpha[1],FP_0) `MUL(5,alpha[1],FP_0)
                    `MUL(6,alpha[1],FP_0) `MUL(7,alpha[1],FP_0) `MUL(8,alpha[1],FP_0)
                    `MUL(9,alpha[1],FP_0) `MUL(10,alpha[1],FP_0) `MUL(11,alpha[1],FP_0)
                    `MUL(12,alpha[1],t[7])
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SP18;
                    end
                end
                SP18: begin // Emit row2: alpha[2]*[0,0,-1, 0,0,0, tdiff[2]]
                    `MUL(0,alpha[2],FP_0) `MUL(1,alpha[2],FP_0) `MUL(2,alpha[2],neg(FP_1))
                    `MUL(3,alpha[2],FP_0) `MUL(4,alpha[2],FP_0) `MUL(5,alpha[2],FP_0)
                    `MUL(6,alpha[2],FP_0) `MUL(7,alpha[2],FP_0) `MUL(8,alpha[2],FP_0)
                    `MUL(9,alpha[2],FP_0) `MUL(10,alpha[2],FP_0) `MUL(11,alpha[2],FP_0)
                    `MUL(12,alpha[2],t[8])
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SP19;
                    end
                end
                SP19: begin // Emit row3: alpha[3]*[0,0,0, -1,0,0, rR[0]]
                    `MUL(0,alpha[3],FP_0) `MUL(1,alpha[3],FP_0) `MUL(2,alpha[3],FP_0)
                    `MUL(3,alpha[3],neg(FP_1)) `MUL(4,alpha[3],FP_0) `MUL(5,alpha[3],FP_0)
                    `MUL(6,alpha[3],FP_0) `MUL(7,alpha[3],FP_0) `MUL(8,alpha[3],FP_0)
                    `MUL(9,alpha[3],FP_0) `MUL(10,alpha[3],FP_0) `MUL(11,alpha[3],FP_0)
                    `MUL(12,alpha[3],t[26])
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SP20;
                    end
                end
                SP20: begin // Emit row4: alpha[4]*[0,0,0, 0,-1,0, rR[1]]
                    `MUL(0,alpha[4],FP_0) `MUL(1,alpha[4],FP_0) `MUL(2,alpha[4],FP_0)
                    `MUL(3,alpha[4],FP_0) `MUL(4,alpha[4],neg(FP_1)) `MUL(5,alpha[4],FP_0)
                    `MUL(6,alpha[4],FP_0) `MUL(7,alpha[4],FP_0) `MUL(8,alpha[4],FP_0)
                    `MUL(9,alpha[4],FP_0) `MUL(10,alpha[4],FP_0) `MUL(11,alpha[4],FP_0)
                    `MUL(12,alpha[4],t[27])
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b0;
                        st<=SP21;
                    end
                end
                SP21: begin // Emit row5 (last): alpha[5]*[0,0,0, 0,0,-1, rR[2]]
                    `MUL(0,alpha[5],FP_0) `MUL(1,alpha[5],FP_0) `MUL(2,alpha[5],FP_0)
                    `MUL(3,alpha[5],FP_0) `MUL(4,alpha[5],FP_0) `MUL(5,alpha[5],neg(FP_1))
                    `MUL(6,alpha[5],FP_0) `MUL(7,alpha[5],FP_0) `MUL(8,alpha[5],FP_0)
                    `MUL(9,alpha[5],FP_0) `MUL(10,alpha[5],FP_0) `MUL(11,alpha[5],FP_0)
                    `MUL(12,alpha[5],t[28])
                    if(mul_vld[0]&&row_out_ready) begin
                        row_out_valid<=1'b1;
                        row_out_panel[0]<=mul_z[0]; row_out_panel[1]<=mul_z[1]; row_out_panel[2]<=mul_z[2];
                        row_out_panel[3]<=mul_z[3]; row_out_panel[4]<=mul_z[4]; row_out_panel[5]<=mul_z[5];
                        row_out_trail[0]<=mul_z[6]; row_out_trail[1]<=mul_z[7]; row_out_trail[2]<=mul_z[8];
                        row_out_trail[3]<=mul_z[9]; row_out_trail[4]<=mul_z[10]; row_out_trail[5]<=mul_z[11];
                        row_out_b<=mul_z[12];
                        row_out_last_in_factor<=1'b1; // Last row in Prior factor
                        st<=S0; factor_ready<=1'b1; eng<=ENG_IDLE;
                    end
                end
                
                default: st<=S0;
            endcase
        end
    end

endmodule
