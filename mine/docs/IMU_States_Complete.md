# IMU 因子完整状态实现 (SI1-SI28)

## 状态SI1-SI11 (已实现)
✅ 计算 vi, rt 完成

## 状态SI12-SI14: 完成C矩阵剩余元素

```systemverilog
SI12: begin // C row1&2 reduction
    if(add_vld[3]&&mul_vld[0]) begin
        t[55]<=add_z[3]; t[56]<=add_z[4]; t[57]<=add_z[5]; // C[0] final
        t[58]<=mul_z[0]; t[59]<=mul_z[1]; t[60]<=mul_z[2];
        t[61]<=mul_z[3]; t[62]<=mul_z[4]; t[63]<=mul_z[5];
        // C row2 start (复用t[0..8])
        t[0]<=mul_z[6]; t[1]<=mul_z[7]; t[2]<=mul_z[8];
        add_a[6]<=t[58]; add_b[6]<=t[59]; add_en[6]<=1'b1;
        add_a[7]<=t[61]; add_b[7]<=t[62]; add_en[7]<=1'b1;
        add_a[8]<=t[0]; add_b[8]<=t[1]; add_en[8]<=1'b1;
        st<=SI13;
    end
end
SI13: begin
    if(add_vld[6]) begin
        t[3]<=add_z[6]; t[4]<=add_z[7]; t[5]<=add_z[8]; // C[1][0..2]
        add_a[6]<=t[3]; add_b[6]<=t[60]; add_en[6]<=1'b1;
        add_a[7]<=t[4]; add_b[7]<=t[63]; add_en[7]<=1'b1;
        add_a[8]<=t[5]; add_b[8]<=t[2]; add_en[8]<=1'b1;
        st<=SI14;
    end
end
SI14: begin // C矩阵9元素全部完成，存为t[55..63]
    if(add_vld[6]) begin
        t[6]<=add_z[6]; t[7]<=add_z[7]; t[8]<=add_z[8]; // C[2][0..2]
        // 现在 C 存储：
        // C[0][0..2] = t[55,56,57]
        // C[1][0..2] = t[48,51,54] (from row1)
        // C[2][0..2] = t[6,7,8]
        st<=SI15; // 开始Re=Rd^T*C
    end
end
```

## 状态SI15-SI19: 计算Re=Rd^T*C (27 mul + 18 add)

```systemverilog
SI15: begin // Re row0 = Rd[0,:]*C[:,0..2] (9 mul)
    mul_a[0]<=Rd[0][0]; mul_b[0]<=t[55]; mul_en[0]<=1'b1; // Rd[0][0]*C[0][0]
    mul_a[1]<=Rd[1][0]; mul_b[1]<=t[48]; mul_en[1]<=1'b1; // Rd[1][0]*C[1][0]
    mul_a[2]<=Rd[2][0]; mul_b[2]<=t[6]; mul_en[2]<=1'b1;  // Rd[2][0]*C[2][0]
    mul_a[3]<=Rd[0][0]; mul_b[3]<=t[56]; mul_en[3]<=1'b1;
    mul_a[4]<=Rd[1][0]; mul_b[4]<=t[51]; mul_en[4]<=1'b1;
    mul_a[5]<=Rd[2][0]; mul_b[5]<=t[7]; mul_en[5]<=1'b1;
    mul_a[6]<=Rd[0][0]; mul_b[6]<=t[57]; mul_en[6]<=1'b1;
    mul_a[7]<=Rd[1][0]; mul_b[7]<=t[54]; mul_en[7]<=1'b1;
    mul_a[8]<=Rd[2][0]; mul_b[8]<=t[8]; mul_en[8]<=1'b1;
    st<=SI16;
end
SI16: begin // Re[0] reduction
    if(mul_vld[0]) begin
        t[9]<=mul_z[0]; t[10]<=mul_z[1]; t[11]<=mul_z[2];
        t[12]<=mul_z[3]; t[13]<=mul_z[4]; t[14]<=mul_z[5];
        t[15]<=mul_z[6]; t[16]<=mul_z[7]; t[17]<=mul_z[8];
        add_a[0]<=t[9]; add_b[0]<=t[10]; add_en[0]<=1'b1;
        add_a[1]<=t[12]; add_b[1]<=t[13]; add_en[1]<=1'b1;
        add_a[2]<=t[15]; add_b[2]<=t[16]; add_en[2]<=1'b1;
        st<=SI17;
    end
end
SI17: begin
    if(add_vld[0]) begin
        t[18]<=add_z[0]; t[19]<=add_z[1]; t[20]<=add_z[2];
        add_a[0]<=t[18]; add_b[0]<=t[11]; add_en[0]<=1'b1;
        add_a[1]<=t[19]; add_b[1]<=t[14]; add_en[1]<=1'b1;
        add_a[2]<=t[20]; add_b[2]<=t[17]; add_en[2]<=1'b1;
        // 开始Re row1 (similar pattern)
        st<=SI18;
    end
end
SI18,SI19: begin // Re row1&2 (类似SI15-17，约10行)
    st<=SI20;
end
```

## 状态SI20-SI23: CORDIC Log(Re)

```systemverilog
SI20: begin // vee(Re - Re^T)/2
    // v[0] = (Re[2][1] - Re[1][2])/2
    sub_a[0]<=t[Re21]; sub_b[0]<=t[Re12]; sub_en[0]<=1'b1;
    sub_a[1]<=t[Re02]; sub_b[1]<=t[Re20]; sub_en[1]<=1'b1;
    sub_a[2]<=t[Re10]; sub_b[2]<=t[Re01]; sub_en[2]<=1'b1;
    st<=SI21;
end
SI21: begin // ||v|| and tr(Re)
    if(sub_vld[0]) begin
        t[30]<=sub_z[0]; t[31]<=sub_z[1]; t[32]<=sub_z[2];
        mul_a[0]<=t[30]; mul_b[0]<=t[30]; mul_en[0]<=1'b1; // v[0]^2
        mul_a[1]<=t[31]; mul_b[1]<=t[31]; mul_en[1]<=1'b1;
        mul_a[2]<=t[32]; mul_b[2]<=t[32]; mul_en[2]<=1'b1;
        mul_a[3]<=t[30]; mul_b[3]<=FP_HALF; mul_en[3]<=1'b1; // v[0]/2
        mul_a[4]<=t[31]; mul_b[4]<=FP_HALF; mul_en[4]<=1'b1;
        mul_a[5]<=t[32]; mul_b[5]<=FP_HALF; mul_en[5]<=1'b1;
        add_a[6]<=t[Re00]; add_b[6]<=t[Re11]; add_en[6]<=1'b1; // tr start
        st<=SI22;
    end
end
SI22: begin // sqrt(v^2), atan2 prep
    if(mul_vld[0]&&add_vld[6]) begin
        t[33]<=mul_z[3]; t[34]<=mul_z[4]; t[35]<=mul_z[5]; // v_half
        add_a[0]<=mul_z[0]; add_b[0]<=mul_z[1]; add_en[0]<=1'b1; // sumv2
        add_a[7]<=add_z[6]; add_b[7]<=t[Re22]; add_en[7]<=1'b1; // tr
    end
    if(add_vld[0]) begin
        add_a[1]<=add_z[0]; add_b[1]<=mul_z[2]; add_en[1]<=1'b1;
        sqrt_in<=add_z[0]; sqrt_en<=1'b1; // sqrt(sumv2)
    end
    if(add_vld[7]&&sqrt_vld) begin
        t[36]<=sqrt_out; // ||v||
        mul_a[9]<=sqrt_out; mul_b[9]<=FP_HALF; mul_en[9]<=1'b1; // s=0.5*||v||
        sub_a[3]<=add_z[7]; sub_b[3]<=FP_1; sub_en[3]<=1'b1; // tr-1
        st<=SI23;
    end
end
SI23: begin // atan2(s,c) → theta, scale
    if(mul_vld[9]&&sub_vld[3]) begin
        t[37]<=mul_z[9]; // s
        mul_a[10]<=sub_z[3]; mul_b[10]<=FP_HALF; mul_en[10]<=1'b1; // c=(tr-1)/2
        atan_y<=t[37]; atan_x<=mul_z[10]; atan_en<=1'b1;
    end
    if(atan_vld) begin
        t[38]<=atan_theta; // theta
        mul_a[11]<=FP_2; mul_b[11]<=t[37]; mul_en[11]<=1'b1; // 2*s
        div_num<=t[38]; div_den<=mul_z[11]; div_en<=1'b1; // theta/(2s)
    end
    if(div_vld) begin
        t[39]<=div_quot; // scale
        mul_a[12]<=div_quot; mul_b[12]<=t[33]; mul_en[12]<=1'b1; // scale*v[0]
        mul_a[13]<=div_quot; mul_b[13]<=t[34]; mul_en[13]<=1'b1;
        mul_a[14]<=div_quot; mul_b[14]<=t[35]; mul_en[14]<=1'b1;
        st<=SI24;
    end
end
```

## 状态SI24-SI26: 计算M和RtK

```systemverilog
SI24: begin // rR完成, 计算M=Rd^T*Ri^T (27 mul, similar to Re)
    if(mul_vld[12]) begin
        t[40]<=mul_z[12]; t[41]<=mul_z[13]; t[42]<=mul_z[14]; // rR[0..2]
        // M计算：类似Re，用Rd^T*R0
        mul_a[0]<=Rd[0][0]; mul_b[0]<=R0[0][0]; mul_en[0]<=1'b1;
        // ... (27 mul for M)
        st<=SI25;
    end
end
SI25: begin // Kvi=skew(vi) 和 RtK=Rd^T*Kvi start
    skew_x<=t[15]; skew_y<=t[16]; skew_z<=t[17]; // use stored vi
    // Wait 1 cycle for skew_K
    st<=SI26;
end
SI26: begin // RtK computation (27 mul)
    // Rd^T * Kvi
    mul_a[0]<=Rd[0][0]; mul_b[0]<=skew_K[0][0]; mul_en[0]<=1'b1;
    // ... (27 mul)
    st<=SI27;
end
```

## 状态SI27: 行白化 (6行×12列=72个元素，需分批白化)

```systemverilog
SI27: begin // Whiten batch1: row0 panel (pose_i)
    mul_a[0]<=alpha[0]; mul_b[0]<=t[M00]; mul_en[0]<=1'b1; // -M[0][0]的负
    mul_a[1]<=alpha[0]; mul_b[1]<=t[M01]; mul_en[1]<=1'b1;
    // ... 白化6个panel元素
    st<=SI28;
end
```

## 状态SI28及后续: 逐行输出 (需6个状态或用计数器)

```systemverilog
SI28: begin // Emit row0
    if(mul_vld[0]&&row_out_ready) begin
        row_out_valid<=1'b1;
        row_out_panel_cols<=6; row_out_trail_cols<=6;
        row_out_local_col_panel<=local_col_panel;
        row_out_local_col_trail<=local_col_trail;
        row_out_panel[0]<=mul_z[0]; // ... panel 6 elements
        row_out_trail[0]<=mul_z[6]; // ... trail 6 elements
        row_out_b<=mul_z[12];
        row_out_last_in_factor<=(emit_cnt==5);
        // 继续row1..row5 (需要状态机或计数器)
    end
end
```

---

## 完整的Verilog代码生成

由于状态太多，建议用**参数化状态生成**：

```systemverilog
// 用reg计数器代替多状态
reg [2:0] imu_phase; // 0=rt, 1=C, 2=Re, 3=Log, 4=M, 5=RtK, 6=emit
reg [3:0] imu_step;  // 子步骤
reg [2:0] emit_row_idx; // 输出行计数

// 在SI12进入sub-FSM
SI12: begin
    case(imu_phase)
        3'd1: // C matrix phase
            case(imu_step)
                4'd0: // C row1&2 (已在SI10-SI11开始)
                4'd1: // ...
            endcase
        3'd2: // Re phase
        3'd3: // Log phase
        3'd4: // M phase
        3'd5: // RtK phase
        3'd6: // Emit phase
            if(row_out_ready) begin
                // 输出当前行
                emit_row_idx <= emit_row_idx + 1;
                if(emit_row_idx==5) begin
                    st<=S0; factor_ready<=1'b1;
                end
            end
    endcase
end
```

---

## 寄存器优化建议

**面积critical时**：
- t[0..63]足够，通过分阶段复用（C算完后t[55..63]存C；Re算完后覆盖存Re；Log只需t[40..42]）
- jac数组可省略：在白化时直接从t取并流式输出
- 当前设计已优化为单tr数组

**性能critical时**：
- 可增加中间寄存避免等待（如C/Re并行计算）
- 但会增加约20个寄存器

---

## 建议的补全方式

### 方式1: 手动逐状态补（最精确）
按本文档SI12-SI28每个状态写case分支（约300行Verilog）

### 方式2: 用子FSM+计数器（推荐，节省状态数）
用phase/step两级计数器，减少状态爆炸，代码约150行

### 方式3: 我继续分批patch补完（约需5-10次patch）
每次补5-6个状态，确保每个细节正确

---

**你希望哪种方式？** 我可以：
A. 继续分批patch把所有状态补完（需10次左右patch）
B. 给你子FSM+计数器优化版（代码更紧凑，约800行总长）
C. 提供状态生成Python脚本，一键生成剩余Verilog代码

