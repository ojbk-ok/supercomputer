# Deepseek Inference
**指 DeepSeek 的 AI 模型（如 DeepSeek-V2）在实际应用中对用户输入进行预测、生成回答的过程。**
##  DeepSeek AI 模型的技术架构设计
### Key Concepts  
1. **Pre-Training Scaling**  
   - 模型预训练阶段的扩展（数据量、算力、参数规模）。  

2. **Post-Training Scaling**  
   - 训练后的优化（微调、对齐、RLHF）。  

3. **Test-Time Scaling ("Long Thinking")**  
   - 推理时通过长上下文/迭代思考提升性能（如128K tokens、多步推理）。  

4. **"INTELLIGENCE"**  
   - 通过三类扩展定律实现更高阶智能（AGI）。  

### DeepSeek V3/R1 模型架构概览

#### DeepSeek-R1 主干结构
- **基础Transformer块**：第1-3层（标准结构）
- **MoE Transformer块**：第4-61层（混合专家模式）

---

#### 标准Transformer块组成
```mermaid
graph LR
    A[输入] --> B[注意力层]
    B --> C[MASNorm]
    C --> D[前馈网络]
    D --> E[MASNorm]
    E --> F[输出]
```

---

### DeepSeekMoE 专家模块
#### 核心组件
- **根专家系统**：
  - 共享根专家(Rooted Expert Shared Expert)
  - 根标记器(Rooted Marker)

#### 隐藏层通道
| 类型       | 数量 | 示例编号              | 备注                     |
|------------|------|-----------------------|--------------------------|
| 输入隐藏层 | 491  | N1-N491               | 存在重复编号（如N478两次）|
| 输出隐藏层 | 462  | N1-N462               | 比输入层少29个通道       |

#### 关键设计特征
1. **非对称压缩**：
   - 输入通道(491) > 输出通道(462)
   - 可能实现特征选择/降维

2. **特殊编号模式**：
   - 断续编号（如N491→N494）
   - 部分专家可能有专用通道

3. **MASNorm应用**：
   - 每个Transformer块内部分别进行两次标准化


### 核心参数配置
- **分词器(Tokenizer)**  
  - 词表大小(vocab size): ~130,000
- **模型维度(Model dimension):** 7,168  
- **帧维度(Frame dimensions):** 18,432  



### 关键组件说明
#### 1. Transformer块结构
- **标准块**: Block 1  
- **MoE块**: Block 4/6（混合专家架构）  
  - 每token激活专家数: 8  
  - 总专家数: 205  

#### 2. 注意力机制
- **多头注意力(Multi-Head Attention)**  
  - 注意力头数: 128 heads  
  - 归一化方法: RMSSform（创新优化层）  

#### 3. 推理过程特征
- **输入隐藏状态**  
  - 动态缓存标记: `Input Hidden in [ ] Gxhed`  
- **输出隐藏状态**  
  - 动态缓存标记: `Output Hidden in [ ] Gxhed`  

---

### 架构创新点
1. **延续性设计**  
   - 继承DeepSeek-V2的MLA（高效注意力）和DeepSeekMoE架构  
2. **经济性优化**  
   - 通过专家混合系统降低训练成本  
3. **高性能推理**  
   - 大模型维度(7k+)配合128头注意力实现长程建模  

### 参数对比表
| 组件                | V2版本特性       | V3/R1改进点         |
|---------------------|-----------------|---------------------|
| 模型维度            | 5,120           | **7,168**           |
| 注意力头数          | 96              | **128**             |
| MoE专家激活策略     | 固定4专家       | **动态8专家**       |
### Multi-head Latent Attention (MLA)

#### 1. 不同注意力机制的KV缓存对比
| 注意力机制                | KV缓存大小                  | 关键特征                                                                 |
|---------------------------|-----------------------------|--------------------------------------------------------------------------|
| Multi-Head Attention (MHA) | $2n_h d_n l$                | 每个头独立维护KV，缓存开销随头数线性增长。                               |
| Grouped-Query Attention (GQA) | $2n_g d_n l$              | 分组查询，每组共享KV，降低缓存但保留多组灵活性。                         |
| Multi-Query Attention (MQA) | $2d_n l$                   | 所有头共享同一KV，缓存最小但可能损失表达能力。                           |
| Multi-Head Latent Attention (MLA) | $(d_c + d_R^R) l = 4.5d_n l$ | 引入压缩潜在KV，通过投影平衡缓存与性能，$d_c=4d_n, d_R^R=0.5d_n$。        |

#### 2. MLA核心公式
- **潜在特征投影**：
  $$ c_t^{KV} = W^{DKV} h_t, \quad [k_{1,i}^C;...;k_{n_h,i}^C] = k_t^C = W^{UK} c_t^{KV} $$
  $$ k_{i,l}^R = \text{RoPE}(W^{KR} h_t), \quad k_{i,l} = [k_{i,l}^C; k_{i,l}^R] $$
- **注意力计算**：
  $$ o_{t,i} = \sum_{j=1}^t \text{Softmax}\left(\frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d_R^R}}\right) v_{j,i}^C $$
- **输出隐藏层**：
  $$ u_t = W^O [o_{t,1}; o_{t,2}; ...; o_{t,n_h}] $$

#### 3. 符号说明
| 符号       | 含义                          |
|------------|-------------------------------|
| $W^{DQ}, W^{UQ}$ | 查询投影权重                  |
| $W^{KR}, W^{UK}$ | 键投影与潜在特征权重          |
| $d_c, d_R^R$ | 潜在特征与旋转编码维度        |
| $n_h$      | 注意力头数                    |
| $l$        | 层数                          |

#### 4. 架构亮点
- **缓存优化**：通过潜在特征压缩KV缓存至$4.5d_n l$，较MHA减少60%以上。
- **RoPE集成**：旋转位置编码增强长序列建模能力。
- **分层投影**：$W^D$下投影降维，$W^U$上投影恢复，平衡计算与表达。
这是关于**Multi-head Latent Attention (MLA)** 的矩阵吸收优化（Mat Absorb）内容，核心是通过合并权重矩阵减少计算量，属于大语言模型（LLM）效率优化技术。

### 1. 核心公式
- **查询与键的潜在特征**：
  $$ c_t^Q = W^{DQ} h_t, \quad [q_{1,i}^C;...;q_{n_h,i}^C] = q_t^C = W^{UQ} c_t^Q $$
  $$ c_t^{KV} = W^{DKV} h_t, \quad [k_{1,i}^C;...;k_{n_h,i}^C] = k_t^C = W^{UK} c_t^{KV} $$
- **旋转编码（RoPE）**：
  $$ [q_{1,i}^R;...;q_{n_h,i}^R] = q_t^R = \text{RoPE}(W^{QR} c_t^Q) $$
  $$ k_i^R = \text{RoPE}(W^{KR} h_t), \quad k_{i,l} = [k_{i,l}^C; k_{i,l}^R] $$
- **注意力与输出**：
  $$ o_{t,i} = \sum_{j=1}^t \text{Softmax}\left(\frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d_R^R}}\right) v_{j,i}^C $$
  $$ u_t = W^O [o_{t,1}; o_{t,2}; ...; o_{t,n_h}] $$

### 2. 矩阵吸收优化
- **Key Mat Absorb**：合并 $W^{UQ}$ 与 $W^{UK}$ 到注意力权重计算中：
  $$ \text{attentionweight} = q'^T k' = (c_t^Q W^{UQ})^T W^{UK} c_t^{KV} = c_t^Q (W^{UQ^T} W^{UK}) c_t^{KV} $$
- **Value Mat Absorb**：合并 $W^{UV}$ 与 $W^O$ 到输出计算中：
  $$ \text{finaloutput} = A_{\text{weight}} (W^{UV} c_t^{KV}) W^O = A_{\text{weight}} c_t^{KV} (W^{UV} W^O) $$

### 3. 符号说明
| 符号       | 含义                          |
|------------|-------------------------------|
| $W^{DQ}, W^{UQ}$ | 查询投影权重                  |
| $W^{DKV}, W^{UK}$ | 键值投影与潜在特征权重        |
| $d_h, d_R^R$ | 头维度与旋转编码维度          |
| $n_h$      | 注意力头数                    |
| $A_{\text{weight}}$ | 注意力权重                    |

### 4. 优化亮点
- **计算量减少**：通过合并权重矩阵（如 $W^{UQ}$ 与 $W^{UK}$），减少矩阵乘法次数。
- **缓存友好**：压缩键值对（$K', V'$）降低KV缓存占用，提升推理速度。
- **RoPE分离**：显式区分无旋转（nope）与旋转（rope）特征，增强位置编码灵活性。
这是关于 **DeepSeek MoE（混合专家模型）** 的技术解析，属于大语言模型（LLM）架构优化范畴。

### 1. 核心架构
- **共享专家（Shared Expert）**：始终激活，捕捉通用特征。
- **路由专家（Routed Expert）**：动态路由，捕捉特定任务特征。
- **DeepSeek v3 MoE 配置**：
  - 1个共享专家 + 256个路由专家。
  - 每个token路由到8个专家。
  - 前3层采用共享专家 + 固定激活专家或全连接MLP。

### 2. 关键公式
- **门控机制**：
  $$ s_{i,t} = \text{Sigmoid}(u_t^T e_i), \quad g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}} $$
  其中，$s_{i,t}$ 是token与专家的亲和力分数，$g_{i,t}$ 是归一化门控值。
- **输出计算**：
  $$ h_t' = u_t + \sum_{i=1}^{N_s} \text{FFN}^{(s)}(u_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}^{(r)}(u_t) $$
  合并共享专家与路由专家的输出。

### 3. 代码片段
```python
class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```
- **核心逻辑**：门控投影（gate_proj）与上投影（up_proj）的逐元素乘积，通过激活函数后下投影（down_proj）输出。

### 4. 优化亮点
- **动态路由**：Top-$K_r$ 选择专家，减少计算量。
- **混合专家**：共享专家保证基础能力，路由专家提升任务特异性。
- **效率提升**：前层采用全连接MLP或固定专家，平衡性能与速度。

### 5. 符号说明
| 符号       | 含义                          |
|------------|-------------------------------|
| $N_s$      | 共享专家数量                  |
| $N_r$      | 路由专家数量                  |
| $\text{FFN}^{(s)}$ | 共享专家前馈网络              |
| $\text{FFN}^{(r)}$ | 路由专家前馈网络              |
| $g_{i,t}$  | 归一化门控值                  |

### 6. 架构演进
- **传统Top-2路由**：简单路由，专家利用率低。
- **细粒度专家分割**：支持更多专家（$K=4$），提升表达能力。
- **共享专家隔离（DeepSeek MoE）**：分离共享与路由专家，平衡通用与特异性。
这是关于 **Multi-token Prediction (MTP)** 的技术解析，属于大语言模型（LLM）生成加速技术。

### 1. 核心机制
- **多token预测**：通过 $D$ 个顺序模块预测 $D$ 个额外token，输入为 $[t_1, t_2, ..., t_{SL}]$，输出为 $[t_{1+1+D}, t_{2+1+D}, ..., t_{SL+1+D}]$。
- **共享层**：嵌入层（Embedding Layer）与输出头（Output Head）与主模型共享。

### 2. 训练与推理
- **训练**：
  - 主模型（Main Model）预测下一个token，损失为 $L_{\text{Main}}$。
  - MTP模块（MTP Module）预测后续token，损失为 $L_{\text{MTP}}^1, L_{\text{MTP}}^2, ...$。
- **推理**：
  - 直接丢弃MTP模块，使用主模型生成。
  - 或复用MTP模块进行投机解码（speculative decoding），减少生成延迟。

### 3. 架构细节
- **主模型**：
  - 结构：Transformer Block × L → Output Head。
  - 输入：$[t_1, t_2, t_3, t_4]$，预测 $t_5$。
- **MTP模块**：
  - 结构：Embedding Layer → RMSNorm → Linear Projection → Transformer Block → Output Head。
  - MTP Module 1输入：$[t_2, t_3, t_4, t_5]$，预测 $t_6$。
  - MTP Module 2输入：$[t_3, t_4, t_5, t_6]$，预测 $t_7$。

### 4. 符号说明
| 符号       | 含义                          |
|------------|-------------------------------|
| $D$        | MTP模块数量                   |
| $SL$       | 输入序列长度                  |
| $L_{\text{Main}}$ | 主模型损失函数                |
| $L_{\text{MTP}}^i$ | 第$i$个MTP模块损失函数        |
| RMSNorm    | 均方根归一化层                |

### 5. 优化亮点
- **训练加速**：通过多token预测增强主模型学习，共享嵌入层与输出头减少参数量。
- **推理灵活**：支持直接丢弃MTP或投机解码，平衡生成质量与速度。
- **并行性**：MTP模块可并行训练，提升训练效率。

### 6. 应用场景
- **长文本生成**：通过MTP模块预测后续token，减少生成步骤。
- **低延迟需求**：投机解码利用MTP模块预生成token，降低延迟。
### DeepSeek MTP 介绍
#### 算法介绍——示例
- **MTP Eagle**
  - **配置**：MTP=3。输入token：$t_0 t_1 t_2 t_3 t_4$
  - **符号说明**：
    - $t_x$：已接受的token
    - $d_x$：未验证的草稿token

#### 生成流程
1. **上下文阶段**
   - 输入：$t_0 t_1 t_2 t_3 t_4$
   - 主模型生成 $t_5$
   - MTP 1 预测草稿token $d_6, d_7, d_8$
2. **生成阶段 1**
   - 输入：$t_5 d_6 d_7 d_8$
   - 验证并接受 $d_6, d_7$ 为 $t_6, t_7$
   - 采样 $t_7$ 的logit得到 $t_8$
   - MTP 1 预测草稿token $d_9, d_{10}, d_{11}$
3. **生成阶段 2**
   - 输入：$t_8 d_9 d_{10} d_{11}$
   - 验证并接受 $d_9$ 为 $t_9$
   - 采样 $t_9$ 的logit得到 $t_{10}$
   - MTP 1 预测草稿token $d_{11}, d_{12}, d_{13}$

#### 关键优化
- **DeepSeek v3 配置**：
  - $D = 1$，使用1个MTP模块
  - 第二个额外token使解码速度提升至1.8倍TPS
## 深度求索推理框架
### 框架对比
#### 关键特性

| 框架   | 并行性 |        |           |        |          | Kernel    |          |          | 量化       |          |                |           |            | Others    |           |              |                   |
|--------|--------|--------|-----------|--------|----------|-----------|----------|----------|------------|----------|----------------|-----------|------------|-----------|-----------|-------------|-------------------|
|        | TP     | EP     | Attn-DP   | PP     | Flash MLA | Deep GEMM | Deep EP  | precision | W4A8 (per tensor) | W4A8 (W: per 1x128, A: per tensor) FP8 A+FP8 KV | FP4       | Cuda graph | Torch compile | MTP      | PD disagg | Attn backend    | Chunked Prefix Cache |
| TRTLLM | Y      | Y      | Y         | Y      | Y        | Y         | N        | FP8/NVFP4 | Y            | WIP               | Y         | Y          | Y          | Y         | Y         | Flash_infer FMHA(WIP) FP8_FMHA(WIP) | Y                   |
| SGLang | Y      | Y      | Y         | Y      | Y        | Y         | Y        | FP8/BF16/AWQ/INT8 | Y            | A:per-token-per-128-channel sub-vector scales W: Per-128x128-block | WIP       | Y          | Y          | Y         | Y         | Pyverbs Mooncake NIXL-INV | Flash_infer Flashattention3 Triton | Y                   |
| vLLM   | Y      | Y      | WIP       | Y      | Y        | WIP       | N        | FP8       | Y            | N                 | N         | Y          |            | Y         | Y         | Flash_infer     | Y                   |

### Timeline Analysis and Kernel Optimizations
#### 性能分析工具界面
- **时间线视图**：显示CUDA硬件（NVIDIA H10-3e）上的GPU活动，包括内核执行和内存使用。
- **内核分布**：
  - `device_kernel`占比38.1%
  - `ncclDevKernel_ReduceScatter_Sum_bf16_R`占比14.9%
  - `flash_fwd_split_mha_kernel`占比11.4%
  - `fp8_gemm_kernel_swapAB`占比11.0%
  - `ncclDevKernel_AllGather_RING_LL`占比5.1%
  - `fp8_gemm_kernel`占比4.5%
  - `finalizeMoERoutingKernel`占比1.3%

#### 关键指标
- **CUDA API跟踪**：显示内核启动和执行时间。
- **GPU内核统计**：按时间、实例数、平均/最大/最小执行时间排序。

### DP - MLA
#### 并行策略
- **TP MLA**：
  - MHA：按GPU分割注意力头、KV缓存和权重。
  - MLA：单注意力头，部分权重无法分割，KV缓存维度为(bs, 1, seq_len, kv_lora_rank)。
- **DP MLA**：
  - 每个GPU维护bs*DP*seq_len的KV缓存。
  - GPU1: kvcache = (bs_1, 1, seq_len, kv_lora_rank)
  - GPU2: kvcache = (bs_2, 1, seq_len, kv_lora_rank)
  - 所有GPU通过Allgather获取完整序列隐藏状态，然后进行TP MoE。
  - 消息大小：(bs, seq_len, hidden_size)

#### 性能优化
- **SGLang v0.4**：相比v0.3，解码吞吐量提升1.9倍。
- **基准测试**：
  - 在H100上的DeepSeekCoder-V2吞吐量基准测试。
  - 蓝色：其他OSS基线
  - 红色：SGLang v0.3
  - 黄色：SGLang v0.4

### EP - MoE
#### 专家并行（EP2）
- **流程**：
  1. GPU0和GPU1分别处理部分token。
  2. 路由（Router）选择TopK专家。
  3. 置换（Permute）和分发（Dispatcher: All to All）。
  4. 专家计算（Expert 0-3）。
  5. 合并（Combine: All to All）和逆置换（Unpermute）。

#### 关键步骤
- **路由**：所有token选择TopK专家。
- **置换**：将token按专家分组。
- **分发**：将token发送到对应专家GPU。
- **合并**：收集所有专家结果。
- **逆置换**：恢复原始token顺序。

### SGLang PD + LargeEP
#### 部署与性能
- **架构**：
  - **Prefill Workers**：3个节点，处理专家子组和注意力计算。
  - **Decode Workers**：9个节点，处理注意力和专家子组计算。
  - **KV缓存传输**：在Prefill和Decode阶段之间传输KV缓存。
- **性能指标**：
  - 每个节点输入吞吐量：52.3k tokens/s
  - 每个节点输出吞吐量：22.3k tokens/s
- **NVIDIA Hopper性能提升**：
  - 4个月内推理吞吐量提升26倍。
  - DeepSeek-R1 671B模型在H100节点（8 GPU）上的吞吐量：
    - 1月：844 tokens/s
    - 5月：22.3k tokens/s

#### 并行策略
- **注意力（Attention）**：
  - 支持混合DP与TP。
  - 消除跨设备KV缓存重复，减少通信开销。
- **密集FFN（Dense FFNs）**：
  - 支持纯DP或纯TP。
  - 引入DP减少碎片化，提高内存和计算效率。
- **稀疏FFN（Sparse FFNs）**：
  - 支持DeepEP的EP并行。
  - 提供Normal和Low-latency模式。
- **LM头（LM Head）**：
  - 使用DP而非传统TP，降低内存开销。

#### 关键特性
- **PD实现**：
  - 非阻塞传输：数据发送和接收在后台线程运行。
  - RDMA传输：利用远程直接内存访问加速数据传输。
  - 灵活API集成：支持Mooncake和NIXL等高性能RDMA库。
- **大规模专家并行**：
  - DeepEP：支持Normal和Low-latency内核。
  - DeepGEMM：支持不同阶段的GEMM内核。

#### 两批重叠（Two-batch Overlap）
- **目标**：通过将批次拆分为微批次，重叠计算和通信。
- **挑战**：代码复杂度和预填充阶段的同步问题。
- **实现**：优先提交计算任务到GPU，减少CPU阻塞通信。
- **性能提升**：吞吐量增加27%到35%。

#### 专家并行负载均衡（EPLB）
- **目标**：最小化专家分布不平衡。
- **策略**：
  - 冗余专家策略：启发式打包重复专家。
  - 增加批次大小：通过扩展集群或使用MTP。
  - 周期性重平衡：定期更新专家排列。
- **实现**：
  - 系统加载阶段：权重从磁盘预加载到主内存。
  - 重平衡准备阶段：使用DMA异步传输权重到设备内存。
  - 重平衡执行阶段：设备间复制更新权重。

### Benchmark Results
#### 预填充性能
- **输入吞吐量**：
  - 1k输入：EP32约60k tokens/s，TP16约20k tokens/s
  - 2k输入：EP32约60k tokens/s，TP16约20k tokens/s
  - 4k输入：DeepSeek约60k tokens/s，EP32约60k tokens/s
- **关键结论**：
  - EP32在2k输入时比TP16快3.3倍。
  - EP32在4k输入时比DeepSeek慢5.6%，比EP32（理想EPLB）慢20%。

#### 解码性能
- **输出吞吐量**：
  - 0.5k输入1.5k输出：约20k tokens/s
  - 1k输入1k输出：约20k tokens/s
  - 2k输入100输出：约20k tokens/s
  - 4k输入100输出（模拟MTP）：约20k tokens/s
- **关键结论**：
  - EP72在2k输入时比TP16快5.2倍。
  - EP72在4k输入时比DeepSeek慢6.6%。
## 深度求索推理基准测试
### Excel 原始性能数据
#### 数据字段
- 量化（Quantization）
- 框架（Framework）
- 客户端（client）
- 版本（version）
- GPU
- 批量GPU数（Num GPUs）
- ISL/OSL
- 并发（concurrency）
- 实际并发（Achieved concurrency）
- 最大批大小（max bs）
- 最大token数（max_num_tokens）
- 内存分数（mem_fraction）
- TP/PP/EP/DP
- 注意力（Attn）
- DeepGEMM/DeepEP
- MTP
- KV缓存对象（KV_cache_objs）
- 输出吞吐量（Output throughput）
- TPOT/TFTT

### 介绍
#### 基础
- **DeepSeek模型**
  - **完整模型**：DeepSeek-V3-671B和DeepSeek-R1-671B（V3和R1结构相同）
  - **蒸馏模型**（结构与Qwen2.5和Llama3相同）
    - Qwen：Distill-Qwen-1.5B、Distill-Qwen-7B、Distill-Qwen-14B、Distill-Qwen-32B
    - Llama：Distill-Llama-8B、Distill-Llama-70B
- **基准指标解释**
  - **并发（Concurrency）**：推理期间同时处理的请求数
  - **吞吐量（Throughput）**：系统处理的总token数，包括输入和输出吞吐量
  - **TOPT, TTFT**：TOPT是每个输出token的时间，TTFT是第一个token的时间
  - **TPS/user**：每个用户的token/s，与在线用户体验相关
  - **MTP**：多token预测，显著加速模型解码速度
- **典型场景**
  - **延迟不敏感**：如文档摘要，可设置最大并发以获得最高系统吞吐量
  - **延迟敏感**：如聊天机器人，需满足TTFT和TOPT阈值

#### 设置
- **硬件**
  - GPU：8xH20(141G)
  - CPU：Intel(R) Xeon(R) Platinum 8468V
  - 系统内存：2T
  - 网络：8x400G IB (E-W), 2x200GE (N-S)
- **软件**
  - TensorRT-LLM torch flow、SGLang、vLLM的最新版本
- **测试用例**
  - 比较IFB性能，不使用静态批处理和PD分解
  - 仅使用ISL/OSL:1000/1000
  - 选择并发8/64/2048
  - 覆盖单节点和双节点
  - 使用帕累托曲线比较性能
- **注意事项**
  - 不同框架使用不同客户端进行基准测试，如TRTLLM使用trtllm-bench，SGLang使用bench-serving

### 20250619性能对比
#### 测试配置
- 模型：deepseekr1_1000/1000_H203e
- 框架版本：
  - TRTLLM: 0.2.10rc2-8H203e, 0.2.10rc2-16H203e
  - SGLang: 0.4.7-8H203e, 0.4.7-16H203e
  - vLLM: 0.9.2.dev98-8H203e, 0.9.2.dev98-16H203e

#### 性能结论
- TRTLLM(SGLang/vLLM)均使用启动服务器 + sglang.bench_serving
- 与trtllm-bench相比，SGLang客户端 + TRTLLM服务性能降低约30%
- 性能比较：TRTLLM(w4a8) > SGLang(fp8) > vLLM(fp8)
- 单节点TPS/GPU更好，多节点引入更多通信开销