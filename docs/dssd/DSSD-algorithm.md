# DSSD 分布式分割推测解码 —— 完整算法流程描述

---

## 系统概述

系统由两个角色组成：

- **Device（设备端）**：运行轻量小语言模型 MqM_q Mq​（SLM，如 OPT-125M），负责 Draft 生成与 Resample
- **Edge/BS（基站端）**：运行大语言模型 MpM_p Mp​（LLM，如 OPT-13B），负责 Accept/Reject 判断

两个模型共享同一词表 V\mathcal{V} V，词表大小为 ∣V∣|\mathcal{V}| ∣V∣。

---

## 核心数据结构

```
prefix       : List[int]          # 当前已确认的 token 序列（设备与基站同步维护）
γ            : int                 # 每轮草稿长度（超参数，需调优）
draft_tokens : List[int]          # SLM 采样出的 γ 个 token 索引
Q_dists      : List[Tensor]       # Q_i(x)，SLM 对每个 draft token 的完整词表分布，shape=(γ, |V|)
q_values     : List[float]        # q_i(x_i)，SLM 分布中对应 draft token 的概率标量值，length=γ
P_dists      : List[Tensor]       # P_j(x)，LLM 对每个位置的完整词表分布，shape=(γ+1, |V|)
```

---

## 完整单轮循环流程

### Phase 1 · Draft（设备端）

```
输入: prefix, Mq, γ

y = []                                   # 本轮已生成的 draft token 列表
for i in 1 .. γ:
    Q_i(x) = Mq(prefix + y)             # 前向推理，得到完整词表概率分布，shape=(|V|,)
    x_i ~ Q_i(x)                        # 按分布采样一个 token（top-k/temperature 等策略）
    y.append(x_i)

draft_tokens = [x_1, ..., x_γ]
Q_dists      = [Q_1(x), ..., Q_γ(x)]
q_values     = [Q_1(x_1), Q_2(x_2), ..., Q_γ(x_γ)]  # 取对应位置的标量概率值
```

---

### Phase 2 · 上行传输（Device → BS）

**DSSD 的关键优化**：上行只传标量概率值，而非完整分布。

```
上行数据包:
  - draft_tokens : List[int]    # γ 个 token 的词表索引
  - q_values     : List[float]  # γ 个标量概率值 q_i(x_i)

上行数据量 ≈ γ × (index_bits + 32bits)   # 极小，约 50 bytes（γ=8 时）
对比 DSD 上行: γ × |V| × 32bits          # 约 61,269 bytes（γ=4，|V|=50257 时）
```

---

### Phase 3 · Verification Accept/Reject（基站端）

```
输入: prefix, draft_tokens=[x_1,...,x_γ], q_values=[q_1(x_1),...,q_γ(x_γ)], Mp

# LLM 一次并行前向推理，得到 γ+1 个位置的分布
[P_1(x), ..., P_γ(x), P_{γ+1}(x)] = Mp(prefix, prefix+[x_1], ..., prefix+[x_1,...,x_γ])
#  P_j(x) 是 LLM 在看到前 j-1 个 draft token 后对第 j 个位置的预测分布

j = 1
Flag = True   # True 表示目前全部接受

while j <= γ and Flag:
    p_j_xj = P_j(x)[x_j]                      # LLM 分布中 x_j 对应的标量概率
    q_j_xj = q_values[j-1]                    # SLM 分布中 x_j 对应的标量概率（上行传来）
    r_j ~ Uniform(0, 1)

    if r_j < min(1, p_j_xj / q_j_xj):        # Accept 条件
        x_j 被接受
        j += 1
    else:                                       # Reject
        Flag = False                            # 停止，记录拒绝位置 j
        # 注意：BS 不做 Resample！这是 DSSD 与 DSD 的核心区别

if Flag == True:                               # 全部 γ 个 token 被接受
    x_{γ+1} ~ P_{γ+1}(x)                     # BS 额外采样第 γ+1 个 token
```

---

### Phase 4 · 下行传输（BS → Device）

**DSSD 的另一关键优化**：只在发生拒绝时才下行传输完整分布，且最多传一个。

```
if Flag == False:   # 发生拒绝，第 j 个 token 被拒绝
    下行数据包:
      - P_j(x) : Tensor[|V|]    # 拒绝位置的 LLM 完整词表分布
      - j       : int            # 拒绝发生的位置索引

if Flag == True:    # 全部接受
    下行数据包:
      - x_{γ+1} : int            # 新采样的 token 索引
      - j = γ+1  : int           # 标记全部接受

下行数据量（拒绝时）= |V| × 32bits ≈ 200KB（最坏情况，仅偶发）
下行数据量（接受时）≈ 4bytes（仅一个 token 索引）
```

---

### Phase 5 · Resample（设备端）

```
if j < γ+1:   # 收到拒绝信号（Flag==False）
    # 设备端用收到的 P_j(x) 和本地保存的 Q_j(x) 执行 Resample
    Q_j(x) = Q_dists[j-1]                              # 本地已有，无需传输
    residual = max(0, P_j(x) - Q_j(x))                 # 逐元素操作，shape=(|V|,)
    x'_j ~ normalize(residual)                          # 按残差分布重采样
    x_j = x'_j                                          # 替换被拒绝的 token

if j == γ+1:  # 全部接受
    无需 Resample
```

---

### Phase 6 · Prefix 同步与下一轮准备

```
# 本轮接受的 token 序列为 [x_1, ..., x_j]（含重采样后的 x_j，若有拒绝）
prefix = prefix + [x_1, ..., x_j]

# 设备与基站均更新 prefix（保持同步）
# 若发生拒绝（Flag==False），设备需在下一轮上行时携带 x'_j，使基站同步
# （算法注释：这个同步可以在下一轮 Draft 的上行包中捎带完成）

进入下一轮，重复 Phase 1~6，直到生成足够 token
```

