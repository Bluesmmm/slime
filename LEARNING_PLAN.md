# Slime 框架 15 天学习计划

> 面向：大模型算法工程师（ToB 交付方向）
> 起点：理论完整（DL/RL/分布式概念），缺代码实践经验
> 终点：能独立使用 Slime 完成 ToB 场景下的 RL post-training 任务
> 仓库：https://github.com/THUDM/slime
> 本地路径：`/home/jeffliu/projects/slime`

---

## 可行性评估

### 你的优势
- Python 熟练，Linux 基本操作没问题
- 深度学习、Transformer 架构基础扎实
- RL 算法原理（PPO/GRPO）已掌握
- 分布式训练概念（TP/PP）已理解

### 需要补强的短板
| 领域 | 差距 | 预计补强时间 |
|------|------|-------------|
| Ray 框架 | 不懂 Ray API 和 placement group | 1 天 |
| SGLang 代码级理解 | 懂原理但没读过代码 | 1 天 |
| Megatron 代码模式 | 懂概念但没写过代码 | 1 天 |
| Slime 框架本身 | 全新框架 | 8-10 天 |

### 15 天后你能做到什么

| 能力 | 程度 |
|------|------|
| 环境搭建 + 模型转换 | 独立完成，闭眼操作 |
| 配置训练参数并跑通 | 独立完成，理解每个参数含义 |
| 自定义 reward function | 独立开发，适配各种 ToB 场景 |
| 自定义数据格式和 prompt | 独立开发 |
| 排错（OOM/不收敛/Ray问题） | 大部分问题能独立定位 |
| 精读核心源码 | 能读懂，理解设计意图 |
| 高级特性（FSDP/多模态/ speculative decoding） | 理解原理，能按文档配置 |

---

## 第一阶段：补强短板（Day 1-3）

### Day 1 — Ray 框架速成

**目标**：掌握 Ray 核心概念，能读懂 Slime 中的 Ray 代码

**学习内容**：

1. **Ray 核心概念**（2-3 小时）
   - Remote functions（`@ray.remote`）
   - Actors（`@ray.remote` class）
   - Object refs（异步数据引用）
   - Task 和 Actor 的资源管理（`num_gpus`, `num_cpus`）

   推荐资源：
   - Ray 官方教程：https://docs.ray.io/en/latest/ray-core/walkthrough.html
   - 重点读 "Remote Functions" 和 "Actors" 两节

2. **Placement Group**（1-2 小时）
   - 理解 `PlacementGroup`：把 GPU 资源打包分组，确保 actor 调度在同一节点
   - 理解 `BUNDLE_RESOURCE_LABEL`：每个 bundle 绑定的资源量
   - 理解策略：`STRICT_SPREAD`（严格分散）、`STRICT_PACK`（严格打包）

   对应 Slime 代码：
   ```
   slime/ray/placement_group.py（190行）
   ```
   - `create_placement_groups()` — GPU 资源分配
   - `create_training_models()` — 训练模型初始化
   - `create_rollout_manager()` — Rollout 管理器创建

3. **Ray 在 Slime 中的用法**（1-2 小时）
   - 读 `slime/ray/ray_actor.py`（11行）— 最简单的 Ray actor
   - 读 `slime/ray/actor_group.py` — Actor 组管理
   - 读 `slime/ray/utils.py` — 工具函数

**检查点**：
- [ ] 能解释 `@ray.remote(num_gpus=4)` 的作用
- [ ] 能解释 placement group 为什么在分布式训练中必要
- [ ] 能读懂 `slime/ray/placement_group.py` 中的资源分配逻辑

---

### Day 2 — SGLang 推理引擎代码级理解

**目标**：理解 SGLang 架构，能读懂 Slime 中的 rollout 代码

**学习内容**：

1. **SGLang 核心概念回顾**（1 小时）
   - RadixAttention：前缀共享的 KV cache 管理
   - Continuous batching：请求动态入队/出队
   - Tokenizer worker：tokenization 并行化
   - Slime 中 SGLang 通过 `--sglang-*` 参数配置

2. **Slime 如何调用 SGLang**（2 小时）
   - 读 `slime/rollout/sglang_rollout.py` — 核心文件，理解 rollout 生成流程
   - 重点理解：
     - 如何启动 SGLang server
     - 如何发送 generate 请求
     - 如何获取 logprobs
     - 如何做 weight sync（训练端 → 推理端）

3. **Router 中间件**（1 小时）
   - 读 `slime/router/router.py`（259行）— HTTP 负载均衡
   - 读 `slime/router/middleware_hub/radix_tree_middleware.py`（170行）— KV cache 缓存

**检查点**：
- [ ] 能画出"训练端权重更新 → SGLang 推理端加载新权重"的流程
- [ ] 理解 router 在 rollout 中的作用（负载均衡 + 请求分发）
- [ ] 知道 `--sglang-mem-fraction-static` 等参数怎么配

---

### Day 3 — Megatron-LM 代码模式

**目标**：通过 Slime 的 backends 模块理解 Megatron 代码模式

**学习内容**：

1. **Megatron 在 Slime 中的角色**（1 小时）
   - Megatron 负责：模型并行（TP/PP/EP）、前向传播、梯度计算、优化器更新
   - Slime 负责：数据编排、rollout 生成、RL loss 计算、训练-推理协调
   - 读 `docs/zh/get_started/usage.md`（380行）— 训练后端配置部分

2. **训练后端代码**（2 小时）
   - 读 `slime/backends/` 目录结构
   - Megatron 后端：理解模型初始化、数据加载、训练循环如何与 Megatron 交互
   - FSDP 后端：读 `slime/backends/fsdp_utils/arguments.py`（99行）— FSDP 配置
   - 读 `slime/backends/fsdp_utils/actor.py`（1145行）— 选读，理解 `FSDPTrainRayActor` 的大致结构

3. **参数系统**（1 小时）
   - 读 `slime/utils/arguments.py`（1731行）— 选读关键部分：
     - 集群资源配置（`--actor-num-nodes`, `--actor-num-gpus-per-node`, `--rollout-num-gpus`）
     - 训练参数（`--train-backend`, `--advantage-estimator`）
     - RL 超参数（`--kl-coef`, `--eps-clip`, `--entropy-coef`）
     - 自定义路径（`--rollout-function-path`, `--custom-rm-path`）

**检查点**：
- [ ] 理解 Slime 和 Megatron 的职责分工
- [ ] 知道三类参数（Megatron/SGLang/Slime）怎么传
- [ ] 能看懂 `scripts/models/glm4-9B.sh` 中每个参数的含义

---

## 第二阶段：环境搭建 + 跑通（Day 4-5）

### Day 4 — Docker 环境搭建 + 模型转换

**目标**：搭建完整训练环境，完成模型权重转换

**学习内容**：

1. **Docker 环境搭建**（1 小时）
   ```bash
   docker pull slimerl/slime:latest
   docker run --rm --gpus all --ipc=host --shm-size=16g \
     --ulimit memlock=-1 --ulimit stack=6710864 \
     -it slimerl/slime:latest /bin/bash
   ```
   - 理解每个参数的必要性（`--ipc=host`, `--shm-size`, `--ulimit`）
   - 读 `docs/zh/get_started/quick_start.md`（587行）— 环境搭建部分

2. **模型和下载数据集**（1 小时）
   ```bash
   # 下载模型权重
   hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414
   # 下载训练数据
   hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
   # 下载评估数据
   hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
   ```

3. **模型权重转换**（2 小时）
   - HF → Megatron torch_dist 格式：
     ```bash
     source scripts/models/glm4-9B.sh
     PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
       ${MODEL_ARGS[@]} \
       --hf-checkpoint /root/GLM-Z1-9B-0414 \
       --save /root/GLM-Z1-9B-0414_torch_dist
     ```
   - 理解为什么需要转换（Megatron 的模型并行需要特定格式）
   - Megatron → HF 格式：读 `tools/convert_torch_dist_to_hf.py`
   - 浏览 `scripts/models/` 下所有 `.sh` 文件（约30个），理解不同模型的配置差异

**检查点**：
- [ ] Docker 容器正常运行，`nvidia-smi` 可见 GPU
- [ ] 模型权重成功转换为 torch_dist 格式
- [ ] 理解 HF 格式和 Megatron 格式的区别

---

### Day 5 — Quick Start 完整跑通 + 参数理解

**目标**：完成一次完整的 RL 训练，理解每个参数含义

**学习内容**：

1. **启动训练**（2 小时）
   - 读 `docs/zh/get_started/quick_start.md` 的训练启动部分
   - 理解训练脚本中的关键参数：
     ```bash
     source scripts/models/glm4-9B.sh

     ray job submit --address="http://127.0.0.1:8265" \
       --python train.py \
       ${MODEL_ARGS[@]} \
       --train-backend megatron \
       --prompt-data /root/dapo-math-17k \
       --apply-chat-template \
       --rollout-batch-size 32 \
       --n-samples-per-prompt 8 \
       --rollout-max-response-len 1024 \
       --global-bs 8 \
       --max-epochs 1 \
       ...
     ```

2. **参数四象限理解**（2 小时）

   **集群资源**：
   | 参数 | 含义 | ToB 场景注意 |
   |------|------|-------------|
   | `--actor-num-nodes` | 训练节点数 | 根据客户GPU资源定 |
   | `--actor-num-gpus-per-node` | 每节点GPU数 | 通常8卡 |
   | `--rollout-num-gpus` | 推理GPU数 | 影响rollout速度 |
   | `--colocate` | 训练推理共用GPU | 节省资源但可能OOM |

   **训练配置**：
   | 参数 | 含义 | ToB 场景注意 |
   |------|------|-------------|
   | `--train-backend` | megatron 或 fsdp | Megatron 性能更好，FSDP 更灵活 |
   | `--advantage-estimator` | grpo/gspo/ppo | GRPO 最常用，不需要value model |
   | `--global-bs` | 全局batch size | 影响训练稳定性 |
   | `--max-epochs` | 训练轮数 | 根据收敛情况调 |

   **Rollout 配置**：
   | 参数 | 含义 | ToB 场景注意 |
   |------|------|-------------|
   | `--rollout-batch-size` | 每次rollout的prompt数 | 越大越稳定，但吃显存 |
   | `--n-samples-per-prompt` | 每个prompt生成几条 | GRPO 通常 8-16 |
   | `--rollout-max-response-len` | 最大生成长度 | 根据任务定 |

   **RL 超参数**：
   | 参数 | 含义 | 典型值 |
   |------|------|--------|
   | `--kl-coef` | KL 散度系数 | 0.001-0.01 |
   | `--eps-clip` | PPO clip 范围 | 0.2 |
   | `--entropy-coef` | 熵奖励系数 | 0.001 |
   | `--reward-type` | 奖励类型 | 见 rm_hub |

3. **监控训练**（1 小时）
   - 使用 wandb 监控 loss/reward/grad_norm
   - 使用 tensorboard 查看训练曲线
   - 读 `docs/zh/get_started/qa.md`（69行）— 常见问题排查

**检查点**：
- [ ] 成功跑完一次完整的 RL 训练
- [ ] 能解释每个关键参数的含义和调优方向
- [ ] 能看懂 wandb/tensorboard 中的训练指标

---

## 第三阶段：核心模块精读（Day 6-9）

### Day 6 — 数据流全景 + 入口文件

**目标**：理解 Slime 的完整数据流转路径

**学习内容**：

1. **画出数据流图**（2 小时）

   ```
   Prompt数据集
       ↓
   DataSource（data_source.py）加载prompt
       ↓
   RolloutActor（ray/rollout.py）调用 rollout
       ↓
   SGLangServer 生成responses → Router 负载均衡
       ↓
   RM Hub（rm_hub/）计算reward
       ↓
   Filter Hub（filter_hub/）过滤低质量样本
       ↓
   DataBuffer 缓存 (prompt, response, reward) 数据
       ↓
   TrainActor（ray/train_actor.py）读取数据训练
       ↓
   Megatron/FSDP 执行前向/反向传播
       ↓
   Weight Sync 训练权重 → SGLang 推理端
       ↓
   下一轮 Rollout
   ```

2. **精读入口文件**（2 小时）
   - 读 `train.py`（102行）— 同步训练循环
     - 理解 `init()` → `generate_rollout()` → `train_step()` → `sync_weight()` 的循环
   - 读 `train_async.py`（78行）— 异步训练循环
     - 理解 `generate_rollout()` 和 `train_step()` 如何并行
     - 理解为什么异步模式下不支持 colocate

**检查点**：
- [ ] 能默画出完整数据流图
- [ ] 能解释同步训练和异步训练的区别
- [ ] 知道每一轮迭代的执行顺序

---

### Day 7 — Rollout 模块精读

**目标**：深入理解 rollout 生成、reward 计算、数据过滤

**学习内容**：

1. **Rollout 核心**（2 小时）
   - 读 `slime/rollout/sglang_rollout.py` — 主力 rollout 实现
     - 理解 `generate_rollout()` 函数的完整流程
     - 理解 prompt → SGLang generate → 收集 response → 计算 reward
   - 读 `slime/rollout/sft_rollout.py`（65行）— SFT rollout
     - 理解 loss mask 在多轮对话中的处理
   - 读 `slime/rollout/sleep_rollout.py`（13行）— 空 rollout

2. **数据源**（1 小时）
   - 读 `slime/rollout/data_source.py`（219行）
     - `DataSource` 基类
     - `RolloutDataSource` — 全局数据集
     - `RolloutDataSourceWithBuffer` — 带缓冲的数据源

3. **Reward Model Hub**（2 小时）— **重点，ToB 核心技能**
   - 读 `slime/rollout/rm_hub/__init__.py`（81行）— reward 分发器
     - 理解 `async_rm()` 和 `batched_async_rm()` 的调用链
   - 读 `slime/rollout/rm_hub/deepscaler.py` — 数学推理 reward
   - 读 `slime/rollout/rm_hub/f1.py` — F1 score reward
   - 读 `slime/rollout/rm_hub/gpqa.py` — 多选 reward
   - 读 `slime/rollout/rm_hub/math_utils.py`（489行）— 数学答案验证（选读）
   - 读 `slime/rollout/rm_hub/ifbench.py` — 指令遵循 reward

4. **Filter Hub**（1 小时）
   - 读 `slime/rollout/filter_hub/__init__.py` — 过滤器分发
   - 读 `slime/rollout/filter_hub/base_types.py`（38行）— 过滤器类型定义
   - 读 `slime/rollout/filter_hub/dynamic_sampling_filters.py` — 动态采样过滤

**检查点**：
- [ ] 能解释 `async_rm()` 的调用链和 reward 计算流程
- [ ] 能说出 5 种以上内置 reward 类型的适用场景
- [ ] 理解 dynamic sampling filter 的作用和配置方法

---

### Day 8 — Ray 编排层精读

**目标**：理解 Slime 如何用 Ray 编排分布式训练

**学习内容**：

1. **Placement Group**（2 小时）
   - 精读 `slime/ray/placement_group.py`（190行）
     - `create_placement_groups()` — GPU 资源分配策略
       - 理解训练组和推理组的资源划分
       - 理解 `STRICT_SPREAD` vs `STRICT_PACK` 策略
     - `create_training_models()` — 训练模型初始化
     - `create_rollout_manager()` — Rollout 管理器创建

2. **Train Actor**（2 小时）
   - 读 `slime/ray/train_actor.py` — 训练 actor
     - 理解训练循环中 Ray actor 的生命周期
   - 读 `slime/ray/rollout.py` — Rollout actor
     - 理解 rollout 生成的分布式调度

3. **Actor Group**（1 小时）
   - 读 `slime/ray/actor_group.py` — Actor 组管理
     - 理解如何管理多个 GPU 上的 actor 实例

**检查点**：
- [ ] 能画出训练 actor 和 rollout actor 的资源分布图
- [ ] 理解 colocate 模式和非 colocate 模式的资源分配差异
- [ ] 知道如何调整 `--actor-num-gpus-per-node` 和 `--rollout-num-gpus`

---

### Day 9 — Router + Data Buffer + Generate Hub

**目标**：理解推理路由、数据缓冲和生成策略

**学习内容**：

1. **Router 系统**（2 小时）
   - 精读 `slime/router/router.py`（259行）
     - `SlimeRouter` 类 — HTTP 负载均衡器
     - `proxy()` — 请求转发逻辑
     - `add_worker()` — Worker 注册与健康检查
   - 读 `slime/router/middleware_hub/radix_tree_middleware.py`（170行）
     - Radix tree 缓存 — 前缀共享 KV cache
     - 权重版本管理

2. **Rollout Buffer Plugin**（2 小时）
   - 读 `slime_plugins/rollout_buffer/buffer.py`（341行）
     - `BufferQueue` — 数据分组和验证
     - `RolloutBuffer` — 线程安全缓冲区管理
     - 理解 FastAPI 服务集成
   - 读 `slime_plugins/rollout_buffer/rollout_buffer_example.py` — 使用示例

3. **Generate Hub + 自定义 Rollout**（1 小时）
   - 读 `slime/rollout/generate_hub/__init__.py`
   - 读 `slime/rollout/generate_hub/benchmarkers.py`
   - 理解 `--rollout-function-path` 如何加载自定义 rollout 函数

**检查点**：
- [ ] 理解 Router 如何在多个 SGLang server 之间分发请求
- [ ] 理解 Radix tree 中间件如何加速 rollout 生成
- [ ] 知道 Rollout Buffer 的使用场景和配置方法

---

## 第四阶段：自定义开发（Day 10-12）

### Day 10 — 自定义 Reward Function（ToB 最重要的技能）

**目标**：能根据客户需求开发自定义 reward function

**学习内容**：

1. **Reward Function 接口规范**（1 小时）
   - 读 `docs/zh/get_started/customization.md`（410行）— 自定义 reward 部分
   - 理解 reward function 的签名和返回值格式
   - 理解 `--custom-rm-path` 参数如何加载自定义 reward

2. **精读内置 Reward 实现**（2 小时）
   - `slime/rollout/rm_hub/deepscaler.py` — 最简单的示例
   - `slime/rollout/rm_hub/f1.py` — 文本匹配类 reward
   - `slime/rollout/rm_hub/gpqa.py` — 多选题类 reward
   - `slime/rollout/rm_hub/ifbench.py` — 指令遵循类 reward

3. **实战：开发3个自定义 Reward**（3 小时）

   **Reward 1：JSON 格式验证**
   - 场景：客户要求模型输出严格 JSON 格式
   - 实现：解析 JSON + 检查字段完整性 + 分数计算

   **Reward 2：业务规则匹配**
   - 场景：客户有特定的业务规则（如客服回复规范）
   - 实现：正则匹配 + 关键词检查 + 权重打分

   **Reward 3：多维度综合评分**
   - 场景：综合准确性、格式、长度等多个维度
   - 实现：加权求和 + 阈值过滤

**检查点**：
- [ ] 能独立写出符合 Slime 接口规范的 reward function
- [ ] 理解 `--custom-rm-path` 的加载机制
- [ ] 能根据客户需求快速设计 reward 方案

---

### Day 11 — 自定义数据格式 + Prompt 模板 + Rollout

**目标**：能适配各种客户数据格式，自定义 rollout 流程

**学习内容**：

1. **数据格式适配**（2 小时）
   - 读 `docs/zh/get_started/customization.md` 的数据部分
   - 理解 `--prompt-data`, `--input-key`, `--label-key` 参数
   - 理解 `--apply-chat-template` 的作用
   - 实战：将客户 CSV/Excel/JSON 数据转换为 Slime 支持的格式

2. **自定义 Rollout Function**（2 小时）
   - 理解 `slime/rollout/base_types.py` 中的 `RolloutFnTrainOutput` 和 `RolloutFnEvalOutput`
   - 读 `examples/` 中的自定义 rollout 示例：
     - `examples/retool/` — 工具调用类 rollout
     - `examples/tau-bench/` — 多轮对话类 rollout
   - 理解 `--rollout-function-path` 如何加载自定义 rollout

3. **Chat Template 定制**（1 小时）
   - 理解不同模型的 chat template 格式
   - 实战：为自定义模型编写 chat template

**检查点**：
- [ ] 能将客户原始数据转为 Slime 训练格式
- [ ] 能编写自定义 rollout function
- [ ] 理解 chat template 的配置和自定义方法

---

### Day 12 — 自定义 Loss Function + Dynamic Sampling Filter

**目标**：掌握训练策略层面的自定义能力

**学习内容**：

1. **自定义 Loss Function**（2 小时）
   - 读 `docs/zh/get_started/customization.md` 的 loss 部分
   - 读 `examples/DrGRPO/` — 自定义 policy gradient reducer 示例
     - 理解如何修改 loss 计算逻辑
     - 理解 `ConstantDivisorReducer` 的实现
   - 实战：实现一个带 length penalty 的 GRPO loss

2. **Dynamic Sampling Filter**（2 小时）
   - 读 `slime/rollout/filter_hub/dynamic_sampling_filters.py`
   - 理解过滤策略：
     - 按 reward 值过滤
     - 按响应长度过滤
     - 按重复率过滤
   - 理解 `--dynamic-sampling-filter-path` 参数

3. **Buffer Filter**（1 小时）
   - 理解 buffer filter 的作用（数据质量控制）
   - 理解 `--buffer-filter-path` 参数

**检查点**：
- [ ] 能独立编写自定义 loss function
- [ ] 理解 dynamic sampling filter 的设计模式和配置
- [ ] 知道如何组合 reward + filter + loss 进行端到端训练优化

---

## 第五阶段：高级场景 + 实战（Day 13-15）

### Day 13 — FSDP Backend + 多模态 VLM 训练

**目标**：掌握非 Megatron 后端和多模态场景

**学习内容**：

1. **FSDP Backend**（2 小时）
   - 读 `slime/backends/fsdp_utils/arguments.py`（99行）— FSDP 配置
   - 读 `slime/backends/fsdp_utils/actor.py`（1145行）— 选读关键部分：
     - `FSDPTrainRayActor` — FSDP 训练 actor
     - `_setup_device_mesh()` — 并行策略配置
     - `apply_fsdp2()` — FSDP wrapper 应用
   - 读 `slime/backends/fsdp_utils/data_packing.py` — 变长数据处理
   - 理解 FSDP vs Megatron 的选择：
     - FSDP：更灵活，支持更多模型架构，不需要权重转换
     - Megatron：性能更好，但需要 torch_dist 格式

2. **非 Megatron 模型支持**（1 小时）
   - 读 `docs/en/advanced/arch-support-beyond-megatron.md`
   - 理解三组件集成系统：Module spec 替换 + HF wrapper + 权重对齐
   - 读 `slime_plugins/models/` 下的模型适配代码

3. **多模态 VLM 训练**（2 小时）
   - 读 `examples/geo3k_vlm/` — 单轮 VLM 训练
   - 读 `examples/geo3k_vlm_multi_turn/` — 多轮 VLM 训练
   - 理解 `examples/true_on_policy_vlm/` — VLM on-policy 训练
   - 理解 VLM 训练的特殊配置（图像处理、视觉编码器等）

**检查点**：
- [ ] 理解 FSDP 和 Megatron 后端的适用场景
- [ ] 知道如何配置 FSDP 后端训练
- [ ] 理解 VLM RL 训练的流程和注意事项

---

### Day 14 — ToB 典型场景实战

**目标**：模拟真实 ToB 场景，端到端完成一次交付

**场景 1：数学推理能力增强（上午）**

基于 `docs/en/examples/qwen3-4b-base-openhermes.md` 或 `docs/en/examples/qwen3-4B.md`：
- 选择基础模型（Qwen3-4B）
- 准备客户数学数据
- 配置 math reward（使用 `rm_hub/math_utils.py`）
- 配置训练参数并启动
- 分析训练曲线和评估结果

**场景 2：工具调用能力训练（下午）**

基于 `examples/retool/` 和 `examples/tau-bench/`：
- 理解工具调用类 reward 设计
- 配置多轮对话 rollout
- 理解环境交互（tool execution feedback）
- 配置并运行训练

**场景 3：代码生成能力（选做）**

基于 `examples/swe-agent/`：
- 理解代码类 reward 设计（编译通过、测试通过）
- 理解 Docker-in-Docker 环境配置

**检查点**：
- [ ] 能从零开始配置并运行一个完整的 ToB RL 训练任务
- [ ] 能根据客户需求选择合适的 reward function
- [ ] 能分析训练结果并给出调优建议

---

### Day 15 — 排错指南 + 性能调优 + 总结复盘

**目标**：掌握排错方法论，建立系统知识框架

**学习内容**：

1. **排错指南**（2 小时）
   - 读 `docs/en/developer_guide/debug.md`
     - 精度对齐验证方法
     - 分离调试模式（`--debug-rollout-only`, `--debug-train-only`）
     - Debug 数据保存和加载
   - 读 `docs/zh/get_started/qa.md`（69行）— 常见问题
     - OOM 解决方案
     - Ray job 挂起排查
     - Grad norm 异常处理
     - SGLang 连接问题

2. **性能调优**（2 小时）
   - 读 `docs/en/advanced/speculative-decoding.md` — 推测解码加速
   - 读 `docs/en/advanced/pd-disaggregation.md` — Prefill/Decode 分离
   - 读 `docs/en/advanced/fault-tolerance.md` — 容错机制
   - 理解关键性能参数：
     - `--rollout-batch-size` vs `--n-samples-per-prompt` 的 trade-off
     - Colocate 模式的资源利用率优化
     - `--sglang-mem-fraction-static` 对显存的影响

3. **总结复盘**（2 小时）
   - 画一张完整的 Slime 架构图（含所有模块和数据流）
   - 整理 ToB 场景 checklist：
     - 需求分析 → 模型选择 → 数据准备 → Reward 设计 → 训练配置 → 评估交付
   - 列出后续深入学习方向（见下方"大师之路"）

**检查点**：
- [ ] 能独立排查常见训练问题
- [ ] 知道如何针对不同场景调优性能
- [ ] 有完整的 Slime 知识框架图

---

## 大师之路：15 天之后

15 天计划完成后，你已经具备 ToB 实战能力。以下方向可以持续精进：

### 短期（第 3-4 周）
- [ ] 深入阅读 `slime/backends/fsdp_utils/actor.py`（1145行）全文
- [ ] 研究 `examples/on_policy_distillation/` — 蒸馏训练
- [ ] 研究 `examples/multi_agent/` — 多智能体训练
- [ ] 研究 `examples/formal_math/` — 形式化数学证明训练
- [ ] 阅读 Slime 官方博客：https://thudm.github.io/slime/blogs/release_v0.1.0.html

### 中期（第 2-3 月）
- [ ] 研究 speculative decoding + EAGLE 算法实现
- [ ] 研究 PD 分离部署模式
- [ ] 研究 Triton kernel 优化（`slime/backends/fsdp_utils/kernels/`）
- [ ] 研究 MoE 模型训练（`slime_plugins/mbridge/`）
- [ ] 尝试给 Slime 提交 PR（文档改进、bug fix）

### 长期
- [ ] 跟进 Slime 版本更新，关注新特性
- [ ] 研究 APRIL 论文 — Rollout 加速优化
- [ ] 研究自研 RL 算法在 Slime 中的实现
- [ ] 参与 Slime 社区讨论和 Issue 回答
- [ ] 积累 5+ 个 ToB RL 训练实战案例

---

## 关键文件速查表

| 用途 | 文件路径 | 行数 | 优先级 |
|------|---------|------|--------|
| 快速开始 | `docs/zh/get_started/quick_start.md` | 587 | P0 |
| 使用文档 | `docs/zh/get_started/usage.md` | 380 | P0 |
| 自定义指南 | `docs/zh/get_started/customization.md` | 410 | P0 |
| 常见问题 | `docs/zh/get_started/qa.md` | 69 | P0 |
| 调试指南 | `docs/en/developer_guide/debug.md` | 50 | P1 |
| 同步训练入口 | `train.py` | 102 | P0 |
| 异步训练入口 | `train_async.py` | 78 | P0 |
| 全部参数定义 | `slime/utils/arguments.py` | 1731 | P1 |
| Rollout 核心 | `slime/rollout/sglang_rollout.py` | - | P0 |
| SFT Rollout | `slime/rollout/sft_rollout.py` | 65 | P1 |
| 数据源 | `slime/rollout/data_source.py` | 219 | P0 |
| 基础类型 | `slime/rollout/base_types.py` | 27 | P0 |
| Reward Hub | `slime/rollout/rm_hub/__init__.py` | 81 | P0 |
| 数学 Reward | `slime/rollout/rm_hub/math_utils.py` | 489 | P1 |
| Filter Hub | `slime/rollout/filter_hub/__init__.py` | 1 | P1 |
| Filter 类型 | `slime/rollout/filter_hub/base_types.py` | 38 | P1 |
| 动态过滤 | `slime/rollout/filter_hub/dynamic_sampling_filters.py` | - | P1 |
| Ray 编排 | `slime/ray/placement_group.py` | 190 | P0 |
| 训练 Actor | `slime/ray/train_actor.py` | - | P1 |
| Rollout Actor | `slime/ray/rollout.py` | - | P1 |
| Actor 组 | `slime/ray/actor_group.py` | - | P2 |
| Router | `slime/router/router.py` | 259 | P1 |
| Radix 缓存 | `slime/router/middleware_hub/radix_tree_middleware.py` | 170 | P2 |
| FSDP Actor | `slime/backends/fsdp_utils/actor.py` | 1145 | P2 |
| FSDP 参数 | `slime/backends/fsdp_utils/arguments.py` | 99 | P1 |
| Rollout Buffer | `slime_plugins/rollout_buffer/buffer.py` | 341 | P2 |
| 模型配置脚本 | `scripts/models/*.sh` | ~30个文件 | P0 |
| 架构扩展 | `docs/en/advanced/arch-support-beyond-megatron.md` | 35 | P2 |
| 推测解码 | `docs/en/advanced/speculative-decoding.md` | 39 | P2 |
| PD 分离 | `docs/en/advanced/pd-disaggregation.md` | 3 | P2 |
| 容错 | `docs/en/advanced/fault-tolerance.md` | 13 | P2 |
