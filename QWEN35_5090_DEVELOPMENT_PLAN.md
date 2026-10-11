# Qwen3.5 / 8×5090 推理优化计划

通过改进 vLLM，让 Qwen3.5-122B FP8 在少量用户同时使用时，更快开始回答、持续输出更流畅。

主要优化并发 1–2，验证到 16。保留 TP8、MTP=3 和已验证的 P2P/1-stage 通信实现。
基于 `origin/qwen35-122B-A10B-FP8`（v0.25.1，`752a3a5044`）。

## 先做什么

先测清时间花在哪里，再减少 MTP 的通信和 CPU 编排，随后优化最耗时的计算。
Mega kernel 与其他方案一起评估：当前安排在基础优化之后，是为了取得公平对照；若初始 trace 显示收益更大，可以提前做原型。

### 1. 建立可比较的基线

目前 P2P 方案的 P90 TTFT 为 640.54 ms、TPOT 为 14.19 ms、ITL 为 138.04 ms。
先补齐这些数据的请求长度、并发和采样条件，记录部署版本、拓扑及通信补丁。

- [ ] 固定测试数据和启动配置，预热后重复测量。
- [ ] 分别测主模型验证、MTP 预测、采样、状态准备及通信的耗时，确认 GPU 等待的原因。

完成后应能回答：下一项改动能减少哪段时间、预计收益上限多大。

### 2. 减少预测阶段的通信，补齐图尺寸

先启用 `use_local_argmax_reduction=true`：每卡只交换候选最大值和 token，避免收集完整词表 logits。Qwen 已有接口，无需移植其他模型的实现。

重新设计 CUDA Graph 尺寸，减少补齐计算和超出范围后的回退：主模型 MTP=3 验证每请求 4 个 token，后续预测每请求 1 个 token，两者分别设计。

- [ ] local argmax 独立 A/B，确认通信减少、预测结果正确。
- [ ] graph 上限先扩到 64；覆盖验证阶段的 4–64（步长 4）和后续预测的 1–16，首步按实际形状处理。

当前尺寸已覆盖并发 1–2 的主模型验证；扩大尺寸主要帮助其他并发。

### 3. 减少 CPU 回传，让全部 draft forward 使用 full graph

这是第一项主要源码工作，分两项独立验证。

- [ ] 减少接受 token 状态和 draft token 的无用 CPU 回传，参考 [#43955](https://github.com/vllm-project/vllm/pull/43955)。该 PR 未合并，需重新核对缓存模式和 CPU 消费者，保留必要状态更新。
- [ ] 将三次 draft forward（首步及后续两步）全部纳入 full graph。首步按多 token/实际输入形状捕获，后续按每请求一个 token 捕获；分别处理图描述、预热、KV 写入隔离和 TP8 通信。参考 [#45258](https://github.com/vllm-project/vllm/issues/45258) 的后续步设计，首步另行实现。

metadata 构建、输入准备和采样仍在图外，不开展 #49488 的整条 chain 捕获。
full forward 内部需要固定地址的 metadata buffer，运行前更新其内容；这属于必要接入。
先验证后续步，再接入首步，但最终交付包含三次 forward。首步需覆盖 prefill、
混合及纯 decode 形状；不支持的 backend/形状保留原路径并明确记录回退范围。

完成后比较每一步的 TPOT、ITL 和接受长度，确认请求变化及拒绝 token 后状态仍正确。

## 接着优化什么

以下都是具体工作项。根据最新 trace 排序，先实现收益最明确的一项；没有足够成本占比的项目暂缓。

### 4. 融合通信与归一化

- [ ] 基于现有 PCIe 通信实现，融合 all reduce、residual add 和 Qwen RMSNorm，减少中间访存及 kernel 边界。

参考 [#49793](https://github.com/vllm-project/vllm/pull/49793) 的尾部通信融合。它主要针对 DeepSeek/GLM，需适配 Qwen 和 SM120；其中 local argmax 已纳入第 2 项。

### 5. 优化小批量专家计算

- [ ] 为主模型验证和 MTP 预测的小尺寸设计 MoE 执行路径，减少排序、padding 和多阶段调度；探索 shared/routed expert 联合执行。

前提是 MoE 在 trace 中占比足够大。保留大批量 prefill 路径，分别验证 TTFT 和 TPOT。

### 6. 融合 GDN 验证计算

- [ ] 将卷积后的门控、递归状态更新和归一化融合，适配 5090。

参考 [#51674](https://github.com/vllm-project/vllm/pull/51674)。其验证硬件是 SM100，需检查 SM120 支持，并核对 [#51812](https://github.com/vllm-project/vllm/pull/51812) 的状态索引修复。以实际 GDN 耗时决定投入顺序。

### 7. 融合输出计算与 token 选择

- [ ] 若 LM-head/argmax 成本仍明显，将局部 logits 计算与最大值选择融合，减少 logits 写回和重读。

它与第 2 项不同：第 2 项减少跨卡通信，本项优化每卡计算与访存。

### 8. 用 mega kernel 扩大融合范围

- [ ] 选择 MTP head、MoE 或 decoder block 做 persistent kernel 原型，让 GPU 内部连续调度计算和通信。
- [ ] 与同范围的 full graph 比较；局部收益成立后，再评估跨层或整模型实现。

目标是减少设备端空隙并重叠任务。是否提前做，取决于这些成本是否大于前面各项，以及能否保持 FP8 kernel 效率和 PCIe 通信进展。参考 [Mirage](https://github.com/mirage-project/mirage) 与 [Hazy TP 实现](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)。

## 首次回答速度单独验证

### 9. 减少多轮请求的重复计算

此项在基线完成后即可开展，无需等待计算内核优化。

- [ ] 用真实多轮代码/工具请求，测前缀命中后实际重算多少 token，检查 MTP 末块回退和混合模型状态对齐的代价。
- [ ] 若重算明显，设计减少重算的状态保存/恢复方案；同时测冷请求及正在输出时插入长输入的表现。

验收是 warm TTFT 改善，缓存状态正确，已有请求的输出停顿不恶化。

## 如何判断完成

每项独立开关、独立对照；保留收益成立的改动，再测试组合效果。

- 并发 1、2 为主，3、8、16 验证；固定请求长度和采样参数，每组至少 3 次。
- 保存 TTFT、TPOT、ITL 的分位数，以及接受长度、显存和 trace；收益应超过基线波动。
- 验证输出及状态，覆盖请求加入/退出、拒绝 token、混合 prefill/decode、缓存命中和 block 边界。
- 源码改动运行对应测试；组合完成后复测 MTP=2/3/4。

当前所有工作均待实现或实测。向上游提交时按 AGENTS.md 完成重复工作检查、人工审查及 AI 使用说明。
