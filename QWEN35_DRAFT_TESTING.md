# Draft 优化开发机验证

这是实验实现，尚未在 CUDA 上验证。原开发计划未修改。

## Backend 是什么

这里指 attention backend，即执行注意力计算的具体实现。原启动命令没有指定它，由 vLLM 根据硬件和模型选择；启动日志中的 attention backend 才是实际结果。主模型的 GDN 实现与 MTP head 的 attention backend 需要分别看。

当前新增路径仅接入 drafter 的 `TRITON_ATTN`，主模型仍自动选择。先比较原自动选择与 Triton，再在相同 Triton 配置上比较 graph 优化，避免把 backend 切换收益算进优化。

## 两个开关

- `full_draft_cudagraph`：捕获首步及后续两个 draft forward；输入准备、metadata 更新和采样仍在图外。当前覆盖均匀 decode 输入，prefill、混合及未捕获形状回退原路径。
- `skip_unused_accepted_tokens_cpu`：仅在异步 speculative decode 且 `mamba_cache_mode=none` 时跳过接受数量的 CPU 副本。`align/all` 保留更新；开启 prefix cache 的默认模式可能不会触发此优化。原有 draft token CPU 副本条件判断继续保留。

## 测试顺序

1. 按仓库 AGENTS.md 在 Linux CUDA 开发机安装环境和项目，运行：

   ```bash
   .venv/bin/python -m pytest tests/v1/spec_decode/test_mtp_cudagraph.py -v
   ```

   这些单测检查 metadata 地址与内容更新、回退条件和接受数量更新，不能替代 GPU 测试。

2. 保留原启动参数，先将 speculative config 改成下面配置作为对照：

   ```json
   {"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}
   ```

3. 仅添加 `"full_draft_cudagraph":true`，比较 graph 开关两组。启动应出现 `Captured ... full MTP forward graphs`；运行时会按 query_len 分别报告 hits/fallbacks（首次及每1000次）。纯 decode 的 query_len=4 和1都应命中；prefill/混合阶段允许回退。CUDA Graph 额外占显存，启动 OOM 时先减少 capture sizes。

4. 保留已测出收益的 local argmax 和其他参数作为新基线；graph 上限设64以覆盖16请求×4 token。drafter 自动补齐并发1–16的精确尺寸，不受主模型把尺寸取整为4倍数的影响；其他均匀尺寸可使用下一档图，补齐行不写KV。先测并发1、2，再测3、8、16。

5. 固定输入、temperature=0 和输出长度，逐请求比较 token IDs；覆盖短/长输入、多轮缓存命中、请求加入退出、混合 prefill/decode 和拒绝 token。输出差异先定位，不直接当成正常浮点差异。保存接受长度及 TTFT/TPOT/ITL，每组至少三次。

6. 用 Nsight Systems 确认纯 decode 时三次 draft forward 都是 full graph replay，采样仍在图外。核对 TP8 custom all reduce 捕获/重放是否正确、有无挂起，以及 KV 写入和拒绝后状态恢复。

CPU 回传开关单独测试：确认异步 speculative decode 已启用、缓存模式为 none 后，分别打开/关闭 `skip_unused_accepted_tokens_cpu`；不要与 prefix cache 模式切换同时比较。先通过正确性检查，再组合两个开关测性能。
