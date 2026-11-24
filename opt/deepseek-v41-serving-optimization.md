# A5 V4.1 Serving 调度优化总览

本文只讨论 A5 V4.1 的 Serving 调度层。需要区分两个问题：

1. **DPLB：新请求首次分配到哪个 DP partition。**
2. **抢占：KV 容量不足时释放哪个已有请求。**

DPLB 与抢占不是同一个机制。DPLB 只做新请求 admission，抢占只处理容量不足时的已有请求释放。

## 1. 当前 Serving 边界

V4.1 使用全局 Scheduler 和 grouped KV cache：

- 全局调度入口：[Scheduler._schedule()](../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L457)
- 新请求进入全局 waiting 队列；
- 请求获得 cache partition 后继续使用原 partition；
- 已绑定请求不能因为负载评分迁移；
- partition 的 KV block 和请求容量仍由 allocator/Runner 做硬校验；
- homogeneous Prefill/Decode 和模型 forward 不在本方案内修改。

关键源码：

- [V4.1 EngineConfig](../../pypto-serving/pypto_serving/cli/main.py#L387)
- [partition 候选过滤](../../pypto-serving/pypto_serving/serving/memory/kv_cache.py#L1427)
- [partition 选择](../../pypto-serving/pypto_serving/serving/memory/kv_cache.py#L1447)
- [V4.1 容量校验](../../pypto-serving/pypto_serving/model/deepseek_v41/npu_runner.py#L778)

## 2. DPLB 问题与联合方案

### 2.1 当前 A5 如何给全新请求选择 DP partition

一个全新请求首先由 `Scheduler.add_request()` 放入全局 `waiting` 队列。它此时没有 `cache_partition`。A5 不是为每个 DP 建立一个 Scheduler，而是由一个全局 Scheduler 在 grouped KV 分配时选择 partition。

当前调用顺序是：

```text
全新请求
  -> Scheduler.waiting
  -> Scheduler._schedule() 的 waiting Phase 2
  -> Prefix Cache lookup
  -> 计算本轮 num_new
  -> Scheduler._try_allocate_request_blocks()
  -> KvCacheManager.ensure_group_blocks()
  -> KvCacheManager._select_group_partition()
  -> 写入 request.cache_partition
```

源码位置：

- 新请求进入 waiting：[Scheduler.add_request()](../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L275)
- waiting admission：[Scheduler._schedule()](../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L591)
- KV 分配：[Scheduler._try_allocate_request_blocks()](../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L1251)
- grouped KV 分配：[KvCacheManager.ensure_group_blocks()](../../pypto-serving/pypto_serving/serving/memory/kv_cache.py#L1491)
- partition 选择：[KvCacheManager._select_group_partition()](../../pypto-serving/pypto_serving/serving/memory/kv_cache.py#L1447)

当前选择分为两种情况。

**情况一：Prefix Cache 命中。**

`acquire_group_prefix_blocks()` 会先在各 partition 查找可复用的 prefix block。命中后，返回值中已经带有 `request.cache_partition`，后续 allocation 会继续使用这个 partition。此时当前代码不会再按请求数重新选择。

**情况二：没有可用 Prefix Cache 命中。**

请求的 `cache_partition` 仍然是 `None`，`ensure_group_blocks()` 内部调用 `_select_group_partition()`。该函数先调用 `_candidate_group_partitions()`：

1. 遍历所有 grouped KV partition；
2. 过滤已经达到 `max_requests_per_partition` 的 partition；
3. 对每个 KV group 检查 `_additional_group_blocks()` 是否不超过该 partition 的 free block；
4. 在剩余候选中统计已有 request 数；
5. 选择 request 数最少的 partition；
6. 如果多个 partition 请求数相同，用 `_next_group_partition` 做 round-robin。

选择结果只有在 `ensure_group_blocks()` 成功后才通过 `_group_request_partitions` 和 `request.cache_partition` 固定下来。已绑定的请求后续不会重新走自由选择。

A5 的额外约束在 Runner 侧再次检查：[_validate_dp_capacity()](../../pypto-serving/pypto_serving/model/deepseek_v41/npu_runner.py#L778) 按 partition 累加请求数和 Prefill token，最多 12 个请求、最多 4096 个 Prefill token。这个校验发生在 admission 之后，是最终硬校验，不参与当前 partition 的评分。

当前策略可以概括为：

```text
硬容量可行性
  -> 已命中 Prefix Cache 时保持命中 partition
  -> 否则按已有请求数最少选择
  -> 平局 round-robin
```

它没有读取 partition 上请求还剩多少 Prefill/Decode token，也没有把 partition 的 KV usage 或同一轮已经投影的新请求纳入评分。

### 2.2 其他框架的 DP 间请求分配调研

本节调研其他框架如何把请求分配到 DP，作为联合 DPLB 的参照，只记录分配策略本身。

#### 2.2.1 DeepInfer（DeepSeek V4.1 生产推理系统）

DeepInfer 引擎仓库本身没有 DP 间分配逻辑：每个 DP rank 是一个独立 Scheduler（SPMD），请求经 `add_sequence()` 进入本 rank 的 waiting 队列后不会跨 rank 移动（[scheduler.hpp:135](../../DeepInfer/csrc/engine/scheduler.hpp#L135)），组内引擎只通过步型投票 allreduce 锁步（[scheduler.hpp:308](../../DeepInfer/csrc/engine/scheduler.hpp#L308)）。分配由生产系统的 Prefill/Decode Load Balancer 完成，引擎仓库中只有部署文档描述这套策略（[架构总览](../../DeepInfer/doc/inference_deployment_qa.md#L33)）。

Prefill Load Balancer 分三层。

第一层跨 DP 组——按长度分桶，引擎按剩余配额拉取：

```text
请求按总上下文长度进 4 个桶: <16K / 16-64K / 64-256K / >=256K
  -> 每桶引擎数按上一轮该桶 compute-quota 需求占比动态再分配
  -> 每桶有独立的 per-batch compute-quota 上限
  -> 桶内引擎按剩余配额拉取（pull），不是中心路由推（push）
```

（[doc:137-139](../../DeepInfer/doc/inference_deployment_qa.md#L137)）

配额是时间量纲，不是请求数或 token 数。拉取时机由引擎自己定：用当前 step 的剩余时间决定何时拉下一批，取请求和准备与当前 step 的计算重叠，step 结束时批已就绪，TTFT 最小化且引擎满载（[doc:159](../../DeepInfer/doc/inference_deployment_qa.md#L159)）。引擎是唯一知道步内剩余时间的一方，这是 prefill 用 pull 不用 push 的原因。

桶间无前缀亲和路由：任意 prefill 机器都能命中任意前缀。前提是 KVCache Storage——block KV 存在专用 3FS 存储集群而非本地内存（[doc:175](../../DeepInfer/doc/inference_deployment_qa.md#L175)）；命中长度在路由之前由 Entrypoint hash-chain 匹配算出（[doc:191](../../DeepInfer/doc/inference_deployment_qa.md#L191)），直接进入成本模型。

对 A5 的映射：

- A5 是单全局 Scheduler 集中可见，不存在 pull/push 之分；2.4 的集中评分与它的“组内最低配额 rank”同构，差异在量纲。
- 它的配额是时间（ns）加成本模型；本方案的 projected token 评分是其粗化版。V4.1 注意力成本对上下文长度非线性，时间量纲更贴近真实步耗时。
- 长度分桶把长短请求的步时间和排队隔离开，这是 per-request 评分替代不了的；A5 当前 4 个 partition 对长度不设防。
- 无前缀亲和依赖 3FS 跨机 KV 存储；A5 只有单机分组 prefix cache，命中请求必须保持原 partition，这一点不可搬。

#### 2.2.2 vLLM（DPLBAsyncMPClient 内部 LB）

vllm-ascend 仓库本身没有 DP 间分配逻辑——它是 vLLM 核心的 Ascend 插件，分配在 vLLM 前端完成。每个 DP rank 是独立 engine 进程加独立 Scheduler（SPMD）；`data_parallel_size > 1` 且不显式开 external LB 时，前端默认使用内部 LB `DPLBAsyncMPClient`（[core_client.py:172-180](../../vllm/vllm/v1/engine/core_client.py#L172)），`data_parallel_external_lb` 默认 False（[parallel.py:178](../../vllm/vllm/config/parallel.py#L178)）。

没有绑定 rank 的新请求按 score 最小选 rank（[get_core_engine_for_request()](../../vllm/vllm/v1/engine/core_client.py#L1821)）：

```text
score = max(client_count * 本client在该rank的inflight, 快照waiting + 快照running)
if waiting > 0:
    score += waiting * 6.0 * max(0, kv_cache_usage - 0.5)
```

- 快照来自独立 DPCoordinator 进程：每个 engine 每步上报 `num_waiting_reqs / num_running_reqs / kv_cache_usage`（[coordinator.py:406-408](../../vllm/vllm/v1/engine/coordinator.py#L406)），最快每 100ms 向前端广播一次（[coordinator.py:152](../../vllm/vllm/v1/engine/coordinator.py#L152)）。
- 快照可能过期，所以用 `client_count * 本 client inflight`（所有前端均匀发包时的全局在途估计）做精确下限兜底，burst 期间也能轮转开（[core_client.py:1856](../../vllm/vllm/v1/engine/core_client.py#L1856)）。
- waiting 惩罚从 50% KV usage 起线性增长，100% 时为 3 倍 waiting：排队出现在 KV 受压 rank 上时，让其他 rank 明显更优（[core_client.py:1864](../../vllm/vllm/v1/engine/core_client.py#L1864)）。
- 平局处理：选中后本地计数立即 +1，同一快照窗口内的后续请求看到更新（[core_client.py:1870](../../vllm/vllm/v1/engine/core_client.py#L1870)）；扫描起点每请求旋转一次，等负载时等价 round-robin 且无固定偏置（[core_client.py:1876](../../vllm/vllm/v1/engine/core_client.py#L1876)）。
- streaming-input 会话亲和原 rank；普通请求选定后粘在该 rank 直到结束，不迁移（[core_client.py:1828](../../vllm/vllm/v1/engine/core_client.py#L1828)）。
- 大 DP 下源码注明待改 P2C（[core_client.py:1839](../../vllm/vllm/v1/engine/core_client.py#L1839)）。

对 A5 的映射：

- 2.4 的 `kv_pressure = 6 * projected_waiting * max(0, kv_usage - 0.5)` 与它的 waiting 惩罚项完全一致，本方案评分即源于此。
- vLLM 前端只能看到 100ms 快照，需要本地 inflight 兜底；A5 单全局 Scheduler 每轮 admission 前重建画像，2.3 的 projected 更新即对应它的本地计数，无陈旧问题。
- 它同样只在 admission 时决策一次、请求粘住不迁移；vllm-ascend 在 rank 级别另加了两个默认关闭的事后修正扩展（balance scheduling 全局冻结闸门、DyntraLB KV 保留式暂停再均衡），说明纯 admission 评分不足以长期维持 DP 均衡——与 A5 已绑定请求不可迁移的问题同源。

### 2.3 我们要实现的联合 DPLB

优化只改变“没有 partition 的全新请求首次 admission”。它不拆分全局 waiting/running 队列，也不迁移已经绑定 partition 的请求。

Scheduler 在每个 `schedule()` 的 waiting admission 前构建 partition 画像。画像必须同时保留以下信息：

```text
partition[p].running_prefill_count
partition[p].running_decode_count
partition[p].waiting_prefill_count
partition[p].waiting_decode_count
partition[p].running_prefill_tokens
partition[p].running_decode_tokens
partition[p].waiting_prefill_tokens
partition[p].waiting_decode_tokens
partition[p].projected_waiting
partition[p].kv_usage
```

字段取值必须来自当前 Scheduler/Request 对象：

- `Scheduler.running` 和 `Scheduler.waiting` 是请求集合；
- `Request.cache_partition` 决定已有请求属于哪个 partition；
- `Request.is_prefill` 区分 Prefill 和 Decode；
- Prefill 使用 `Request.num_new_tokens_needed` 作为剩余 token；
- 普通 Decode demand 为 1；
- DSpark Decode 使用 Scheduler 为本轮预留的 `1 + num_speculative_tokens`，并且只计入 token load 一次；
- `kv_usage` 从 grouped KV 各 partition 的 free queue 计算，取各 KV group 中压力最高者。

没有 `cache_partition` 的新请求仍保留在全局 waiting，不写入任何真实 partition 画像。评估候选 partition `p` 时，才临时加入该请求：

```text
projected_pending_tokens[p]
    = current_pending_tokens[p] + demand(new_request)

projected_waiting[p]
    = current_projected_waiting[p] + 1
```

如果同一轮已经成功为前一个新请求选择了 partition，后面的新请求必须看到前一个请求更新后的 projected 值。

### 2.4 联合评分和候选顺序

对每个候选 partition，先继续执行当前 `_candidate_group_partitions()` 的硬过滤。只有通过 block/request capacity 的 partition 才计算 score：

queue_pressure = current_request_count + projected_waiting

token_pressure = projected_pending_tokens / token_scale

kv_pressure = 6 * projected_waiting * max(0, kv_usage - 0.5)

score = queue_pressure + token_pressure + kv_pressure

其中：

- `current_request_count` 来自 grouped KV 当前已经绑定的请求；
- `projected_waiting` 只表示本轮已投影的新请求，不把全局 waiting backlog 重复加到每个 partition；
- `projected_pending_tokens` 同时包含 running/waiting 已绑定请求的剩余 token 和候选新请求 demand；
- Prefill demand 是剩余 Prefill token；
- 普通 Decode demand 是 1；
- DSpark Decode demand 是本轮预留的 token 数；
- `token_scale` 使用 A5 的 `SchedulerConfig.max_num_scheduled_tokens`，也就是 CLI 计算出的 `step_tokens`，只用于归一化；
- 4096 是 Runner 的单 partition Prefill 硬校验，不能写成新的 Scheduler 每轮预算；
- Decode 已包含在 projected token 中，不能再次增加独立 decode 项。

score 最小的合法 partition 获选，score 相同才使用现有 round-robin。Prefix Cache 命中请求保持命中 partition；只有 grouped hit 因容量不足释放、重新走 cold fallback 时，才作为无 partition 请求重新评分。

完整字段、代码 patch 和验证用例见 [DPLB 实现方案](./DPLB-realize/implementation-plan.md)。

## 3. 抢占问题与修复方案

### 3.1 当前 A5 的抢占流程

抢占发生在已经属于 `running` 的请求继续申请 KV block 时，不发生在 DPLB 的首次 partition 选择阶段。当前调用链是：

```text
Scheduler._schedule() Phase 1
  -> 处理 running 请求
  -> _try_allocate_request_blocks()
  -> grouped KV allocation 失败
  -> _preempt_lowest_priority()
  -> 释放一个 victim
  -> 再次 _try_allocate_request_blocks()
  -> 仍失败则本轮 defer 当前请求
```

源码位置：

- running 调度和首次抢占：[Scheduler._schedule()](../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L466)
- KV 分配：[Scheduler._try_allocate_request_blocks()](../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L1251)
- victim 选择：[Scheduler._preempt_lowest_priority()](../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L1283)
- block 释放：[Scheduler._free_request_blocks()](../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L1355)

当前逻辑一次只尝试一个 victim：如果释放后第二次分配仍失败，当前请求就留在 `running_to_keep`，等待后续 scheduler step。一个请求需要释放多个 victim 时，不能在同一轮完成分配。

### 3.2 当前 victim 的合法性和选择规则

`_preempt_lowest_priority()` 不是任意选择请求。当前候选必须满足：

1. 不是触发分配失败的 `exclude` 请求；
2. `_recompute_rejection(request) is None`，请求可以通过本地重算恢复；
3. `not request.prefill_complete`；
4. `not request.scheduling_held`；
5. grouped KV 下，满足触发请求的 arrival-time 保护；
6. grouped KV 下，如果触发请求已有 partition，victim 必须属于同一 partition。

过滤之后当前使用：

```python
victim = max(candidates, key=lambda request: request.arrival_time)
```

也就是说，当前主要按 arrival time 选择，没有比较 prompt 长度、已经生成 token 数或预计 replay 工作量。

### 3.3 抢占后的实际状态清理

当前 victim 会经过 `_free_request_blocks()`，释放 generic/grouped KV block，并清空 `cache_partition`。随后 `_preempt_lowest_priority()` 还会重置：

```text
status = PREEMPTED
num_replay_tokens = len(output_token_ids)
num_computed_tokens = 0
cached_block_ids / allocated_block_ids
allocated_group_block_ids
cache_partition
num_blocks_cached / num_group_blocks_cached
prefix_cache_hit_tokens
num_output_placeholders
terminal_prefill_in_flight
```

victim 会被放回公共 `waiting` 队列。当前同一轮通过 `preempted_this_step` 跳过它，避免旧 worker 结果尚未处理时立即重新 admission。后续重新 admission 时，Prefix Cache 再决定哪些 immutable prompt 页面可以复用，不能假设所有上下文都能直接复用。

### 3.4 其他框架的抢占策略对比

本节对照 vLLM v1 与 DeepInfer 的抢占策略，作为 3.5 修复方案的参照。

#### 3.4.1 vLLM v1

running 请求分配 KV 失败后进入抢占循环（[scheduler.py:873](../../vllm/vllm/v1/core/sched/scheduler.py#L873)）：

- **多 victim 循环**：抢一个 victim 后立刻重试分配，仍失败继续抢，直到成功或没有合法 victim，不存在“释放一个仍不够就 defer”的问题。
- **victim 选择**：默认 FCFS 抢 running 队尾（最晚进入 running，抢新护老）；PRIORITY 策略下抢 `max(running, key=(priority, arrival_time))`（[scheduler.py:891](../../vllm/vllm/v1/core/sched/scheduler.py#L891)）。完全不看 victim 的重算成本。
- **被抢者一律重算**：无 stop 模式，释放全部 block、`num_computed_tokens = 0`、插回 waiting 队头（[scheduler.py:1786](../../vllm/vllm/v1/core/sched/scheduler.py#L1786)）。PD 分离部署的 decode 侧同样是全功能 Scheduler，被抢后也是重算，差异只在 in-flight 输出经 `drop_stale_output` 丢弃、connector 仍持有外部 KV 时重新 admission 可再拉取（[base.py:205](../../vllm/vllm/v1/distributed/kv_transfer/kv_connector/v1/base.py#L205)）。
- **已调度 victim 回滚** token/encoder 预算（[scheduler.py:910](../../vllm/vllm/v1/core/sched/scheduler.py#L910)）；connector 场景另有 pending-free 检查和延迟释放保护（[scheduler.py:901](../../vllm/vllm/v1/core/sched/scheduler.py#L901)），A5 没有 external connector，不需要对应物。

#### 3.4.2 DeepInfer

DeepInfer 在抢占发生前先有事前防线（[scheduler.hpp](../../DeepInfer/csrc/engine/scheduler.hpp)）：

- **事前防线**：KV 使用率超过 `GPU_CACHE_PREFILL_WATERMARK = 0.9` 时本轮跳过 prefill 步，先让 decode 消化存量（常量定义 [scheduler.hpp:80](../../DeepInfer/csrc/engine/scheduler.hpp#L80)，投票判断 [scheduler.hpp:313](../../DeepInfer/csrc/engine/scheduler.hpp#L313)）；另有 Σseqlen_k 批预算与 prompt 超长直接拒收（[scheduler.hpp:112](../../DeepInfer/csrc/engine/scheduler.hpp#L112)）。A5 和 vLLM 都没有这一层。
- **victim 选择**：running 按已生成长度降序排序，从最长的开始连抢（[scheduler.hpp:622](../../DeepInfer/csrc/engine/scheduler.hpp#L622)），一次 victim 释放最多 KV。
- **被抢者两种下场**（[scheduler.hpp:766](../../DeepInfer/csrc/engine/scheduler.hpp#L766)）：NORMAL 引擎 reset 后回 waiting 队头重算；COMPLETION 引擎（生产 PD 分离的 decode 侧）强制 `stop_preempted_request=true`（[scheduler.hpp:102](../../DeepInfer/csrc/engine/scheduler.hpp#L102)），被抢请求直接终止、不重算。

#### 3.4.3 三方对照

| 维度 | A5 现状 | A5 修复方案 | vLLM v1 | DeepInfer |
| --- | --- | --- | --- | --- |
| victim 数量 | 1 个就停 | 循环抢到够 | 循环抢到够 | 循环抢到够 |
| victim 选择 | 最晚 arrival | replay cost 最小，平局最晚 arrival | 最晚 arrival / priority | 已生成最多 |
| 被抢者下场 | 重算 | 重算 | 重算 | 重算或直接终止 |
| 事前防线 | 无 | 无（不在本次范围） | 无 | watermark 0.9 + 预算 |

对 3.5 修复方案的映射：

- **循环框架取自 vLLM**：victim 从 running 删除并回 waiting 后，循环不会重复选中同一请求；两家都按这个前提实现。
- **victim 度量三方各自自洽**：vLLM 完全不看成本；DeepInfer 抢已生成最多的（有 stop 模式兜底，抢最长的释放最多内存）；A5 被抢者必须全量重算，牺牲 replay cost 最小的请求浪费最小，所以反向取最小。
- **反复牺牲风险**：A5 没有 `num_preemptions` 字段，replay cost 最小的短请求可能被反复牺牲；平局时的最晚 arrival 规则只能在成本相同的请求间轮转牺牲对象，vLLM 的“永远抢最新”反而天然分散。
- **共同缺口**：只有 DeepInfer 有事前防线，A5 与 vLLM 都在分配失败后才动作；本次修复不引入 watermark/预算，不作为目标。

### 3.5 修复方案

抢占修复只改变“释放谁以及释放几次”，不改变 DPLB 的新请求 partition 评分。

分配失败后改成循环：

```text
尝试分配当前请求
  -> 选择一个合法 victim
  -> 释放 victim
  -> 回收本轮已调度 token budget
  -> 再次尝试分配当前请求
  -> 仍不足则继续选择下一个 victim
  -> 成功或候选耗尽时结束
```

victim 仍然必须经过当前的所有合法性过滤。合法候选的排序改为：

```python
victim = min(
    candidates,
    key=lambda request: (
        request.num_prompt_tokens + len(request.output_token_ids),
        -request.arrival_time,
    ),
)
```

含义是优先选择 replay cost 较小的请求；成本相同时保留原来优先选择较晚 arrival 请求的规则。Grouped KV 仍然不能跨 partition 抢占，也不能搬运 live request 的 KV 或 DSpark state。

如果 victim 已经在当前 schedule 的 output 中被调度，现有回滚逻辑必须保留：从 scheduled request 集合移除、返还 token budget、释放 snapshot，并回滚 Prefill/Decode 统计。

完整 patch 和验证用例见 [抢占修复方案](./抢占修复/preemption-repair-plan.md)。

## 4. 代码改动范围

A5 Serving 只改通用调度层：

- pypto_serving/serving/memory/kv_cache.py
- pypto_serving/serving/sched/scheduler.py

不改：

- V4.1 model forward；
- page table、slot、generation 和 DSpark ABI；
- pypto-lib 计算图；
- V4.1 Runner 的硬容量约束。


