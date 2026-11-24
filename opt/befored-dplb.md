A3 DPLB 调度现状与优化建议
本文只讨论 A3 的请求调度和 DP partition 选择，不讨论模型算子和通信实现细节。
当前结构
1.1 一个 Scheduler，四个 DP partition
A3 V4.1 使用 DP4/TP4/EP16 和 placement_mode="overlapped"。四个 DP 是同一个 16-rank native worker 内的逻辑 partition，不是四个独立的 serving replica。
overlapped 模式下，ParallelConfig.num_replicas 固定为 1：parallel.py:107。
AsyncLLMEngine 因此只创建一个 ReplicaEngineCore：async_engine.py:1726。
一个 ReplicaEngineCore 只创建一个通用 Scheduler：async_engine.py:247。
这个 Scheduler 维护一套全局 waiting 队列、running 列表和全局 max_num_scheduled_tokens：scheduler.py:455。
因此，A3 当前没有“每个 DP 一个 Scheduler”的 DPLB。它是：
一个全局 Scheduler
    -> 统一选择本轮请求
    -> 为请求选择一个 cache partition（DP0~DP3）
    -> 一个 native session 让 16 个 rank 共同执行

1.2 当前请求分配流程
新请求进入全局 waiting 队列。Scheduler 先按全局 token budget 选择请求；Grouped KV Cache 在为请求分配 KV block 时，才选择具体 partition：
Scheduler.schedule() -> _try_allocate_request_blocks() -> KvCacheManager._select_group_partition()
当前 partition 选择主要考虑：
partition 是否有足够的 KV block；
partition 是否达到 request slot 上限；
已绑定到该 partition 的请求数是否最少；
请求数相同则轮询选择。
_select_group_partition() 的实现只统计 request_partitions 的数量，没有统计该 partition 上所有请求还剩多少 Prefill/Decode token：kv_cache.py:1300。
请求一旦绑定 partition，A3 native ledger 会继续绑定 dp、slot、generation。活跃请求的 DP 不能直接改变：runner.py:86、runner.py:105。
1.3 Prefill/Decode 约束
A3 配置要求一个 Scheduler step 只能包含一种 phase：
requires_homogeneous_prefill_decode = True
supports_async_scheduling = False

见 cli.py:74。
当全局同时存在 Prefill 和 Decode 时，Scheduler 在两个 phase 之间轮转，而不是让不同 DP partition 同时运行不同 phase：scheduler.py:816。
例如：
本轮选择 Decode：DP0 有 Decode，DP1~DP3 inactive
下一轮选择 Prefill：DP1 有 Prefill，DP0、DP2、DP3 inactive

空闲 partition 仍要进入同一次 native collective，但不会执行另一个 phase 的请求。
当前 DPLB 问题
当前真正的负载均衡位置不是“多个 Scheduler 之间路由”，而是“一个 Scheduler 为新请求选择 DP partition”。目前的选择指标接近：
partition 请求数 + KV block 可用性 + 轮询

这会产生以下问题：
两个 partition 都有 2 个尚未完成的 Prefill 请求，但其中一个 partition 可能还剩大量 token，另一个只剩很少 token；当前逻辑仍认为两者同样繁忙。
长 Prefill 连续到达时，按请求数均衡不等于按 token 工作量均衡。
请求数看起来均衡，但某个 partition 的后续 native dispatch 更容易超过 dispatch_tokens_per_dp，从而产生更多串行 wave。
已经绑定 partition 的请求不能迁移，因此首次分配错误会持续影响整个请求生命周期。
优化点
3.1 在首次 admission 时做 token-aware partition 选择
保留 Scheduler 原有的一套全局 waiting 队列和 running 列表，不把它们拆成四个 DP 队列。新增的是一份按 DP partition 聚合的负载画像，用于新请求首次绑定 partition 时做决策。
画像的来源分为两类：
已处理且未被抢占的请求已经有 cache_partition，下一轮仍归属于原 DP；它们无论本轮是否被选中执行，都要按原 partition 计入画像；
只有首次到达、尚未绑定 partition 的新请求没有 DP 归属。它们先留在全局 waiting 队列，路由计算时才对候选 DP 形成“投影负载”；
如果请求被抢占，Scheduler 会释放它的 KV 并清空 cache_partition，它重新排队后才重新选择 DP。
对一个全新请求，先从全局负载画像计算候选 DP 的得分，选定 DP 后再进入现有的 KV block 分配和 Scheduler 流程。
这里的“真实负载”和“投影负载”含义如下：
已有 partition 的请求：
    计入对应 DP 的真实负载

没有 partition 的新请求：
    继续留在全局 waiting 队列
    只在评估候选 DP 时临时加入计算
    选择完成并且 KV 分配成功后，才正式绑定 cache_partition

例如当前状态为：
DP0：请求 A，已绑定 DP0，剩余 3000 个 Prefill token
DP1：请求 B，已绑定 DP1，剩余 500 个 Prefill token
全局 waiting：新请求 C，尚未绑定 DP，需要 1000 个 Prefill token

调度器不会先把 C 放入某个 DP 的 waiting 队列，而是分别计算：
假设 C 放到 DP0：DP0 预测负载 = 3000 + 1000 = 4000
假设 C 放到 DP1：DP1 预测负载 =  500 + 1000 = 1500

因此 C 选择 DP1。KV block 分配成功后，C 才获得 cache_partition=1，从下一轮开始作为 DP1 的真实负载统计。若 KV 分配失败，则可以尝试其他满足硬容量约束的候选 DP。
如果同一轮连续接纳多个新请求，还要维护临时的 projected_waiting[p]。例如第一个新请求已经被预测分配给 DP1，那么评估第二个新请求时，DP1 的预测负载必须先加上第一个请求，避免多个新请求同时判断 DP1 最空闲并集中到同一个 partition。
第一步：硬约束过滤
候选 partition 必须同时满足：
KV block 足够
active request 数 < slots_per_dp
本轮分配 token 不超过 dispatch_tokens_per_dp

第二步：计算每个 partition 的当前负载
对 partition p 统计：
running_prefill_count[p]
running_decode_count[p]
running_prefill_tokens[p] = Σ(max(0, num_prefill_tokens - num_computed_tokens))
running_decode_tokens[p]  = running_decode_count[p]       # 无投机解码时每个为 1
pending_tokens[p] = running_prefill_tokens[p] + running_decode_tokens[p]

对于全局 waiting 队列，额外维护：
global_waiting_prefill_count
global_waiting_decode_count
global_waiting_prefill_tokens
global_waiting_decode_tokens

当评估新请求 r 是否分配给 partition p 时，先计算将 r 放入 p 后的投影值：
projected_running_count[p] = running_count[p] + 1
projected_pending_tokens[p] = pending_tokens[p] + demand(r)

demand(r) =
    prompt_remaining(r)       # 新 Prefill
    或 1                      # 无投机 Decode

如果同一轮连续路由多个新请求，还要把已经选定但尚未真正下发的请求记录为 projected_waiting[p]，避免它们在同一轮都被判断为落在最空闲的 DP。
然后采用 vLLM DPLB 的“队列压力 + KV 压力”思路，并加入 A3 的 token 工作量：
queue_pressure[p] =
    max(inflight_steps[p],
        running_count[p] + projected_waiting[p])

score(p) =
      queue_pressure[p]
    + a * projected_pending_tokens[p] / dispatch_tokens_per_dp
    + b * projected_waiting[p] * max(0, kv_usage[p] - 0.5)
    + c * projected_decode_count[p]

选择 score 最小的 partition；分数相同再轮询。
global_waiting_* 反映全局 backlog，projected_waiting[p] 才用于比较不同 DP 的投影差异。kv_usage[p] 需要按 partition 的 block 使用情况计算；当前代码只用剩余 block 判断候选是否可用。
这里的 vLLM 公式是参考其 Engine 级路由思路，不是说 vLLM 当前已经按 token 做 DPLB。vLLM 当前内部 LB 的实际评分是：
score = max(client_count * local_inflight, waiting + running)
if waiting > 0:
    score += waiting * 6 * max(0, kv_cache_usage - 0.5)

源码见 core_client.py:1821。该公式使用 waiting、running、KV usage 和 in-flight，不使用请求剩余 token；A3 需要在此基础上增加 partition 级 pending_tokens。
3.2 不改变的部分
这个优化只改变新请求第一次绑定哪个 partition，不改变以下约束：
已经进入 native ledger 的请求不迁移；
dp、slot、generation 必须保持一致；
一个 Scheduler step 仍然只能是 Prefill 或 Decode；
16 个 rank 仍共同执行 native collective；
dispatch_tokens_per_dp 仍是 native 层的硬容量限制。
如果发生抢占，Scheduler 会释放原请求的 KV、清空 cache_partition 并重新排队：scheduler.py:1249。此时请求需要重新 Prefill，才可能被分配到另一个 partition。
可能有收益的场景
4.1 长短 Prefill 混合
DP0：2 个长 Prefill，剩余 6000 token
DP1：2 个短 Prefill，剩余 200 token
DP2：空闲
DP3：空闲

当前按请求数选择时，DP0 和 DP1 都是“2 个请求”。加入 pending_tokens 后，后续新 Prefill 会优先进入 DP2/DP3，减少长 Prefill 继续集中到 DP0 的概率。
4.2 并发长 Prefill 到达
多个长 prompt 在短时间内到达时，按请求数轮询可能让不同 partition 的 token 总量差异很大。token-aware 选择可以使各 partition 的剩余工作量更接近，降低某个 partition 形成连续多 wave 的概率。
4.3 Decode 负载较重的 partition 接收新 Prefill
某个 partition 可能请求数不多，但其中已有请求还要持续 Decode。在下一次 Prefill phase 进行新请求 admission 时，加入 pending_decode 或 decode_requests 可以避免新 Prefill 继续压到这个 partition，减少后续 Decode backlog 对整体吞吐的影响。这里不表示同一个 Scheduler step 同时执行 Prefill 和 Decode。
4.4 KV 压力不均
两个 partition 的请求数和 token 数接近，但 KV block 使用率不同。加入 kv_usage 后，可以优先使用 KV 更充足的 partition，减少因局部 KV 不足导致的等待或抢占。
收益边界与验证指标
收益主要出现在“各请求长度差异明显、多个 partition 同时有可用 slot、且当前存在长 Prefill 或 Decode backlog”的场景。
以下场景收益有限：
所有请求 prompt 和生成长度几乎相同；
只有一个 partition 有可用 KV/slot；
请求数很少，partition 本身没有形成明显不均衡；
性能瓶颈来自全局 homogeneous phase 或 16-rank collective，而不是 partition 分配。
验证时至少记录：
每个 partition 的 active_requests
每个 partition 的 pending_prefill/pending_decode/pending_tokens
每个 partition 的 KV 使用率
每个 Scheduler step 的 native wave 数
每个 partition 的实际 dispatch token 数
Prefill 和 Decode 的等待时间

只有同时观察这些指标，才能确认收益来自 DP partition 负载均衡，而不是仅仅减少了 host 调度开销。