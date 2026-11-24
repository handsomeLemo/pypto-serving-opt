# A5 V4.1 Serving：抢占修复代码方案

本文只讨论抢占。抢占的职责是：KV 容量不足时释放已有请求，让当前请求继续分配。新请求首次选择 partition 属于 DPLB，见 [DPLB 联合评分实现](../DPLB-realize/implementation-plan.md)。

## 1. 当前源码问题

在 [_schedule()](../../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L457) 中，running 请求分配 KV 失败后只调用一次 [_preempt_lowest_priority()](../../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L1283)。释放一个 victim 后仍不足时，本轮直接 defer。

当前合法候选仍必须满足：

- _recompute_rejection(request) is None；
- 未完成 Prefill；
- 未处于 scheduling hold；
- grouped KV 下与触发请求属于同一 partition；
- arrival time 保护条件。

这些过滤条件必须保留。

## 2. 多 victim patch

将当前单次逻辑改为：

~~~python
allocated = self._try_allocate_request_blocks(
    request, scheduled_tokens, scheduled=scheduled
)
while not allocated:
    preempted = self._preempt_lowest_priority(
        request,
        scheduled_req_ids,
        num_scheduled_tokens,
        output,
    )
    if preempted is None:
        break

    token_budget += preempted.get("returned_tokens", 0)
    output.preempted_requests.append(preempted["request"])
    allocated = self._try_allocate_request_blocks(
        request, scheduled_tokens, scheduled=scheduled
    )

if not allocated:
    reason = self._recompute_rejection(request)
    if reason is not None:
        stalled_non_recomputable.append((request, reason))
    running_to_keep.append(request)
    continue
~~~

每次 victim 都从 self.running 删除并放回 self.waiting，所以循环不会重复选择同一个请求。

如果 victim 已经在本轮 output 中被调度，现有 _preempt_lowest_priority() 的以下逻辑必须保留：

- 从 scheduled_req_ids 移除；
- 从 num_scheduled_tokens 取回 token budget；
- 释放 scheduled snapshot；
- 从 output.scheduled_requests 删除；
- 回滚 num_prefill_tokens 或 num_decode_tokens。

## 3. replay cost 选择 patch

保留现有候选过滤，只把：

~~~python
victim = max(candidates, key=lambda r: r.arrival_time)
~~~

改成：

~~~python
victim = min(
    candidates,
    key=lambda r: (
        r.num_prompt_tokens + len(r.output_token_ids),
        -r.arrival_time,
    ),
)
~~~

含义：

- 优先选择 prompt + 已生成 token 较少的请求；
- replay cost 相同，继续优先选择 arrival time 较晚者；
- 不改变 grouped KV 的同 partition 约束；
- 不跨 partition 搬运 KV 或 DSpark state。

## 4. 状态清理必须保留

现有 _preempt_lowest_priority() 中的清理代码不能删除：

~~~python
self._free_request_blocks(victim)
victim.status = RequestStatus.PREEMPTED
victim.num_replay_tokens = len(victim.output_token_ids)
victim.num_computed_tokens = 0
victim.cached_block_ids = []
victim.allocated_block_ids = []
victim.allocated_group_block_ids = {}
victim.cache_partition = None
victim.num_blocks_cached = 0
victim.num_group_blocks_cached = {}
victim.prefix_cache_hit_tokens = 0
victim.num_output_placeholders = 0
victim.terminal_prefill_in_flight = False
~~~

A5 的 Prefix Cache 只能由后续 admission 重新判断可复用页面，不能直接假设所有 prompt KV 都能复用。

## 5. 不做的改动

- 不建立每个 DP 一个 Scheduler；
- 不把抢占变成跨 partition 迁移；
- 不增加 A5 不存在的 native wave 或 dispatch_tokens_per_dp；
- 不在当前 async_scheduling=False 配置下引入 in-flight step 假设；
- 不修改 V4.1 Runner 的 slot/generation ABI。

## 6. 验证用例

1. 当前请求需要释放两个 victim 时，同一个 schedule() 内完成分配；
2. replay cost 较小的合法 victim 优先被选择；
3. 不同 partition 的 victim 不会被互相抢占；
4. 已调度 victim 的 token budget、output 和 snapshot 正确回滚；
5. victim 重新 admission 后 Prefix Cache 和 num_replay_tokens 状态正确；
6. 没有合法 victim 时仍按原逻辑 defer/reject。

