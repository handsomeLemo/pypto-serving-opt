# A5 V4.1 Serving：DPLB 联合评分代码方案

本文给出可直接在当前 A5 Serving 源码上实现的 patch 方案，但只写入文档，不直接修改源码。DPLB 的职责是给没有 partition 的新请求选择首次 cache partition。抢占见 [抢占修复方案](../抢占修复/preemption-repair-plan.md)。

## 1. 真实接入边界

A5 V4.1 在 [_build_v41_engine_config()](../../../pypto-serving/pypto_serving/cli/main.py#L380) 中构造一个全局 Scheduler：

- DP/EP 为 4 或 8，TP 为 1；
- grouped KV 的 partition 数等于 DP 数；
- max_num_batched_tokens 被收敛为 step_tokens；
- requires_homogeneous_prefill_decode=True；
- async_scheduling=False。

Runner 最终仍会在 [_validate_dp_capacity()](../../../pypto-serving/pypto_serving/model/deepseek_v41/npu_runner.py#L778) 校验每个 partition 的请求数和 Prefill token 数。因此 DPLB 只能改变 admission 选择，不能绕过 Runner 或 allocator。

## 2. 当前源码路径

当前 waiting 请求在 [Scheduler._schedule()](../../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L457) 的 Phase 2 处理：

1. Prefix Cache lookup；
2. 计算本轮 Prefill chunk；
3. 调用 [_try_allocate_request_blocks()](../../../pypto-serving/pypto_serving/serving/sched/scheduler.py#L1251)；
4. grouped KV 下进入 [ensure_group_blocks()](../../../pypto-serving/pypto_serving/serving/memory/kv_cache.py#L1491)；
5. 没有已有 partition 时由 [_select_group_partition()](../../../pypto-serving/pypto_serving/serving/memory/kv_cache.py#L1447) 选择。

当前 _select_group_partition() 只按已绑定请求数选择最小 partition，再 round-robin。

## 3. 联合画像字段

在 Scheduler 内增加本轮临时画像，不增加每个 DP 的独立队列。画像必须区分 running/waiting 和 Prefill/Decode：

~~~python
@dataclass
class PartitionLoad:
    running_prefill_count: int = 0
    running_decode_count: int = 0
    waiting_prefill_count: int = 0
    waiting_decode_count: int = 0
    running_prefill_tokens: int = 0
    running_decode_tokens: int = 0
    waiting_prefill_tokens: int = 0
    waiting_decode_tokens: int = 0
    projected_waiting: int = 0
    kv_usage: float = 0.0

    @property
    def running_count(self):
        return self.running_prefill_count + self.running_decode_count

    @property
    def waiting_count(self):
        return self.waiting_prefill_count + self.waiting_decode_count

    @property
    def pending_tokens(self):
        return (
            self.running_prefill_tokens
            + self.running_decode_tokens
            + self.waiting_prefill_tokens
            + self.waiting_decode_tokens
        )
~~~

字段来源必须对应当前源码：

| 画像字段 | 源码来源 | 统计规则 |
| --- | --- | --- |
| running/waiting | `Scheduler.running` / `Scheduler.waiting` | 遍历当前请求对象 |
| partition | `Request.cache_partition` | 有值才进入真实 partition 画像 |
| Prefill/Decode | `Request.is_prefill` | `num_computed_tokens < num_prefill_tokens` 为 Prefill |
| Prefill 剩余 token | `Request.num_new_tokens_needed` | 对 Prefill 计入剩余 token |
| 普通 Decode token | `1` | 无投机时每个 Decode 请求一次 token |
| DSpark Decode token | `1 + speculative_reserved` | 按当前 Scheduler 本轮预留数量计入一次 |
| KV usage | `KvCacheManager` grouped pool free queue | 按 partition 计算各 group usage，取最高压力 |

画像构建代码应明确区分 running 和 waiting：

~~~python
def _partition_demand(self, request):
    if request.is_prefill:
        return max(1, request.num_new_tokens_needed)
    speculative = 0
    if request.temperature <= 0.0:
        speculative = min(
            self.config.num_speculative_tokens,
            max(0, self.config.max_seq_len - request.num_computed_tokens - 1),
        )
    return 1 + speculative

def _add_profile_request(self, profile, request, *, waiting):
    if request.status.is_finished or request.cache_partition is None:
        return
    item = profile[request.cache_partition]
    demand = self._partition_demand(request)
    if request.is_prefill:
        if waiting:
            item.waiting_prefill_count += 1
            item.waiting_prefill_tokens += demand
        else:
            item.running_prefill_count += 1
            item.running_prefill_tokens += demand
    else:
        if waiting:
            item.waiting_decode_count += 1
            item.waiting_decode_tokens += demand
        else:
            item.running_decode_count += 1
            item.running_decode_tokens += demand

def _build_partition_profile(self):
    profile = {
        p: PartitionLoad()
        for p in range(self.kv_cache_manager.group_partition_count)
    }
    for request in self.running:
        self._add_profile_request(profile, request, waiting=False)
    for request in self.waiting:
        self._add_profile_request(profile, request, waiting=True)
    for partition, metrics in self.kv_cache_manager.group_partition_metrics().items():
        profile[partition].kv_usage = metrics["kv_usage"]
    return profile
~~~

这里的 waiting 只统计**已经有 partition 的 waiting 请求**。没有 partition 的新请求仍然保留在公共 waiting 队列，不写入任何真实 DP 画像；它只在候选评分时形成 projected load。

全局 waiting backlog 另行统计，不能冒充某个 partition 的真实 waiting：

~~~python
global_waiting_prefill_count = sum(
    request.is_prefill for request in self.waiting
    if request.cache_partition is None
)
global_waiting_decode_count = sum(
    not request.is_prefill for request in self.waiting
    if request.cache_partition is None
)
global_waiting_prefill_tokens = sum(
    request.num_new_tokens_needed for request in self.waiting
    if request.cache_partition is None and request.is_prefill
)
global_waiting_decode_tokens = sum(
    self._partition_demand(request) for request in self.waiting
    if request.cache_partition is None and not request.is_prefill
)
~~~

这些全局值用于记录 backlog 和监控；候选 partition 间的差异使用 `projected_waiting[p]`，不能把所有全局 waiting 重复加到每个 partition。

## 4. KvCacheManager patch

在 group_partition_count 后增加只读 partition metrics：

~~~python
def group_partition_metrics(self):
    if not self._group_pools:
        return {0: {"request_count": 0, "kv_usage": self.usage}}

    owned = tuple(self._group_request_partitions.values())
    metrics = {}
    for partition in range(self.group_partition_count):
        request_count = sum(owner == partition for owner in owned)
        usages = []
        for pool in self._group_pools.values():
            total = pool.blocks_per_partition
            free = pool.num_free_blocks_in(partition)
            usages.append((total - free) / total if total else 0.0)
        metrics[partition] = {
            "request_count": request_count,
            "kv_usage": max(usages, default=0.0),
        }
    return metrics
~~~

将原 _select_group_partition() 改成委托函数，并增加评分入口：

~~~python
def _select_group_partition(self, request_id, token_count):
    return self.select_group_partition(request_id, token_count)

def select_group_partition(
    self,
    request_id,
    token_count,
    *,
    projected_token_load=None,
    projected_waiting=None,
    token_scale=1,
):
    candidates = self._candidate_group_partitions(request_id, token_count)
    if not candidates:
        return None

    assigned = self._group_request_partitions.get(request_id)
    if assigned is not None:
        return assigned

    metrics = self.group_partition_metrics()
    projected_token_load = projected_token_load or {}
    projected_waiting = projected_waiting or {}
    token_scale = max(1, int(token_scale))

    def score(partition):
        metric = metrics[partition]
        waiting = projected_waiting.get(partition, 0)
        queue_pressure = metric["request_count"] + waiting
        token_pressure = projected_token_load.get(partition, 0) / token_scale
        kv_pressure = 6.0 * waiting * max(0.0, metric["kv_usage"] - 0.5)
        return queue_pressure + token_pressure + kv_pressure

    minimum = min(score(partition) for partition in candidates)
    least_loaded = {
        partition for partition in candidates
        if math.isclose(score(partition), minimum, abs_tol=1e-12, rel_tol=0.0)
    }
    for offset in range(self.group_partition_count):
        partition = (self._next_group_partition + offset) % self.group_partition_count
        if partition in least_loaded:
            self._next_group_partition = (partition + 1) % self.group_partition_count
            return partition
    raise RuntimeError("group partition selection became inconsistent")
~~~

这里仍然先执行 _candidate_group_partitions()，所以：

- 请求数上限不被绕过；
- KV block 不足不会因为评分而强行选择；
- 已绑定 partition 不会迁移；
- 原 round-robin 仍只负责最终平局。

## 5. Scheduler admission patch

让 _try_allocate_request_blocks() 接收一个可选 partition，但必须保留当前函数中的 Prefill history、snapshot 和异常回收逻辑。只修改 grouped KV 的调用参数：

~~~python
def _try_allocate_request_blocks(
    self, request, num_new_tokens, *, scheduled=None, partition=None,
):
    if self.kv_cache_manager.has_groups:
        total_tokens = request.num_computed_tokens + num_new_tokens
        read_blocks, read_snapshot = ({}, None)
        if scheduled is not None and scheduled.is_prefill:
            read_blocks, read_snapshot = self.kv_cache_manager.retain_group_prefill_history(
                request.request_id, request.num_computed_tokens,
            )
        selected = request.cache_partition if partition is None else partition
        try:
            request.allocated_group_block_ids = self.kv_cache_manager.ensure_group_blocks(
                request.request_id,
                total_tokens,
                partition=selected,
            )
            request.cache_partition = self.kv_cache_manager.group_request_partition(
                request.request_id
            )
        except Exception as error:
            if read_snapshot is not None:
                self.kv_cache_manager.release_group_block_snapshot(read_snapshot)
            if isinstance(error, KVCacheCapacityError):
                return False
            raise
        if scheduled is not None:
            scheduled.read_block_ids_by_group = read_blocks
            scheduled.group_read_snapshot = read_snapshot
        return True
    return self._try_allocate_blocks(request, self._blocks_needed(request, num_new_tokens))
~~~

在 _schedule() Phase 2 开始处初始化：

~~~python
profile = self._build_partition_profile()
projected_token_load = {
    partition: item.pending_tokens
    for partition, item in profile.items()
}
projected_waiting = {
    partition: item.projected_waiting
    for partition, item in profile.items()
}
~~~

在 Prefix Cache lookup 和 num_new 计算之后，对没有 partition 的新请求形成候选投影：

~~~python
was_unassigned = request.cache_partition is None
partition_hint = None

if self.kv_cache_manager.has_groups and was_unassigned:
    demand = self._partition_demand(request)
    projected = dict(projected_token_load)
    projected_waiting_for_score = dict(projected_waiting)

    for partition in projected:
        projected[partition] += demand
        projected_waiting_for_score[partition] += 1

    partition_hint = self.kv_cache_manager.select_group_partition(
        request.request_id,
        request.num_computed_tokens + num_new,
        projected_token_load=projected,
        projected_waiting=projected_waiting_for_score,
        token_scale=self.config.max_num_scheduled_tokens,
    )

allocated = self._try_allocate_request_blocks(
    request,
    num_new,
    scheduled=scheduled,
    partition=partition_hint,
)
~~~

只有 allocation 成功后才能更新本轮投影：

~~~python
if was_unassigned and request.cache_partition is not None:
    partition = request.cache_partition
    projected_token_load[partition] += self._partition_demand(request)
    projected_waiting[partition] += 1
~~~

Prefix Cache 命中时 request.cache_partition 已经存在，必须跳过评分并保持命中 partition。只有 grouped hit 因容量不足被释放、重新走 cold fallback 时，才重新执行上述选择。

## 6. 不能做的改动

- 不能把 waiting 队列拆成多个真实 DP 队列；
- 不能迁移已经绑定 partition 的 live request；
- 不能把 4096 写成 Scheduler 每轮固定 token budget；
- 不能把 Decode 同时计入 token_load 和独立 decode 项；
- 不能用评分替代 can_ensure_group_blocks()；
- 不能修改 V4.1 forward、page table 或 DSpark ABI。

## 7. 验证用例

至少增加以下单元测试：

1. 两个 partition 请求数相同，但 token_load 不同，新请求选择 token_load 较低者；
2. token_load 接近时，KV usage 较低者得分更低；
3. 同一 schedule 连续接纳两个新请求，第二个请求能看到第一个的 projected_waiting；
4. Prefix Cache 命中请求保持原 partition；
5. capacity 不足时评分不绕过 allocator，request 留在 waiting；
6. DSpark speculative demand 只计算一次。

