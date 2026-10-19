"""CPU paged K/V storage with reference counting, prefix forks and copy on write."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch


class PagedKVCache:
    def __init__(self, pages, block_size, key_width, value_width, dtype=torch.float64):
        if any(not isinstance(x, int) or x < 1 for x in (pages, block_size, key_width, value_width)):
            raise ValueError("All cache dimensions must be positive integers")
        if not dtype.is_floating_point:
            raise ValueError("Cache dtype must be floating point")
        self.block_size = block_size
        self.keys = torch.empty(pages, block_size, key_width, dtype=dtype)
        self.values = torch.empty(pages, block_size, value_width, dtype=dtype)
        self.free = list(reversed(range(pages)))
        self.refs = [0] * pages
        self.tables = {}
        self.lengths = {}
        self.copied_token_slots = 0
        self.allocations = 0

    def start(self, name):
        if name in self.tables:
            raise ValueError("Sequence already exists")
        self.tables[name] = []
        self.lengths[name] = 0

    def _allocate(self):
        if not self.free:
            raise MemoryError("No free K/V pages")
        page = self.free.pop()
        self.refs[page] = 1
        self.allocations += 1
        return page

    def fork(self, parent, child, prefix_tokens=None):
        length = self.lengths[parent] if prefix_tokens is None else prefix_tokens
        if child in self.tables or not isinstance(length, int) or not 0 <= length <= self.lengths[parent]:
            raise ValueError("Child must be new and prefix length within parent")
        table = self.tables[parent][:(length + self.block_size - 1) // self.block_size]
        self.tables[child] = table.copy()
        self.lengths[child] = length
        for page in table:
            self.refs[page] += 1

    def append(self, name, key, value):
        length = self.lengths[name]
        for tensor, width in ((key, self.keys.shape[2]), (value, self.values.shape[2])):
            if tensor.shape != (width,) or tensor.dtype != self.keys.dtype or tensor.device.type != "cpu":
                raise ValueError("Append expects CPU vectors with cache widths and dtype")
        offset = length % self.block_size
        table = self.tables[name]
        if offset == 0:
            table.append(self._allocate())
        elif self.refs[table[-1]] > 1:
            old = table[-1]
            new = self._allocate()  # OOM happens before changing existing state.
            self.keys[new, :offset].copy_(self.keys[old, :offset])
            self.values[new, :offset].copy_(self.values[old, :offset])
            self.copied_token_slots += offset
            self.refs[old] -= 1
            table[-1] = new
        page = table[-1]
        self.keys[page, offset].copy_(key)
        self.values[page, offset].copy_(value)
        self.lengths[name] += 1

    def gather(self, name):
        length = self.lengths[name]
        table = self.tables[name]
        if length == 0:
            return self.keys.new_empty(0, self.keys.shape[2]), self.values.new_empty(0, self.values.shape[2])
        # Teaching implementation materializes contiguous K/V. A GPU paged kernel
        # would traverse the block table directly instead of doing this gather.
        return (self.keys[table].flatten(0, 1)[:length],
                self.values[table].flatten(0, 1)[:length])

    def attend(self, name, query):
        keys, values = self.gather(name)
        if not len(keys):
            raise ValueError("Attention requires a nonempty context")
        return torch.softmax(query @ keys.T / keys.shape[1]**0.5, dim=-1) @ values

    def release(self, name):
        for page in self.tables.pop(name):
            self.refs[page] -= 1
            if self.refs[page] == 0:
                self.free.append(page)
        del self.lengths[name]

    def check_invariants(self):
        counted = [0] * len(self.refs)
        for name, table in self.tables.items():
            assert len(table) == (self.lengths[name] + self.block_size - 1) // self.block_size
            assert len(set(table)) == len(table)
            for page in table:
                counted[page] += 1
        assert counted == self.refs
        assert len(set(self.free)) == len(self.free)
        assert set(self.free) == {page for page, refs in enumerate(counted) if refs == 0}

    def stats(self):
        used = len(self.refs) - len(self.free)
        slot_bytes = (self.keys.shape[2] + self.values.shape[2]) * self.keys.element_size()
        return {"active_pages": used, "free_pages": len(self.free),
                "active_page_bytes": used * self.block_size * slot_bytes,
                "pool_tensor_bytes": self.keys.numel()*self.keys.element_size() + self.values.numel()*self.values.element_size(),
                "logical_token_references": sum(self.lengths.values()),
                "allocations": self.allocations, "copied_token_slots": self.copied_token_slots}


def run():
    generator = torch.Generator().manual_seed(33)
    cache = PagedKVCache(16, 4, 8, 6)
    keys = torch.randn(9, 8, generator=generator, dtype=torch.float64)
    values = torch.randn(9, 6, generator=generator, dtype=torch.float64)
    cache.start("prompt")
    for key, value in zip(keys[:6], values[:6]):
        cache.append("prompt", key, value)
    cache.fork("prompt", "branch-a")
    cache.fork("prompt", "branch-b")
    shared = cache.stats()
    errors = []
    for name, index in (("branch-a", 6), ("branch-b", 7)):
        cache.append(name, keys[index], values[index])
        query = torch.randn(8, generator=generator, dtype=torch.float64)
        expected = torch.softmax(query @ torch.cat((keys[:6], keys[index:index+1])).T / 8**0.5, -1) @ torch.cat((values[:6], values[index:index+1]))
        actual = cache.attend(name, query)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        errors.append((actual-expected).abs().max().item())
    cache.check_invariants()
    divergent = cache.stats()
    torch.testing.assert_close(cache.gather("prompt")[0], keys[:6])
    for name in ("prompt", "branch-a", "branch-b"):
        cache.release(name)
        cache.check_invariants()
    return {"shared_prefix": shared, "after_divergence": divergent, "after_release": cache.stats(),
            "max_abs_attention_error": max(errors), "prefix_tokens": 6, "block_size": 4,
            "scope": "Real CPU tensors and allocator; gather materializes K/V; no GPU paged-attention kernel, radix tree, LLM or throughput claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/33_paged_kv_cache.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "device": "cpu", "gpu_status": "skipped_by_user_request", **run()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
