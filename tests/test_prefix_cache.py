"""Prefix caching: allocator invariants, and cache hits must not change a single token."""
from tensorline.batch_engine import BatchEngine
from tensorline.engine import GenStats, SamplingParams
from tensorline.paged import BlockAllocator, block_hash
from tests.test_batching import NO_STOP, TINY, run_concurrent


def test_allocator_refcount_lru_and_eviction():
    a = BlockAllocator(4)
    x = a.allocate(2)
    h0 = block_hash(None, [1, 2])
    a.register(x[0], h0)
    a.free(x)
    assert a.num_free == 4 and a.num_cached == 1          # freed but still cached
    assert a.lookup(h0) == x[0] and a.num_free == 3       # revived from the free list
    assert a.lookup(h0) == x[0]                           # shared: refcount 2
    a.free([x[0]])
    assert a.num_free == 3                                # still referenced once
    a.free([x[0]])
    assert a.num_free == 4
    a.allocate(4)                                         # memory pressure evicts it
    assert a.num_cached == 0 and a.lookup(h0) is None


def test_block_hash_chains_over_the_whole_prefix():
    assert block_hash(block_hash(None, [1]), [2]) != block_hash(block_hash(None, [9]), [2])


def _engine(**kw):
    return BatchEngine(TINY, device="cpu", block_size=4, num_blocks=128, **kw)


def _gen(eng, prompt, n=12):
    st = GenStats()
    text = "".join(eng.stream(prompt, SamplingParams(n, stop_token_ids=NO_STOP), st))
    return text, st


SHARED = "you are a careful assistant that answers questions about distributed systems " * 3


def test_shared_prefix_hits_and_matches_uncached():
    questions = [SHARED + q for q in ("what is raft", "what is paxos", "define a quorum",
                                      "what is a lease", "explain gossip")]
    ref_eng = _engine(prefix_caching=False)
    ref = [_gen(ref_eng, q)[0] for q in questions]

    eng = _engine()
    first, st0 = _gen(eng, questions[0])                  # warms the cache
    assert first == ref[0] and st0.cached_tokens == 0
    texts, stats = run_concurrent(eng, [(q, 12) for q in questions[1:]])
    assert texts == ref[1:]
    shared_blocks = len(eng.tokenizer(SHARED).input_ids) // 4
    assert all(s.cached_tokens >= (shared_blocks - 1) * 4 for s in stats)
    assert eng.prefix_hit_tokens > 0
    assert eng.alloc.num_free == eng.alloc.num_blocks     # nothing leaked; cached blocks are free


def test_identical_prompt_fully_cached_recomputes_last_block():
    eng = _engine()
    a, _ = _gen(eng, SHARED)
    b, st = _gen(eng, SHARED)
    assert a == b
    assert 0 < st.cached_tokens < st.prompt_tokens        # last block recomputed to get logits


def test_preemption_with_prefix_cache_matches_reference():
    ref = _engine(prefix_caching=False)
    jobs = [(SHARED + str(i), 10) for i in range(6)]
    want, _ = run_concurrent(ref, jobs)
    eng = BatchEngine(TINY, device="cpu", block_size=4, num_blocks=24)
    got, _ = run_concurrent(eng, jobs)
    assert got == want and eng.preemptions > 0
    assert eng.alloc.num_free == eng.alloc.num_blocks
