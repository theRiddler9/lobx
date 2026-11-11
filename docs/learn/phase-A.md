# Phase A: C++ Batch API — `submit_many` and the Python/C++ Boundary

> *Plain-language notes written after the phase. Read these after you have looked at the code.*

## What we built

One new method on `NativeOrderBook`: `submit_many(orders: list[Order]) -> list[Fill]`.

It is equivalent to calling `submit(o)` for every order in sequence and
concatenating the fills, but it crosses the Python/C++ boundary **once per
batch** instead of **once per order**.

## Why the boundary matters

The C++ matching engine runs at about **5.2 million orders per second** when
called in pure C++ (no Python in the loop). The Python reference engine runs
at about **430 thousand orders per second**. That is a 12× gap from the
engine alone.

But when we call the C++ engine from Python one order at a time, the gap
shrinks to only **1.6×**. Here is why:

```
One call to book.submit(order):

  Python code builds an Order dataclass         ~100 ns
  Python function call overhead                 ~200 ns
  pybind11 unpacks arguments (ints, optionals)  ~300 ns
  GIL is re-acquired / released                 ~100 ns
  Actual C++ matching                           ~100-200 ns  ← this is the fast part
  pybind11 packs fill tuples into Python list   ~200 ns per fill
  Python code converts tuples to Fill dataclass ~100 ns per fill
  ------------------------------------------------------------------
  Total per order:                              ~1,000-1,200 ns
```

The matching itself is about **10-15%** of that total. Everything else is the
**language boundary cost**: argument packing, type conversion, GIL management.

## How `submit_many` fixes it

```
One call to book.submit_many(batch_of_500_orders):

  Python builds a list of tuples                ~50 ns per order (less per-call overhead)
  pybind11 receives the list once               ~300 ns (one call)
  GIL released for the entire C++ loop         0 extra cost
  C++ processes 500 orders natively             ~500 × 120 ns = 60,000 ns
  pybind11 packs all fills into one list        ~200 ns per fill
  Python converts tuples to Fill dataclasses    ~100 ns per fill
  ------------------------------------------------------------------
  Amortized per order:                          ~700-800 ns
```

The fixed costs of calling across the boundary (GIL, argument unpacking, one
function dispatch) are paid **once** for 500 orders instead of 500 times.

## What we measured (Windows 11, AMD Ryzen, Python 3.15)

Command: `python benchmarks/bench_submit_many.py --ops 200000 --batch-size 500`

| mode | ops/s | median ns/order | speedup vs Python-seq |
|---|---:|---:|---:|
| python-seq | 427,735 | 1,600 | 1.0× |
| native-seq | 702,334 | 1,100 | 1.6× |
| native-batch (size=500) | 1,231,512 | 763 | **2.9×** |

**Honest reading:**

- Batch gives **2.9× over Python sequential** and **1.75× over native per-order**.
- The batch speedup is real but not dramatic on this machine, because the
  Python tuple-building step is itself not trivially fast: constructing 500
  `(id, agent, side, type, price, qty, ts)` tuples still touches Python's
  memory allocator before the C++ call.
- The claim of ">14.8M orders/s" that was previously in the README was not
  reproducible. It has been removed. The real numbers are above.
- On Linux with a colder cache, the speedup ratio for batch vs. sequential
  native tends to be **2× to 4×**, depending on batch size and cache state.

## Why the GIL matters for correctness, not just speed

Python's Global Interpreter Lock (GIL) ensures that only one thread runs
Python bytecode at a time. When pybind11 binds a C++ function with
`py::call_guard<py::gil_scoped_release>()`, the GIL is released during the
C++ call. This means:

1. **Another Python thread can run** while the C++ matching loop is running.
2. The C++ code must not call back into Python (it doesn't — it only touches
   `std::map`, `std::deque`, and `std::vector`).
3. If the C++ code had data races (two threads mutating the book), releasing
   the GIL would be unsafe. The single-threaded `submit_many` has no threads,
   so releasing the GIL is safe and free.

## The differential test

`tests/test_batch_submit_many.py` feeds the same list of orders to both
`PythonOrderBook` (sequential) and `NativeOrderBook.submit_many` (batch) and
checks:

- Every fill (maker id, taker id, side, price, quantity, timestamp) is
  identical.
- Best bid, best ask, and depth-5 snapshot agree after the batch.
- Edge cases: empty batch, single-order batch, crossing within the batch,
  partial fill, market order, all-resting (no fills) batch.

The test runs on seeds 0–9 with 2,000 operations and on a single 100,000-order
batch. If a future change to the C++ `submit_many` changes behaviour, all 23
tests catch it.

## What this phase did *not* change

- The simulator still calls `submit()` one order at a time. Integrating
  `submit_many` into the simulator would require grouping each agent's orders
  into batches per time tick, which is a simulator redesign. The benefit for
  the current simulator would be small: the bottleneck is model prediction
  (59% of evaluation time) and feature extraction (26%), not the matching
  engine (5%).
- The C++ `submit_many` implementation is simply a loop calling `submit()`
  inside the C++ layer. It is **not** a vectorized matching algorithm. A
  proper vectorized batch could match orders within the batch before they are
  inserted, reducing the fill count — but it would change correctness semantics
  and is not worth it for this project.

---

## Three practice exercises

**Exercise 1 — Understand the boundary cost:**
Modify `bench_submit_many.py` to add a fourth mode: `native-batch (size=1)`,
which calls `submit_many([o])` for each order individually. Predict the
outcome before running it. Run it. Explain the result.

*(Hint: what is the difference between one call with one element and one
call with zero elements?)*

**Exercise 2 — Write a streaming batch variant:**
Implement a generator version:

```python
def submit_many_gen(book: NativeOrderBook, orders: Iterable[Order],
                    batch_size: int = 500) -> Iterator[Fill]:
    ...
```

It should yield fills as they are produced from each batch, so the caller
does not have to wait for all fills before processing the first one. Write
a test that checks the total fills produced equal those from sequential
`submit()`.

**Exercise 3 — Measure the allocation cost:**
The current wrapper builds a `list[tuple]` in Python before each C++ call.
An alternative is to use a `numpy` structured array or a `ctypes` buffer to
pass orders without Python-object allocation. Sketch the interface (you do not
need to implement it, but you should be able to describe the pybind11 binding
changes required). What would you measure to decide if it is worth the added
complexity?
