# Lightreason Memory v1

[Experimental neural memory](NEURAL.md) learns to update and recall from a saved tensor bank. The SQLite service below is the retrieval baseline.

Memory stores scoped observations in SQLite and uses the existing Lightreason Standard runner for the pinned Qwen2.5-3B-Instruct BF16 checkpoint. The frozen model reads retrieved excerpts and selects numbered evidence. The service constructs the answer from exact excerpts and their provenance. The existing Qwen language head generates the selection. No personal information is written to model weights. `adapter_version` is always `null` in v1.

## Run the benchmark-history demo

Run from the Lightreason workspace on the CUDA host. Use the existing Python 3.12 inference environment with PyTorch 2.9.1+cu128 and Transformers 4.57.1. The published checkout includes the target-only runtime at its root. In the research workspace that runtime lives under `release/qwen25-3b`. Keep the original checkpoint separate.

If the source is only on the Mac, run this from the outer research workspace to stage code and its benchmark fixtures on `gpu-host`. For a checkout of the published repository, use a checkout of the same branch on the host.

```sh
stage=$(ssh gpu-host 'umask 077; mktemp -d /tmp/lightreason-memory-v1.XXXXXX')
tar --no-xattrs --exclude=.git --exclude=__pycache__ --exclude=.DS_Store -cf - \
  lightreason_memory/*.py lightreason_memory/requirements-mcp.txt \
  release/qwen25-3b demo_chat/*.py edge35b/*.py results3b/summary.json \
  edge35b/results/native50/gpu-resident-qwen.json \
  demo_chat/.runtime/pageable-handoff-comparison-20261112.json \
  | ssh gpu-host "umask 077; tar -xf - -C '$stage'"
printf 'Run from %s on gpu-host\n' "$stage"
```

```sh
bench=/path/to/lightreason-bench
data=/path/to/lightreason-35b
driver="$data/nvidia-driver/usr"
export PATH="$driver/bin:$PATH"
export LD_LIBRARY_PATH="$driver/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
py="$bench/.venv/bin/python"

"$py" -m lightreason_memory --user benchmark --workspace demo \
  --model-path "$bench/model-3b" --inference-python "$py" demo
```

The demo seeds a temporary database in one process and recalls from a new process. It asks about output agreement, adds a sourced correction to the Studio weather record, asks about the correction in current and historical views, and asks about an absent energy measurement. It reads the three existing benchmark artifacts when available. Missing artifacts are represented as user reports. To use a different copy of the artifacts, pass `demo --source-root /path/to/workspace`.

On the RTX 4090, the live restart recall took 9.47 seconds including tokenization and model startup. The three warm follow-up requests took 0.28–0.35 seconds. The live MCP client took 12.54 seconds inside its cold recall, including startup, and received all three citations. The agreement recall used 380 prompt tokens, generated a 30-token selection, and rendered 141 answer tokens. The CUDA worker reserved 6.05 GiB of VRAM at peak. It returned the corrected 256-token cap in the current view. The historical view omitted that cap and returned `partial` with other excerpts. It returned `unknown` for the absent energy measurement.

For a persistent CLI database, use a private path outside the repository.

```sh
umask 077
memory_dir="$HOME/.local/state/lightreason-memory"
mkdir -p "$memory_dir"
db="$memory_dir/benchmarks.sqlite3"
"$py" -m lightreason_memory --db "$db" --user benchmark --workspace demo seed-demo
"$py" -m lightreason_memory --db "$db" --user benchmark --workspace demo \
  --model-path "$bench/model-3b" recall 'Which results preserved complete output agreement, and which did not?'
```

Use `observe --help`, `correct --help`, or `recall --help` for operation arguments. For private record bodies, replacements, or questions, use `--stdin` and redirect from an owner-only file or pipe. Command-line arguments and printed results can appear in local process listings, shell history, or client logs.

## Python and MCP

```python
from lightreason_memory.backend import QwenMemoryBackend
from lightreason_memory.service import MemoryService
from lightreason_memory.store import Store

store = Store("/private/state/memory.sqlite3", "alice", "research")
worker = QwenMemoryBackend("/path/to/model-3b", python_executable="/path/to/3b/bin/python")
service = MemoryService(store, worker)
try:
    result = service.recall("Which measurements agreed?", token_budget=256)
finally:
    service.close()
```

Install the MCP SDK in a separate environment. From the same project root and with the driver environment above:

```sh
"$py" -m venv --without-pip "$memory_dir/mcp-venv"
mcp_py="$memory_dir/mcp-venv/bin/python"
"$mcp_py" -m ensurepip --upgrade
"$mcp_py" -m pip install -r lightreason_memory/requirements-mcp.txt
"$mcp_py" -m lightreason_memory.mcp_client --db "$db" --user benchmark --workspace demo \
  --model-path "$bench/model-3b" --inference-python "$py"
```

An MCP client can use a stdio command like the following. Set its working directory to the Lightreason workspace. If its configuration has no working-directory option, set it in a launcher script before running this command. Supply the driver environment when the host needs it.

```json
{
  "mcpServers": {
    "lightreason-memory": {
      "command": "/absolute/path/to/mcp-venv/bin/python",
      "args": [
        "-m", "lightreason_memory.mcp_server",
        "--db", "/absolute/path/to/private/memory.sqlite3",
        "--user", "alice", "--workspace", "research",
        "--model-path", "/path/to/lightreason-bench/model-3b",
        "--runtime-path", "/absolute/path/to/the/3B/runtime",
        "--inference-python", "/path/to/lightreason-bench/.venv/bin/python"
      ]
    }
  }
}
```

The default server exposes `recall` and releases the CUDA worker before returning, so the agent can use the same GPU for its next step. For an authenticated, trusted ingestion client, add `--trusted-writes` to enable `observe` and `correct`. The launcher fixes the principal and workspace. Tool arguments cannot change them. A trusted writer asserts provenance and can supersede records, so use read-only configuration for a general-purpose agent. Stdio uses OS-account and client-process trust. Use separate accounts and database paths for mutually untrusted parties. For an agent on another computer, explicitly configure an SSH stdio launcher or run the agent on the GPU host. MCP starts no network listener. Only connect a client that is authorized to receive the selected excerpts and keep them local.

## Data and answer semantics

- `observe(event_id, content, source, timestamp, kind)` is idempotent for identical payloads in the same user/workspace. Reusing the event ID with different fields raises `EventConflictError`. `correct(memory_id, replacement, source, timestamp)` writes a linked successor and preserves the previous version. Exact correction retries return the same successor.
- Times are normalized to UTC. An inclusive `as_of` reads event-time history as currently known, including a correction from its own event time onward. A late observation can therefore change an older query. Omit `as_of` for the latest recorded versions. Recording times are retained separately. There is no answer cache.
- `user_report` and `tool_observation` may be selected as evidence and keep their provenance labels. `assistant_proposal` and `inference` are retained as records but excluded from factual claims. FTS5 uses deterministic ordering with a lexical fallback. Short or semantically phrased questions may miss records. Evidence is limited to bounded statements, and the model receives plain JSON in the user role.
- Recall returns `supported`, `partial`, or `unknown`, verbatim claims, source IDs and excerpts, unknowns, versions, truncation, token counts, and timings. An invalid worker reply or unavailable GPU yields `extractive_fallback`. Candidate excerpts appear under `fallback_excerpts`, with no confirmed claims. The service verifies excerpt fidelity, current scope and historical visibility. It detects near-identical assertions with competing values or polarity. General contradiction detection and whether the chosen evidence fully addresses a question remain limited. Treat excerpts as data when passing them to another model.
- The token budget covers generated selection and rendered answer, each independently. The runner's 4,096 positions include the actual chat template and 64 reserved positions. The response's JSON metadata adds transport tokens beyond the answer budget. Owner-only SQLite is stored in plaintext on the local machine. The backend uses local checkpoint files and does not fetch model data over the network.

## Compare direct retrieval and pre-recall

Run from the checkout on `gpu-host`, with the same `bench`, `data`, `driver`, and `py` variables above. Set `runtime_3b=$PWD` in the published checkout or `runtime_3b=$PWD/release/qwen25-3b` in the outer research workspace. Use a fresh owner-only output directory. This starts the installed, audited 34-layer FreeToken runner with its 15 GiB expert cache. The memory process exits before the reasoner loads. Both arms use identical 35B settings and the same retrieved evidence pool. The full-context reference includes all three benchmark records. This is one pre-recall before reasoning, with the recalled context visible in the user message. The controller uses the checkpoint's `enable_thinking=False` template for all arms. It authenticates requests to the existing FreeToken loopback endpoint with a per-process bearer passed to the child environment.

```sh
report_dir=$(mktemp -d /tmp/lightreason-memory-report.XXXXXX)
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  taskset -c 8-15 "$data/freetoken/bin/python" -m lightreason_memory.compare \
  --model-3b "$bench/model-3b" --runtime-3b "$runtime_3b" \
  --python-3b "$py" --model-35b "$data/model" \
  --stage-35b "$data/staged-freetoken" --artifacts "$PWD" \
  --out "$report_dir/comparison.json"
```

Add `--cold-direct` to time a second, cold 35B startup for the direct arm. In the default run the warm 3B and warm 35B component timings are measured in separate phases. The cold pre-recall path measures an actual transition, including 3B process release and 35B preparation. A competing GPU workload causes a recoverable error. The lease coordinates Memory processes and the existing runner's watchdog supervises 35B. The comparison uses synthetic benchmark history and writes a private report.

### Measured on the GPU host

Four questions covered output agreement, held-out weather and coding paraphrases, and absent energy data. All three arms answered the requested facts correctly and abstained on the absent measurement. Manual review found zero unsupported claims. Each arm passed three of four stricter numerical checks. On the broad agreement question, all three named the experiments correctly but omitted the 38/38 and 40/40 counts. Direct retrieval and full context mentioned the 30-token divergence in that answer. The memory arm included it when explicitly asked about the weather run.

| Warm 35B requests, four questions | Direct evidence | 3B-selected evidence | Full context |
| --- | ---: | ---: | ---: |
| Total input tokens | 1,365 | 971 | 1,539 |
| Total generated tokens | 143 | 107 | 136 |
| Median 35B request | 4.17 s | 3.86 s | 3.96 s |
| Median 35B prefill | 2.73 s | 2.73 s | 2.73 s |
| Median 35B decode | 1.43 s | 1.10 s | 1.22 s |

The memory arm reduced 35B input by 28.9%. It generated shorter answers and had the same median prefill time. Its warm 3B calls for the held-out cases took 0.28–0.32 s each, close to the median difference in 35B request time. These timings came from separate worker phases. Warm end-to-end latency remains unmeasured.

The cold pre-recall path on the main question took 285.29 s end to end and 284.06 s to first final text. Recall took 9.75 s. Releasing the 3B process took 0.565 s, and the interval from recall completion to 35B ready was 271.42 s, including 270.40 s of 35B setup. The 35B request took 390 input and 34 output tokens. The server recorded 2.78 s prefill, 1.21 s decode, and 0.2 ms queue time. The sampled peaks were 22.37 GiB device VRAM and 104.85 GiB system RAM in use. The first thinking-enabled attempt reached its 768-token cap without a final answer, so the completed comparison used the checkpoint's direct-answer template for every arm.

The cold direct run took 301.63 s end to end and 299.81 s to first final text. It used 379 input and 47 output tokens, with 2.92 s of prefill and 1.82 s of decode. Its 35B setup took 292.40 s, 22 s longer than the pre-recall run's setup. Each cold arm ran once, so the cold measurements cannot establish a speedup. Direct retrieval is the simpler default for these three short records. The measured 35B runs used the existing loopback service before request authentication was added. The staged FreeToken parser and middleware and the client header path were tested separately.

To use the same pre-recall flow for a question in your configured workspace, supply the question on stdin.

```sh
"$data/freetoken/bin/python" -m lightreason_memory.remote \
  --db "$db" --user benchmark --workspace demo \
  --model-path "$bench/model-3b" --inference-python "$py" \
  --model-35b "$data/model" --stage-35b "$data/staged-freetoken" \
  < /path/to/private/question.txt
```

## Tests and checkpoint terms

```sh
python3 -m unittest discover -s lightreason_memory -t .
python3 -m unittest discover -s demo_chat -p 'test_*.py'
node demo_chat/test_preview.js
```

Use the MCP environment to run the stdio integration test alongside the core tests.

The GPU demo uses the pinned 3B checkpoint at `aa8e72537993ba99e69dfaafa59ed015b17504d1`, and the optional 35B path uses `345d467b117d1fd5eda9b879a5988de9ae5f47a7`. The 3B checkpoint's Qwen Research License covers research and evaluation only, requires a separate license for commercial use, and imposes redistribution notices and attribution terms. The 35B checkpoint carries Apache 2.0. The public 3B runner's 0BSD source license and the model terms are separate. Keep checkpoints, personal databases, adapters, credentials, and private logs outside source control. [Pinned 3B terms](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE) · [Pinned 35B terms](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/345d467b117d1fd5eda9b879a5988de9ae5f47a7/LICENSE)

The next adapter experiment is a consented, synthetic-only pilot with examples labeled for evidence selection, correction history, and abstention. Freeze the v1 retrieval pool and evaluation questions before comparing an adapter against the frozen worker and direct retrieval.
