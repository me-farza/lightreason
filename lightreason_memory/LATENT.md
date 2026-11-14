# Direct handoff to the 35B reasoner

This experiment lets the local Qwen3.6-35B-A3B runner consume remembered values as embedding rows. The 3B memory worker still identifies the requested object from the question. `recall_vectors` returns the matched rows before the memory value decoder. The 35B model generates the visible answer.

There are two vector formats:

- **Projected.** Existing 256-dimensional value codes go through a learned 256×2,048 projection. Fitting uses shared lexical forms and both pinned checkpoints' embedding tables. The lantern fixture's component token IDs are excluded from fitting.
- **Writer-aligned.** While the original approved observation is present, the writer also saves the remembered phrase's native 35B input vectors in the same slot. It saves continuous BF16 rows, with their count, alongside the original 256-dimensional code. It reads those rows directly after a restart. Existing state files need a new write in this format.

The local runner clones the audited 34-layer FreeToken stage and replaces reserved placeholder embedding rows during uncached prefill. The model's checkpoint and configured 15 GiB expert cache stay the same. The prefill hook uses a private one-shot tensor sidecar with a receipt binding the actual request, positions, and embedding digest. Ordinary requests continue through the existing text path. The experiment compares both direct-vector formats with empty state and the existing decoded-text handoff.

## Lantern correction on the RTX 4090

All arms asked “Which access phrase does the lantern use now?” The writes and reads used separate processes. The corrected vector request contained three BF16 rows at prefill positions 56–58. The text control occupied the same three positions in an 80-position prompt. Both returned the exact corrected phrase.

| Input to 35B | Before correction | After correction |
| --- | --- | --- |
| Native 35B vectors saved at write time | `cotton walnut` | `satin ochre` |
| Learned projection from the existing 256D state | `unknown` | `den ochbeh` |
| Decoded-text control | — | `satin ochre` |
| Empty-state control | — | `unknown` |

The passing format uses a 445,216-byte state with the extra target rows. The test found neither observation text nor plaintext answers in that state or the vector prompt. The projected rows were delivered successfully, but the learned bridge failed to transfer the exact values. Its offline mean cosine was 0.262 on 1,024 held-out vocabulary rows, so that score cannot substitute for a generation test. A single warm request per arm measured 1.486 seconds for corrected vectors and 1.486 seconds for decoded text. These timings exclude the earlier 3B work and the 271.8-second 35B startup.

## Reproduce on the GPU host

Run from the full research workspace with the pinned 3B environment, the checkpoint trained in the [neural-memory experiment](NEURAL.md), and the staged 35B environment:

```sh
umask 077
bench=/path/to/lightreason-bench
root=/path/to/lightreason-35b
driver="$root/nvidia-driver/usr"
export PATH="$driver/bin:$PATH"
export LD_LIBRARY_PATH="$driver/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py="$bench/.venv/bin/python"
checkpoint=/path/to/prompted.safetensors
run=$(mktemp -d /tmp/lightreason-latent.XXXXXX)

"$py" -m lightreason_memory.latent_bridge fit \
  --model-3b "$bench/model-3b" --memory-checkpoint "$checkpoint" \
  --model-35b "$root/model" --out "$run/bridge.safetensors" --report "$run/bridge.json"

OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  taskset -c 8-15 "$root/freetoken/bin/python" -m lightreason_memory.latent_demo \
  --model-3b "$bench/model-3b" --memory-python "$py" --checkpoint "$checkpoint" \
  --projection "$run/bridge.safetensors" --model-35b "$root/model" \
  --stage-35b "$root/staged-freetoken" --report "$run/comparison.json"
```

The demo uses separate processes to write the initial lantern phrase and the correction, and to read back each state. Its vector workers fail if they call the value decoder. The final question, 35B weights, decoding settings, and instruction are shared across the arms. With `--prepare-only`, the tool verifies the restart path without starting 35B. For timed or repeated trials, adding `--save-prepared` saves the synthetic vector inputs and restart verification for a subsequent `--prepared` run. Treat such files like the state itself because they can disclose remembered content.

For a caller that already owns a local 35B runner, build the question's `PromptText` with one literal `[[MEMORY]]` placeholder. Use `NeuralMemory.recall_vectors(question, target_aligned=True)` from a freshly loaded target-aligned state. Transfer the returned CPU BF16 rows, let the memory process exit, then call `LatentReasoner35B.answer_with_vectors(prompt, token_budget, rows)`. An empty read returns `None`. Ask the reasoner with the memory field empty in that case. Ensure the 3B process has exited before loading 35B on the shared GPU.

## ZnO annealing correction

Use the same local runtime and checkpoint setup above. This fixture uses the compact temperature and ramp-rate values recorded in the synthetic ZnO workflow. The local Qwen identifies the schedule in the observation and in a differently worded question. It reads saved destination-model rows after restart. The 35B prompt gets those rows in the two vector arms, an empty memory field in the empty arm, and the small model's decoded value in the text arm.

```sh
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  taskset -c 8-15 "$root/freetoken/bin/python" -m lightreason_memory.latent_demo --fixture zno \
  --model-3b "$bench/model-3b" --memory-python "$py" --checkpoint "$checkpoint" \
  --model-35b "$root/model" --stage-35b "$root/staged-freetoken" \
  --reasoner-tokens 32 --report "$run/zno.json"
```

The report records the four actual answers, per-process state hashes, and the prefill receipts. Run the CLI directly to exercise this route. The scientist notebook's evidence lookup uses its SQLite store.
