# Reward-trained local memory

A small trainable policy uses frozen Qwen2.5-3B token features. It chooses whether to store an observation, which input spans form the memory key and value, and which saved slot to read or whether to abstain. The earlier trained codec converts selected text spans to continuous values. Corrections replace the slot selected by cosine key matching. Runtime state contains only key and value tensors, occupancy, and a checkpoint binding. The reader receives the current question and that state.

Training starts with imitation, then samples writer and reader actions over complete synthetic episodes. REINFORCE updates the policy from later recall outcomes and a small write cost, with a leave-one-out batch baseline. Incorrect answers cost more than abstaining. The policy sees individual utterance features. The environment retains answer labels for scoring and never gives them to the sampled policy as inputs. [Williams, 1992](https://doi.org/10.1007/BF00992696)

## Replicate on the RTX 4090

```sh
umask 077
bench=/path/to/lightreason-bench
driver=/path/to/lightreason-35b/nvidia-driver/usr
export PATH="$driver/bin:$PATH"
export LD_LIBRARY_PATH="$driver/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
py="$bench/.venv/bin/python"
codec=/path/to/prompted.safetensors
run=$(mktemp -d /tmp/lightreason-rl-run.XXXXXX)

"$py" -m lightreason_memory.rl_memory --model "$bench/model-3b" --codec "$codec" \
  fit --out "$run/checkpoints" --warmstart-steps 35 --rl-steps 350 --batch-size 32 --aligned-teacher
"$py" -m lightreason_memory.rl_demo --model "$bench/model-3b" --codec "$codec" \
  --policy "$run/checkpoints/rl.safetensors" --transient --report "$run/restart.json"
```

Run the command from the Lightreason workspace. The `fit` command creates its output directory and saves untrained, supervised warmstart, matched imitation, and RL snapshots. Matched imitation and RL fork at the same warmstart and use the same number of updates, learning rate, and sampled episodes.

## Held-out synthetic results on `gpu-host`

Each of two seeds used 128 unseen test episodes and 2,048 queries with disjoint object–value pairs and distinct phrasings. The values are 12 single-token colors. The large test scores the selected latent value against the synthetic catalog. The restart demonstration uses the full Qwen output head to decode actual answers.

| Policy or control | Exact overall | Corrections | Temporary facts written |
| --- | ---: | ---: | ---: |
| Matched imitation | 87.5% | 100% | 100% |
| Imitation with explicit durable-write labels | 83.1–92.0% | 71.1–78.1% | 0% |
| Reward-trained policy | 99.1–100% | 100% | 0% |
| Reward-trained policy, empty state | 37.5% | 0% | — |
| Reward-trained policy, shuffled query keys | 35.1–35.2% | — | — |

The basic imitation teacher labels fact spans for writing even when the text marks them as temporary. The explicit-label teacher learns from corrected write labels. Both that teacher and RL rejected all 128 temporary facts in the test. The explicit-label teacher's lower overall score came from other queries. The reward treats temporary examples as absent from durable memory and penalizes confident recall of those examples. On the simpler task without temporary examples, the warmstart and RL both reached 100%. The first, weaker feature configuration regressed under RL.

In the fresh-process demo, empty state returned `unknown`. The policy then remembered the lantern was `purple`, applied a correction, and recalled `gray` with the original question and a paraphrase. It retained the mug and bicycle colors and declined to retain the temporary green jacket. The state was 6,520 bytes. The basic imitation control stored the temporary observation and returned `green` even after restart.

The pilot updates the attached policy weights. The Qwen transformer and codec stay frozen. The policy handles this synthetic single-token domain and uses fixed slot replacement and capacity rules. The `vectors` CLI returns 256-dimensional value rows before decoding. Connecting those rows reliably to 35B remains a separate problem, since the previous compact-to-35B projection failed to preserve exact values. [35B handoff results](LATENT.md)

## Train the memory policy every two messages

On `absorb`, Qwen encodes the current message and the policy can update the neural bank immediately. A trusted client can attach verified single-token key and value labels or mark an observation as a verified ignore. Verified writes apply to the bank immediately and supply a target action for the next training attempt. The service saves the features, vectors, and target positions for replay.

Set `--dream-every 2` to count absorbs and both kinds of recall. On each boundary, one queued verified observation triggers a bounded CPU policy update from those saved vectors. A checkpoint replaces the active policy only if the greedy policy meets the new target and preserves decisions on the recent replay captures. The file commits the active policy, counters, replay window, and bank together. A process restart resumes the count and loads the active policy. `dream-status` reads progress without counting as a message or loading Qwen.

Run from the CUDA workspace with the environment above and an existing trained checkpoint:

```sh
policy="$run/checkpoints/rl.safetensors"
session=$(mktemp -d /tmp/lightreason-dream.XXXXXX)
state="$session/personal.safetensors"
memory() {
  action=$1
  shift
  "$py" -m lightreason_memory.rl_memory --model "$bench/model-3b" --codec "$codec" \
    --dream-every 2 "$action" --policy "$policy" --state "$state" "$@"
}
printf '%s' '{"text":"I can confirm the lantern is purple.","expected_key":"lantern","expected_value":"purple"}' |
  memory absorb --feedback-stdin
printf '%s' 'Can you tell me the hue of my lantern?' |
  memory vectors --out "$session/answer.safetensors"
memory dream-status
```

The second operation writes a decoder-free vector file. `awaiting_feedback` means the boundary arrived with no verified training target. `unchanged` means the current policy already handles the target. `promoted` means training changed the policy and it passed the replay gate. Chatter can advance the count, and a trusted caller supplies the verification needed for training. The replay window holds up to eight verified observations and four recent read decisions. Treat both banks and replay tensors as private.

For a trusted MCP client with the SDK installed, launch from the same workspace:

```sh
"$mcp_py" -m lightreason_memory.dream_mcp \
  --model "$bench/model-3b" --codec "$codec" --policy "$policy" --state "$state" \
  --inference-python "$py" --allow-absorb --dream-every 2
```

The trusted client forwards observations with `observe(experience, expected_key, expected_value)` or `observe(experience, ignore=true)`. MCP `recall(question)` is the text diagnostic. Use the Python `PolicyMemory.vectors` API or the `vectors` CLI for continuous output. The cadence only sees requests forwarded to this memory backend.

## Pass learned memory to 35B

Set `--model-35b "$root/model"` on the writer, or on the MCP launcher, to store native BF16 embedding rows alongside each value. The writer gets these rows from the approved observation while its span is still available. The trained policy selects the same slot on recall. Use `vectors --target-aligned --out PATH` or `PolicyMemory.vectors(question, target_aligned=True)` for the 2,048-dimensional rows. Existing compact-only entries gain native rows the next time the writer observes those facts with the destination model configured.

`lightreason_memory.dream_reason.answer_with_memory` takes the question and paths `model_3b`, `codec`, `policy`, `state`, `model_35b`, and `stage_35b`. Give it the 3B Python path as `memory_python` and run the calling process in the staged 35B environment. Use `memory_query` to provide a short lookup question separately from the caller's prompt. The reasoner controller defaults to read-only recall. Pass a positive `dream_every` to opt into the message-driven pilot. The 3B worker exits before 35B loads. On a hit, 35B injects the rows into prefill and the controller verifies the receipt. On a miss, the ordinary 35B runner receives the caller's original prompt, system instructions, and token budget. [Audited runner setup](LATENT.md)

### Live checks on the GPU host

The first process stored purple for the lantern. A fresh process returned one native BF16 row. A verified correction to gray replaced the row, and another fresh process returned the corrected row. The same 35B question returned `purple` with the original row, `gray` with the corrected row, `unknown` with empty memory, and `gray` with decoded text. The policy already chose the verified write actions, so it needed zero optimizer steps at both boundaries.

In a separate session, verified ignore feedback triggered an update at message two. The candidate passed after 38 steps, and a fresh process used generation one to ignore the same observation without a feedback label. Another session rejected a candidate after 48 steps when it could not satisfy the feedback and keep the prior lantern decisions.

### Swap two independent memories

The swap check writes two different lantern values into independent state files. Fresh 3B workers receive the same question and one state path each, including an empty-state control. The check disables the memory value encoder and decoder during recall, verifies that each returned BF16 row matches the saved bytes, and validates the 35B prefill receipt. The populated runs use identical prompt token IDs. The empty run sends the original question without a memory marker. All three use one warm 35B runner. Add `--prepare-only` to audit the memory workers without starting the 35B model.

Use the environments above and point `policy` at the trained policy and `root` at the staged 35B data directory.

```sh
taskset -c 8-15 "$root/freetoken/bin/python" -m lightreason_memory.dream_swap \
  --model-3b "$bench/model-3b" --memory-python "$py" \
  --codec "$codec" --policy "$policy" --model-35b "$root/model" \
  --stage-35b "$root/staged-freetoken" --report "$session/swap.json"
```

## Nightly training design

During the day, capture approved conversation events, tool results, and wiki edits. Store their source, time, and scope. Write compact memory state for immediate recall. Record verified corrections as revisions. A user-stated mood can inform the current situation and carry an expiry. A lasting preference has an explicit scope and can be revised.

A local job turns the day's evidence into episodes. For each episode, it builds questions about a fact, an ordered procedure and its exceptions, a preference and its scope, or context that may expire. Each answer key cites the source events that establish it. Include stale and unsupported questions that should produce a memory miss. Hold out whole events or days so that training and validation do not share the same evidence.

The small model learns to encode experiences into persistent state and emit a bounded set of useful 35B context vectors for a query. Runtime recall uses the query and saved neural state. The wiki is unavailable in the latent-only evaluation. The current controller's compatible vector format is the bridge for an initial rollout. Its write-time lexical rows are the baseline for a learned encoder that composes context across several observations.

Reinforcement learning supplies the objective. A policy head or a LoRA adapter carries the learned parameters. Score useful retrieval, correct withholding, current revision, and scope against grounded answers. Compare exact facts, step order, and exceptions where deterministic scoring is possible. Use a small downstream 35B eval to verify that the context vectors actually improve the answer over the same prompt without memory. Rehearse earlier evidence and gate promotion on retention and correction tests as well as new cases.

The existing scoped SQLite record and search layer can back the detailed wiki. Build pages for procedures, preferences, and project decisions with versioned sections and source links. 35B requests exact steps, wording, or history explicitly through a search or page read. The same evidence can supply targets during training.

The first experiment can replay one simulated day with a procedure and exception, a corrected project preference, a temporary state, and an unrelated task. Then run one explicit consolidation job, restart, and ask new phrasings. Compare the current checkpoint, the candidate, an empty bank, and a populated bank containing unrelated information. Check that both misses use the ordinary request path with identical input IDs and deterministic output. Score exact wiki details separately. The nightly job, richer episodes, learned encoder, and wiki still need implementation.
