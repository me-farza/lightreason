# Lightreason neural-memory experiment

[Direct vector handoff to 35B](LATENT.md) uses the matched memory slot as continuous input to the larger model.

[Reward-trained memory policy](RL.md) learns what to write and retrieve from episodic feedback.

The experimental backend learns to write and read a fixed-size personal state. An approved observation arrives through `absorb`. The pinned original Qwen2.5-3B-Instruct identifies a named object and copies its access phrase from the current observation. The trained writer projects those spans into continuous address and value tensors and commits them to one private state file. A new observation at the same address replaces the older value.

On `recall`, the reader receives a question and the saved state. It projects the question's target into the same address space, checks for a close match, and decodes the selected value vectors through the original tied Qwen output head. The state file contains eight slots of fixed-shape tensors and occupancy metadata. The implementation stores neither original text nor token IDs. No text database or retrieval service participates in absorb or recall.

Backbone weights stay frozen. Training learns the address and codec projections separately. The current task is one access phrase per distinct named object in English. The writer expects one fact per approved observation. General conversation summarization and periodic consolidation are outside the measured prototype. Once all eight slots are occupied, a new distinct address replaces the least recently written slot. Use a separate owner-only directory and state path for each principal. Treat the neural tensors as private data because the reader can reconstruct the stored values. [MemoryLLM architectural reference](https://proceedings.mlr.press/v235/wang24s.html)

## Train and exercise the restart gate

The original Qwen checkpoint and pinned Python environment stay in place. From the project root on `gpu-host`:

```sh
umask 077
bench=/path/to/lightreason-bench
py="$bench/.venv/bin/python"
model="$bench/model-3b"
driver=/path/to/lightreason-35b/nvidia-driver/usr
export PATH="$driver/bin:$PATH"
export LD_LIBRARY_PATH="$driver/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
experiment=$(mktemp -d /tmp/lightreason-neural.XXXXXX)
checkpoint="$experiment/writer-reader.safetensors"

"$py" -m lightreason_memory.neural_train --model-path "$model" --prompted-spans \
  --checkpoint "$checkpoint" --report "$experiment/training.json"
"$py" -m lightreason_memory.neural_demo --model-path "$model" \
  --checkpoint "$checkpoint" --report "$experiment/acceptance.json"
```

The parent process generates held-out synthetic observations and provides them only to writers. Every recall runs in a new child with the question, pinned model, trained sidecar, and one state path. It also evaluates the identical final questions with an empty state. A failed gate exits nonzero and records the failed cases. The pinned Qwen checkpoint is subject to its research license. [Default 3B model terms](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE)

### RTX 4090 results

- Trained 1,310,721 projection and codec parameters for 900 steps on 6,000 synthetic episodes. Training took 7.8 seconds including setup. The Qwen weights stayed frozen.
- The primary held-out suite and `--split alternate` each passed all 12 saved-state checks and all six empty-state checks. This covered paraphrased recall, two subsequent observations, replacement on correction, retention of the other values, and abstention for an unseen object or an unrelated attribute.
- The state stayed at 51,704 bytes and contained no original experience strings or plaintext answer phrases. Recall workers received only questions plus model, checkpoint, and state paths. They used the host's normal filesystem permissions. State and checkpoint digests stayed unchanged during recall.
- Median recall after load was 0.12–0.14 seconds. Cold recall-worker phases took 9.5–9.9 seconds including integrity checks, startup, and the phase's multiple questions. The learned contextual span tagger scored 6/12 in the primary loaded-state test. The passing variant uses the original Qwen model to extract spans and the separately trained neural projections to store and reconstruct them.
- General wording is still a source of false abstentions. After a successful write for a synthetic kettle, “What is the access phrase for the kettle?” returned the value, while “What word pair should I use to unlock the kettle?” returned `unknown`. The two fixed suites establish state-only recall for these tasks and phrasings.

## Use the experimental backend

```sh
state="$experiment/personal.safetensors"
printf '%s' 'I set the access phrase for the lantern to coral amber.' |
  "$py" -m lightreason_memory.neural_cli --model-path "$model" \
  --checkpoint-path "$checkpoint" --state-path "$state" absorb
printf '%s' 'What access phrase did I choose for the lantern?' |
  "$py" -m lightreason_memory.neural_cli --model-path "$model" \
  --checkpoint-path "$checkpoint" --state-path "$state" recall
```

The MCP launcher exposes `recall(question)` by default. A trusted application can forward approved one-fact observations to `absorb(experience)` with `--allow-absorb`. The configured state path belongs to the launcher. The tool caller supplies question or observation text.

```sh
"$mcp_py" -m lightreason_memory.neural_mcp --model-path "$model" \
  --checkpoint-path "$checkpoint" --state-path "$state" --inference-python "$py"
```

Set the MCP server working directory to the project root and provide the driver environment above. The isolated MCP Python environment uses `lightreason_memory/requirements-mcp.txt`. The launcher releases the inference process before returning the tool result. A consuming agent can pass the returned `status` and `answer` as user context through `lightreason_memory.reasoning.reasoner_prompt`.

```python
from lightreason_memory.reasoning import reasoner_prompt

result = await session.call_tool("recall", {"question": question})
if result.isError:
    raise RuntimeError("Neural memory recall failed")
reply = result.structuredContent
prompt = reasoner_prompt(question, {
    "status": reply["status"],
    "neural_memory_answer": reply["answer"],
})
# Send prompt to your existing reasoner through its supported user/system interface.
```
