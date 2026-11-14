# Vicena Lightreason scientist notebook

Run from the Lightreason checkout with Python 3 and SQLite.

```sh
python3 -m scientist_ui.server --db "$PWD/.runtime/scientist-demo.sqlite3" --port 8766
```

Open [the local notebook](http://127.0.0.1:8766/notebook). The server seeds two synthetic workflows into the supplied database. Use a fresh database path to replay their corrections from the beginning.

1. Open the ZnO workflow and inspect Step 4 of the original protocol.
2. Apply the recorded correction. Step 4 changes from 350 °C at 10 °C/min to 425 °C at 3 °C/min.
3. Switch to Original and run the historical Step 4 lookup. Expand the cited procedure to see the source record.
4. Switch to TiO₂ and repeat the revision check. Restart the server with the same database path to verify that the corrections persist.

The notebook uses scoped SQLite records to show exact procedures, revision history, and source passages. Example questions open those passages and free-form questions search them. This browser app runs without a model or cloud API. The local Qwen neural memory experiment runs through the separate [decoder-free benchmark](../lightreason_memory/RL.md#swap-two-independent-memories).


## Chat on a local GPU host

Use the pinned Qwen2.5-3B checkpoint and Python environment from the Lightreason setup. Run the server with `--chat-model /path/to/model-3b` and use the home page for a live conversation. For a separate GPU host, forward the server port to your browser machine over SSH. The Notebook link opens the synthetic workflows.

The chat keeps recent turns for the current browser tab, including across a reload. Regular chat messages queue locally for Dream by default. Uncheck **Save for Dream** to skip a message, or attach a TXT or Markdown document. Open **Dream**, then click **Run Dream** to process approved material. Each new batch increments the Dream number. The workflow shows the source material, training questions, sampled rewards, validation checks, and whether a checkpoint was saved. **Morning** opens a fresh chat after the completed night when there is no unsent draft. Use **Inspect memory** under a reply to see its memory route.

One explicit lesson is enough to attempt training. Dream turns it into a question with an expected answer, a differently worded recall check, and an unrelated check that should recall nothing. Dream skips clear chat-only turns and uses the newest version of each lesson in a batch. Vague or lengthy notes can still be saved as evidence. Dream asks for the missing detail when it cannot turn a note into a trainable example. Entering Dream with an empty queue keeps the last report available.

Use the note field in Dream to queue new material from the same screen. The sample uses a synthetic optics procedure. With Gemini configured, Dream shares bounded excerpts of approved notes and their grounded lessons so Gemini can propose question variants and organize the memory wiki. Vicena assigns the expected answers, trains locally, and holds candidates that fail validation. If planning fails, Dream can use its local questions. The training panel charts sampled mean reward and mean policy entropy before the recorded updates. Held-out and retention checks determine whether Dream saves a candidate.

Open **Memory wiki** to see the Dream queue count and search processed notes, including the exact source text and any recorded corrections. Ask **Which memories do you have?** for a local inventory of saved notes, older versions, and notes waiting for Dream.

The [two-day demo](DEMO.md) includes paste-ready lessons and questions for checking correction and retention. Dream can train from explicit, compact notes such as the following.

```markdown
# Procedure: baseline drift
- Use a fresh reference first.

# Procedure: visual scoring
- Blind the scorer first.
```

For the next night, save a message such as `Correction: for baseline drift, use a sealed blank first.` Dream trains a local recall policy over frozen Qwen 3B features and tests whether the candidate selects the expected memory or abstains. The current decoder supports compact payloads of up to 12 native rows. Complete uploaded documents remain available through the saved-evidence chat mode.

For the 35B vector route, additionally provide the pinned 35B checkpoint, staged runner, codec, policy, and a state path inside an owner-only directory.

```sh
python -m scientist_ui.server --db "$run/chat.sqlite3" --port 8767 \
  --chat-model "$bench/model-3b" --chat-model-35b "$root/model" \
  --chat-stage-35b "$root/studio-shared34.d3DPj1" \
  --dream-codec "$codec" --dream-policy "$policy" --dream-state "$run/personal.safetensors"
```

The vector route reads the persisted policy state on CPU and hands vectors to the resident 35B model. Saved evidence mode uses text records. Dream uses the existing reward/replay policy for verified single-token color examples. For a compact procedure or experiment status, it also trains a small retrieval head on frozen 3B features. A positive paraphrase and an unrelated question check whether it can select the saved native 35B rows and abstain. The procedure checkpoint persists independently of the SQLite evidence index.

To reproduce the procedure demo, upload a Markdown or text file containing `The ZnO anneal schedule specifies 425C at 3C/min.` and run Dream. The current procedure trainer accepts compact explicit statements with a value that fits 12 native input embedding rows. Open **Local 35B · neural vectors** and ask about the temperature and ramp rate. **Inspect memory** shows the 35B prefill receipt. Longer narrative and complete multi-step procedures remain available through saved evidence.

Quick chat is available with `--openrouter-model` and `--openrouter-key-file` or the `OPENROUTER_API_KEY` environment variable. Its conversation stays separate from local chat and its generation request contains no saved notes or neural vectors. Its outgoing user messages also queue locally for Dream, unless you uncheck **Save for Dream**.

## Gemini with local Dream memory

Pass `--gemini-model gemini-3.8-flash` with `GEMINI_API_KEY` in the service environment or a private `--gemini-key-file`. Keep the local 3B checkpoint, Dream codec, policy, and state. Pass the pinned destination embedding table and tokenizer through `--dream-destination-model`. The Gemini route works without `--chat-model-35b` or `--chat-stage-35b`.

The source tree includes the training code. Use the pinned [3B download](../README.md#run-on-a-gpu-machine), train a [memory codec](../lightreason_memory/NEURAL.md#train-and-exercise-the-restart-gate) and a [recall policy](../lightreason_memory/RL.md#replicate-on-the-rtx-4090), and supply the compatible [35B destination artifacts](../docs/qwen35b.md). The Gemini route uses the destination embeddings and tokenizer for memory decoding.

```sh
umask 077
state_dir="$HOME/.local/state/vicena"
mkdir -p "$state_dir"
python -m scientist_ui.server --port 8767 --db "$state_dir/vicena.sqlite3" \
  --chat-model /path/to/model-3b --gemini-model "$VICENA_GEMINI_MODEL" \
  --dream-codec /path/to/codec.safetensors --dream-policy /path/to/rl.safetensors \
  --dream-destination-model /path/to/model-35b --dream-state "$state_dir/personal.safetensors"
```

Set `GEMINI_API_KEY` in the server environment and `VICENA_GEMINI_MODEL` to a model available to your project. Open [Vicena](http://127.0.0.1:8767/) and try the [two-day demo](DEMO.md).

Gemini becomes the default in the same chat. For each request, the local 3B memory selects relevant rows from its saved checkpoint. The local decoder maps those rows back to text. Gemini receives the visible chat context and that short readout. An abstention omits the readout. Dream training, the tensor checkpoints, and the evidence notebook stay local. **Inspect memory** shows the exact readout supplied to Gemini.

## Two-day Dream rehearsal

The CLI rehearsal remains available as a hardware regression test with fixed lessons. It checks transfer, retention, and abstention and asks each test question without the teaching turns in its context. The saved skill checkpoint contains the learned policy and native 35B rows.

Use a separate database and Dream state path for the rehearsal. A deployment can pass a verified direct-vector report via `--rehearsal-proof` to show the recorded 35B post-restart checks in chat. Vicena displays the report only when its checkpoint hash matches the promoted Day 2 checkpoint.


For a fresh hardware check, give the server a new private state directory and pass `--rehearsal-proof "$run/restart-proof.json"`. With the server running on port 8767, teach both days through one conversation:

```sh
python -m scientist_ui.run_rehearsal teach --run-dir "$run"
```

Restart that server with the same database and model state, then run:

```sh
python -m scientist_ui.run_rehearsal verify --run-dir "$run"
```

Verification checks the changed decision, retained skill, and abstention. For each memory hit, it compares the hash of the rows restored by the local 3B with the rows received at 35B prefill. The first 35B request after a restart can take several minutes. The recorded proof appears under Dream after verification.

For a Gemini-only deployment, pass `--mode gemini` to both commands. Use `--base-url` for a different loopback port. The Gemini verifier checks the decoded text and Gemini answer and writes `gemini-restart-proof.json` separately from the vector proof.
