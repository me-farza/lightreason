# Lightreason

Lightreason is a local inference lab and interactive Studio for running Qwen faster on one RTX 4090. The experiments use the original BF16 weights for **Qwen2.5-3B-Instruct** (3.09B parameters) and the text model of **Qwen3.6-35B-A3B** (34.66B parameters).

![Decode throughput in tokens per second for both models on one RTX 4090. Qwen2.5-3B goes from 63.0 to 139.6 in the exact target-only suite and reaches 525.2 on a complete matching weather response with a draft. Qwen3.6-35B weather decoding goes from 7.79 to 46.80 across versions, with different output from Scheduled. The packed runner reaches 51.08 on coding and matches its frozen eight-position control.](assets/speedup-history.png)

- **3B · the easiest comparison to run.** Lightreason Exact cut warmed prompt-plus-generation time by **2.19×** versus Hugging Face, with **30/30** timing outputs matching. The separate draft-assisted runner reached **8.88×** on a 492-token weather dashboard through the model’s natural stop. Its best **10.79×** exact result was a **128-token coding slice**. [3B evidence and methods](docs/performance.md#qwen25-3b)
- **35B · 51.08 tok/s on coding.** The fastest retained packed runner matched the frozen eight-position verifier on **20/20** comparisons. On the same 256-token weather prompt, recorded decode rose from **7.79 to 46.80 tok/s** across versions. Those weather runs came from separate sessions and the output changed at token 31 versus the early Scheduled runner. [35B evidence and methods](docs/performance.md#qwen36-35b)

The larger model’s text weights occupy 64.56 GiB. Lightreason keeps most expert weights in system RAM and moves the ones Qwen actually uses. When a small draft model can predict several tokens, Qwen checks them together and can reuse some of the same weights across those checks. The smaller model fits on the GPU, where reusing prepared GPU work and checking several tokens together saves time. Only the original Qwen target commits output.

Live Studio is a simple, interactive way to test speed. Build and preview a web page, then run the same prompt through Standard and Lightreason Exact to watch each response stream and compare its timings.

[Run 3B Studio](#run-on-a-gpu-machine) · [Reproduce the 35B benchmarks](docs/qwen35b.md) · [Port to other hardware](docs/porting.md) · [Benchmark data and chart source](docs/performance.md)

## Run on a GPU machine

The public **3B Studio** quickstart uses Linux with an RTX 4090 and a driver compatible with CUDA 12.8. Standard and Lightreason Exact use only the Qwen target. Your browser can run on a different machine. The 35B runners use a separate benchmark setup with a larger system RAM requirement. [35B setup](docs/qwen35b.md)

Install Python 3.12, its venv package and headers, and a C compiler for Triton. On Ubuntu 24.04, install `build-essential python3.12-venv python3.12-dev`. Run `nvidia-smi` on the GPU machine to confirm the driver sees the card. The download is about 6.2 GB and the downloader requires another 2 GiB of free space. Budget separately for Python packages and compilation caches.

```bash
git clone https://github.com/me-farza/lightreason.git
cd lightreason
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c 'import torch; name = torch.cuda.get_device_name(); print(name); assert "4090" in name and torch.version.hip is None and torch.cuda.is_bf16_supported()'
python download_3b.py --destination model-3b
python -m demo_chat.server --model-path model-3b
```

The setup pins PyTorch 2.9.1 with CUDA 12.8 and Transformers 4.57.1. `download_3b.py` checks the official objects against the pinned [download plan](results3b/download-plan.json). Read the [Qwen license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE) before downloading or using the model. The original release source is tagged [`v0.1.0`](https://github.com/me-farza/lightreason/releases/tag/v0.1.0).

Leave the server running and open **<http://127.0.0.1:8765/>**. For a remote GPU, first run this tunnel in a separate terminal on your browser machine.

```bash
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 YOUR_GPU_HOST
```

1. Wait for Qwen to load. You can check for `"ready": true` with `curl -fsS http://127.0.0.1:8765/api/status`.
2. Open **Compare runners**, select **Qwen · Lightreason Exact**, choose a sample prompt, and click **Run comparison**. Studio runs Standard first and then Exact on the same GPU and compares the full token sequences.
3. Repeat the same prompt to see warm performance. Exact prepares graphs automatically, and Studio reports preparation separately from generation. Use **Save results** to export the timings and match result. Stop the server with `Ctrl+C`.

For a headless smoke test, run the same prompt and token cap through each CLI engine. The JSON records preparation, generation, token IDs, and stop reason. Use Studio or the repeated benchmark harness for speed measurements.

```bash
python run_model_3b.py --model-path model-3b --engine original --tokens 128 --json --prompt 'Explain why the sky is blue.'
python run_model_3b.py --model-path model-3b --engine bucketed --tokens 128 --json --prompt 'Explain why the sky is blue.'
```

## Understand the speedup and port it

**Lightreason Exact** records and replays GPU work, keeps a compact attention cache, combines BF16 projections, and screens vocabulary candidates before scoring survivors with the original output weights. The **draft-assisted** 3B runner verifies candidate trees and coordinates acceptance on the GPU. On 36 held-out comparisons it was **1.062×** faster with the same outputs as the preceding draft-assisted runner. It matched Hugging Face on **12/36**. [Implementation and reproduction](docs/benchmarks.md)

For **35B**, the biggest gain came from reducing expert reads and overlapping CPU and GPU work. The latest retained runner uses seven DFlash2 proposals, an eight-position verifier, packed host banks on 34 layers, and a lossless GPU replica tier. It reconstructs the original BF16 bits before the expert math. [Runtime, profiles, and reproduction](docs/qwen35b.md)

For other hardware, measure the unmodified model on your device first. Use the [porting guide](docs/porting.md) to identify whether GPU work, memory, or transfers limit the full request. From the cloned repository, ask Codex to **use `$lightreason-hardware`**. With another coding agent, point it at the [hardware skill](.agents/skills/lightreason-hardware/SKILL.md).

## Edit a page

Open **Build preview** to generate single-file HTML and render it in the sandboxed preview. Studio reports generation and browser rendering separately.

**New app** generates a page. **Edit current app** supplies its HTML and asks Qwen for the full update. **Patch edit (experimental)** asks Qwen for one small JSON search-and-replace, checks it against the current HTML, and renders the reconstructed document after successful completion. Use the full edit path for larger changes or to retry a rejected patch. A successful patch currently reloads the sandboxed preview, which restarts page state.

The two patch formats, their local HTTP benchmarks, and known failures are described in [the benchmark notes](docs/benchmarks.md).

## Optional draft-assisted research

Standard and Exact run with the target checkpoint alone. The optional Turbo runner and the GPU coordination benchmark additionally require the pinned AMD PARD2 draft directory and the matching trained LoRA/projector checkpoint supplied locally. See [the optional setup and reproduction commands](docs/benchmarks.md). That adapter is separate from this source release.

Run the CPU-side Studio contract tests from the repository root:

```bash
python -m unittest discover -s demo_chat -p 'test_*.py'
node demo_chat/test_preview.js
```

## License

Original Lightreason source and documentation use [0BSD](LICENSE). FreeToken patches and the Qwen checkpoints have separate upstream terms. The repository contains source, benchmark records, and input fixtures. Download the checkpoints separately. [Third-party terms and credits](THIRD_PARTY.md) · [Lightreason logo](demo_chat/static/lightreason.svg)
