<img src="assets/lightreason-banner.svg" alt="Lightreason" width="620">

# Lightreason

Lightreason runs the original BF16 **Qwen2.5-3B-Instruct** on NVIDIA CUDA.

Live Studio is a simple, interactive demo. Build and preview a web page, then try the same prompt with the standard and Lightreason Exact runners to see how fast each one responds.

The source is [0BSD](LICENSE). [Transparent logo](demo_chat/static/lightreason.svg). The pinned Qwen checkpoint has separate [research-only terms](THIRD_PARTY.md). This release contains code and input fixtures, with model downloads handled separately.

## Run on a GPU machine

Live Studio requires a **Linux machine with an RTX 4090** and a driver compatible with CUDA 12.8. Standard and Lightreason Exact use only the Qwen target. Your browser can run on a different machine.

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

**Lightreason Exact** speeds up target-only decoding with CUDA graphs, a compact KV cache, packed BF16 projections, and a screened output head that scores survivors with the original BF16 weights. **Draft-assisted research** uses an auxiliary proposer and verifies candidate trees with the original Qwen target. The roughly 9× lab figures apply to that research runner. It was 1.062× faster than the prior optimized runner with identical tokens on all 36 held-out comparisons. The benchmark against original Hugging Face includes outputs that differ. The [implementation walkthrough and reproduction commands](docs/benchmarks.md) explain the changes, timings, and adapter requirement.

For other hardware, follow the [porting guide](docs/porting.md). Start with the Standard CLI on other BF16-capable NVIDIA CUDA GPUs and establish a baseline on that device. AMD, Apple Silicon, and CPU require backend work. From the cloned repository, ask Codex to **use `$lightreason-hardware`**. With another coding agent, point it at the [hardware skill](.agents/skills/lightreason-hardware/SKILL.md).

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
