# Third-party components and model terms

The repository's [0BSD license](LICENSE) covers original Lightreason source and documentation. Dependencies are installed separately under their upstream terms.

| External component | Upstream terms |
| --- | --- |
| Qwen2.5-3B-Instruct at `aa8e72537993ba99e69dfaafa59ed015b17504d1` | [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE). Research/evaluation only under this agreement. Commercial use requires a separate license from Alibaba Cloud. |
| Qwen3.6-35B-A3B | [Apache-2.0, upstream license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE). The exact benchmark revision and download command are in [35B reproduction](docs/qwen35b.md). |
| Inco AI DFlash2 draft for Qwen3.6-35B-A3B | [Apache-2.0, upstream model card](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2). |
| FreeToken and FreeToken-derived patches | [Apache-2.0, upstream license](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE). The patches modify FreeToken source in a separate checkout. The [Apache-2.0 text](edge35b/LICENSE-APACHE) accompanies the 35B research code. |
| Optional AMD PARD2 draft | [MIT, upstream model card](https://huggingface.co/amd/PARD2-Qwen3-8B). |
| Optional auxiliary Qwen3-0.6B tokenizer | [Apache-2.0, upstream model card](https://huggingface.co/Qwen/Qwen3-0.6B). |
| PyTorch | [BSD-style license and notices](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE). CUDA binaries also contain separately licensed components. |
| Transformers | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.57.1/LICENSE). |
| Triton | [MIT](https://github.com/triton-lang/triton/blob/v3.5.1/LICENSE). |

Model weights, tokenizer files, the trained LoRA/projector adapter, datasets, and dependency binaries are downloaded or supplied separately. 0BSD does not replace their terms. The download manifest points to the original Qwen license and files at the pinned revision.

The Scheduled baseline in `edge35b/expert_stream.py` adapts the eager Qwen MoE expert calculation from [Transformers 5.7.0](https://github.com/huggingface/transformers/blob/v5.7.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py). That upstream code is Apache-2.0.

The search-and-replace prototype follows the edit format described by [Aider](https://aider.chat/docs/more/edit-formats.html). The implementation is part of Lightreason; no Aider code is bundled.
