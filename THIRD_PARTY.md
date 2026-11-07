# Third-party components and model terms

The repository's [0BSD license](LICENSE) covers original Lightreason source and documentation. Dependencies are installed separately under their upstream terms.

| External component | Upstream terms |
| --- | --- |
| Qwen2.5-3B-Instruct at `aa8e72537993ba99e69dfaafa59ed015b17504d1` | [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE). Research/evaluation only under this agreement. Commercial use requires a separate license from Alibaba Cloud. |
| Optional AMD PARD2 draft | [MIT, upstream model card](https://huggingface.co/amd/PARD2-Qwen3-8B). |
| Optional auxiliary Qwen3-0.6B tokenizer | [Apache-2.0, upstream model card](https://huggingface.co/Qwen/Qwen3-0.6B). |
| PyTorch | [BSD-style license and notices](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE). CUDA binaries also contain separately licensed components. |
| Transformers | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.57.1/LICENSE). |
| Triton | [MIT](https://github.com/triton-lang/triton/blob/v3.5.1/LICENSE). |

Model weights, tokenizer files, the trained LoRA/projector adapter, datasets, and dependency binaries are downloaded or supplied separately. 0BSD does not replace their terms. The download manifest points to the original Qwen license and files at the pinned revision.

The search-and-replace prototype follows the edit format described by [Aider](https://aider.chat/docs/more/edit-formats.html). The implementation is part of Lightreason; no Aider code is bundled.
