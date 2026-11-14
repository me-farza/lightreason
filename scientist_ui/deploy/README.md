# Vicena Lightreason on CapRover

Build from the **Lightreason repository root**. Configure CapRover's Captain Definition Path as `scientist_ui/deploy/captain-definition`. For a prebuilt image, build it from that same checkout and select that image in the CapRover app's deployment configuration. The existing MirrorThink checkout does not need to contain Lightreason.

```sh
docker build --platform linux/amd64 -f scientist_ui/deploy/Dockerfile -t vicena-lightreason .
```

Use a dedicated app with one replica. Set its **Container HTTP Port** to `8080`. That port serves the invite gateway, which forwards authorized requests to the backend inside the container at `127.0.0.1:8783`. Do not publish a host port directly for either process. CapRover terminates TLS. Use `VICENA_PUBLIC_ORIGIN=https://vicena-tmp.mirrorthink.ai` and `VICENA_PUBLIC_PATH=/lightreason` when the public site routes this separate app under the existing Vicena hostname. The routing layer must preserve both that path prefix and the public Host. For a dedicated hostname, leave `VICENA_PUBLIC_PATH` unset.

| Container path | Required content |
| --- | --- |
| `/data` | Persistent volume writable by container UID/GID `10001:10001`. Startup creates a private `/data/demo` folder for SQLite and Dream checkpoints. If using a bind mount, make the mount root writable by that UID. |
| `/models/qwen3b` | Original pinned Qwen2.5-3B-Instruct checkpoint with all model shards, index, tokenizer and config files. |
| `/models/qwen35b-input` | Original pinned Qwen3.6-35B-A3B config, tokenizer, download manifest, weight index and the complete original shard that contains the input embedding tensor. |
| `/models/codec.safetensors`, `/models/policy.safetensors` | Matching Lightreason Dream codec and policy artifacts. |
| `/run/vicena/gemini-key`, `/run/vicena/invite` | Optional runtime credential files, owned by UID `10001` with mode `0600`. |

Mount the model and credential folders read-only where the hosting platform supports that. See the [3B download setup](../../README.md#run-on-a-gpu-machine) and [35B download manifest](../../docs/qwen35b.md#download-and-verify-the-models). Model and policy artifacts are provisioned separately from the image. The 35B input shard supports the local decoder. This deployment runs the real 3B Dream policy on CPU and uses Gemini for generation and optional question planning.

Alternatively set `GEMINI_API_KEY` and `VICENA_DEMO_INVITE` as runtime app variables. Startup passes both secrets to child processes via private ephemeral files. Use a randomly generated URL-safe invite at least 32 characters long. The same invite unlocks one shared fictional workspace. Keep the invite and API key out of the image, build context, repository, and public launch URL. `VICENA_GEMINI_MODEL` optionally overrides the default `gemini-3.8-flash`.

Verify the unauthenticated public URL serves the invite landing and the wiki API returns 401. Then enter an invite, queue a fictional note, run Dream, and check that the note and recall survive replacing the container with the same volume. Allow initial model file verification and CPU inference to complete before timing. The container healthcheck checks access gating and app readiness. A real Dream and recall exercise validates the model artifacts and Gemini path.
