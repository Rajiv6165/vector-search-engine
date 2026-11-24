# Production Deployment & Operations Guide

This guide covers production configuration, process safety rules, environment variables, and deployment instructions for `vsearch`.

---

## 1. Runtime Configuration & Environment Variables

All runtime behavior is governed via environment variables.

| Variable | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | `8000` | Positive integer | The port on which Uvicorn binds and listens. |
| `VSEARCH_DATA_DIR` | `./vsearch_data` | Valid filesystem path | Directory storing Write-Ahead Log (WAL), snapshot checkpoints, and memory-mapped vector files. (Aliases `VSEARCH_PERSIST_DIR` for backward compatibility). |
| `VSEARCH_API_KEY` | `None` (fails closed) | String | **Write API Key**. Required for state-modifying operations (`POST /vectors`, `POST /vectors/batch`, `DELETE /vectors/{id}`). Also permits read queries. Refuses to start if unset unless `VSEARCH_ALLOW_INSECURE=1`. |
| `VSEARCH_READ_KEY` | `None` (unauthenticated) | String | **Read API Key**. Permits read-only operations (`POST /search`, `POST /search/batch`, `GET /stats`). Strictly rejected on write endpoints with HTTP 401. |
| `VSEARCH_ALLOW_INSECURE` | `0` | `0`, `1` | If `1`, permits unauthenticated local development without setting `VSEARCH_API_KEY`. Defaults to `0` (fails closed). |
| `VSEARCH_MAX_VECTORS` | `None` (unlimited) | Positive integer | Hard limit on indexed vectors. When reached, further inserts return HTTP 400. |
| `VSEARCH_MAX_K` | `1000` | Positive integer | Maximum nearest neighbors ($k$) allowed per search request. Requests exceeding this return HTTP 400. |
| `VSEARCH_MAX_EF` | `500` | Positive integer | Maximum search queue depth ($ef$) allowed per search request. Requests exceeding this return HTTP 400. |
| `VSEARCH_MAX_BATCH` | `1000` | Positive integer | Maximum number of vectors allowed in a single batch insert or batch search request. Returns HTTP 400 if exceeded. |
| `VSEARCH_MAX_REQUEST_BYTES`| `10485760` (10 MB) | Positive integer | Maximum allowed HTTP request payload size in bytes. Requests exceeding this return HTTP 413 Payload Too Large. |
| `VSEARCH_RATE_LIMIT_RPS` | `100` | Positive integer | Sliding-window per-IP rate limit on `/search` and `/search/batch`. Returns HTTP 429 Too Many Requests if exceeded. |
| `VSEARCH_SEED_ON_START` | `0` | `0`, `1` | If `1`, automatically generates and seeds ~20,000 clustered demo vectors on startup (ideal for ephemeral cloud tiers). |
| `VSEARCH_SNAPSHOT_INTERVAL`| `300` (seconds) | Positive integer | Background periodic snapshot checkpoint interval in seconds. |

---

## 2. Single-Process Architecture & Concurrency Safety

> [!CRITICAL]
> **Single-Process Constraint**: `vsearch` must strictly run as a **single process** (`--workers 1`).

If `WEB_CONCURRENCY > 1` or `--workers > 1` is detected in the environment or command-line arguments, `vsearch` **refuses to start and aborts immediately** with a descriptive `RuntimeError`:

```
RuntimeError: vsearch must run as a single process (WEB_CONCURRENCY=2 detected). 
Multi-worker configurations corrupt in-memory HNSW graphs and memmap WAL state. 
Use single-process concurrency.
```

### Exclusive Directory Lockfile (`vsearch.lock`)
To prevent accidental dual-process attachment to the same data directory across multiple containers or processes, `vsearch` acquires an exclusive filesystem lock on `vsearch.lock` inside `VSEARCH_DATA_DIR` at startup:
- **POSIX / Linux**: Managed via `fcntl.flock(fd, LOCK_EX | LOCK_NB)`.
- **Windows**: Managed via `msvcrt.locking(fd, LK_NBLCK, 1)`.

If a second process attempts to bind to an already attached data directory, startup is rejected immediately:
```
RuntimeError: Another vsearch process is currently running on data directory './vsearch_data'. 
Exclusive lock acquisition failed; only one process may access the data directory at a time.
```

### VM Sizing & Cold-Start Benchmarks
Peak resident memory (RSS) was measured during snapshot serialization at $N=20,000$ (dim=64):
- **Base RSS**: 105.14 MB
- **Peak RSS during snapshot**: 105.88 MB
- **Peak Memory Spike**: +0.74 MB
- **Serialization duration**: 0.248 s
- **Remaining headroom in 512 MB VM**: 406.12 MB (79.3% free)

**Conclusion**: A **512 MB VM** provides ample headroom for production deployments up to 50,000 vectors.

**CPU-Throttled Cold Start (`--cpus=0.25`)**:
- Cold container boot to `/health` (HTTP 200): **5.046 s**
- Cold container boot to first successful `/search` (HTTP 200): **5.110 s** (snapshot loads in ~64 ms)

---

## 3. Endpoints & Authentication Matrix

| Endpoint | Method | Required Permission | Accepted Key |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | **Public** | None required (always open) |
| `/stats` | `GET` | **Read** | `VSEARCH_READ_KEY` or `VSEARCH_API_KEY` |
| `/search` | `POST` | **Read** | `VSEARCH_READ_KEY` or `VSEARCH_API_KEY` |
| `/search/batch` | `POST` | **Read** | `VSEARCH_READ_KEY` or `VSEARCH_API_KEY` |
| `/vectors` | `POST` | **Write** | `VSEARCH_API_KEY` only |
| `/vectors/batch` | `POST` | **Write** | `VSEARCH_API_KEY` only |
| `/vectors/{id}` | `DELETE` | **Write** | `VSEARCH_API_KEY` only |

---

## 4. Platform Deployment

### Option A: Fly.io (with Persistent Volume)

Fly.io provides global container deployment with dedicated volume persistence.

1. **Verify configuration (`fly.toml`)**:
   ```toml
   app = "vsearch-engine"
   primary_region = "iad"

   [build]
     dockerfile = "Dockerfile"

   [env]
     PORT = "8080"
     VSEARCH_DATA_DIR = "/data"
     VSEARCH_SNAPSHOT_INTERVAL = "300"
     VSEARCH_MAX_VECTORS = "100000"
     VSEARCH_MAX_K = "1000"
     VSEARCH_MAX_BATCH = "1000"
     VSEARCH_SEED_ON_START = "0"

   [mounts]
     source = "vsearch_data"
     destination = "/data"

   [http_service]
     internal_port = 8080
     force_https = true
     auto_stop_machines = "stop"
     auto_start_machines = true
     min_machines_running = 1
     processes = ["app"]

   [[vm]]
     size = "shared-cpu-1x"
     memory = "1gb"
   ```

2. **Deploy Commands**:
   ```bash
   # 1. Authenticate
   fly auth login

   # 2. Create app
   fly apps create vsearch-engine

   # 3. Create persistent storage volume (1 GB in region iad)
   fly volumes create vsearch_data --size 1 --region iad

   # 4. Set platform secrets
   fly secrets set VSEARCH_API_KEY="your-secret-write-key" \
                   VSEARCH_READ_KEY="your-public-read-key"

   # 5. Deploy container
   fly deploy
   ```

---

### Option B: Render (Web Service / Blueprint)

Render allows instantaneous Docker deployments connected directly to GitHub.

1. **Verify Blueprint (`render.yaml`)**:
   ```yaml
   services:
     - type: web
       name: vsearch-demo
       runtime: docker
       plan: free
       healthCheckPath: /health
       region: singapore
       branch: main
       autoDeploy: true
       envVars:
         - key: PORT
           value: 8000
         - key: VSEARCH_DATA_DIR
           value: /app/vsearch_data
         - key: VSEARCH_SNAPSHOT_INTERVAL
           value: "300"
         - key: VSEARCH_SEED_ON_START
           value: "0"
         - key: VSEARCH_MAX_VECTORS
           value: "50000"
         - key: VSEARCH_MAX_K
           value: "100"
         - key: VSEARCH_MAX_EF
           value: "500"
         - key: VSEARCH_MAX_BATCH
           value: "100"
         - key: VSEARCH_API_KEY
           sync: false
         - key: VSEARCH_READ_KEY
           sync: false
   ```

2. **Deploy Steps**:
   - Navigate to [Render Dashboard](https://dashboard.render.com/) -> **New +** -> **Blueprint**.
   - Connect the `vector-search-engine` repository.
   - Render will parse `render.yaml`.
   - Enter secret values for `VSEARCH_API_KEY` and `VSEARCH_READ_KEY`.
   - Click **Apply**.
   > *Note on Render Free Tier*: Persistent disks require a paid Starter instance. On the Free tier, setting `VSEARCH_SEED_ON_START=1` ensures the 20,000-vector demo dataset is automatically generated on cold-start boots.

---

## 5. Verification & Smoke Testing

Once deployed, run the comprehensive verification suite against the live deployment URL:

```bash
python scripts/smoke_test.py --url https://your-deployment-url.com \
                            --write-key "your-secret-write-key" \
                            --read-key "your-public-read-key"
```

The smoke test verifies:
1. `GET /health` returns `200` with `status: ok` without credentials.
2. `GET /stats` returns `200` with `native: true` and current version.
3. Unfiltered search returns exactly $k$ nearest neighbors.
4. Filtered search returns only vectors satisfying the categorical metadata constraint.
5. `VSEARCH_READ_KEY` is strictly blocked from write endpoints (`POST /vectors`, `DELETE /vectors/{id}`).
6. `VSEARCH_API_KEY` can perform writes and deletes.
