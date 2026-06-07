# OTN RCA — Multi-Cluster Kubernetes Platform

Infrastructure scaffolding that productionizes the OTN RCA pipeline on
**3× RTX 6000 Ada** as a multi-cluster platform with custom controllers and a
dedicated vLLM serving cluster. Full design rationale is in
`~/.claude/plans/nested-gliding-snowflake.md`.

## Topology (2 + 1)

| Box | GPUs | Cluster | Runs |
|-----|------|---------|------|
| box-1 | GPU0 + GPU1 | **A: training + control** | Kubeflow Pipelines, Training Operator, Volcano, MinIO, MLflow, ArgoCD, custom controllers, sim/train/eval Jobs |
| box-2 | GPU2 | **B: dedicated vLLM serving** | KServe + vLLM (Qwen2.5-7B + LoRA), rca-agent-api, rca-tools-svc |

Hardware constraints baked in: **no MIG** (whole-GPU allocation; time-slicing is
dev-only, see `gpu/`), **no NVLink** (single-GPU 7B; TP=2 only for 14B), single
serving GPU (1 vLLM replica; concurrency via continuous batching).

## Layout

```
deploy/
├── crds/                  DatasetBuild + AdapterPromotion CRDs (+ samples/)
├── controllers/           kubebuilder reconcile-logic spec (Go; kopf alt)
├── pipelines/             KFP DSL: simulate→data→train→eval
├── serving/               KServe vLLM runtime + InferenceService + agent/tools + KEDA
├── gpu/                   NVIDIA GPU Operator time-slicing config (dev only)
└── argocd/                ApplicationSet fanning out to both clusters
```

Repo-side code these manifests run:
- `serving/rca_agent_api.py` — FastAPI ReAct service (LLM via vLLM).
- `serving/rca_tools_api.py` — optional standalone tools service.
- `docker/Dockerfile.cpu` (sim/data/serving) and `docker/Dockerfile.gpu`
  (train/eval).

## End-to-end flow

```
DatasetBuild CR ─► (controller) CPU sim fan-out ─► versioned dataset in MinIO
       │
KFP pipeline ─► SFT/GRPO PyTorchJob (box-1 GPUs) ─► adapter + reward curve (MLflow/MinIO)
       │
AdapterPromotion CR ─► evaluate.py gate ─► patch KServe on box-2 (canary→100%)
       │
Serving (box-2): vLLM + rca-agent-api answer live RCA  ─►  alarm-flow CSV
```

## Bring-up order (maps to plan phases P0–P6)

```bash
# P0 bootstrap (both clusters): GPU Operator, MinIO, ArgoCD, monitoring.
kubectl apply -f deploy/gpu/time-slicing-configmap.yaml

# P2 dataset controller
kubectl apply -f deploy/crds/datasetbuild.yaml
kubectl apply -f deploy/crds/adapterpromotion.yaml
# (deploy the controllers from deploy/controllers per its README)

# P3 training
python deploy/pipelines/rca_pipeline.py   # compile, then upload to KFP

# P4 serving (cluster B)
kubectl apply -f deploy/serving/vllm-servingruntime.yaml
kubectl apply -f deploy/serving/inferenceservice-rca-7b.yaml
kubectl apply -f deploy/serving/rca-agent-api.yaml
kubectl apply -f deploy/serving/keda-scaledobject.yaml
# optional: kubectl apply -f deploy/serving/rca-tools-svc.yaml

# P5 promotion
kubectl apply -f deploy/crds/samples/adapterpromotion-7b.yaml

# P6 GitOps everything
kubectl apply -f deploy/argocd/applicationset.yaml
```

## Verify

```bash
# data
kubectl apply -f deploy/crds/samples/datasetbuild-grpo.yaml
kubectl get dsb grpo-nightly -w          # → Ready, status.datasetURI set

# serving
curl -s http://<rca-agent-api>/readyz
curl -s -XPOST http://<rca-agent-api>/rca \
  -H 'content-type: application/json' \
  -d '{"user_prompt":"<observations...>","data_dir":"/data/run_1"}'
# → {predicted_event, predicted_board, alarm_flow_csv, ...}
```

Replace `REGISTRY/` and the box-2 API server URL before applying.
