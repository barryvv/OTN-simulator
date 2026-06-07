# Custom controllers (kubebuilder / Go)

Two thin controllers fill the gaps Kubeflow leaves. Scaffold with:

```bash
kubebuilder init --domain rca.dev --repo github.com/barryvv/otn-platform
kubebuilder create api --group otn --version v1alpha1 --kind DatasetBuild
kubebuilder create api --group otn --version v1alpha1 --kind AdapterPromotion
```

The CRD schemas are already authored in `deploy/crds/*.yaml`; generate the Go
types to match, then implement the reconcile loops below. (A `kopf`/Python
implementation is a drop-in alternative if the team prefers the repo language.)

## DatasetBuild reconciler

State machine: `Pending → Simulating → Generating → Ready` (`Failed` on error).

1. **Pending → Simulating:** compute a content hash of
   `(spec.type, topology, simulatorOverrides, numFailures)`. If
   `s3://<bucket>/<type>/<hash>/train.jsonl` already exists → jump to `Ready`
   (idempotent). Else create an indexed `Job` (or Argo Workflow) with
   `completions=numFailures`, `parallelism=spec.parallelism`, CPU image,
   running `run.py → convert → otn_simulator.py --seed $JOB_COMPLETION_INDEX`,
   syncing each run to MinIO.
2. **Simulating → Generating:** when the sim Job completes, launch a single
   `Job` running the matching generator (`generate_sft_data.py` /
   `generate_grpo_data.py` / `sfc_bfl.run_all`) over the synced runs, writing
   `train.jsonl` to the hashed prefix.
3. **Generating → Ready:** set `status.datasetURI`, `status.contentHash`,
   `status.stats` (example count from the JSONL); register lineage in MLflow.

Watch owned Jobs; surface failures to `status.phase=Failed` + `message`.

## AdapterPromotion reconciler (cross-cluster)

State machine: `Pending → Evaluating → Gated → Canary → Promoted`
(`RolledBack` / `Failed`).

1. **Pending → Evaluating:** create a GPU `Job` (GPU image) running
   `evaluate.py --run-inference --adapter <adapterURI> --test-file <evalDataset>`,
   writing `results_summary.csv` to MinIO. Parse `eventAcc`, `boardF1`,
   `e2eScore` into `status.metrics`.
2. **Evaluating → Gated:** read current production metrics
   (`status.baselineMetrics`) from the live endpoint's last promotion. If any
   `spec.gate.*` threshold fails OR the candidate regresses vs baseline → set
   `phase=Gated`, stop (a human can override).
3. **Gated(pass) → Canary:** using the kubeconfig from the cross-cluster Secret
   (`spec.servingCluster`), patch the serving cluster's KServe
   `InferenceService/<endpoint>`: append `name=<adapterURI>` to the
   `otn.rca.dev/lora-modules` annotation and set canary traffic to
   `spec.canaryPercent`. Record `status.loraModuleName`.
4. **Canary → Promoted:** if `spec.autoPromote` and canary error rate / latency
   stay healthy for the soak window, set traffic to 100% and update
   `baselineMetrics`. On regression → revert the annotation/traffic →
   `RolledBack`.

### RBAC / wiring
- In-cluster: `jobs`, `pods/log`, `configmaps`, plus the two CRDs +
  `*/status`.
- Cross-cluster: a `Secret` (type `Opaque`) holding `serving.kubeconfig`,
  scoped (via the remote ServiceAccount) to `get/patch` on
  `inferenceservices` in the `serving` namespace only.
- Artifacts: S3 creds for MinIO (read adapters/datasets, write reports).
