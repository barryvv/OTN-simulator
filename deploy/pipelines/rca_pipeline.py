"""Kubeflow Pipelines DSL: simulate → data → train → eval for OTN RCA.

This is the training-side orchestration on the control/training cluster (box-1).
It wraps the repo's existing CLIs as containerized steps. The DatasetBuild and
AdapterPromotion *controllers* handle the parts KFP doesn't (CPU fan-out with
content-hash versioning, and cross-cluster eval-gated promotion); this pipeline
covers the linear train path and is what a CronWorkflow re-triggers.

Compile:
    python deploy/pipelines/rca_pipeline.py    # -> rca_pipeline.yaml
Then upload to KFP (UI or `kfp pipeline upload`).

Notes:
- GPU steps request a WHOLE GPU (no MIG on RTX 6000 Ada).
- The GRPO step carries the learning fixes from POST_TRAINING_DEBUGGING.md:
  Dr. GRPO advantages (no --scale-rewards), token-level loss, --reward-fn
  continuous, --group-size 8.
"""

from kfp import dsl

CPU_IMAGE = "REGISTRY/otn-rca-cpu:latest"
GPU_IMAGE = "REGISTRY/otn-rca-gpu:latest"


@dsl.container_component
def simulate(num_failures: int, out_uri: str):
    return dsl.ContainerSpec(
        image=CPU_IMAGE,
        command=["bash", "-lc"],
        args=[
            f"python run.py && "
            f"python convert_graphml_to_topology.py --out topology.yaml && "
            f"for i in $(seq 1 {num_failures}); do "
            f"  python otn_simulator.py --seed $i "
            f"    --output-dir /work/runs/run_$i; done && "
            f"aws s3 sync /work/runs {out_uri}/runs"
        ],
    )


@dsl.container_component
def build_grpo_data(runs_uri: str, dataset_uri: str):
    return dsl.ContainerSpec(
        image=CPU_IMAGE,
        command=["bash", "-lc"],
        args=[
            f"aws s3 sync {runs_uri}/runs /work/runs && "
            f"python generate_grpo_data.py --runs-dir /work/runs "
            f"  --outdir /work/grpo && "
            f"aws s3 cp /work/grpo/train.jsonl {dataset_uri}"
        ],
    )


@dsl.container_component
def train_grpo(dataset_uri: str, adapter_uri: str, max_steps: int):
    return dsl.ContainerSpec(
        image=GPU_IMAGE,
        command=["bash", "-lc"],
        args=[
            f"aws s3 cp {dataset_uri} /work/train.jsonl && "
            f"python train_grpo_scratch.py "
            f"  --base-model Qwen/Qwen2.5-7B-Instruct "
            f"  --train-file /work/train.jsonl "
            f"  --output-dir /work/adapter "
            f"  --device cuda --dtype bfloat16 "
            f"  --group-size 8 --max-steps {max_steps} "
            f"  --lr 1e-4 --lr-scheduler cosine --warmup-steps 10 "
            f"  --reward-fn continuous && "
            f"aws s3 sync /work/adapter/final {adapter_uri}"
        ],
    )


@dsl.container_component
def evaluate(adapter_uri: str, eval_uri: str, report_uri: str):
    return dsl.ContainerSpec(
        image=GPU_IMAGE,
        command=["bash", "-lc"],
        args=[
            f"aws s3 sync {adapter_uri} /work/adapter && "
            f"aws s3 cp {eval_uri} /work/eval.jsonl && "
            f"python evaluate.py --run-inference "
            f"  --base-model Qwen/Qwen2.5-7B-Instruct "
            f"  --adapter /work/adapter "
            f"  --test-file /work/eval.jsonl "
            f"  --results-dir /work/results && "
            f"aws s3 sync /work/results {report_uri}"
        ],
    )


@dsl.pipeline(name="otn-rca-train", description="simulate→data→train→eval")
def rca_pipeline(
    num_failures: int = 200,
    max_steps: int = 300,
    bucket: str = "s3://datasets/grpo/run",
    adapter_bucket: str = "s3://adapters/qwen7b-grpo/run",
    eval_uri: str = "s3://datasets/eval/holdout.jsonl",
    report_bucket: str = "s3://reports/qwen7b-grpo/run",
):
    sim = simulate(num_failures=num_failures, out_uri=bucket)

    data = build_grpo_data(runs_uri=bucket, dataset_uri=f"{bucket}/train.jsonl")
    data.after(sim)

    tr = train_grpo(
        dataset_uri=f"{bucket}/train.jsonl",
        adapter_uri=adapter_bucket,
        max_steps=max_steps,
    )
    tr.after(data)
    tr.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)

    ev = evaluate(adapter_uri=adapter_bucket, eval_uri=eval_uri, report_uri=report_bucket)
    ev.after(tr)
    ev.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    # Promotion is intentionally NOT a pipeline step — create an
    # AdapterPromotion CR pointing at `adapter_bucket` to gate + canary across
    # clusters.


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(rca_pipeline, "rca_pipeline.yaml")
    print("[ok] wrote rca_pipeline.yaml")
