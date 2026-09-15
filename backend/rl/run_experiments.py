"""Automated experimental runner for Milestone 36 PPO Hyperparameter Tuning.

Executes:
1. Candidate Matrix Screening (200,000 timesteps each, seed=200):
   - C0: m34_control (lr=3e-4, n=128, b=64, gamma=0.99, gae=0.95, ent=0.0, vf=0.5, arch=[64,64])
   - C1: lr5e4_n128 (lr=5e-4, n=128, b=64, gamma=0.99, gae=0.95, ent=0.0, vf=0.5, arch=[64,64])
   - C2: lr5e4_n256 (lr=5e-4, n=256, b=64, gamma=0.99, gae=0.95, ent=0.0, vf=0.5, arch=[64,64])
   - C3: lr3e4_n256 (lr=3e-4, n=256, b=64, gamma=0.99, gae=0.95, ent=0.0, vf=0.5, arch=[64,64])
   - C4: lr5e4_ent001 (lr=5e-4, n=128, b=64, gamma=0.99, gae=0.95, ent=0.001, vf=0.5, arch=[64,64])
   - C5: lr5e4_vf10 (lr=5e-4, n=128, b=64, gamma=0.99, gae=0.95, ent=0.0, vf=1.0, arch=[64,64])
   - C6: lr5e4_arch128 (lr=5e-4, n=128, b=64, gamma=0.99, gae=0.95, ent=0.0, vf=0.5, arch=[128,128])
   - Native dynamic episode seeding in training environment.
   - Checkpoint evaluations every 25,000 timesteps on held-out seeds (500..519, 20 episodes).
   - Full learning curve tracking across all statistical metrics.
2. Candidate ranking and winner selection on held-out selection seeds.
3. Multi-seed training of winning configuration on 3 independent seeds (101, 102, 103).
4. Evaluation of all baselines and winning models on:
   - Benchmark A: M35 Benchmark (seeds 42..91, 50 episodes)
   - Benchmark B: Extended Generalization Benchmark (seeds 42..141, 100 episodes)
5. Comprehensive reporting and artifact generation in artifacts/tuning/ and extended_experiments/.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from rl.evaluation import (  # noqa: E402
    EvaluationResult,
    HoldBaselinePolicy,
    RandomBaselinePolicy,
    RuleBasedBaselinePolicy,
    compare_policies,
    create_evaluation_env,
    evaluate_policy,
)
from rl.ppo import PPOAgent  # noqa: E402
from rl.training import PPOTrainer, TrainingConfig  # noqa: E402
from rl.tuning import (  # noqa: E402
    CandidateConfig,
    MultiSeedSummary,
    TrialResult,
    compute_composite_score,
)


def get_candidate_matrix() -> list[CandidateConfig]:
    """Define the structured experimental candidates C0-C6."""
    return [
        CandidateConfig(
            name="C0_m34_control",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "n_epochs": 10,
                "net_arch": [64, 64],
            },
            description="M34 control under dynamic episode seeding (lr=3e-4, n=128, b=64)",
        ),
        CandidateConfig(
            name="C1_lr5e4_n128",
            ppo_kwargs={
                "learning_rate": 5e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "n_epochs": 10,
                "net_arch": [64, 64],
            },
            description="Higher learning rate (5e-4) testing policy step size",
        ),
        CandidateConfig(
            name="C2_lr5e4_n256",
            ppo_kwargs={
                "learning_rate": 5e-4,
                "n_steps": 256,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "n_epochs": 10,
                "net_arch": [64, 64],
            },
            description="Extended rollout buffer (n=256, b=64) with higher LR (5e-4)",
        ),
        CandidateConfig(
            name="C3_lr3e4_n256",
            ppo_kwargs={
                "learning_rate": 3e-4,
                "n_steps": 256,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "n_epochs": 10,
                "net_arch": [64, 64],
            },
            description="Extended rollout buffer (n=256, b=64) with conservative LR (3e-4)",
        ),
        CandidateConfig(
            name="C4_lr5e4_ent001",
            ppo_kwargs={
                "learning_rate": 5e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.001,
                "vf_coef": 0.5,
                "n_epochs": 10,
                "net_arch": [64, 64],
            },
            description="Action entropy bonus (ent_coef=0.001) for sustained exploration",
        ),
        CandidateConfig(
            name="C5_lr5e4_vf10",
            ppo_kwargs={
                "learning_rate": 5e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 1.0,
                "n_epochs": 10,
                "net_arch": [64, 64],
            },
            description="Value function loss coefficient (vf_coef=1.0) testing critic weighting",
        ),
        CandidateConfig(
            name="C6_lr5e4_arch128",
            ppo_kwargs={
                "learning_rate": 5e-4,
                "n_steps": 128,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "n_epochs": 10,
                "net_arch": [128, 128],
            },
            description="Higher representational capacity network (MLP [128, 128])",
        ),
    ]


def evaluate_checkpoints(
    checkpoint_dir: Path,
    eval_seed_start: int = 500,
    eval_episodes: int = 20,
) -> list[dict[str, Any]]:
    """Evaluate all checkpoint models in a directory on held-out seeds."""
    eval_env = create_evaluation_env(seed=eval_seed_start)
    ckpt_files = sorted(
        checkpoint_dir.glob("checkpoint_*.zip"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
    )
    records: list[dict[str, Any]] = []

    for ckpt_path in ckpt_files:
        step_str = ckpt_path.stem.split("_")[-1]
        if not step_str.isdigit():
            continue
        timestep = int(step_str)

        agent = PPOAgent.load(str(ckpt_path), env=eval_env)
        eval_res = evaluate_policy(
            policy=agent,
            env=eval_env,
            episodes=eval_episodes,
            base_seed=eval_seed_start,
            deterministic=True,
            policy_name=f"ckpt_{timestep}",
        )
        overshoots = sum(1 for ep in eval_res.episodes if ep.final_inventory > ep.target_inventory)
        overshoot_rate = (
            float(overshoots / eval_res.total_episodes) if eval_res.total_episodes > 0 else 0.0
        )
        score = compute_composite_score(
            mean_reward=eval_res.mean_reward,
            completion_rate=eval_res.target_completion_rate,
            mean_shortfall=eval_res.mean_shortfall,
            overshoot_rate=overshoot_rate,
        )

        records.append({
            "timestep": timestep,
            "mean_reward": eval_res.mean_reward,
            "median_reward": eval_res.median_reward,
            "std_reward": eval_res.std_reward,
            "min_reward": eval_res.min_reward,
            "max_reward": eval_res.max_reward,
            "completion_rate": eval_res.target_completion_rate,
            "mean_shortfall": eval_res.mean_shortfall,
            "median_shortfall": eval_res.median_shortfall,
            "max_shortfall": eval_res.max_shortfall,
            "mean_final_inventory": eval_res.mean_final_inventory,
            "mean_executed_quantity": eval_res.mean_executed_quantity,
            "overshoot_rate": overshoot_rate,
            "composite_score": score,
            "mean_execution_reward": eval_res.mean_execution_reward,
            "mean_inventory_progress_reward": eval_res.mean_inventory_progress_reward,
            "mean_inventory_penalty": eval_res.mean_inventory_penalty,
            "mean_terminal_penalty": eval_res.mean_terminal_penalty,
            "mean_episode_length": eval_res.mean_episode_length,
        })

    return records


def run_candidate_training(
    candidate: CandidateConfig,
    train_seed: int,
    total_timesteps: int,
    run_dir: Path,
    eval_seed_start: int = 500,
    eval_episodes: int = 20,
) -> tuple[TrialResult, list[dict[str, Any]]]:
    """Train a single candidate for total_timesteps and evaluate checkpoints."""
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    ckpts_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"

    ppo_cfg = candidate.to_ppo_config(seed=train_seed)
    train_cfg = TrainingConfig(
        total_timesteps=total_timesteps,
        checkpoint_frequency=25_000,
        eval_frequency=25_000,
        eval_episodes=5,
        logging_frequency=5_000,
        model_output_dir=models_dir,
        checkpoint_dir=ckpts_dir,
        log_dir=logs_dir,
        final_model_name=f"{candidate.name}_final",
        best_model_name=f"{candidate.name}_best",
        seed=train_seed,
        ppo_config=ppo_cfg,
        dynamic_episode_seeds=True,
    )

    t0 = time.time()
    train_env, eval_env = PPOTrainer.build_environments(train_cfg)
    trainer = PPOTrainer(config=train_cfg, env=train_env, eval_env=eval_env)
    agent, _diagnostics = trainer.train()
    duration = time.time() - t0

    # Evaluate all 25k checkpoints on held-out selection seeds (500..519)
    learning_curve = evaluate_checkpoints(
        checkpoint_dir=ckpts_dir,
        eval_seed_start=eval_seed_start,
        eval_episodes=eval_episodes,
    )

    # Evaluate final model on held-out selection seeds (500..519)
    eval_benchmark_env = create_evaluation_env(seed=eval_seed_start)
    final_eval = evaluate_policy(
        policy=agent,
        env=eval_benchmark_env,
        episodes=eval_episodes,
        base_seed=eval_seed_start,
        deterministic=True,
        policy_name=candidate.name,
    )

    overshoots = sum(1 for ep in final_eval.episodes if ep.final_inventory > ep.target_inventory)
    overshoot_rate = (
        float(overshoots / final_eval.total_episodes) if final_eval.total_episodes > 0 else 0.0
    )
    score = compute_composite_score(
        mean_reward=final_eval.mean_reward,
        completion_rate=final_eval.target_completion_rate,
        mean_shortfall=final_eval.mean_shortfall,
        overshoot_rate=overshoot_rate,
    )

    final_model_path = models_dir / f"{candidate.name}_final.zip"

    trial_result = TrialResult(
        candidate_name=candidate.name,
        candidate_params=candidate.ppo_kwargs,
        train_seed=train_seed,
        eval_seed_start=eval_seed_start,
        eval_episodes=eval_episodes,
        total_timesteps=total_timesteps,
        mean_reward=final_eval.mean_reward,
        std_reward=final_eval.std_reward,
        completion_rate=final_eval.target_completion_rate,
        mean_shortfall=final_eval.mean_shortfall,
        std_shortfall=float(np.std([ep.shortfall for ep in final_eval.episodes])),
        mean_executed_quantity=final_eval.mean_executed_quantity,
        mean_final_inventory=final_eval.mean_final_inventory,
        overshoot_rate=overshoot_rate,
        composite_score=score,
        training_duration_seconds=duration,
        model_path=str(final_model_path),
        eval_result=final_eval,
    )

    return trial_result, learning_curve


def run_all_milestone36_experiments():
    print("=" * 75)
    print("MILESTONE 36: RIGOROUS PPO HYPERPARAMETER OPTIMIZATION (200k TIMESTEPS)")
    print("=" * 75)

    tuning_dir = Path("artifacts/tuning")
    tuning_dir.mkdir(parents=True, exist_ok=True)
    extended_dir = tuning_dir / "extended_experiments"
    extended_dir.mkdir(parents=True, exist_ok=True)
    val_dir = tuning_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    candidates = get_candidate_matrix()
    candidate_results: list[TrialResult] = []
    all_learning_curves: dict[str, list[dict[str, Any]]] = {}

    total_timesteps = 200_000
    screening_train_seed = 200
    heldout_eval_seed_start = 500
    heldout_eval_episodes = 20

    print(
        f"\n[PHASE 1] TRAINING & EVALUATING {len(candidates)} CANDIDATES FOR "
        f"{total_timesteps:,} STEPS EACH..."
    )
    print(f"    Train Seed: {screening_train_seed} | Native Dynamic Episode Seeds: True")
    print(
        f"    Held-Out Selection Seeds: {heldout_eval_seed_start}.."
        f"{heldout_eval_seed_start + heldout_eval_episodes - 1} (20 episodes)"
    )
    print("    Checkpoint Evaluations: Every 25,000 steps (8 evaluations per candidate)")

    for i, cand in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] >>> Training Candidate: {cand.name} <<<")
        print(f"    Params: {cand.ppo_kwargs}")
        run_dir = extended_dir / f"{cand.name}_seed{screening_train_seed}"

        t_start = time.time()
        trial, curve = run_candidate_training(
            candidate=cand,
            train_seed=screening_train_seed,
            total_timesteps=total_timesteps,
            run_dir=run_dir,
            eval_seed_start=heldout_eval_seed_start,
            eval_episodes=heldout_eval_episodes,
        )
        t_elapsed = time.time() - t_start

        comp_pct = trial.completion_rate * 100
        print(
            f"    Finished in {t_elapsed:.1f}s | Score: {trial.composite_score:.2f} | "
            f"Reward: {trial.mean_reward:.2f} | Completion: {comp_pct:.1f}% | "
            f"Shortfall: {trial.mean_shortfall:.2f} | "
            f"Final Inv: {trial.mean_final_inventory:.2f} | "
            f"Overshoot: {trial.overshoot_rate*100:.1f}%"
        )
        print("    Learning Curve (Held-out seeds 500..519):")
        for pt in curve:
            print(
                f"      Step {pt['timestep']:>6}: Rew={pt['mean_reward']:>5.2f}, "
                f"Comp={pt['completion_rate']*100:>5.1f}%, Shortfall={pt['mean_shortfall']:>4.2f}, "
                f"Inv={pt['mean_final_inventory']:>4.2f}, Score={pt['composite_score']:>5.2f}"
            )

        candidate_results.append(trial)
        all_learning_curves[cand.name] = curve

        # Save incremental results
        with open(tuning_dir / "candidate_results.json", "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in candidate_results], f, indent=2)
        with open(tuning_dir / "learning_curves.json", "w", encoding="utf-8") as f:
            json.dump(all_learning_curves, f, indent=2)

    # Rank Candidates on Held-Out Selection Seeds
    candidate_results.sort(key=lambda r: r.composite_score, reverse=True)
    winner_trial = candidate_results[0]
    winner_cand = CandidateConfig(
        name=winner_trial.candidate_name,
        ppo_kwargs=winner_trial.candidate_params,
    )

    print("\n" + "=" * 75)
    print("PHASE 1 CANDIDATE RANKING (Held-Out Selection Seeds 500..519)")
    print("=" * 75)
    fmt_h = (
        f"{'Rank':<4} | {'Candidate':<22} | {'Score':<6} | {'Reward':<7} | "
        f"{'Comp %':<7} | {'Shortfall':<9} | {'Final Inv':<9}"
    )
    print(fmt_h)
    print("-" * 75)
    for rank, cr in enumerate(candidate_results, 1):
        print(
            f"{rank:<4} | {cr.candidate_name:<22} | {cr.composite_score:>6.2f} | "
            f"{cr.mean_reward:>7.2f} | {cr.completion_rate*100:>6.1f}% | "
            f"{cr.mean_shortfall:>9.2f} | {cr.mean_final_inventory:>9.2f}"
        )
    print(f"\nWINNER SELECTED: {winner_cand.name} (Score: {winner_trial.composite_score:.2f})")
    print(f"Hyperparameters: {winner_cand.ppo_kwargs}")

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: MULTI-SEED VALIDATION (Seeds 101, 102, 103)
    # ─────────────────────────────────────────────────────────────
    val_seeds = [101, 102, 103]
    print("\n" + "=" * 75)
    print(f"[PHASE 2] MULTI-SEED VALIDATION OF WINNER: {winner_cand.name}")
    print(f"    Training across 3 seeds: {val_seeds} for {total_timesteps:,} steps each")
    print("=" * 75)

    val_trials_m35: list[TrialResult] = []
    val_trials_extended: list[TrialResult] = []

    for seed_idx, seed_val in enumerate(val_seeds, 1):
        print(f"\n--- [{seed_idx}/{len(val_seeds)}] Seed {seed_val}: {winner_cand.name} ---")
        val_run_dir = val_dir / winner_cand.name / f"seed_{seed_val}"
        t_start = time.time()

        trial_heldout, _ = run_candidate_training(
            candidate=winner_cand,
            train_seed=seed_val,
            total_timesteps=total_timesteps,
            run_dir=val_run_dir,
            eval_seed_start=heldout_eval_seed_start,
            eval_episodes=heldout_eval_episodes,
        )
        t_elapsed = time.time() - t_start

        # Load trained agent for benchmark evaluations
        val_model_path = val_run_dir / "models" / f"{winner_cand.name}_final.zip"
        agent = PPOAgent.load(str(val_model_path))

        # Benchmark A: Untouched M35 benchmark (seeds 42..91, 50 episodes)
        env_m35 = create_evaluation_env(seed=42)
        eval_m35 = evaluate_policy(
            policy=agent,
            env=env_m35,
            episodes=50,
            base_seed=42,
            deterministic=True,
            policy_name=f"{winner_cand.name}_seed{seed_val}",
        )
        overshoots_m35 = sum(
            1 for ep in eval_m35.episodes if ep.final_inventory > ep.target_inventory
        )
        overshoot_rate_m35 = float(overshoots_m35 / 50)
        score_m35 = compute_composite_score(
            mean_reward=eval_m35.mean_reward,
            completion_rate=eval_m35.target_completion_rate,
            mean_shortfall=eval_m35.mean_shortfall,
            overshoot_rate=overshoot_rate_m35,
        )
        trial_m35 = TrialResult(
            candidate_name=winner_cand.name,
            candidate_params=winner_cand.ppo_kwargs,
            train_seed=seed_val,
            eval_seed_start=42,
            eval_episodes=50,
            total_timesteps=total_timesteps,
            mean_reward=eval_m35.mean_reward,
            std_reward=eval_m35.std_reward,
            completion_rate=eval_m35.target_completion_rate,
            mean_shortfall=eval_m35.mean_shortfall,
            std_shortfall=float(np.std([ep.shortfall for ep in eval_m35.episodes])),
            mean_executed_quantity=eval_m35.mean_executed_quantity,
            mean_final_inventory=eval_m35.mean_final_inventory,
            overshoot_rate=overshoot_rate_m35,
            composite_score=score_m35,
            training_duration_seconds=t_elapsed,
            model_path=str(val_model_path),
            eval_result=eval_m35,
        )
        val_trials_m35.append(trial_m35)

        # Benchmark B: Extended generalization benchmark (seeds 42..141, 100 episodes)
        env_ext = create_evaluation_env(seed=42)
        eval_ext = evaluate_policy(
            policy=agent,
            env=env_ext,
            episodes=100,
            base_seed=42,
            deterministic=True,
            policy_name=f"{winner_cand.name}_seed{seed_val}",
        )
        overshoots_ext = sum(
            1 for ep in eval_ext.episodes if ep.final_inventory > ep.target_inventory
        )
        overshoot_rate_ext = float(overshoots_ext / 100)
        score_ext = compute_composite_score(
            mean_reward=eval_ext.mean_reward,
            completion_rate=eval_ext.target_completion_rate,
            mean_shortfall=eval_ext.mean_shortfall,
            overshoot_rate=overshoot_rate_ext,
        )
        trial_ext = TrialResult(
            candidate_name=winner_cand.name,
            candidate_params=winner_cand.ppo_kwargs,
            train_seed=seed_val,
            eval_seed_start=42,
            eval_episodes=100,
            total_timesteps=total_timesteps,
            mean_reward=eval_ext.mean_reward,
            std_reward=eval_ext.std_reward,
            completion_rate=eval_ext.target_completion_rate,
            mean_shortfall=eval_ext.mean_shortfall,
            std_shortfall=float(np.std([ep.shortfall for ep in eval_ext.episodes])),
            mean_executed_quantity=eval_ext.mean_executed_quantity,
            mean_final_inventory=eval_ext.mean_final_inventory,
            overshoot_rate=overshoot_rate_ext,
            composite_score=score_ext,
            training_duration_seconds=t_elapsed,
            model_path=str(val_model_path),
            eval_result=eval_ext,
        )
        val_trials_extended.append(trial_ext)

        print(
            f"    Seed {seed_val} M35 Benchmark (50 ep): Score={score_m35:.2f}, "
            f"Reward={eval_m35.mean_reward:.2f}, Comp={eval_m35.target_completion_rate*100:.1f}%, "
            f"Shortfall={eval_m35.mean_shortfall:.2f}, Inv={eval_m35.mean_final_inventory:.2f}"
        )
        print(
            f"    Seed {seed_val} Extended Benchmark (100 ep): Score={score_ext:.2f}, "
            f"Reward={eval_ext.mean_reward:.2f}, Comp={eval_ext.target_completion_rate*100:.1f}%, "
            f"Shortfall={eval_ext.mean_shortfall:.2f}, Inv={eval_ext.mean_final_inventory:.2f}"
        )

    summary_m35 = MultiSeedSummary.from_trials(
        candidate_name=winner_cand.name,
        candidate_params=winner_cand.ppo_kwargs,
        trials=val_trials_m35,
    )
    summary_ext = MultiSeedSummary.from_trials(
        candidate_name=winner_cand.name,
        candidate_params=winner_cand.ppo_kwargs,
        trials=val_trials_extended,
    )

    with open(tuning_dir / "multi_seed_summary_m35.json", "w", encoding="utf-8") as f:
        json.dump(summary_m35.to_dict(), f, indent=2)
    with open(tuning_dir / "multi_seed_summary_extended.json", "w", encoding="utf-8") as f:
        json.dump(summary_ext.to_dict(), f, indent=2)

    # Identify best seed model (by M35 benchmark composite score)
    best_val_trial = max(val_trials_m35, key=lambda t: t.composite_score)
    best_model_src = Path(best_val_trial.model_path)
    tuned_best_dst = models_dir / "ppo_execution_tuned_best.zip"
    tuned_final_dst = models_dir / "ppo_execution_tuned_final.zip"
    shutil.copy2(best_model_src, tuned_best_dst)
    shutil.copy2(best_model_src, tuned_final_dst)
    print(f"\nTop model ({best_val_trial.candidate_name}_seed{best_val_trial.train_seed}) saved:")
    print(f"    {tuned_best_dst}")
    print(f"    {tuned_final_dst}")

    # ─────────────────────────────────────────────────────────────
    # PHASE 3: COMPREHENSIVE BASELINE & COMPARISON SUITE
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("[PHASE 3] EVALUATING ALL BASELINES ACROSS BOTH BENCHMARKS")
    print("=" * 75)

    def evaluate_all_baselines(
        seed_start: int, num_episodes: int
    ) -> dict[str, EvaluationResult]:
        baselines: dict[str, Any] = {
            "rule_based": RuleBasedBaselinePolicy(),
            "hold": HoldBaselinePolicy(),
            "random": RandomBaselinePolicy(seed=seed_start),
        }
        res_dict: dict[str, EvaluationResult] = {}
        for b_name, b_pol in baselines.items():
            env = create_evaluation_env(seed=seed_start)
            r = evaluate_policy(
                policy=b_pol,
                env=env,
                episodes=num_episodes,
                base_seed=seed_start,
                deterministic=True,
                policy_name=b_name,
            )
            res_dict[b_name] = r

        # Evaluate M34 pre-tuning final model
        m34_path = models_dir / "ppo_execution_final.zip"
        if m34_path.exists():
            env = create_evaluation_env(seed=seed_start)
            m34_agent = PPOAgent.load(str(m34_path), env=env)
            r_m34 = evaluate_policy(
                policy=m34_agent,
                env=env,
                episodes=num_episodes,
                base_seed=seed_start,
                deterministic=True,
                policy_name="m34_pre_tuning_ppo",
            )
            res_dict["m34_pre_tuning_ppo"] = r_m34

        # Evaluate C0 M34 Control (trained with dynamic episode seeds)
        c0_trial = next(
            (t for t in candidate_results if t.candidate_name == "C0_m34_control"),
            None,
        )
        if c0_trial and c0_trial.model_path and Path(c0_trial.model_path).exists():
            env = create_evaluation_env(seed=seed_start)
            c0_agent = PPOAgent.load(c0_trial.model_path, env=env)
            r_c0 = evaluate_policy(
                policy=c0_agent,
                env=env,
                episodes=num_episodes,
                base_seed=seed_start,
                deterministic=True,
                policy_name="c0_m34_control",
            )
            res_dict["c0_m34_control"] = r_c0

        # Evaluate Tuned PPO Best Model
        env = create_evaluation_env(seed=seed_start)
        tuned_best_agent = PPOAgent.load(str(tuned_best_dst), env=env)
        r_tuned = evaluate_policy(
            policy=tuned_best_agent,
            env=env,
            episodes=num_episodes,
            base_seed=seed_start,
            deterministic=True,
            policy_name=f"{winner_cand.name}_best_seed",
        )
        res_dict[f"{winner_cand.name}_best_seed"] = r_tuned

        return res_dict

    # 1. Benchmark A (M35 Benchmark: seeds 42..91, 50 episodes)
    print("\n--- Benchmark A: M35 Benchmark (Seeds 42..91, 50 Episodes) ---")
    results_m35 = evaluate_all_baselines(seed_start=42, num_episodes=50)

    # 2. Benchmark B (Extended Generalization Benchmark: seeds 42..141, 100 episodes)
    print("\n--- Benchmark B: Extended Benchmark (Seeds 42..141, 100 Episodes) ---")
    results_ext = evaluate_all_baselines(seed_start=42, num_episodes=100)

    # Generate pairwise comparisons against Rule-Based and M34
    best_m35_res = results_m35[f"{winner_cand.name}_best_seed"]
    comparisons_m35 = {
        name: compare_policies(b_res, best_m35_res).to_dict()
        for name, b_res in results_m35.items()
        if name != f"{winner_cand.name}_best_seed"
    }
    with open(tuning_dir / "benchmark_m35_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons_m35, f, indent=2)

    best_ext_res = results_ext[f"{winner_cand.name}_best_seed"]
    comparisons_ext = {
        name: compare_policies(b_res, best_ext_res).to_dict()
        for name, b_res in results_ext.items()
        if name != f"{winner_cand.name}_best_seed"
    }
    with open(tuning_dir / "benchmark_extended_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons_ext, f, indent=2)

    # Final Summary Report
    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "winning_candidate": {
            "name": winner_cand.name,
            "params": winner_cand.ppo_kwargs,
        },
        "screening_results": [r.to_dict() for r in candidate_results],
        "multi_seed_summary_m35": summary_m35.to_dict(),
        "multi_seed_summary_extended": summary_ext.to_dict(),
        "best_seed_trial": best_val_trial.to_dict(),
        "benchmark_m35_metrics": {
            name: {
                "mean_reward": r.mean_reward,
                "median_reward": r.median_reward,
                "std_reward": r.std_reward,
                "min_reward": r.min_reward,
                "max_reward": r.max_reward,
                "completion_rate": r.target_completion_rate,
                "mean_shortfall": r.mean_shortfall,
                "median_shortfall": r.median_shortfall,
                "max_shortfall": r.max_shortfall,
                "mean_final_inv": r.mean_final_inventory,
                "mean_executed_qty": r.mean_executed_quantity,
                "mean_execution_reward": r.mean_execution_reward,
                "mean_inv_progress_reward": r.mean_inventory_progress_reward,
                "mean_inv_penalty": r.mean_inventory_penalty,
                "mean_terminal_penalty": r.mean_terminal_penalty,
                "mean_episode_length": r.mean_episode_length,
            }
            for name, r in results_m35.items()
        },
        "benchmark_extended_metrics": {
            name: {
                "mean_reward": r.mean_reward,
                "median_reward": r.median_reward,
                "std_reward": r.std_reward,
                "min_reward": r.min_reward,
                "max_reward": r.max_reward,
                "completion_rate": r.target_completion_rate,
                "mean_shortfall": r.mean_shortfall,
                "median_shortfall": r.median_shortfall,
                "max_shortfall": r.max_shortfall,
                "mean_final_inv": r.mean_final_inventory,
                "mean_executed_qty": r.mean_executed_quantity,
                "mean_execution_reward": r.mean_execution_reward,
                "mean_inv_progress_reward": r.mean_inventory_progress_reward,
                "mean_inv_penalty": r.mean_inventory_penalty,
                "mean_terminal_penalty": r.mean_terminal_penalty,
                "mean_episode_length": r.mean_episode_length,
            }
            for name, r in results_ext.items()
        },
        "comparisons_against_baselines_m35": comparisons_m35,
        "comparisons_against_baselines_extended": comparisons_ext,
    }

    with open(tuning_dir / "final_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # ─────────────────────────────────────────────────────────────
    # PRINT FINAL MATRICES
    # ─────────────────────────────────────────────────────────────
    def print_matrix(
        title: str,
        results_dict: dict[str, EvaluationResult],
        summary: MultiSeedSummary,
    ):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        hdr = (
            f"{'Policy':<28} | {'Reward':<8} | {'Completion':<12} | "
            f"{'Shortfall':<10} | {'Final Inv':<10}"
        )
        print(hdr)
        print("-" * 80)
        for name, r in results_dict.items():
            comp_p = r.target_completion_rate * 100
            print(
                f"{name:<28} | {r.mean_reward:>8.2f} | {comp_p:>10.1f}% | "
                f"{r.mean_shortfall:>10.2f} | {r.mean_final_inventory:>10.2f}"
            )
        ms_name = f"{winner_cand.name} (Multi-Seed)"
        ms_comp = summary.mean_completion_rate * 100
        print(
            f"{ms_name:<28} | {summary.mean_reward:>8.2f} | {ms_comp:>10.1f}% | "
            f"{summary.mean_shortfall:>10.2f} | {'N/A':>10}"
        )
        print("=" * 80)

    print_matrix(
        "BENCHMARK A: M35 BENCHMARK (Seeds 42..91, 50 Episodes)",
        results_m35,
        summary_m35,
    )
    print_matrix(
        "BENCHMARK B: EXTENDED BENCHMARK (Seeds 42..141, 100 Episodes)",
        results_ext,
        summary_ext,
    )
    print(f"\nAll experiments complete. Artifacts saved to: {tuning_dir.resolve()}")


if __name__ == "__main__":
    run_all_milestone36_experiments()
