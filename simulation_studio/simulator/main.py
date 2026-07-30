"""
main.py — Command-line Runner
Quebec City Blood Bank Simulation

Usage:
    python main.py                        # interactive menu
    python main.py --scenario donor_decrease --strategy mass_campaign
    python main.py --scenario all         # run all scenarios and compare
    python main.py --list                 # list available scenarios & strategies

Options:
    --scenario   KEY        Scenario key (see --list)
    --strategy   KEY        Override strategy (see --list)
    --seed       INT        Random seed (default 42)
    --hours      INT        Override simulation duration
    --output     DIR        Output directory for reports (default ./reports)
    --list                  List all scenarios and strategies
    --quiet                 Suppress console output and detailed event-history capture
    --fast                  Skip detailed history and report-only tracking for faster runs
"""

import argparse
import os
import sys
from typing import Optional

# ── Load road network once ──────────────────────────────
print("⏳ Loading Quebec City road network (this may take ~1 min) …")
from core import load_city_graph

G, NORTH, SOUTH, EAST, WEST = load_city_graph()
print("✅ Network ready.\n")

from engine import (
    DREAMER_CONTROLLER_KEYS,
    generate_report,
    load_dreamerv4_agent,
    load_official_dreamerv3_agent,
    run_scenario,
    summarize_state,
)
from scenarios import SCENARIOS, STRATEGIES


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def list_all():
    print("\n" + "=" * 65)
    print("AVAILABLE SCENARIOS")
    print("=" * 65)
    for key, sc in SCENARIOS.items():
        print(f"  {key:<25} — {sc.name}")
        print(f"    {sc.description[:75]}")
    print()
    print("=" * 65)
    print("AVAILABLE STRATEGIES")
    print("=" * 65)
    for key, st in STRATEGIES.items():
        print(f"  {key:<20} — {st.name}")
        desc_lines = [
            st.description[i : i + 60] for i in range(0, len(st.description), 60)
        ]
        for l in desc_lines:
            print(f"    {l}")
    print()


def interactive_menu() -> tuple[str, str]:
    print("\n" + "=" * 65)
    print("  QUEBEC CITY BLOOD SUPPLY CHAIN SIMULATION")
    print("=" * 65)
    print("\nSelect a scenario:\n")
    keys = list(SCENARIOS.keys())
    for i, key in enumerate(keys, 1):
        sc = SCENARIOS[key]
        print(f"  [{i}] {sc.name}")
        print(f"       {sc.description[:70]}")
    print()
    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(keys):
                sc_key = keys[choice - 1]
                break
        except ValueError:
            pass
        print("Invalid choice.")

    print("\nSelect a strategy:\n")
    st_keys = list(STRATEGIES.keys())
    for i, key in enumerate(st_keys, 1):
        st = STRATEGIES[key]
        print(f"  [{i}] {st.name}")
    print()
    while True:
        try:
            choice = int(input("Enter number (or 0 to use scenario default): "))
            if choice == 0:
                return sc_key, SCENARIOS[sc_key].strategy_key
            if 1 <= choice <= len(st_keys):
                return sc_key, st_keys[choice - 1]
        except ValueError:
            pass
        print("Invalid choice.")


def finish_episode(state):
    if state.env.now < state.params.sim_hours:
        state.env.run(until=state.params.sim_hours)
    return state


def resolve_selected_dreamer_checkpoint(
    strategy_key: str,
    dreamerv3_checkpoint: Optional[str] = None,
    dreamerv4_checkpoint: Optional[str] = None,
) -> Optional[str]:
    if strategy_key == "dreamerv4":
        return dreamerv4_checkpoint
    if strategy_key == "dreamerv3_official":
        return dreamerv3_checkpoint
    return None


def run_and_report(
    sc_key: str,
    strategy_key: str,
    seed: int,
    hours_override: Optional[int],
    output_dir: str,
    dreamer_checkpoint: Optional[str] = None,
    dreamer_agent=None,
    enable_logs: bool = True,
    fast_mode: bool = False,
):
    import copy

    params = copy.deepcopy(SCENARIOS[sc_key])
    params.strategy_key = strategy_key
    if hours_override:
        params.sim_hours = hours_override
    use_dreamer = strategy_key in DREAMER_CONTROLLER_KEYS or dreamer_agent is not None

    state = run_scenario(
        params,
        G,
        NORTH,
        SOUTH,
        EAST,
        WEST,
        seed=seed,
        enable_logs=enable_logs,
        fast_mode=fast_mode,
        official_dreamerv3_agent=dreamer_agent,
        official_dreamerv3_checkpoint=(
            dreamer_checkpoint if use_dreamer else None
        ),
    )
    finish_episode(state)

    report_path = os.path.join(output_dir, f"report_{sc_key}_{strategy_key}.txt")
    generate_report(state, output_path=report_path)
    return state, report_path


def run_all_scenarios(
    seed: int,
    output_dir: str,
    strategy_override: Optional[str] = None,
    hours_override: Optional[int] = None,
    dreamerv3_checkpoint: Optional[str] = None,
    dreamerv4_checkpoint: Optional[str] = None,
    enable_logs: bool = True,
    fast_mode: bool = False,
):
    """Run every scenario with its default strategy, then print comparison table."""
    results = {}
    shared_dreamer_agents: dict[str, object] = {}
    for sc_key in SCENARIOS:
        import copy

        params = copy.deepcopy(SCENARIOS[sc_key])
        if strategy_override:
            params.strategy_key = strategy_override
        if hours_override:
            params.sim_hours = hours_override

        dreamer_agent = None
        if params.strategy_key in DREAMER_CONTROLLER_KEYS:
            dreamer_checkpoint = resolve_selected_dreamer_checkpoint(
                params.strategy_key,
                dreamerv3_checkpoint=dreamerv3_checkpoint,
                dreamerv4_checkpoint=dreamerv4_checkpoint,
            )
            if params.strategy_key not in shared_dreamer_agents:
                load_dreamer_agent = (
                    load_dreamerv4_agent
                    if params.strategy_key == "dreamerv4"
                    else load_official_dreamerv3_agent
                )
                shared_dreamer_agents[params.strategy_key] = load_dreamer_agent(
                    dreamer_checkpoint,
                    seed=seed,
                )
            dreamer_agent = shared_dreamer_agents[params.strategy_key]

        state = run_scenario(
            params,
            G,
            NORTH,
            SOUTH,
            EAST,
            WEST,
            seed=seed,
            enable_logs=enable_logs,
            fast_mode=fast_mode,
            official_dreamerv3_agent=dreamer_agent,
            official_dreamerv3_checkpoint=(
                resolve_selected_dreamer_checkpoint(
                    params.strategy_key,
                    dreamerv3_checkpoint=dreamerv3_checkpoint,
                    dreamerv4_checkpoint=dreamerv4_checkpoint,
                )
                if params.strategy_key in DREAMER_CONTROLLER_KEYS
                and dreamer_agent is None
                else None
            ),
        )
        finish_episode(state)
        report_path = os.path.join(
            output_dir, f"report_{sc_key}_{params.strategy_key}.txt"
        )
        generate_report(state, output_path=report_path)
        results[sc_key] = state

    # Comparison table
    print("\n" + "=" * 90)
    print("SCENARIO COMPARISON TABLE")
    print("=" * 90)
    print(
        f"{'Scenario':<28} {'Strategy':<22} {'Donated':>8} {'Shortage%':>10} {'Reward':>10} {'Expired':>8}"
    )
    print("-" * 90)
    for sc_key, state in results.items():
        summary = summarize_state(state)
        donated = summary["total_donated"]
        expired = summary["total_expired"]
        s_rate = summary["shortage_rate"]
        reward = summary["reward"].total
        strat = state.params.strategy.name[:20]
        print(
            f"  {SCENARIOS[sc_key].name[:26]:<26} {strat:<22} {donated:>8} "
            f"{s_rate:>9.1f}% {reward:>9.1f} {expired:>8}"
        )
    print("=" * 90)
    comp_path = os.path.join(output_dir, "comparison_table.txt")
    # Write comparison to file too
    with open(comp_path, "w") as f:
        f.write("SCENARIO COMPARISON\n")
        f.write(
            f"{'Scenario':<28} {'Strategy':<22} {'Donated':>8} {'Shortage%':>10} {'Reward':>10} {'Expired':>8}\n"
        )
        for sc_key, state in results.items():
            summary = summarize_state(state)
            donated = summary["total_donated"]
            expired = summary["total_expired"]
            s_rate = summary["shortage_rate"]
            reward = summary["reward"].total
            strat = state.params.strategy.name[:20]
            f.write(
                f"  {SCENARIOS[sc_key].name[:26]:<26} {strat:<22} {donated:>8} "
                f"{s_rate:>9.1f}% {reward:>9.1f} {expired:>8}\n"
            )
    print(f"📄 Comparison saved → {comp_path}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
from typing import Optional


def run():
    parser = argparse.ArgumentParser(description="Quebec City Blood Bank Simulation")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--output", default="reports")
    parser.add_argument(
        "--dreamerv3-checkpoint",
        default=None,
        help=(
            "Official DreamerV3 checkpoint path used by the `dreamerv3_official` "
            "strategy. Defaults to simulator/dreamerv3_runs/all_scenarios/ckpt."
        ),
    )
    parser.add_argument(
        "--dreamerv4-checkpoint",
        default=None,
        help=(
            "DreamerV4-compatible checkpoint path used by the `dreamerv4` "
            "strategy. Defaults to simulator/dreamerv4_runs/all_scenarios/ckpt "
            "and falls back to the standard Dreamer checkpoint when needed."
        ),
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output and skip detailed event-history capture.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip detailed history and report-only tracking for maximum speed.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.list:
        list_all()
        sys.exit(0)

    if args.strategy and args.strategy not in STRATEGIES:
        print(f"Unknown strategy '{args.strategy}'. Use --list to see options.")
        sys.exit(1)

    if args.scenario == "all":
        run_all_scenarios(
            args.seed,
            args.output,
            strategy_override=args.strategy,
            hours_override=args.hours,
            dreamerv3_checkpoint=args.dreamerv3_checkpoint,
            dreamerv4_checkpoint=args.dreamerv4_checkpoint,
            enable_logs=not args.quiet,
            fast_mode=args.fast,
        )
        return

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario '{args.scenario}'. Use --list to see options.")
            sys.exit(1)
        sc_key = args.scenario
        st_key = args.strategy or SCENARIOS[sc_key].strategy_key
    else:
        sc_key, st_key = interactive_menu()

    if st_key not in STRATEGIES:
        print(f"Unknown strategy '{st_key}'. Use --list to see options.")
        sys.exit(1)

    state, report_path = run_and_report(
        sc_key,
        st_key,
        args.seed,
        args.hours,
        args.output,
        dreamer_checkpoint=resolve_selected_dreamer_checkpoint(
            st_key,
            dreamerv3_checkpoint=args.dreamerv3_checkpoint,
            dreamerv4_checkpoint=args.dreamerv4_checkpoint,
        ),
        enable_logs=not args.quiet,
        fast_mode=args.fast,
    )
    print(f"\n✅ Done. Full report: {report_path}")


if __name__ == "__main__":
    run()
