"""
ws_simulation.py – WebSocket-based step-by-step simulation runner.

Exposes a FastAPI router with:
  • GET  /api/sim/setup  – available scenarios & strategies for the UI
  • WS   /api/sim/ws     – interactive step-by-step simulation control

Client → Server commands:
  {"command": "start", "scenario_key": "...", "strategy_key": "...",
   "seed": 100, "hours_override": 168, "step_hours": 6, "speed": 1.0}
  {"command": "pause"}
  {"command": "resume"}
  {"command": "stop"}
  {"command": "set_speed", "speed": 2.0}

Server → Client messages:
  {"type": "init",     ...}         after initialization, before first step
  {"type": "step",     ...}         after each simulation step
  {"type": "complete", "report": {}} when simulation finishes
  {"type": "error",    "message": "..."} on error
  {"type": "paused"}  / {"type": "resumed"} / {"type": "stopped"}
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# ---------------------------------------------------------------------------
# Ensure the simulator package is importable (mirrors simulator_service.py).
# ---------------------------------------------------------------------------
STUDIO_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_ROOT = STUDIO_ROOT / "simulator"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

import simpy  # noqa: E402
from engine import (  # noqa: E402
    DonationCenter,
    LEARNED_CONTINUOUS_CONTROLLER_KEYS,
    SimState,
    WeatherEngine,
    budget_monitor,
    calculate_reward,
    demand_forecast_process,
    donor_generator,
    hospital_demand,
    initialize_forecast_state,
    load_learned_continuous_agent,
    load_official_dreamerv3_agent,
    monitor,
    ppo_shortage_controller,
    prestock_centers,
    regional_replenishment,
    resolve_learned_continuous_model_path,
    step_simulation,
    summarize_state,
    total_transfused_units,
    track_episode_score,
    trained_ppo_controller,
    trained_ppo_continuous_controller,
    trained_official_dreamerv3_controller,
)
from eval_metrics import extract_eval_metrics  # noqa: E402
from scenarios import SCENARIOS, ScenarioParams  # noqa: E402

from .center_loader import resolve_center_configs  # noqa: E402
from .operational_seed import (  # noqa: E402
    build_operational_seed_plan,
    build_simulation_centers,
    load_operational_simulation_config,
    prestock_centers_from_operational,
)
from .forecast_ws import (  # noqa: E402
    build_forecast_panel,
    forecast_panel_meta,
    resolve_forecast_job_id,
    valid_forecast_job_id,
)
from .policy_snapshot import summarize_policy_events  # noqa: E402
from .simulator_service import (  # noqa: E402
    custom_scenario_editor_config,
    discover_dreamerv3_runs,
    ensure_strategy,
    get_sim_context,
    list_scenarios,
    list_strategies,
    public_dreamerv3_run_payload,
    resolve_dreamerv3_run,
    resolve_scenario,
    scenario_payload,
    serialize_center,
    strategy_payload,
    utc_now_iso,
)

logger = logging.getLogger("ws_simulation")

router = APIRouter(prefix="/api/sim", tags=["simulation"])

# ═══════════════════════════════════════════════════════════════════════════
# 1. initialize_simulation
# ═══════════════════════════════════════════════════════════════════════════


def load_operational_payload() -> dict[str, Any] | None:
    try:
        from .studio_routes import capture_dashboard_operational_rows

        rows = capture_dashboard_operational_rows()
        payload = rows.normalized or rows.redacted or {}
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.info("Operational seed unavailable: %s", exc)
        return None


def initialize_simulation_from_params(
    params: ScenarioParams,
    *,
    seed: int = 42,
    step_hours: float = 6.0,
    dreamerv3_run_key: Optional[str] = None,
    use_operational_seed: bool = True,
    operational_payload: dict[str, Any] | None = None,
) -> SimState:
    """
    Set up the full SimPy environment and register all processes from an
    already-prepared ``ScenarioParams`` instance.
    """
    strategy = params.strategy
    random.seed(seed)
    env = simpy.Environment()
    weather = WeatherEngine(forced_state=params.forced_weather)

    # Build centres -------------------------------------------------------
    seed_config = load_operational_simulation_config()
    seed_plan = None
    center_configs, center_source = resolve_center_configs(
        seed_config,
        operational_payload if use_operational_seed else None,
    )
    if center_source == "operational_rows" and operational_payload:
        seed_plan = build_operational_seed_plan(operational_payload, config=seed_config)

    centers = build_simulation_centers(env, strategy, center_configs=center_configs)

    # SimState ------------------------------------------------------------
    G, north, south, east, west = get_sim_context()

    state = SimState(
        env,
        params,
        centers,
        G,
        north,
        south,
        east,
        west,
        weather,
        rng_demand=random.Random(seed + 10_001),
        rng_process=random.Random(seed + 20_003),
        rng_transport=random.Random(seed + 30_007),
        enable_logs=False,
        fast_mode=False,  # we want timeline data for the UI
    )

    # Processes -----------------------------------------------------------
    initialize_forecast_state(state)
    env.process(donor_generator(state))
    env.process(hospital_demand(state))
    env.process(regional_replenishment(state))
    env.process(demand_forecast_process(state))
    env.process(budget_monitor(state))
    env.process(monitor(state))

    # Controller ----------------------------------------------------------
    if strategy.controller_key == "ppo_shortage_minimizer":
        state.runtime_controller = "heuristic_ppo"
        state.runtime_controller_step_hours = float(strategy.controller_interval_h)
        env.process(ppo_shortage_controller(state))

    # DreamerV3 controller
    elif strategy.controller_key == "dreamerv3_official":
        try:
            run_info = resolve_dreamerv3_run(dreamerv3_run_key)
            checkpoint_path = run_info["checkpoint_input"]
            agent = load_official_dreamerv3_agent(
                checkpoint_path,
                seed=seed,
                step_hours=step_hours,
            )
            state.runtime_controller = "official_dreamerv3"
            state.runtime_controller_step_hours = float(strategy.controller_interval_h)
            env.process(
                trained_official_dreamerv3_controller(
                    state,
                    agent,
                    step_hours=strategy.controller_interval_h,
                )
            )
        except Exception as exc:
            logger.warning(
                "DreamerV3 agent loading failed: %s — running without controller", exc
            )

    elif strategy.controller_key in LEARNED_CONTINUOUS_CONTROLLER_KEYS:
        try:
            resolved_model = resolve_learned_continuous_model_path(
                strategy.controller_key
            )

            if resolved_model is not None:
                agent = load_learned_continuous_agent(
                    strategy.controller_key,
                    resolved_model,
                    params=params,
                    G=G,
                    north=north,
                    south=south,
                    east=east,
                    west=west,
                )
                state.runtime_controller = getattr(
                    agent,
                    "_controller_name",
                    f"trained_{strategy.controller_key}",
                )
                state.runtime_controller_step_hours = float(
                    strategy.controller_interval_h
                )
                env.process(
                    trained_ppo_continuous_controller(
                        state,
                        agent,
                        step_hours=strategy.controller_interval_h,
                    )
                )
            else:
                logger.warning(
                    "%s model not found — running without controller",
                    strategy.controller_key,
                )
        except Exception as exc:
            logger.warning(
                "%s agent loading failed: %s — running without controller",
                strategy.controller_key,
                exc,
            )

    # Reward tracking setup -----------------------------------------------
    reward_step_hours = 6.0
    if strategy.controller_key == "ppo_shortage_minimizer":
        reward_step_hours = float(strategy.controller_interval_h)
    elif (
        strategy.controller_key in LEARNED_CONTINUOUS_CONTROLLER_KEYS
        or strategy.controller_key == "dreamerv3_official"
    ):
        reward_step_hours = float(strategy.controller_interval_h)

    seeded_from_operational = False
    if seed_plan and seed_plan.enabled and seed_plan.supplies:
        seeded_from_operational = prestock_centers_from_operational(
            centers,
            env,
            seed_plan.supplies,
            config=seed_config,
        )
    if not seeded_from_operational:
        prestock_centers(centers, env, inventory_days=params.initial_inventory_days)
    state.episode_step_hours = reward_step_hours
    state.episode_prev_transfused = total_transfused_units(state)
    state.episode_prev_reward_total = float(
        calculate_reward(
            state,
            prev_transfused=state.episode_prev_transfused,
            update_prev_transfused=False,
        ).total
    )
    env.process(track_episode_score(state, step_hours=reward_step_hours))

    return state


def initialize_simulation(
    scenario_key: str,
    strategy_key: str,
    seed: int = 42,
    hours_override: Optional[int] = None,
    step_hours: float = 6.0,
    dreamerv3_run_key: Optional[str] = None,
    *,
    use_operational_seed: bool = True,
    operational_payload: dict[str, Any] | None = None,
) -> SimState:
    """
    Set up the full SimPy environment and register all processes, but
    do **not** call ``env.run()``.  The caller advances the simulation
    one step at a time via :func:`step_simulation`.
    """
    params: ScenarioParams = resolve_scenario(scenario_key)
    params.strategy_key = strategy_key
    ensure_strategy(strategy_key)

    if hours_override is not None:
        params.sim_hours = int(hours_override)

    return initialize_simulation_from_params(
        params,
        seed=seed,
        step_hours=step_hours,
        dreamerv3_run_key=dreamerv3_run_key,
        use_operational_seed=use_operational_seed,
        operational_payload=operational_payload,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. build_snapshot
# ═══════════════════════════════════════════════════════════════════════════


def _center_snapshot(center: DonationCenter) -> dict[str, Any]:
    inv = center.inventory_by_component()
    return {
        "name": center.name,
        "type": getattr(center, "original_type", None) or center.ctype,
        "role": center.ctype,
        "external_id": getattr(center, "external_id", None),
        "lat": center.lat,
        "lon": center.lon,
        "inventory": {
            "RBC": inv.get("RBC", 0),
            "PLATELETS": inv.get("PLATELETS", 0),
            "PLASMA": inv.get("PLASMA", 0),
        },
        "stats": {
            "donated": center.stats["donated"],
            "rejected": center.stats["rejected"],
            "no_show": center.stats["no_show"],
            "transfused": center.stats["transfused"],
            "expired": center.stats["expired"],
        },
        "total_units": len(center.inventory),
    }


def build_snapshot(
    state: SimState,
    step_number: int,
    *,
    prev_donated: int = 0,
    prev_shortage: int = 0,
    policy_log_since_index: int = 0,
) -> dict[str, Any]:
    """
    Build a JSON-serialisable snapshot of the current simulation state.

    ``prev_donated`` / ``prev_shortage`` are cumulative values from the
    *previous* step so that deltas (recent_donations / recent_shortages) can
    be computed.
    """
    total_donated = sum(c.stats["donated"] for c in state.centers)
    total_transfused = sum(c.stats["transfused"] for c in state.centers)
    total_expired = sum(c.stats["expired"] for c in state.centers)
    total_shortage = state.total_shortage_units

    fulfilled_plus_shortage = total_transfused + total_shortage
    shortage_rate = (
        total_shortage / fulfilled_plus_shortage * 100.0
        if fulfilled_plus_shortage > 0
        else 0.0
    )

    current_hour = state.env.now
    total_hours = state.params.sim_hours
    progress = min(current_hour / max(total_hours, 1) * 100.0, 100.0)

    # Active actions
    active_actions = [
        {
            "key": aa.action.key,
            "name": aa.action.name,
            "intensity": float(getattr(aa, "intensity", 1.0)),
        }
        for aa in state.active_actions
    ]

    # Latest inventory snapshot from the monitor process
    inventory_snapshot: dict[str, Any] | None = None
    if state.hourly_inventory:
        inventory_snapshot = state.hourly_inventory[-1]

    runtime_controller = getattr(state, "runtime_controller", None)
    controller_decision = summarize_policy_events(
        state.policy_log,
        policy_log_since_index,
        runtime_controller=runtime_controller,
    )

    return {
        "type": "step",
        "step": step_number,
        "hour": round(current_hour, 2),
        "total_hours": total_hours,
        "progress": round(progress, 2),
        "score": round(float(state.episode_score), 4),
        "shortage_rate": round(shortage_rate, 2),
        "total_donated": total_donated,
        "total_transfused": total_transfused,
        "total_expired": total_expired,
        "total_shortage": total_shortage,
        "budget_remaining": round(float(state.budget_remaining), 2),
        "budget_total": round(float(state.budget_total), 2),
        "budget_spent": round(float(state.budget_spent), 2),
        "weather": state.weather.state,
        "donor_count_by_center": [
            {
                "name": c.name,
                "active_donors": c.nurses.count if hasattr(c.nurses, "count") else 0,
                "queue_length": len(c.nurses.queue)
                if hasattr(c.nurses, "queue")
                else 0,
            }
            for c in state.centers
        ],
        "active_actions": active_actions,
        "controller_decision": controller_decision,
        "centers": [_center_snapshot(c) for c in state.centers],
        "inventory_snapshot": inventory_snapshot,
        "recent_donations": max(total_donated - prev_donated, 0),
        "recent_shortages": max(total_shortage - prev_shortage, 0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. build_report
# ═══════════════════════════════════════════════════════════════════════════


def build_report(state: SimState) -> dict[str, Any]:
    """
    Final comprehensive report produced when the simulation completes.
    Mirrors what ``run_single`` returns in *simulator_service.py*.
    """
    summary = summarize_state(state)
    reward = summary["reward"]
    metrics = extract_eval_metrics(summary, state)

    return {
        "generated_at": utc_now_iso(),
        "hours": state.params.sim_hours,
        "summary": {
            **metrics,
            "active_action_keys": list(summary["active_action_keys"]),
            "shortage_by_component": dict(summary["shortage_by_component"]),
            "shortage_by_hospital": dict(summary["shortage_by_hospital"]),
        },
        "reward_terms": {k: float(v) for k, v in reward.terms.items()},
        "inventory_timeline": state.hourly_inventory,
        "centers": [serialize_center(c) for c in state.centers],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. build_init_message
# ═══════════════════════════════════════════════════════════════════════════


def build_init_message(
    state: SimState,
    scenario_key: str,
    strategy_key: str,
    seed: int,
) -> dict[str, Any]:
    """
    The first message sent over the WebSocket after the simulation has been
    initialised (before any steps are run).
    """
    params = state.params

    scenario_info = scenario_payload(
        scenario_key,
        params,
        source="custom" if scenario_key not in SCENARIOS else "built_in",
        default_strategy_key=strategy_key,
    )
    strategy_info = strategy_payload(strategy_key)

    runtime_controller = getattr(state, "runtime_controller", None)
    controller_step_hours = getattr(state, "runtime_controller_step_hours", None)

    return {
        "type": "init",
        "generated_at": utc_now_iso(),
        "scenario": scenario_info,
        "strategy": strategy_info,
        "runtime_controller": runtime_controller,
        "controller_step_hours": controller_step_hours,
        "seed": seed,
        "total_hours": params.sim_hours,
        "step_hours": float(state.episode_step_hours or 6.0),
        "total_steps": _total_steps(params.sim_hours, state.episode_step_hours or 6.0),
        "budget_total": round(float(state.budget_total), 2),
        "weather": state.weather.state,
        "centers": [_center_snapshot(c) for c in state.centers],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. WebSocket endpoint
# ═══════════════════════════════════════════════════════════════════════════


def _total_steps(sim_hours: float, step_hours: float) -> int:
    if step_hours <= 0:
        return 0
    import math

    return math.ceil(sim_hours / step_hours)


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Send JSON; silently swallow connection-closed errors."""
    try:
        await ws.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        pass


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Interactive step-by-step simulation over WebSocket.

    The endpoint accepts a *start* command, then pushes ``init`` → N×``step``
    → ``complete`` messages.  The client may pause / resume / stop / change
    speed at any time.
    """
    await websocket.accept()

    # ── Shared mutable state between the listener and the runner ────────
    pause_event = asyncio.Event()
    pause_event.set()  # starts in "not paused" (event is set → no wait)

    stop_requested = False
    speed: float = 1.0
    sim_running = False  # whether the simulation loop is active
    forecast_job_id: str | None = None
    forecast_tick_state: dict[str, Any] = {
        "tick_number": 0,
        "simulated_hour": 0.0,
        "simulated_state": {},
        "centers": [],
    }

    # ── Background listener task ────────────────────────────────────────
    async def _listen() -> dict[str, Any] | None:
        """
        Return the next JSON message from the client, or *None* if the
        connection was closed.
        """
        try:
            return await websocket.receive_json()
        except (WebSocketDisconnect, RuntimeError):
            return None

    async def _command_listener() -> None:
        nonlocal stop_requested, speed, forecast_job_id

        while True:
            msg = await _listen()
            if msg is None:
                stop_requested = True
                pause_event.set()  # unblock runner if paused
                return

            cmd = msg.get("command")

            if cmd == "pause":
                pause_event.clear()
                await _send_json(websocket, {"type": "paused"})

            elif cmd == "resume":
                pause_event.set()
                await _send_json(websocket, {"type": "resumed"})

            elif cmd == "stop":
                stop_requested = True
                pause_event.set()  # unblock
                await _send_json(websocket, {"type": "stopped"})
                return

            elif cmd == "set_speed":
                new_speed = msg.get("speed", 1.0)
                try:
                    speed = max(0.1, float(new_speed))
                except (TypeError, ValueError):
                    speed = 1.0

            elif cmd == "set_random_event_rate":
                await _send_json(
                    websocket,
                    {
                        "type": "info",
                        "message": (
                            "Random event rate adjustments are available in "
                            "Digital Twin mode."
                        ),
                        "rate": msg.get("rate"),
                    },
                )

            elif cmd == "set_forecast_model":
                candidate = valid_forecast_job_id(str(msg.get("job_id") or ""))
                if not candidate:
                    await _send_json(
                        websocket,
                        {"type": "error", "message": "Invalid or disabled forecast job_id."},
                    )
                    continue
                forecast_job_id = candidate
                panel = await asyncio.to_thread(
                    build_forecast_panel,
                    forecast_job_id,
                    tick_number=int(forecast_tick_state.get("tick_number") or 0),
                    simulated_hour=float(forecast_tick_state.get("simulated_hour") or 0),
                    simulated_state=dict(forecast_tick_state.get("simulated_state") or {}),
                    snapshot=None,
                    centers=list(forecast_tick_state.get("centers") or []),
                )
                await _send_json(
                    websocket,
                    {"type": "forecast_panel_updated", "forecast_panel": panel},
                )

            elif cmd == "start" and not sim_running:
                # Will be handled by the main loop; push it back by
                # re-processing inside _run_simulation.
                # Actually, the first "start" is awaited in the main body,
                # so any *start* arriving here while a sim is running is
                # simply ignored.
                pass

    # ── Main control loop ───────────────────────────────────────────────
    try:
        # Wait for the initial "start" command.
        start_msg: dict[str, Any] | None = None
        while True:
            msg = await _listen()
            if msg is None:
                return  # client disconnected
            if msg.get("command") == "start":
                start_msg = msg
                break
            else:
                await _send_json(
                    websocket,
                    {
                        "type": "error",
                        "message": "Send a 'start' command to begin the simulation.",
                    },
                )

        # Extract parameters from start message
        scenario_key: str = start_msg.get("scenario_key", "baseline")
        strategy_key: str = start_msg.get("strategy_key", "baseline")
        seed: int = int(start_msg.get("seed", 100))
        hours_override: int | None = start_msg.get("hours_override")
        step_hours: float = float(start_msg.get("step_hours", 6.0))
        dreamerv3_run_key: str | None = start_msg.get("dreamerv3_run_key")
        speed = max(0.1, float(start_msg.get("speed", 1.0)))
        use_operational_seed = bool(start_msg.get("use_operational_seed", True))
        forecast_job_id = resolve_forecast_job_id(start_msg)

        if hours_override is not None:
            hours_override = int(hours_override)

        # Initialise simulation in a thread (CPU-bound) ------------------
        try:
            state: SimState = await asyncio.to_thread(
                lambda: initialize_simulation(
                    scenario_key,
                    strategy_key,
                    seed,
                    hours_override,
                    step_hours,
                    dreamerv3_run_key,
                    use_operational_seed=use_operational_seed,
                )
            )
        except Exception as exc:
            await _send_json(websocket, {"type": "error", "message": str(exc)})
            return

        # Send init message ----------------------------------------------
        init_msg = build_init_message(state, scenario_key, strategy_key, seed)
        init_msg["forecast_panel"] = forecast_panel_meta()
        init_msg["forecast_job_id"] = forecast_job_id
        init_centers = [_center_snapshot(center) for center in state.centers]
        forecast_tick_state["centers"] = init_centers
        forecast_tick_state["simulated_hour"] = float(state.env.now)
        if forecast_job_id:
            init_msg["forecast_live"] = await asyncio.to_thread(
                build_forecast_panel,
                forecast_job_id,
                tick_number=0,
                simulated_hour=float(state.env.now),
                simulated_state={},
                snapshot=None,
                centers=init_centers,
            )
        await _send_json(websocket, init_msg)

        # Launch the command listener in the background -------------------
        sim_running = True
        listener_task = asyncio.create_task(_command_listener())

        # ── Step-by-step simulation loop ────────────────────────────────
        actual_step_hours = step_hours
        total_hours = state.params.sim_hours
        step_number = 0

        # Cumulative counters for computing deltas
        prev_donated = sum(c.stats["donated"] for c in state.centers)
        prev_shortage = state.total_shortage_units
        policy_log_cursor = 0

        try:
            while state.env.now < total_hours and not stop_requested:
                # Honour pause
                await pause_event.wait()
                if stop_requested:
                    break

                # Clamp last step so we don't overshoot sim_hours
                remaining = total_hours - state.env.now
                if remaining <= 0:
                    break
                effective_step = min(actual_step_hours, remaining)

                # Run one step in a worker thread
                try:
                    await asyncio.to_thread(step_simulation, state, effective_step)
                except Exception as exc:
                    await _send_json(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Simulation step error: {exc}",
                        },
                    )
                    break

                step_number += 1

                # Build & send snapshot
                snapshot = build_snapshot(
                    state,
                    step_number,
                    prev_donated=prev_donated,
                    prev_shortage=prev_shortage,
                    policy_log_since_index=policy_log_cursor,
                )
                forecast_tick_state["tick_number"] = step_number
                forecast_tick_state["simulated_hour"] = float(state.env.now)
                forecast_tick_state["centers"] = list(snapshot.get("centers") or [])
                if forecast_job_id:
                    forecast_panel = await asyncio.to_thread(
                        build_forecast_panel,
                        forecast_job_id,
                        tick_number=step_number,
                        simulated_hour=float(state.env.now),
                        simulated_state={},
                        snapshot=None,
                        centers=forecast_tick_state["centers"],
                    )
                    if forecast_panel:
                        snapshot["forecast_panel"] = forecast_panel
                await _send_json(websocket, snapshot)

                # Update delta baselines for next step
                prev_donated = sum(c.stats["donated"] for c in state.centers)
                prev_shortage = state.total_shortage_units
                policy_log_cursor = len(state.policy_log)

                # Inter-step delay (adjustable via speed)
                if not stop_requested and state.env.now < total_hours:
                    delay = max(0.05, 0.5 / speed)
                    await asyncio.sleep(delay)

            # ── Simulation finished (or was stopped) ────────────────────
            if not stop_requested:
                try:
                    report = await asyncio.to_thread(build_report, state)
                except Exception as exc:
                    await _send_json(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Report generation error: {exc}",
                        },
                    )
                    return

                await _send_json(
                    websocket,
                    {
                        "type": "complete",
                        "report": report,
                    },
                )

        finally:
            sim_running = False
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.exception("Unexpected error in websocket_endpoint")
        try:
            await _send_json(websocket, {"type": "error", "message": str(exc)})
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# 6 & 7. Setup data helpers + HTTP endpoint
# ═══════════════════════════════════════════════════════════════════════════


def get_scenarios_and_strategies() -> dict[str, Any]:
    """
    Return available scenarios and strategies for the front-end setup screen.
    """
    return {
        "scenarios": list_scenarios(),
        "strategies": list_strategies(),
    }


@router.get("/setup")
def get_setup_data() -> dict[str, Any]:
    """
    ``GET /api/sim/setup`` – scenarios & strategies for the UI setup screen.
    """
    dreamerv3_runs = discover_dreamerv3_runs()
    return {
        **get_scenarios_and_strategies(),
        "custom_scenario_editor": custom_scenario_editor_config(),
        "dreamerv3_runs": [public_dreamerv3_run_payload(r) for r in dreamerv3_runs],
        "default_dreamerv3_run_key": dreamerv3_runs[0]["key"]
        if dreamerv3_runs
        else None,
        "forecast_panel": forecast_panel_meta(),
    }
