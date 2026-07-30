from core import load_city_graph
from engine import generate_report, run_with_ppo_agent
from scenarios import SCENARIOS
from stable_baselines3 import PPO

# ─────────────────────────────────────────
# 1. Load trained model
# ─────────────────────────────────────────
model = PPO.load("ppo_fast_model")
print("✅ Loaded PPO model")

# ─────────────────────────────────────────
# 2. Load scenario
# ─────────────────────────────────────────
params = SCENARIOS["baseline"]

# 🚨 CRITICAL FIX: switch to PPO controller
params.strategy_key = "ppo_shortage_minimizer"

# ─────────────────────────────────────────
# 3. Load map
# ─────────────────────────────────────────
G, north, south, east, west = load_city_graph()

# ─────────────────────────────────────────
# 4. Run simulation WITH PPO
# ─────────────────────────────────────────
state = run_with_ppo_agent(
    params,
    G,
    north,
    south,
    east,
    west,
    agent=model,  # ✅ pass SB3 model directly
    seed=42,
)

# ─────────────────────────────────────────
# 5. Generate report
# ─────────────────────────────────────────
generate_report(state)
