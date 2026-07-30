from main import run  # adjust if needed


def run_simulation(params, controller, seed=0):
    return run(
        params=params,
        controller=controller,
        seed=seed,
        fast_mode=True,  # IMPORTANT for speed
        enable_logs=False,
    )
