class SB3Adapter:
    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action), None
