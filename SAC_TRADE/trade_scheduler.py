"""The paper's loss-EMA / strict g_m(k) rebound rule, on environment steps."""
import copy
import math


class TRADEScheduler:
    protocol = "loss-ema-env-step-v1"

    def __init__(self, c_min, smoothing=0.99, loss_floor=1e-8):
        if int(c_min) != c_min or c_min < 1:
            raise ValueError("c_min must be a positive number of environment steps")
        if not 0 <= smoothing < 1 or loss_floor <= 0:
            raise ValueError("Invalid smoothing coefficient or numerical loss floor")
        self.c_min = int(c_min)
        self.smoothing = float(smoothing)
        self.loss_floor = float(loss_floor)
        self.cycle = 0
        self.last_refresh_step = None
        self.last_observation_step = None
        self.initial_loss = None
        self.smoothed_loss = None
        self.best_score = math.inf
        self.history = []
        self.last_diagnostic = None

    def update(self, loss, env_step):
        """Observe one pre-update loss, then decide after its optimization block.

        The first fitting cycle starts when learning starts, after warm-up.
        Later k values include intervening environment interactions, even when
        the backbone performs gradient updates only every fourth step.
        """
        env_step = int(env_step)
        if env_step < 1 or not math.isfinite(float(loss)) or loss < 0:
            raise ValueError("Expected a finite nonnegative loss and positive env_step")
        if self.last_observation_step is not None and env_step <= self.last_observation_step:
            raise ValueError("Call once per optimization block at increasing steps")
        if self.last_refresh_step is None:
            self.last_refresh_step = env_step - 1
        self.last_observation_step = env_step
        value = max(float(loss), self.loss_floor)
        if self.initial_loss is None:
            self.initial_loss = self.smoothed_loss = value
        else:
            self.smoothed_loss = (self.smoothing * self.smoothed_loss
                                  + (1 - self.smoothing) * value)
        k = env_step - self.last_refresh_step
        score = None
        refresh = False
        if k >= self.c_min:
            score = (math.log(self.smoothed_loss) - math.log(self.initial_loss)) / (2 * k)
            if score < self.best_score:
                self.best_score = score
            refresh = score > self.best_score
        self.last_diagnostic = dict(step=env_step, cycle=self.cycle, interval=k,
                                    loss=value, ema=self.smoothed_loss, score=score,
                                    best_score=self.best_score, refresh=refresh)
        if refresh:
            self.history.append(dict(self.last_diagnostic))
            self.cycle += 1
            self.last_refresh_step = env_step
            self.initial_loss = self.smoothed_loss = None
            self.best_score = math.inf
        return refresh

    def state_dict(self):
        return dict(protocol=self.protocol, state=copy.deepcopy(self.__dict__))

    def load_state_dict(self, payload):
        if payload.get("protocol") != self.protocol:
            raise ValueError("Checkpoint uses a different TRADE trigger protocol")
        state = payload["state"]
        for key in ["c_min", "smoothing", "loss_floor"]:
            if state[key] != getattr(self, key):
                raise ValueError(f"Checkpoint {key} differs from requested configuration")
        self.__dict__.update(copy.deepcopy(state))
