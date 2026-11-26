"""Timer state for the agent. Knows nothing about the LLM or JSON: methods take and return plain values."""

import time
from collections.abc import Callable
from dataclasses import dataclass

SECONDS_PER_MINUTE = 60


@dataclass
class Timer:
    label: str
    length_seconds: float
    started_at: float   # seconds on the store's clock

    def remaining_seconds(self, now: float) -> float:
        return max(0.0, self.started_at + self.length_seconds - now)


class TimerStore:
    """All timers of one session, keyed by label (case-insensitive).

    Every method returns a dict. A problem (no such timer, duplicate label, bad length) comes back as
    {"error": "..."} instead of an exception, so the model can tell the user what went wrong.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock   # injectable so tests can move time forward without waiting
        self.timers: dict[str, Timer] = {}

    def start(self, label: str, minutes: float) -> dict:
        label = label.strip()
        if not label:
            return {"error": "A timer needs a label."}
        if minutes <= 0:
            return {"error": "The length must be more than zero minutes."}
        if label.lower() in self.timers:
            return {"error": f"There is already a timer called {label}. Change or cancel it instead."}
        timer = Timer(label, minutes * SECONDS_PER_MINUTE, self.clock())
        self.timers[label.lower()] = timer
        return {"started": self._describe(timer)}

    def list(self) -> dict:
        return {"timers": [self._describe(timer) for timer in self.timers.values()]}

    def set_length(self, minutes: float, label: str | None = None) -> dict:
        """Give the timer a new total length, still counted from when it was started."""
        if minutes <= 0:
            return {"error": "The length must be more than zero minutes."}
        timer = self._find(label)
        if isinstance(timer, dict):
            return timer
        timer.length_seconds = minutes * SECONDS_PER_MINUTE
        return {"changed": self._describe(timer)}

    def extend(self, minutes: float, label: str | None = None) -> dict:
        if minutes <= 0:
            return {"error": "The extra time must be more than zero minutes."}
        timer = self._find(label)
        if isinstance(timer, dict):
            return timer
        timer.length_seconds += minutes * SECONDS_PER_MINUTE
        return {"extended": self._describe(timer)}

    def cancel(self, label: str | None = None) -> dict:
        timer = self._find(label)
        if isinstance(timer, dict):
            return timer
        del self.timers[timer.label.lower()]
        return {"cancelled": timer.label}

    def _find(self, label: str | None) -> Timer | dict:
        """The timer with this label, or the only timer when no label is given; otherwise an error dict."""
        if label and label.strip():
            timer = self.timers.get(label.strip().lower())
            if timer is None:
                return {"error": f"No timer called {label.strip()}.", **self.list()}
            return timer
        if len(self.timers) == 1:
            return next(iter(self.timers.values()))
        if not self.timers:
            return {"error": "There are no timers."}
        return {"error": "Several timers are running; say which one.", **self.list()}

    def _describe(self, timer: Timer) -> dict:
        remaining = timer.remaining_seconds(self.clock())
        return {
            "label": timer.label,
            "length_minutes": round(timer.length_seconds / SECONDS_PER_MINUTE, 2),
            "remaining_seconds": round(remaining),
            "finished": remaining == 0,
        }
