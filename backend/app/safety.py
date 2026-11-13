"""Safety layer: no setting reaches the astronaut or the ESP32 without passing here."""
from .config import FREQUENCY_STEP, LIMITS, WAVEFORMS


def clamp_settings(frequency_khz: float, amplitude_percent: float,
                   timer_minutes: float, waveform: str) -> dict:
    """Force any raw model output into the allowed ranges."""
    f_lo, f_hi = LIMITS["frequency_khz"]
    a_lo, a_hi = LIMITS["amplitude_percent"]
    t_lo, t_hi = LIMITS["timer_minutes"]

    freq = round(float(frequency_khz) / FREQUENCY_STEP) * FREQUENCY_STEP
    return {
        "frequency_khz": int(min(max(freq, f_lo), f_hi)),
        "amplitude_percent": int(min(max(round(amplitude_percent), a_lo), a_hi)),
        "timer_minutes": int(min(max(round(timer_minutes), t_lo), t_hi)),
        "waveform": waveform if waveform in WAVEFORMS else "sine",
    }


def validate_settings(s: dict) -> None:
    """Strict check for settings coming back from the frontend. Raises ValueError."""
    for key, (lo, hi) in LIMITS.items():
        value = s.get(key)
        if value is None or not (lo <= value <= hi):
            raise ValueError(f"{key} must be between {lo} and {hi}")
    if s.get("waveform") not in WAVEFORMS:
        raise ValueError(f"waveform must be one of {WAVEFORMS}")
