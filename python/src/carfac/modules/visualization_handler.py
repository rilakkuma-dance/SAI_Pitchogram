import numpy as np
from dataclasses import dataclass

@dataclass
class SAIParams:
    num_channels: int = 200
    sai_width: int = 400
    future_lags: int = 5
    num_triggers_per_frame: int = 10
    trigger_window_width: int = 20
    input_segment_width: int = 30
    channel_smoothing_scale: float = 0.5

class VisualizationHandler:
    def __init__(self, sample_rate_hz: int, sai_params: SAIParams):
        self.sample_rate_hz = sample_rate_hz
        self.sai_params = sai_params
        self.output = np.zeros(200)
        self.img = np.zeros((200, 200, 3), dtype=np.uint8)
        self.vowel_coords = np.zeros((2, 1), dtype=np.float32)

    def get_vowel_embedding(self, nap):
        if nap.shape[0] > 0:
            self.vowel_coords[0, 0] = np.mean(nap[:10, :]) if nap.shape[1] > 0 else 0
            self.vowel_coords[1, 0] = np.mean(nap[-10:, :]) if nap.shape[1] > 0 else 0
        return self.vowel_coords

    def run_frame(self, sai_frame: np.ndarray):
        if sai_frame.size > 0:
            self.output = sai_frame.mean(axis=0)[:len(self.output)]
        return self.output

    def draw_column(self, column_ptr: np.ndarray):
        v = np.ravel(self.vowel_coords)
        tint = np.array([
            0.5 - 0.6 * (v[1] if len(v) > 1 else 0),
            0.5 - 0.6 * (v[0] if len(v) > 0 else 0),
            0.35 * (v[0] + v[1] if len(v) > 1 else 0) + 0.4
        ], dtype=np.float32)
        k_scale = 0.5 * 255
        tint *= k_scale
        
        for i in range(min(len(self.output), len(column_ptr))):
            column_ptr[i] = np.clip(np.int32((tint * self.output[i])), 0, 255)