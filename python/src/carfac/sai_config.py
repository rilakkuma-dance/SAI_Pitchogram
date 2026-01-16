# sai_config.py
from modules.visualization_handler import SAIParams

# ==========================================
# SAI VISUALIZATION SETTINGS
# ==========================================
SAI_WIDTH = 400             
TRIGGERS_PER_FRAME = 2      

def get_sai_params(n_channels, chunk_size, smoothing_scale=0.5):
    """
    Creates and returns a configured SAIParams object.
    
    Args:
        n_channels (int): From AudioProcessor.
        chunk_size (int): Audio buffer size.
        smoothing_scale (float): 0.1 for sharp/fast, 0.5 for smoother/slower.
    """
    return SAIParams(
        num_channels=n_channels,
        sai_width=SAI_WIDTH,
        future_lags=SAI_WIDTH - 1,
        num_triggers_per_frame=TRIGGERS_PER_FRAME,
        trigger_window_width=chunk_size + 1,
        input_segment_width=chunk_size,
        channel_smoothing_scale=smoothing_scale
    )