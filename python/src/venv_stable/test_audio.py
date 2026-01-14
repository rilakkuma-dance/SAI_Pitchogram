import sounddevice as sd

print("--- Audio Host APIs ---")
try:
    # This lists the drivers (MME, DirectSound, WASAPI, ASIO)
    print(sd.query_hostapis())
except Exception as e:
    print(f"Error querying Host APIs: {e}")

print("\n--- Audio Devices ---")
try:
    # This lists the actual speakers/mics
    print(sd.query_devices())
except Exception as e:
    print(f"Error querying Devices: {e}")