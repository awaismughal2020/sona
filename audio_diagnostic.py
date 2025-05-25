"""
© 2025 Awais Mughal. All rights reserved.
Unauthorized commercial use is prohibited.
"""

import sounddevice as sd
import numpy as np

print("🎤 Audio System Check")
print("=" * 30)

try:
    # List all devices
    devices = sd.query_devices()
    print(f"Total devices: {len(devices)}")

    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            print(f"[{i}] {device['name']} - {device['max_input_channels']} channels")

    print(f"\nFound {len(input_devices)} input devices")

    # Test each input device
    for device_id, device in input_devices:
        try:
            print(f"\nTesting device {device_id}: {device['name']}")
            recording = sd.rec(
                int(0.5 * 16000),  # 0.5 seconds
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                device=device_id
            )
            sd.wait()

            max_level = np.max(np.abs(recording))
            print(f"✅ Success! Max level: {max_level:.4f}")
            print(f"💡 Use device={device_id} in your code")

        except Exception as e:
            print(f"❌ Failed: {e}")

except Exception as e:
    print(f"❌ Error: {e}")
