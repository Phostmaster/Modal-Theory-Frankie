import os
import time

for adapter in ["adapters/analytic", "adapters/engagement"]:
    path = f"{adapter}/adapter_model.safetensors"
    if os.path.exists(path):
        t = os.path.getmtime(path)
        print(f"{adapter}: {time.ctime(t)}")
    else:
        print(f"{adapter}: NOT FOUND")