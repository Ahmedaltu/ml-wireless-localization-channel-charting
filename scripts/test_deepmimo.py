import deepmimo as dm

print("Generating dataset...")
dataset = dm.generate("asu_campus_3p5")

print("\n=== TYPE ===")
print(type(dataset))

# Try dict-like inspection first
try:
    keys = list(dataset.keys())
    print("\n=== TOP-LEVEL KEYS ===")
    print(keys)

    for key in keys:
        print(f"\n--- KEY: {key} ---")
        try:
            val = dataset[key]
            print("TYPE:", type(val))

            if hasattr(val, "shape"):
                print("SHAPE:", val.shape)
            elif isinstance(val, (list, tuple)):
                print("LEN:", len(val))
            elif isinstance(val, dict):
                print("DICT KEYS:", list(val.keys())[:20])
            else:
                print("VALUE:", val)
        except Exception as e:
            print("Could not inspect key:", e)

except Exception as e:
    print("\nDataset is not dict-like or keys() failed:", e)

    print("\n=== DIR(dataset) first 100 ===")
    try:
        print(dir(dataset)[:100])
    except Exception as e2:
        print("dir() failed:", e2)

# Try common key access safely (NOT hasattr)
common_keys = ["channel", "channels", "csi", "rx_pos", "tx_pos", "paths", "scene"]

print("\n=== COMMON KEY CHECK ===")
for key in common_keys:
    try:
        val = dataset[key]
        print(f"\nFOUND KEY: {key}")
        print("TYPE:", type(val))
        if hasattr(val, "shape"):
            print("SHAPE:", val.shape)
        else:
            print("VALUE:", val)
    except Exception:
        pass