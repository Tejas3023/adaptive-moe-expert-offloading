from collections import Counter, defaultdict

from src.modeling.olmoe_loader import load_olmoe_model

model = load_olmoe_model()
counts = defaultdict(Counter)
examples = defaultdict(list)

for name, module in model.named_modules():
    if not hasattr(module, "_hf_hook"):
        continue

    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "model":
        continue
    if parts[1] != "layers" or not parts[2].isdigit():
        continue

    layer_id = int(parts[2])
    hook = module._hf_hook
    status = "offload" if getattr(hook, "offload", False) else "resident"

    counts[layer_id][status] += 1
    examples[layer_id].append((name, status))

print("\nHOOK COUNTS BY TRANSFORMER LAYER")
for layer_id in range(16):
    offload_count = counts[layer_id]["offload"]
    resident_count = counts[layer_id]["resident"]
    print(
        f"Layer {layer_id:2}: "
        f"offload={offload_count:3}, "
        f"resident={resident_count:3}, "
        f"total={offload_count + resident_count:3}"
    )

print("\nEXAMPLE OFFLOADED MODULES")
shown = 0
for layer_id in range(16):
    for name, status in examples[layer_id]:
        if status == "offload" and (
            ".experts." in name or name.endswith(".gate")
        ):
            print(name)
            shown += 1
            if shown == 12:
                break
    if shown == 12:
        break
