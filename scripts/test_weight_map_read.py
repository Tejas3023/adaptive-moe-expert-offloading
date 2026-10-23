from src.modeling.olmoe_loader import load_olmoe_model

model = load_olmoe_model()
modules = dict(model.named_modules())

name = "model.layers.2.mlp.experts.0.gate_proj"
hook = modules[name]._hf_hook
weight_map = hook.weights_map

print("Module:", name)
print("Weight map type:", type(weight_map).__name__)
print("Weight map prefix:", repr(getattr(weight_map, "prefix", None)))

dataset = getattr(weight_map, "dataset", None)
print("Underlying dataset type:", type(dataset).__name__)

prefix = getattr(weight_map, "prefix", "")
candidate = name + ".weight"

if prefix and candidate.startswith(prefix):
    key = candidate[len(prefix):]
else:
    key = candidate

print("Candidate lookup key:", repr(key))
print("Underlying full key:", repr(prefix + key))

try:
    weight = weight_map[key]
    print("Read successful")
    print("Shape:", tuple(weight.shape))
    print("Dtype:", weight.dtype)
    print("Device:", weight.device)
except Exception as exc:
    print("Read failed:", type(exc).__name__, str(exc))