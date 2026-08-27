from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights


MODEL_NAME = "allenai/OLMoE-1B-7B-0924"


def separator():
    print("=" * 80)


def main():
    separator()
    print("OLMOE ARCHITECTURE INSPECTION")
    separator()

    print(f"\nModel: {MODEL_NAME}")

    # Load configuration only.
    config = AutoConfig.from_pretrained(MODEL_NAME)

    print("\nCreating model architecture on the META device...")
    print("No model weights will be downloaded or loaded into GPU memory.")

    # Creates the architecture without allocating real parameter memory.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    print("\nArchitecture created successfully.")

    separator()
    print("TOP-LEVEL MODEL STRUCTURE")
    separator()

    print(f"\nModel class: {model.__class__.__name__}")

    print("\nTop-level children:")
    for name, module in model.named_children():
        print(f"  {name}: {module.__class__.__name__}")

    separator()
    print("TRANSFORMER LAYER INFORMATION")
    separator()

    layers = model.model.layers

    print(f"\nNumber of transformer layers: {len(layers)}")

    print("\nFirst layer structure:")
    first_layer = layers[0]

    for name, module in first_layer.named_children():
        print(f"  {name}: {module.__class__.__name__}")

    separator()
    print("MLP / MOE STRUCTURE")
    separator()

    moe_module = first_layer.mlp

    print(f"\nMoE module class: {moe_module.__class__.__name__}")

    print("\nChildren of the MoE module:")

    for name, module in moe_module.named_children():
        print(f"  {name}: {module.__class__.__name__}")

    separator()
    print("ROUTER / GATE INFORMATION")
    separator()

    if hasattr(moe_module, "gate"):
        gate = moe_module.gate

        print(f"\nRouter module name: gate")
        print(f"Router class: {gate.__class__.__name__}")

        print("\nRouter parameters:")

        for name, parameter in gate.named_parameters():
            print(
                f"  {name}: "
                f"shape={tuple(parameter.shape)}, "
                f"device={parameter.device}"
            )
    else:
        print("\nNo attribute named 'gate' found.")
        print("Available attributes:")
        print(dir(moe_module))

    separator()
    print("EXPERT INFORMATION")
    separator()

    if hasattr(moe_module, "experts"):
        experts = moe_module.experts

        print(f"\nTotal experts in first MoE layer: {len(experts)}")

        print("\nFirst expert:")
        print(experts[0])

        print("\nExpert 0 named modules:")

        for name, module in experts[0].named_children():
            print(f"  {name}: {module.__class__.__name__}")
    else:
        print("\nNo attribute named 'experts' found.")

    separator()
    print("MODEL MEMORY CHECK")
    separator()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"\nTotal architecture parameters: {total_parameters:,}")
    print("Parameter device: meta")
    print("Real model weights allocated: No")

    separator()
    print("ARCHITECTURE INSPECTION COMPLETE")
    separator()


if __name__ == "__main__":
    main()