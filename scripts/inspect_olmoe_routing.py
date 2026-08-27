import inspect

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock


def separator():
    print("=" * 80)


def main():
    separator()
    print("OLMOE ROUTING IMPLEMENTATION INSPECTION")
    separator()

    print("\nInspecting OlmoeSparseMoeBlock.forward()...\n")

    source = inspect.getsource(OlmoeSparseMoeBlock.forward)

    print(source)

    separator()
    print("ROUTING INSPECTION COMPLETE")
    separator()


if __name__ == "__main__":
    main()