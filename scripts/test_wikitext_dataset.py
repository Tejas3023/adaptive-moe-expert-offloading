from datasets import load_dataset


DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"


def main():
    print("=" * 80)
    print("WIKITEXT DATASET LOADING TEST")
    print("=" * 80)

    print()
    print("Dataset:")
    print(DATASET_NAME)

    print("Configuration:")
    print(DATASET_CONFIG)

    print()
    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="test",
    )

    print()
    print("Dataset loaded successfully.")

    print()
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)

    print()
    print(f"Number of examples: {len(dataset)}")

    print()
    print(f"Columns: {dataset.column_names}")

    print()
    print("=" * 80)
    print("FIRST 10 RAW EXAMPLES")
    print("=" * 80)

    for index in range(10):
        print()
        print(f"Example {index}:")
        print(repr(dataset[index]["text"]))

    print()
    print("=" * 80)
    print("FIRST 5 NON-EMPTY EXAMPLES")
    print("=" * 80)

    non_empty_count = 0

    for example in dataset:
        text = example["text"].strip()

        if text:
            print()
            print(
                f"Non-empty example "
                f"{non_empty_count + 1}:"
            )
            print(text[:500])

            non_empty_count += 1

            if non_empty_count == 5:
                break

    print()
    print("=" * 80)
    print("WIKITEXT DATASET TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()