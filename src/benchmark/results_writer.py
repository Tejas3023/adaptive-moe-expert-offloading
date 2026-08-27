import csv
from pathlib import Path


class ResultsWriter:
    """
    Writes experiment results to CSV files.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def write(self, results: list[dict]) -> None:
        """
        Write a list of result dictionaries to a CSV file.
        """

        if not results:
            raise ValueError(
                "Cannot write an empty results list."
            )

        # Create parent directories if they do not exist.
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(results[0].keys())

        with open(
            self.output_path,
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(results)

    def read(self) -> list[dict]:
        """
        Read experiment results back from the CSV file.
        """

        with open(
            self.output_path,
            mode="r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            return list(reader)