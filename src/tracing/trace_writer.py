import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from .expert_trace_logger import ExpertRoutingTrace


class TraceWriter:
    """
    Writes expert routing traces to JSONL files.

    JSONL format:
    One routing trace per line.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

        # Create the parent directory if it does not exist.
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(
        self,
        traces: Iterable[ExpertRoutingTrace],
        append: bool = False,
    ) -> int:
        """
        Write routing traces to a JSONL file.

        Parameters
        ----------
        traces:
            Iterable of ExpertRoutingTrace objects.

        append:
            If False, overwrite the existing file.
            If True, append traces to the existing file.

        Returns
        -------
        int
            Number of traces written.
        """

        mode = "a" if append else "w"
        count = 0

        with self.output_path.open(
            mode,
            encoding="utf-8",
        ) as file:

            for trace in traces:
                trace_dict = asdict(trace)

                json.dump(
                    trace_dict,
                    file,
                )

                file.write("\n")

                count += 1

        return count

    def read(self) -> List[dict]:
        """
        Read all routing traces from the JSONL file.

        Returns
        -------
        List[dict]
            List of routing trace dictionaries.
        """

        if not self.output_path.exists():
            raise FileNotFoundError(
                f"Trace file does not exist: {self.output_path}"
            )

        traces = []

        with self.output_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                line = line.strip()

                if line:
                    traces.append(
                        json.loads(line)
                    )

        return traces