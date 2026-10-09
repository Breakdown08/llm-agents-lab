from collections import Counter
from typing import Any

from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer


class UpdateActivityTransformer(StreamTransformer):
    """Считает события updates по именам узлов."""

    required_stream_modes = ("updates",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.totals: StreamChannel[dict[str, int]] = StreamChannel()
        self.counts: Counter[str] = Counter()

    def init(self) -> dict[str, Any]:
        return {"update_totals": self.totals}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "updates":
            return True

        data = event["params"]["data"]
        if not isinstance(data, dict):
            return True

        for node_name in data:
            self.counts[node_name] += 1

        return True

    def finalize(self) -> None:
        self.totals.push(dict(self.counts))
