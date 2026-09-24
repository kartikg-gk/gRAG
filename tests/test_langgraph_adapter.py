import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from graphrag.tracing import load


@unittest.skipUnless(importlib.util.find_spec("langchain_core"), "adapter extra not installed")
class AdapterTests(unittest.TestCase):
    def test_adapter_writes_scored_v4_trace(self):
        from graphrag.adapters.langgraph import LangGraphTracer, save_run

        with TemporaryDirectory() as directory:
            tracer = LangGraphTracer()
            trace, destination = save_run(
                tracer, query="Who owns the service?", answer="Ada owns the service.",
                path=Path(directory) / "trace.json",
            )
            self.assertEqual(destination, Path(directory) / "trace.json")
            self.assertEqual(load(destination).query, trace.query)
            self.assertEqual(load(destination).answer, trace.answer)
