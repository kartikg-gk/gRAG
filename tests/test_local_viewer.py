import json
import subprocess
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from tempfile import TemporaryDirectory
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from graphrag.tracing import Retrieval, Span, Trace, TraceEdge, TraceItem, to_dict, trace_from_dict
from graphrag.tracing.store import save
from graphrag.tracing.web import discover, make_handler


def sample_trace(query="who owns it?"):
    return Trace(
        query=query, answer="Ada owns it", duration_ms=12,
        retrievals=[Retrieval(query=query, arm="graph", items=[
            TraceItem(id="ada", label="Ada", kind="Person", content="Ada owns it", source="repo", score=0.9, overlap=1),
        ], edges=[TraceEdge("ada", "service", "OWNS", 0.9)])],
        spans=[Span(id="retrieve", name="retrieve", kind="retriever", start_ms=0, end_ms=12, status="ok")],
    )


class LocalViewerTests(unittest.TestCase):
    def test_v4_contains_graph_retrieval_timeline_and_metrics(self):
        payload = to_dict(sample_trace())
        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["graph"]["nodes"][0]["id"], "ada")
        self.assertEqual(payload["graph"]["edges"][0]["relation"], "OWNS")
        self.assertEqual(payload["retrievals"][0]["items"][0]["score"], 0.9)
        self.assertEqual(payload["spans"][0]["end_ms"], 12)
        self.assertEqual(payload["metrics"]["duration_ms"], 12)
        self.assertEqual(to_dict(trace_from_dict(payload)), payload)
        explicit = sample_trace()
        explicit.graph_nodes = []
        self.assertEqual(to_dict(explicit)["graph"]["nodes"], [])

    def test_directory_history_and_loopback_server(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save(sample_trace("first"), directory / "one.json")
            save(sample_trace("second"), directory / "two.json")
            (directory / "other.json").write_text("{}", encoding="utf-8")
            runs = discover(directory)
            self.assertEqual([trace.query for _, trace in runs], ["second", "first"])
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runs))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/api/local/runs") as response:
                    self.assertEqual(len(json.load(response)), 2)
                with urlopen(base + "/api/local/traces/0") as response:
                    self.assertEqual(json.load(response)["query"], "second")
                with urlopen(base + "/") as response:
                    self.assertIn(b"__GRAPHRAG_LOCAL_VIEWER__=true", response.read())
                with self.assertRaises(HTTPError) as rejected:
                    urlopen(base + "/assets/..%5c..%5cREADME.md")
                self.assertEqual(rejected.exception.code, 404)
                for headers in ({"Host": "untrusted.example"}, {"Origin": "https://untrusted.example"}):
                    with self.assertRaises(HTTPError) as forbidden:
                        urlopen(Request(base + "/api/local/traces/0", headers=headers))
                    self.assertEqual(forbidden.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_installed_cli_opens_a_trace_without_browser(self):
        with TemporaryDirectory() as temporary:
            trace_file = save(sample_trace(), Path(temporary) / "trace.json")
            executable = Path(sys.executable).with_name(
                "graphweave.exe" if sys.platform == "win32" else "graphweave"
            )
            command = [str(executable), str(trace_file), "--port", "0", "--no-browser"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                line = process.stdout.readline()
                self.assertIn("http://127.0.0.1:", line)
                address = line.split(" at ", 1)[1].strip()
                with urlopen(address + "api/local/traces/0") as response:
                    self.assertEqual(json.load(response)["query"], "who owns it?")
            finally:
                process.terminate()
                process.communicate(timeout=5)
