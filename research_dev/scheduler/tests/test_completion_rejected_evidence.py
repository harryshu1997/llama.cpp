"""A non-200 ``/completion`` answer carries the server's error text as helper-loss evidence.

Fault run p6f1flt (2026-10-03): the Pixel worker was killed while two Qwen requests shared a server.
The mid-stream request got the SSE error chunk "Compute aborted." and recovered on the desktop; the
request that had just been submitted got a bare HTTP 500, which the classifier did not read as
server-side evidence, so it had no fallback and the run ended. ``CompletionRejectedError`` closes
that gap without touching the classifier: it is a ``CompletionStreamError`` with the body's text.
"""

from __future__ import annotations

import http.server
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.adapters.contracts import (  # noqa: E402
    CompletionRejectedError,
    CompletionStreamError,
    PhysicalAdapterError,
)
from research_dev.scheduler.adapters.heterogeneous_rig_ops.observations import (  # noqa: E402
    _server_side_failure,
    classify_helper_failure,
)
from research_dev.scheduler.adapters.http_backend import (  # noqa: E402
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
    _response_error_message,
)

ABORT_BODY = {"error": {"code": 500, "message": "Compute aborted.", "type": "server_error"}}


class _Handler(http.server.BaseHTTPRequestHandler):
    body: bytes = json.dumps(ABORT_BODY).encode()
    status = 500

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *_args):
        return


class _Response:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self, _limit=None):
        return self._raw


class CompletionRejectedTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def payload(self):
        return LlamaCppCompletionPayload(
            request_id="req-500", expected_model_alias="m", input_tokens=2, output_tokens=4,
            prompt_tokens=(1, 2), seed=0, stream_path=Path(self.directory.name) / "req-500.raw",
            on_first_token=lambda _ns: None)

    def test_a_500_answer_carries_the_servers_error_text(self):
        endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        with self.assertRaises(CompletionRejectedError) as caught:
            LlamaCppHttpClient().complete(endpoint, self.payload(), lambda: None)
        error = caught.exception
        self.assertIsInstance(error, CompletionStreamError)
        self.assertEqual((error.status, error.server_error_message, str(error)),
                         (500, "Compute aborted.", "completion status is 500"))
        # the drop-recovery classifier reads the server log for a server-side failure only
        self.assertTrue(_server_side_failure((error,)))
        self.assertFalse(_server_side_failure((PhysicalAdapterError("completion status is 500"),)))
        # with the log line of the loss, the co-tenant's failure names the lost helper
        helpers = (("op15", "op15-phone", "functionfs-usb"), ("pixel10pro", "pixel10pro-phone", "tcp"))
        devices, evidence = classify_helper_failure(
            ["S41SERVERFFNERROR helper=pixel10pro detail=FFN split EXECUTE header exchange failed"],
            error.server_error_message, helpers)
        self.assertEqual(devices, ("pixel10pro-phone",))
        self.assertIn("helper=pixel10pro", evidence)

    def test_body_parsing_falls_back_to_text_and_tolerates_an_empty_body(self):
        self.assertEqual(_response_error_message(_Response(json.dumps(ABORT_BODY).encode())), "Compute aborted.")
        self.assertEqual(_response_error_message(_Response(b'{"error": "slot unavailable"}')), "slot unavailable")
        self.assertEqual(_response_error_message(_Response(b"  Internal Server Error \n")), "Internal Server Error")
        self.assertIsNone(_response_error_message(_Response(b"")))
        self.assertEqual(_response_error_message(_Response(b"x" * 1000)), "x" * 200)
        with self.assertRaises(PhysicalAdapterError):
            CompletionRejectedError("500")


if __name__ == "__main__":
    unittest.main()
