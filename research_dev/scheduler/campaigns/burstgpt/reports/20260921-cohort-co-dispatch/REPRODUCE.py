import faulthandler
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

from research_dev.scheduler._internal.runtime_queue import RuntimeDispatchQueue
from research_dev.scheduler.tests.test_adaptive_runtime import AdaptiveRuntimeIntegrationTests

output = Path(sys.argv[1])
records = []
original_wait = RuntimeDispatchQueue.wait_ready
original_commit = RuntimeDispatchQueue.commit_ready

def record(queue, stage, request_id):
    with queue._condition:
        rows = []
        for rid, entry in queue._entries.items():
            rows.append({
                'request_id': rid, 'state': entry.state,
                'causal_ready': queue._causal_ready(entry),
                'active_conflict': queue._active_conflict(entry, excluding_request_id=rid),
                'earlier_queued_conflict': queue._earlier_queued_conflict(entry),
                'predecessors': sorted(entry.predecessor_request_ids),
                'cohort': getattr(entry, 'decode_cohort', None).to_json() if getattr(entry, 'decode_cohort', None) else None,
                'leases': [{'token': lease.token, 'owner': lease.owner_id,
                            'resource': lease.resource_id, 'lanes': lease.lanes}
                           for lease in entry.decision.leases],
            })
        records.append({'stage': stage, 'request_id': request_id, 'rows': rows})
    output.write_text(json.dumps(records, indent=2) + '\n')

def wait(queue, request_id, epoch_ns):
    record(queue, 'before_wait', request_id)
    result = original_wait(queue, request_id, epoch_ns)
    record(queue, 'ready_' + result.status, request_id)
    return result

def commit(queue, receipt):
    result = original_commit(queue, receipt)
    record(queue, 'after_commit_' + (result.status if result else 'retry'), receipt.request_id)
    return result

faulthandler.dump_traceback_later(10)
with mock.patch.object(RuntimeDispatchQueue, 'wait_ready', wait), mock.patch.object(RuntimeDispatchQueue, 'commit_ready', commit):
    suite = unittest.TestSuite([AdaptiveRuntimeIntegrationTests('test_cold_phone_cohort_shares_transition_identity')])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
faulthandler.cancel_dump_traceback_later()
raise SystemExit(0 if result.wasSuccessful() else 1)
