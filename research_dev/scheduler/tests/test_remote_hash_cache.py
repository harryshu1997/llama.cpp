#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock

from research_dev.scheduler.adapters.phone_session import DirectPhoneFfnSession
from research_dev.scheduler.adapters.remote_hash_cache import (
    RemoteFileIdentity,
    cached_remote_hashes,
    update_remote_hash_cache,
)


class RemoteHashCacheTests(unittest.TestCase):
    @staticmethod
    def identity(
        boot_id: str,
        stat_identity: str = "1:2:2000000000:4:5",
    ):
        return MappingProxyType({
            "/data/model": RemoteFileIdentity(
                boot_id=boot_id,
                stat_identity=stat_identity,
            )
        })

    def test_cache_requires_exact_boot_and_file_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remote-hashes.json"
            identities = self.identity("boot-a")
            update_remote_hash_cache(
                path,
                serial="PHONE1",
                identities=identities,
                hashes={"/data/model": "sha256:" + "1" * 64},
            )
            self.assertEqual(
                cached_remote_hashes(
                    path,
                    serial="PHONE1",
                    identities=identities,
                )["/data/model"],
                "sha256:" + "1" * 64,
            )
            self.assertEqual(
                dict(cached_remote_hashes(
                    path,
                    serial="PHONE1",
                    identities=self.identity("boot-b"),
                )),
                {},
            )
            self.assertEqual(
                dict(cached_remote_hashes(
                    path,
                    serial="PHONE1",
                    identities=self.identity(
                        "boot-a", "1:2:2000000000:4:6"
                    ),
                )),
                {},
            )

    def test_new_session_reuses_only_boot_bound_remote_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "remote-hashes.json"
            identities = self.identity("boot-a")
            first = object.__new__(DirectPhoneFfnSession)
            first.configuration = SimpleNamespace(
                remote_hash_cache_path=cache_path,
                serial="PHONE1",
            )
            first._verified_remote_hash_by_path = {}
            with mock.patch.object(
                first,
                "_remote_file_identities",
                return_value=identities,
            ), mock.patch.object(
                first,
                "_adb",
                return_value=("1" * 64) + "  /data/model\n",
            ) as adb:
                result = first._remote_hashes({"model": "/data/model"})
            self.assertEqual(result["model"], "sha256:" + "1" * 64)
            adb.assert_called_once()

            second = object.__new__(DirectPhoneFfnSession)
            second.configuration = first.configuration
            second._verified_remote_hash_by_path = {}
            with mock.patch.object(
                second,
                "_remote_file_identities",
                return_value=identities,
            ), mock.patch.object(
                second,
                "_adb",
                side_effect=AssertionError("unexpected remote SHA-256"),
            ):
                cached = second._remote_hashes({
                    "model": "/data/model"
                })
            self.assertEqual(cached, result)

    def test_remote_identity_command_executes_boot_and_stat_probes(self) -> None:
        session = object.__new__(DirectPhoneFfnSession)
        with mock.patch.object(
            session,
            "_adb",
            return_value=(
                "BOOT boot-a\n"
                "FILE0 65102:313288:23832065056:1786054983:1786055501\n"
            ),
        ) as adb:
            identities = session._remote_file_identities(
                ("/data/model",), root=False, timeout_s=30
            )
        command = adb.call_args.args[0]
        self.assertIn(
            "printf 'BOOT '; cat /proc/sys/kernel/random/boot_id",
            command,
        )
        self.assertIn(
            "; printf 'FILE0 '; stat -c '%d:%i:%s:%Y:%Z' /data/model",
            command,
        )
        self.assertEqual(identities["/data/model"].boot_id, "boot-a")
        self.assertEqual(
            identities["/data/model"].stat_identity,
            "65102:313288:23832065056:1786054983:1786055501",
        )

    def test_small_remote_software_is_rehashed_each_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "remote-hashes.json"
            identities = self.identity("boot-a", "1:2:1000:4:5")
            session = object.__new__(DirectPhoneFfnSession)
            session.configuration = SimpleNamespace(
                remote_hash_cache_path=cache_path,
                serial="PHONE1",
            )
            session._verified_remote_hash_by_path = {}
            with mock.patch.object(
                session,
                "_remote_file_identities",
                return_value=identities,
            ), mock.patch.object(
                session,
                "_adb",
                return_value=("2" * 64) + "  /data/model\n",
            ) as adb:
                result = session._remote_hashes({
                    "worker": "/data/model"
                })
            self.assertEqual(result["worker"], "sha256:" + "2" * 64)
            adb.assert_called_once()
            self.assertFalse(cache_path.exists())


if __name__ == "__main__":
    unittest.main()
