#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


PATH = pathlib.Path(__file__).with_name("build_corpus.py")
SPEC = importlib.util.spec_from_file_location("s33_build_corpus", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class BuildCorpusTests(unittest.TestCase):
    def test_candidate_selection(self):
        self.assertEqual(
            MOD.candidate_rows(["", " = title = ", " useful text ", "next"]),
            [(2, "useful text"), (3, "next")],
        )

    def test_token_parser(self):
        self.assertEqual(MOD.parse_token_ids(b"[2, 3, 5]\n"), [2, 3, 5])

    def test_token_parser_rejects_bool(self):
        with self.assertRaises(MOD.CorpusError):
            MOD.parse_token_ids(b"[2, true]")

    def test_token_parser_rejects_code(self):
        with self.assertRaises(MOD.CorpusError):
            MOD.parse_token_ids(b"__import__('os').getcwd()")

    def test_canonical_is_ascii_jsonl(self):
        self.assertEqual(MOD.canonical({"b": 2, "a": 1}), b'{"a":1,"b":2}\n')


if __name__ == "__main__":
    unittest.main()
