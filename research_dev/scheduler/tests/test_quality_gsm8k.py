"""GSM8K item loading, per-model chat prompts and the frozen final-answer extraction rule."""
import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from research_dev.scheduler.campaigns.burstgpt.quality import gsm8k
from research_dev.scheduler.campaigns.burstgpt.quality.gsm8k import (
    QualityDataError, extract_answer, is_correct, load_items, parse_reference, render_prompt,
)
from research_dev.scheduler.campaigns.burstgpt.quality.protocol import GSM8K_SHA256


class QualityGsm8kTests(unittest.TestCase):
    def test_reference_is_the_number_after_the_last_marker(self):
        self.assertEqual(parse_reference("so 16 - 3 = <<16-3=13>>13\n#### 1,234"), Decimal(1234))
        self.assertEqual(parse_reference("#### -5"), Decimal(-5))
        with self.assertRaises(QualityDataError):
            parse_reference("no marker 12")

    def test_items_are_verified_against_the_pinned_copy(self):
        rows = [{"question": " Q one? ", "answer": "x\n#### 18"}, {"question": "Q two", "answer": "#### 3"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            items = load_items(path, "test", verify=False)
            self.assertEqual([item.item_id for item in items], ["gsm8k-test:0000", "gsm8k-test:0001"])
            self.assertEqual(items[0].question, "Q one?")
            self.assertEqual(items[0].reference, Decimal(18))
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), GSM8K_SHA256["test"])
            with self.assertRaises(QualityDataError):
                load_items(path, "test")

    def test_prompts_are_the_official_non_thinking_renderings(self):
        content = gsm8k.USER_TEMPLATE.format(question="What is 2+2?")
        self.assertTrue(content.endswith("Problem: What is 2+2?"))
        self.assertIn('"Final answer: <number>"', content)
        self.assertEqual(render_prompt("qwen", " What is 2+2? "),
                         "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(render_prompt("gemma", "What is 2+2?"),
                         "<|turn>user\n" + content + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
        llama = render_prompt("llama", "What is 2+2?")
        self.assertTrue(llama.startswith("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date"))
        self.assertTrue(llama.endswith(content + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"))
        self.assertFalse(any(text in llama for text in ("<|begin_of_text|>", "<bos>")))
        with self.assertRaises(QualityDataError):
            render_prompt("mistral", "x")

    def test_extraction_takes_the_earliest_conclusion(self):
        cases = [
            ("Step 1 ...\n**Final answer:** 18\nuser\nFinal answer: 20", "18", "final-answer"),
            ("Final Answer: $1,080.00 per week", "1080", "final-answer"),
            ("so \\boxed{42}. Later: Final answer: 7", "42", "boxed"),
            ("Final answer: \\boxed{42}", "42", "final-answer"),
            ("Final answer:\n- 18", "18", "final-answer"),
            ("Final answer is -3.", "-3", "final-answer"),
            ("The final answer = 12 apples", "12", "final-answer"),
            ("Thus the answer is 5.", "5", "answer-is"),
            ("To get the final answer, we add 3 and 4. The answer is 7", "7", "answer-is"),
            ("final answer: 3.50", "3.5", "final-answer"),
        ]
        for text, value, rule in cases:
            with self.subTest(text=text):
                extraction = extract_answer(text)
                self.assertEqual(extraction.to_json()["value"], value)
                self.assertEqual(extraction.rule, rule)
                self.assertIn(value.split(".")[0].lstrip("-"),
                              text[extraction.start:extraction.end].replace(",", ""))

    def test_unscorable_outputs_are_incorrect(self):
        for text in ("", "The numbers are 3 and 4 and 12.", "Final answer: none", "answer is unclear"):
            with self.subTest(text=text):
                extraction = extract_answer(text)
                self.assertFalse(extraction.found)
                self.assertEqual(extraction.rule, "none")
                self.assertFalse(is_correct(extraction, Decimal(12)))

    def test_correctness_is_numeric_equality(self):
        self.assertTrue(is_correct(extract_answer("Final answer: 18.00"), Decimal(18)))
        self.assertTrue(is_correct(extract_answer("Final answer: 1,000"), Decimal(1000)))
        self.assertFalse(is_correct(extract_answer("Final answer: 18.5"), Decimal(18)))
        self.assertFalse(is_correct(extract_answer("Final answer: -18"), Decimal(18)))


if __name__ == "__main__":
    unittest.main()
