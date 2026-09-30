"""Offline checks for model inventory deltas and failed-provider handling."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import model_watch


class ModelWatchTests(unittest.TestCase):
    def test_anthropic_inventory_follows_pages(self) -> None:
        pages = [
            {"data": [{"id": "claude-a", "max_input_tokens": 200000}],
             "has_more": True, "last_id": "claude-a"},
            {"data": [{"id": "claude-b", "max_input_tokens": 1000000}],
             "has_more": False},
        ]
        with patch.object(model_watch, "fetch_json", side_effect=pages) as fetch:
            found = model_watch.inventory("anthropic", "secret")
        self.assertEqual(set(found), {"claude-a", "claude-b"})
        self.assertIn("after_id=claude-a", fetch.call_args_list[1].args[0])

    def test_baseline_then_new_model_and_metadata_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "watch.json"
            first = {
                "gpt-5.4-mini": {"created": 1},
                "gpt-4.1-nano": {"created": 1},
                "gpt-6-luna": {"created": 1},
                "gpt-6-sol": {"created": 1},
            }
            second = dict(first, **{"gpt-6-luna": {"created": 2},
                                    "gpt-6-new": {"created": 2}})
            with patch.dict(os.environ, {"OPENAI_API_KEY": "secret",
                                      "ANTHROPIC_API_KEY": "",
                                      "DEEPSEEK_API_KEY": "",
                                      "SYS_ENV_FILE": "/does/not/exist"}), \
                 patch.object(model_watch, "inventory", side_effect=[first, second, second]):
                baseline = model_watch.run(state)
                changed = model_watch.run(state)
                steady = model_watch.run(state)
            self.assertEqual(baseline["providers"]["openai"]["status"], "baseline")
            self.assertFalse(baseline["needs_review"])
            openai = changed["providers"]["openai"]
            self.assertEqual(openai["status"], "checked")
            self.assertEqual([item["id"] for item in openai["new_models"]],
                             ["gpt-6-new"])
            self.assertTrue(openai["new_models"][0]["name_prefix_match"])
            self.assertEqual([item["id"] for item in openai["metadata_changes"]],
                             ["gpt-6-luna"])
            self.assertTrue(changed["needs_review"])
            self.assertTrue(changed["new_signal"])
            self.assertFalse(steady["new_signal"])
            self.assertTrue(changed["coverage_incomplete"])
            self.assertNotIn("secret", state.read_text())

    def test_failed_provider_keeps_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "watch.json"
            visible = {name: {} for name in model_watch.agent.PROVIDER_MODELS["deepseek"]}
            with patch.dict(os.environ, {"OPENAI_API_KEY": "",
                                      "ANTHROPIC_API_KEY": "",
                                      "DEEPSEEK_API_KEY": "secret",
                                      "SYS_ENV_FILE": "/does/not/exist"}), \
                 patch.object(model_watch, "inventory", side_effect=[
                     visible, TimeoutError(), TimeoutError(),
                     dict(visible, **{"deepseek-new": {}})]):
                model_watch.run(state)
                failed = model_watch.run(state)
                failed_again = model_watch.run(state)
                resumed = model_watch.run(state)
            self.assertEqual(failed["providers"]["deepseek"]["status"], "error")
            self.assertEqual(failed["providers"]["deepseek"]["consecutive_failures"], 1)
            self.assertFalse(failed["providers"]["deepseek"]["persistent_failure"])
            self.assertEqual(failed_again["providers"]["deepseek"]["consecutive_failures"], 2)
            self.assertTrue(failed_again["providers"]["deepseek"]["persistent_failure"])
            self.assertTrue(failed_again["coverage_needs_repair"])
            self.assertTrue(failed["coverage_incomplete"])
            self.assertEqual(resumed["providers"]["deepseek"]["new_models"][0]["id"],
                             "deepseek-new")
            self.assertEqual(resumed["providers"]["deepseek"]["consecutive_failures"], 0)
            self.assertFalse(resumed["coverage_needs_repair"])


if __name__ == "__main__":
    unittest.main()
