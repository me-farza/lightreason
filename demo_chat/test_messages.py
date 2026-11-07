"""CPU-only contracts for immutable, request-local real chat roles."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import unittest

from demo_chat.messages import (PromptText, chat_messages, DEFAULT_DESIGN_PROMPT,
                                MAX_SYSTEM_PROMPT_CHARACTERS, MAX_USER_PROMPT_CHARACTERS)


class MessageTests(unittest.TestCase):
    def test_plain_user_prompt_is_unchanged(self):
        prompt = "  Make a page.\n"
        self.assertEqual(chat_messages(prompt), [{"role": "user", "content": prompt}])

    def test_explicit_system_instruction_is_a_real_distinct_role(self):
        prompt = PromptText("Build a task board.", "  Use responsive grids.\n")
        self.assertEqual(chat_messages(prompt), [
            {"role": "system", "content": "  Use responsive grids.\n"},
            {"role": "user", "content": "Build a task board."}])
        self.assertNotIn("responsive", str(prompt))

    def test_empty_system_instructions_leave_original_user_only_template(self):
        for system in (None, "", " \n\t"):
            self.assertEqual(chat_messages(PromptText("Build a page.", system)),
                             [{"role": "user", "content": "Build a page."}])

    def test_instructions_cannot_change_after_submission(self):
        prompt = PromptText("Build a page.", "First style")
        for name in ("system_prompt", "_system_prompt"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(prompt, name, "Second style")
            with self.subTest(delete=name), self.assertRaises(AttributeError):
                delattr(prompt, name)
        self.assertEqual(prompt.system_prompt, "First style")

    def test_message_lists_are_fresh_not_shared_state(self):
        prompt = PromptText("Build a page.", "First style")
        messages = chat_messages(prompt)
        messages[0]["content"] = "Changed outside"
        messages.append({"role": "assistant", "content": "Old answer"})
        self.assertEqual(chat_messages(prompt), [{"role": "system", "content": "First style"},
                                                 {"role": "user", "content": "Build a page."}])

    def test_concurrent_request_messages_do_not_leak_instructions(self):
        prompts = [PromptText("Build page " + str(i), "Style " + str(i)) for i in range(40)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(chat_messages, prompts * 3))
        for prompt, messages in zip(prompts * 3, results):
            self.assertEqual(messages[0], {"role": "system", "content": prompt.system_prompt})
            self.assertEqual(messages[1], {"role": "user", "content": str(prompt)})

    def test_size_and_type_errors_are_bad_request_values(self):
        for user, system in ((None, None), (" ", None), ("x", 3), ("x", False),
                             ("x", "s" * (MAX_SYSTEM_PROMPT_CHARACTERS + 1)),
                             ("x" * (MAX_USER_PROMPT_CHARACTERS + 1), None)):
            with self.subTest(user=str(user)[:20], system=str(system)[:20]), self.assertRaises(ValueError):
                PromptText(user, system)
        valid = PromptText("x" * MAX_USER_PROMPT_CHARACTERS, "s" * MAX_SYSTEM_PROMPT_CHARACTERS)
        self.assertEqual(len(chat_messages(valid)[0]["content"]), MAX_SYSTEM_PROMPT_CHARACTERS)

    def test_json_logs_need_explicit_system_field_and_copy_keeps_context(self):
        prompt = PromptText("Build a page.", "Use responsive grids.")
        self.assertEqual(json.loads(json.dumps(prompt)), "Build a page.")
        self.assertEqual(copy.copy(prompt).system_prompt, prompt.system_prompt)
        self.assertEqual(copy.deepcopy(prompt).system_prompt, prompt.system_prompt)

    def test_capacity_count_must_include_system_messages(self):
        def encoded_length(messages):
            return sum(len(m["content"]) + 3 for m in messages)
        user = "Build a page."
        bound = PromptText(user, "Use responsive grids.")
        self.assertGreater(encoded_length(chat_messages(bound)), encoded_length(chat_messages(user)))

    def test_default_design_instructions_are_shared_valid_system_text(self):
        messages = chat_messages(PromptText("Build a shop.", DEFAULT_DESIGN_PROMPT))
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("responsive", messages[0]["content"])
        self.assertIn("Make visible controls work", messages[0]["content"])
        self.assertLessEqual(len(DEFAULT_DESIGN_PROMPT), MAX_SYSTEM_PROMPT_CHARACTERS)


if __name__ == "__main__":
    unittest.main()
