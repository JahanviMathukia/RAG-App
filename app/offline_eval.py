import os
import json
from typing import List, Dict, Any
from .rag_core import init_app_state, answer_question

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TESTS_PATH = os.path.join(BASE_DIR, "tests", "tests.json")

def load_tests():
    if not os.path.exists(TESTS_PATH):
        raise RuntimeError(f"tests.json not found at {TESTS_PATH}")
    with open(TESTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_offline_eval():
    tests = load_tests()
    state = init_app_state()

    total = len(tests)
    passed = 0

    print(f"Running {total} offline tests...\n")

    for t in tests:
        q = t["question"]
        expected = t["expect_contains"]
        test_id = t["id"]

        print(f"--- Test {test_id} ---")
        print(f"Q: {q}")

        answer = answer_question(state, q)
        print(f"A: {answer}\n")

        if expected.lower() in answer.lower():
            print("✅ PASS\n")
            passed += 1
        else:
            print(f"❌ FAIL — expected substring: {expected}\n")

    pass_rate = 100 * passed / total
    print(f"Final score: {passed}/{total} passed ({pass_rate:.1f}%).")

if __name__ == "__main__":
    run_offline_eval()
