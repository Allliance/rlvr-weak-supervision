from pathlib import Path
import sys
import unittest
import importlib.util

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from generate_science_noise import boxed_answer, scores, verified_target


class Answers(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('math_verify'),'requires the pinned training verifier')
    def test_compound_target_with_the_correct_answer_is_not_cached(self):
        target=r'P = 525, \quad x = 100'
        self.assertEqual(scores(target,'100'),[0,1,0])
        self.assertFalse(verified_target(target,'100'))
        self.assertTrue(verified_target('7','6'))
        self.assertFalse(verified_target('6','6'))

    def test_last_complete_nested_box_is_the_target(self):
        self.assertEqual(boxed_answer(r'Earlier \boxed{0}, final \boxed{\frac{v_0}{2}}.'),r'\frac{v_0}{2}')

    def test_empty_unfinished_and_unboxed_outputs_are_rejected(self):
        for text in ['42',r'\boxed{}',r'\boxed{\frac{1}{2}']:
            self.assertIsNone(boxed_answer(text))


if __name__=='__main__':unittest.main()
