from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from generate_science_noise import boxed_answer


class Answers(unittest.TestCase):
    def test_last_complete_nested_box_is_the_target(self):
        self.assertEqual(boxed_answer(r'Earlier \boxed{0}, final \boxed{\frac{v_0}{2}}.'),r'\frac{v_0}{2}')

    def test_empty_unfinished_and_unboxed_outputs_are_rejected(self):
        for text in ['42',r'\boxed{}',r'\boxed{\frac{1}{2}']:
            self.assertIsNone(boxed_answer(text))


if __name__=='__main__':unittest.main()
