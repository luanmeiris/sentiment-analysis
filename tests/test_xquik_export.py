import unittest

import pandas as pd

from xquik_export import load_xquik_texts


class XquikExportTests(unittest.TestCase):
    def test_loads_text_from_xquik_tweet_column(self):
        frame = pd.DataFrame({"tweet": [" Great match ", "", None], "id": [1, 2, 3]})

        result = load_xquik_texts(frame)

        self.assertEqual(result, ["Great match"])

    def test_returns_empty_list_for_unknown_schema(self):
        frame = pd.DataFrame({"score": [0.8]})

        result = load_xquik_texts(frame)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
