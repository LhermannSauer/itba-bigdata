# Unit tests for the model training module.
"""Unit tests for the model training module."""

import unittest

try:
    import src.data_ingestion as di
except ImportError:
    # Try direct import for Databricks workspace
    import data_ingestion as di


class TestDataIngestion(unittest.TestCase):
    """Unit tests for data_ingestion.py."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        self.assertTrue(hasattr(di, "SOURCE_PATH"))
        self.assertTrue(hasattr(di, "BRONZE_PATH"))
        self.assertTrue(hasattr(di, "SILVER_PATH"))
        self.assertTrue(hasattr(di, "GOLD_OUT_PATH"))

        # Check they are strings
        self.assertIsInstance(di.SOURCE_PATH, str)
        self.assertIsInstance(di.BRONZE_PATH, str)

    def test_required_fields(self):
        """Test that REQUIRED_FIELDS contains correct fields."""
        expected_fields = [
            "review_id",
            "product_id",
            "customer_id",
            "star_rating",
            "review_date",
            "review_body",
        ]

        self.assertEqual(di.REQUIRED_FIELDS, expected_fields)
        self.assertEqual(len(di.REQUIRED_FIELDS), 6)

    def test_sentiment_mapping(self):
        """Test sentiment mapping is correct."""
        self.assertEqual(di.SENTIMENT_MAP["negative"], [1, 2])
        self.assertEqual(di.SENTIMENT_MAP["neutral"], [3])
        self.assertEqual(di.SENTIMENT_MAP["positive"], [4, 5])

        # Test all keys exist
        self.assertIn("negative", di.SENTIMENT_MAP)
        self.assertIn("neutral", di.SENTIMENT_MAP)
        self.assertIn("positive", di.SENTIMENT_MAP)

    def test_functions_exist(self):
        """Test that all main functions exist."""
        # Check main functions exist
        self.assertTrue(hasattr(di, "bronze_ingestion"))
        self.assertTrue(hasattr(di, "bronze_validation"))
        self.assertTrue(hasattr(di, "silver_ingestion"))
        self.assertTrue(hasattr(di, "gold_ingestion"))
        self.assertTrue(hasattr(di, "main"))

        # Check they are callable
        self.assertTrue(callable(di.bronze_ingestion))
        self.assertTrue(callable(di.main))

        # Check helper functions exist
        self.assertTrue(callable(di.basic_text_cleaning))
        self.assertTrue(callable(di.map_ratings_to_labels))


# Run tests if executed directly
if __name__ == "__main__":
    unittest.main()
