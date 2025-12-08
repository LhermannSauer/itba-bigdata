import pytest
from unittest.mock import MagicMock
import pyspark.sql.functions as F

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import module
import src.data_ingestion as di


def test_basic_text_cleaning():
    """Test text cleaning logic using a mocked Spark DataFrame."""
    spark = MagicMock()
    df = MagicMock()

    df.withColumn.return_value = df  # allow chaining

    result = di.basic_text_cleaning(df, text_col="review_body", out_col="clean_text")

    # Ensure the function returns a DataFrame-like object
    assert result is not None
    df.withColumn.assert_called()


def test_map_ratings_to_labels():
    """Ensure mapping logic does not crash and applies the mapping expression."""
    df = MagicMock()
    df.withColumn.return_value = df

    result = di.map_ratings_to_labels(df)

    assert result is not None
    df.withColumn.assert_called()


def test_required_constants():
    """Ensure required constants exist."""
    assert isinstance(di.REQUIRED_FIELDS, list)
    assert "review_id" in di.REQUIRED_FIELDS