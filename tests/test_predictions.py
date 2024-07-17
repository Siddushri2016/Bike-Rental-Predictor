"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from bikeshare_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 3476

    # When
    result = make_prediction(input_data=sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    #assert isinstance(predictions[0], np.int64)
    #assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data[1]
    r2score =  r2_score(y_true, _predictions)
    print("R2 score:", r2score)
    assert r2score > 0.9
    mse = mean_squared_error(y_true, _predictions)
    mae = mse**0.5
    print("Mean squared error:", mse)
    assert mse < 2000
    assert mae < 0.20*y_true.mean()

