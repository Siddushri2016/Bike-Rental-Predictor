
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import math
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler


def test_weekday_imputer_transformer(sample_input_data):
    # Given
    transformer = WeekdayImputer(
        variables=[config.model_config.weekday_var, config.model_config.dtedays_var] # [weekday, dteday]
    )
    assert np.isnan(sample_input_data[0].loc[7046,'weekday'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[7046,'weekday'] == 'Wed'
    
def test_weathersit_imputer_transformer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        variables=config.model_config.weathersit_var # [weathersit]
    )
    assert np.isnan(sample_input_data[0].loc[12230,'weathersit'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[12230,'weathersit'] == 'Clear'
    
def test_mapper_transformer(sample_input_data):
    # Given
    transformer = Mapper(
        config.model_config.mnth_var, config.model_config.mnth_mappings # [mnth]
    )
    assert sample_input_data[0].loc[12830,'mnth'] == 'April'

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[12830,'mnth'] == 4
    
def test_outlier_transformer(sample_input_data):
    # Given
    transformer = OutlierHandler(
        variables=[config.model_config.windspeed_var] # [mnth]
    )
    assert round(sample_input_data[0].loc[11376,'windspeed'], 4) == 43.0006

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert int(math.ceil(subject.loc[11376,'windspeed'])) == 17