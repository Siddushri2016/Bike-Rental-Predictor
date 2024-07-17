import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikeshare_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    validated_data=validated_data.reindex(columns=config.model_config.features)
    
    print("validated_data\n", validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    print(results)
    if not errors:

        predictions = bikeshare_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":

    data_in={"dteday": ["2012-09-08"], "season": ["fall"], "hr": ["4pm"], "holiday": ["No"], "weekday": ["Sat"], "workingday": ["No"],
            "weathersit": ["Light Rain"], "temp": [18.32], "atemp": [19], "hum": [88], "windspeed": [13], "casual": [19], "registered": [95]}
    
    make_prediction(input_data=data_in)
