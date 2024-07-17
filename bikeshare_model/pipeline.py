import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import drop_columns
from bikeshare_model.processing.features import CustomOneHotEncoder

bikeshare_pipe = Pipeline(
    steps=[
        ('weekday_imputation', WeekdayImputer(variables=config.model_config.weekdayimputer_fields)),
        ('weathersit_imputation', WeathersitImputer(variables=config.model_config.weathersit_var)),
        ('outlier_transform', OutlierHandler(variables=config.model_config.num_features)),
        ('drop_columns', FunctionTransformer(drop_columns, kw_args={config.model_config.col_drop_var: config.model_config.unused_fields})),
        ('preprocessor', ColumnTransformer(
            transformers=[
                #('weekday_onehot', OneHotEncoder(handle_unknown="ignore"), [config.model_config.weekday_var]),
                ('trans_pipeline',
                    Pipeline(steps=[
                        ('map_yr', Mapper(config.model_config.yr_var, config.model_config.yr_mappings)),
                        ('map_season', Mapper(config.model_config.season_var, config.model_config.season_mappings)),
                        ('map_holiday', Mapper(config.model_config.holiday_var, config.model_config.holiday_mappings)),
                        ('map_workingday', Mapper(config.model_config.workingday_var, config.model_config.workingday_mappings)),
                        ('map_weathersit', Mapper(config.model_config.weathersit_var, config.model_config.weathersit_mappings)),
                        ('map_mnth', Mapper(config.model_config.mnth_var, config.model_config.mnth_mappings)),
                        ('map_hr', Mapper(config.model_config.hr_var, config.model_config.hr_mappings)),
                        ('map_weekday', Mapper(config.model_config.weekday_var, config.model_config.weekday_mappings)),
                        ('cat_scaler', StandardScaler())
                        #('cat_onehot', CustomOneHotEncoder(variables=config.model_config.cat_features)),
                    ]), config.model_config.cat_features),
                ('num_scaler', StandardScaler(), config.model_config.num_features)  
            ], remainder='passthrough'
        )),
        ('model_rf', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
    ]
)