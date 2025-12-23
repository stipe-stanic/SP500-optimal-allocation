# %%
import os

import joblib
import polars as pl
import torch

from kaggle_evaluation import default_inference_server
from settings import DirectorySettings
from train import Config, CustomModel, InferenceEnv, Preprocessor, load_best_model, load_position_params

config = Config()
settings = DirectorySettings(exp_name=config.exp_name)

feature_cols = joblib.load(settings.ARTIFACT_EXP_DIR / 'feature_cols.pkl')
target_cols = joblib.load(settings.ARTIFACT_EXP_DIR / 'target_cols.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
raw_train_df = pl.read_csv(settings.COMP_DATASET_DIR / 'train.csv')
train_df = raw_train_df.with_columns(
    [
        pl.col('forward_returns').shift(1).alias('lagged_forward_returns'),
        pl.col('risk_free_rate').shift(1).alias('lagged_risk_free_rate'),
        pl.col('market_forward_excess_returns').shift(1).alias('lagged_market_forward_excess_returns'),

    ]
).drop(target_cols)

preprocessor = Preprocessor(config=config)
preprocessor.load(settings.ARTIFACT_EXP_DIR)
position_params = load_position_params(settings.ARTIFACT_EXP_DIR, config)

best_models = [
    load_best_model(
        model=CustomModel(
            config=config,
            num_input_channels=len(feature_cols),
            num_output_channels=2,
        ),
        model_path=settings.ARTIFACT_EXP_DIR / f"seed_{seed}" / "best_model.pth"
        if not config.full_train
        else settings.ARTIFACT_EXP_DIR / "full_training" / f"seed_{seed}" / "best_model.pth",
        device=device,
    )
    for seed in config.seeds
]

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_env = InferenceEnv(
        config=config,
        preprocessor=preprocessor,
        models=best_models,
        device=device,
        train_df=train_df,
        feature_cols=feature_cols,
        position_params=position_params,
        latest_date_id=8809,
    )

    def predict(test: pl.DataFrame) -> float:
        return inference_env.predict(test)

    inference_server = default_inference_server.DefaultInferenceServer(predict)
    inference_server.serve()
else:
    inference_env = InferenceEnv(
        config=config,
        preprocessor=preprocessor,
        models=best_models,
        device=device,
        train_df=train_df,
        feature_cols=feature_cols,
        position_params=position_params,
        latest_date_id=8979,
    )

    def predict(test: pl.DataFrame) -> float:
        return inference_env.predict(test)

    inference_server = default_inference_server.DefaultInferenceServer(predict)
    inference_server.run_local_gateway((settings.COMP_DATASET_DIR,))

print("Inference completed.")
