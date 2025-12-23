# %%

import abc
import json
import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from numpy.typing import NDArray
from pydantic import BaseModel, model_validator
from schedulefree import RAdamScheduleFree
from sklearn.preprocessing import RobustScaler
from timm.utils import ModelEmaV3
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from customs.metrics import score_fn_v2
from kaggle_evaluation import default_inference_server
from models.itransformer.model.itransformer import ITransformer
from settings import DirectorySettings
from src.utils.plot import plot_training_history, plot_prediction_distribution, plot_prediction_variance, \
    plot_position_distribution, plot_prediction_timeseries, plot_correlation_heatmap
from utils.seed import seed_everything


# %%
class Config(BaseModel):
    exp_name: str = 'baseline'

    sliding_window_view_size: int = 300
    feature_encoders: list[dict] = [
        {
            "name": "RawEncoder",
            "params": {
                "log_transform": False,
                "exclude_features" : [
                    "date_id",
                    "lagged_forward_returns",
                    "lagged_risk_free_rate",
                    "lagged_market_forward_excess_returns",
                ],
            },
        },
        # {
        #     "name": "ShiftEncoder",
        #     "params": {
        #         "target_cols": [
        #             "lagged_forward_returns",
        #             "lagged_risk_free_rate",
        #             "lagged_market_forward_excess_returns",
        #         ],
        #         "shifts": [0],
        #     },
        # },
    ]

    use_fp16: bool = False
    max_grad_norm: float = 1.0
    batch_size: int = 256
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    num_epochs: int = 64
    use_ema: bool = False
    ema_use_warmup: bool = True
    ema_decay: float = 0.9999

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 0

    num_recent_steps: int = 0
    d_model: int = 200
    no_embd: bool = False
    use_norm: bool = True

    default_risk_aversion: float = 10.0
    default_scale_factor: float = 1.0
    default_base_position: float = 1.0

    feature_prefix: str = 'f_'
    debug: bool = False

    seeds: list[int] = [1, 2, 3]

    only_test: bool = False  # skip training, only test
    full_train: bool = False # no validation, train on all data

    # Online training parameters
    online_training_frequency: int = 0  # 0 means no online training
    online_training_epochs: int = 1
    online_training_lr: float = 1e-5
    online_training_batch_size: int = 32

    # --- fixed parameters ---
    # train last date_id = 9047
    test_window_size: int = 180
    test_start_date_id: int = 9047 - test_window_size

    @model_validator(mode='after')
    def after_init(self) -> "Config":
        if self.debug:
            self.num_epochs = 4
            self.seeds = [0, 1]

        if self.only_test:
            self.full_train = False

        return self


class BaseEncoder:
    _fitted: bool
    feature_prefix: str

    @abc.abstractmethod
    def fit(self, df: pl.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)


class RawEncoder(BaseEncoder):
    def __init__(
            self,
            feature_prefix: str = 'f_',
            exclude_features: list[str] = ['date_id'],
            log_transform: bool = False,
            suffix: str = '',
    ):
        self.feature_prefix = feature_prefix
        self.exclude_features = exclude_features
        self.log_transform = log_transform
        self._fitted = False
        self.suffix = suffix

    def fit(self, df: pl.DataFrame) -> None:
        self._fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        expr = (
            pl.exclude(self.exclude_features)
            if not self.log_transform
            else pl.exclude(self.exclude_features).log().name.suffix('__log')
        )
        feature_df = df.select(expr.name.prefix(self.feature_prefix))
        feature_cols = [x for x in feature_df.columns if x.startswith(self.feature_prefix)]
        feature_df = feature_df.select(pl.col(feature_cols).name.suffix(self.suffix))
        return feature_df


class ShiftEncoder(BaseEncoder):
    def __init__(
        self,
        feature_prefix: str = "f_",
        time_col: str = "date_id",
        target_cols: list[str] = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"],  # noqa
        shifts: list[int] = [1, 2, 3],  # noqa
        suffix: str = "",
    ):
        self.feature_prefix = feature_prefix
        self.time_col = time_col
        self.target_cols = target_cols
        self.shifts = shifts
        self.suffix = suffix
        self._fitted = False

    def fit(self, df: pl.DataFrame) -> None:
        self._fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        feature_exprs = []

        for col in self.target_cols:
            if col not in df.columns:
                # If column doesn't exist, skip
                continue

            for shift in self.shifts:
                feature_name = f"{self.feature_prefix}{col}__shift_{shift}{self.suffix}"
                feature_exprs.append(pl.col(col).shift(shift).alias(feature_name))

        result_df = df.select(feature_exprs)
        return result_df


class Preprocessor:
    def __init__(self, config: Config, **kwargs):
        self.config = config

        # only transform encoders
        self.encoders = []
        for encoder_setting in self.config.feature_encoders:
            encoder_name = encoder_setting['name']
            encoder_params = encoder_setting.get('params', {})
            if encoder_name == 'RawEncoder':
                encoder = RawEncoder(
                    feature_prefix=self.config.feature_prefix,
                    **encoder_params,
                )
            elif encoder_name == 'ShiftEncoder':
                encoder = ShiftEncoder(
                    feature_prefix=self.config.feature_prefix,
                    **encoder_params,
                )
            else:
                raise ValueError(f'Unknown encoder: {encoder_name}')
            self.encoders.append(encoder)

        self.scaler = RobustScaler()

    def fit(self, raw_df: pl.DataFrame) -> None:
        output_df = raw_df.select(pl.col('date_id'))
        for encoder in self.encoders:
            feature_df = encoder.fit_transform(raw_df)
            output_df = pl.concat([output_df, feature_df], how='horizontal')

        feature_cols = [x for x in output_df.columns if x.startswith(self.config.feature_prefix)]
        self.scaler.fit(
            output_df.filter(pl.col('date_id') < self.config.test_start_date_id).select(feature_cols).to_numpy()
        )

    def transform(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        base_df = raw_df.select(['date_id'])
        output_df = pl.DataFrame()
        for encoder in self.encoders:
            feature_df = encoder.transform(raw_df)
            output_df = pl.concat([output_df, feature_df], how='horizontal')

        feature_cols = [x for x in output_df.columns if x.startswith(self.config.feature_prefix)]
        output_df = pl.DataFrame(self.scaler.transform(output_df.select(feature_cols).to_numpy()), schema=feature_cols)
        output_df = pl.concat([base_df, output_df], how='horizontal')
        return output_df

    def fit_transform(self, raw_df: pl.DataFrame) -> pl.DataFrame:
        self.fit(raw_df)
        return self.transform(raw_df)

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)

        # save scaler
        self.scaler = joblib.dump(self.scaler, dirpath / 'scaler.pkl')

    def load(self, dirpath: Path) -> None:
        # load scaler
        self.scaler = joblib.load(dirpath / 'scaler.pkl')


def create_sliding_window_x(
        X: NDArray[np.floating], # (N, F)
        seq_length: int,
        y: NDArray[np.floating] | None = None  # (N, L)
) -> tuple[NDArray[np.floating], NDArray[np.floating]] | NDArray[np.floating]:
    """X: sliding window, y: scaler target"""
    X_seq = np.lib.stride_tricks.sliding_window_view(X, seq_length, axis=0)
    # (n_samples, F, seq_length) -> (n_samples, seq_length, F)
    X_seq = X_seq.transpose(0, 2, 1)

    if y is None:
        return X_seq # ((N-seq_length), seq_length, F)

    y_seq = y[seq_length - 1 :]
    return X_seq, y_seq  # ((N-seq_length), seq_length, F), ((N-seq_length), L)


class TrainDataset(Dataset):
    def __init__(self, X_seq: NDArray[np.floating], y: NDArray[np.floating]):
        self.X_seq = X_seq  # (n_samples, seq_length, num_input_channels)
        self.y = y  # (n_samples, num_target_channels)

        # target_values
        #   - index 0: forward returns
        #   - index 1: risk free rate
        #   - index 2: market_forward_excess_returns

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        # observed mask
        past_values = torch.tensor(self.X_seq[idx], dtype=torch.float32)  # (sequence_length, num_input_channels)
        past_observed_mask = ~torch.isnan(past_values)  # (sequence_length, num_input_channels)
        past_values = torch.nan_to_num(past_values, nan=0)  # (sequence_length, num_input_channels)

        target_values = torch.tensor(self.y[idx], dtype=torch.float32)  # (num_target_channels,)
        target_mask = ~torch.isnan(target_values)  # (num_target_channels,)
        target_values = torch.nan_to_num(target_values, nan=0)  # (num_target_channels,)
        return {
            "past_values": past_values,  # ( sequence_length, num_input_channels)
            "past_observed_mask": past_observed_mask,  # ( sequence_length, num_input_channels)
            "target_values": target_values,  # (num_target_channels,)
            "target_mask": target_mask,  # (num_target_channels,)
        }


class TestDataset(Dataset):
    def __init__(self, X_seq: NDArray[np.floating]):
        self.X_seq = X_seq # (n_samples, seq_length, num_input_channels)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        # observed_mask
        past_values = torch.tensor(self.X_seq[idx], dtype=torch.float32)  # (sequence_length, num_input_channels)
        past_observed_mask = ~torch.isnan(past_values)  # (sequence_length, num_input_channels)
        past_values = torch.nan_to_num(past_values, nan=0)  # (sequence_length, num_input_channels)

        return {
            "past_values": past_values,  # ( sequence_length, num_input_channels)
            "past_observed_mask": past_observed_mask,  # ( sequence_length, num_input_channels)
        }


class CustomModel(nn.Module):
    def __init__(
            self,
            config: Config,
            num_input_channels: int = 94,
            num_output_channels: int = 2,
    ):
        super().__init__()

        self.config = config

        self.model = ITransformer(
            seq_len=self.config.sliding_window_view_size,
            d_model=getattr(self.config, 'd_model', 128),
            n_heads=getattr(self.config, 'num_attention_heads', 4),
            e_layers=getattr(self.config, 'num_hidden_layers', 3),
            d_ff=getattr(self.config, 'ffn_dim', 512),
            dropout=getattr(self.config, 'dropout', 0.1),
            activation='relu',
            no_embd=getattr(self.config, 'no_embd', False),
            use_norm=getattr(self.config, 'use_norm', True),
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.num_recent_steps = getattr(self.config, 'num_recent_steps', 3)

        # ITransformer [B, N, E]
        self.transformer_dim = num_input_channels * self.model.d_model

        if self.num_recent_steps > 0:
            recent_input_dim = num_input_channels * self.num_recent_steps
            recent_feature_dim = getattr(self.config, 'recent_feature_dim', 512)

            self.recent_feature_extractor = nn.Sequential(
                nn.Linear(recent_input_dim, recent_feature_dim),
                nn.ReLU(),
                nn.Dropout(getattr(self.config, 'dropout', 0.0)),
                nn.Linear(recent_feature_dim, recent_feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(getattr(self.config, 'dropout', 0.0)),
                nn.Linear(recent_feature_dim // 2, recent_input_dim)
            )

            self.head_dim = self.transformer_dim + recent_feature_dim
        else:
            self.recent_feature_extractor = None
            self.head_dim = self.transformer_dim

        self.regression_head = nn.Linear(self.head_dim, num_output_channels)

    def forward(self, batch: dict) -> torch.Tensor:
        hidden_states = self.model(batch['past_values'], x_mark_enc=None) # [B, N, E]

        transformer_features = self.flatten(hidden_states)  # (B, num_channels * d_model)

        if self.num_recent_steps > 0:
            recent_features = batch['past_values'][:, -self.num_recent_steps :, :]
            recent_features = self.flatten(recent_features)  # (B, num_recent_steps * num_channels)
            recent_features = self.recent_feature_extractor(recent_features)
            combined_features = torch.cat([transformer_features, recent_features], dim=1)
        else:
            combined_features = transformer_features

        x = self.regression_head(combined_features)
        mean = x[:, 0]
        var = F.softplus(x[:, 1]) + 1e-6
        return torch.stack([mean, var], dim=1)


def loss_fn(
        config: Config,
        batch: dict,
        outputs: torch.Tensor
) -> torch.Tensor:
    gaussian_nll_loss = nn.GaussianNLLLoss()
    gaussian_loss_value = gaussian_nll_loss(
        outputs[:, 0],
        batch['target_values'][:, 2],  # market_forward_excess_returns
        outputs[:, 1],  # prediction_vars
    )

    return gaussian_loss_value


def load_best_model(model: nn.Module, model_path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


class AverageMeter:
    """Compute and stores average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _to_device(batch: dict, device: torch.device) -> dict:
    item_candidates = ['past_values', 'past_observed_mask', 'target_values']
    for k in item_candidates:
        if k in batch:
            batch[k] = batch[k].to(device)
    return batch


def train_fn(
    config: Config,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_ema: ModelEmaV3 | None = None,
) -> float:
    model.train()
    optimizer.train()
    losses = AverageMeter()

    scaler = GradScaler() if config.use_fp16 and torch.cuda.is_available() else None
    use_fp16 = config.use_fp16 and torch.cuda.is_available()
    max_grad_norm = getattr(config, "max_grad_norm", 1.0)

    pbar = tqdm(dataloader, desc="batch")
    for batch in pbar:
        batch = _to_device(batch, device)
        batch_size = batch["target_values"].size(0)

        with autocast(enabled=use_fp16, device_type=device.type):
            outputs = model(batch)
            loss = loss_fn(
                config=config,
                batch=batch,
                outputs=outputs,
            )

        losses.update(loss.item(), batch_size)
        optimizer.zero_grad()

        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Update EMA model after optimizer step
        if model_ema is not None:
            model_ema.update(model)

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{losses.avg:.4f}",
                "grad_norm": f"{grad_norm:.4f}",
            }
        )
    pbar.close()

    return losses.avg


def valid_fn(
        config:Config,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
) -> dict:
    model.eval()
    predictions = []
    losses = AverageMeter()

    pbar = tqdm(dataloader, desc='batch', leave=False)
    for batch in pbar:
        batch = _to_device(batch, device)
        batch_size = batch['target_values'].size(0)

        with torch.inference_mode():
            outputs = model(batch)
            loss = loss_fn(
                config=config,
                outputs=outputs,
                batch=batch,
            )
            losses.update(loss.item(), batch_size)
            predictions.append(outputs.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{losses.avg:.4f}"})

    predictions = np.vstack(predictions)
    targets = [batch['target_values'].cpu().numpy() for batch in dataloader]
    target_masks = [batch['target_mask'].cpu().numpy() for batch in dataloader]

    outputs = {
        'predictions': predictions,
        'targets': np.vstack(targets),
        'target_masks': np.vstack(target_masks),
        'loss': losses.avg,
    }
    return outputs


def inference_fn(
        config: Config,
        models: list[nn.Module],
        dataloader: DataLoader,
        device: torch.device,
        verbose: bool = True,
) -> np.ndarray:
    for model in models:
        model.eval()

    all_predictions = []

    progress_bar = tqdm(dataloader, desc="Inference", leave=True) if verbose else dataloader
    for batch in progress_bar:
        batch = _to_device(batch, device)

        batch_predictions = []
        with torch.inference_mode():
            for model in models:
                outputs = model(batch)
                batch_predictions.append(outputs.cpu().numpy())

        batch_avg = np.mean(batch_predictions, axis=0)  # (batch_size, num_output_channels)
        all_predictions.append(batch_avg)

    predictions = np.vstack(all_predictions)
    return predictions


def predictions_to_position(
        predictions: np.ndarray,
        prediction_vars: np.ndarray,
        risk_aversion: float,
        scale_factor: float,
        base_position: float = 1.0,
) -> np.ndarray:
    scores = predictions / (risk_aversion * prediction_vars) if risk_aversion > 0 else predictions
    scores = scores / scale_factor

    positions = base_position + scores
    positions = np.clip(positions, 0, 2)

    return positions


def load_position_params(output_dir: Path, config: Config) -> dict:
    params_path = output_dir / 'position_params.json'
    if params_path.exists():
        with open(params_path) as f:
            return json.load(f)
    return {
        'risk_aversion': config.default_risk_aversion,
        'scale_factor': config.default_scale_factor,
        'base_position': config.default_base_position,
    }


def optimize_position_params(
        predictions: np.ndarray,
        prediction_vars: np.ndarray,
        forward_returns: np.ndarray,
        risk_free_rate: np.ndarray,
) -> dict:
    risk_aversion = np.concatenate(
        [
            np.arange(0.0, 2, 0.1),
            np.arange(2, 10, 0.2),
            np.arange(10, 51, 5),
            np.arange(60, 101, 5),
        ]
    )

    scale_factors = np.concatenate(
        [
            np.arange(0.1, 5, 0.1),
            np.arange(5, 100, 5),
        ]
    )

    base_positions = np.concatenate(
        [
            np.arange(0.0, 2, 0.1),
        ]
    )

    best_score = -float('inf')
    best_params = {}
    total_combinations = len(risk_aversion) * len(scale_factors) * len(base_positions)

    with tqdm(total=total_combinations, desc="Optimizing parameters") as pbar:
        for ra in risk_aversion:
            for sf in scale_factors:
                for bp in base_positions:
                    positions = predictions_to_position(predictions, prediction_vars, ra, sf, bp)
                    scores = score_fn_v2(positions, forward_returns, risk_free_rate)
                    score = scores['adjusted_sharpe']

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'risk_aversion': float(ra),
                            'scale_factor': float(sf),
                            'base_position': float(bp),
                            'best_score': score,
                        }
                        pbar.set_postfix({'best_sharpe': f'{score:.4f}'})

                    pbar.update(1)

    return  best_params


def train_loop(
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        best_model_path: Path,
        device: torch.device,
) -> pl.DataFrame:
    model = CustomModel(
        config=config,
        num_input_channels=train_dataloader.dataset.X_seq.shape[2],
        num_output_channels=2,
    ).to(device)

    optimizer = RAdamScheduleFree(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
    )

    model_ema = (
        ModelEmaV3(
            model,
            use_warmup=getattr(config, 'ema_use_warmup', True),
            decay=getattr(config, 'ema_decay', 0.999),
            warmup_power=getattr(config, 'ema_warmup_power', 1.0),
            warmup_gamma=getattr(config, 'ema_warmup_gamma', 1.0),
            exclude_buffers=True,
        )
        if getattr(config, 'user_ema', False)
        else None
    )

    patience = max(0, getattr(config, "early_stopping_patience", 0))
    min_delta = getattr(config, "early_stopping_min_delta", 0.0)
    min_epochs = max(0, getattr(config, "early_stopping_min_epochs", 0))

    best_score = -float('inf')
    best_epoch = 0
    epochs_without_improve = 0
    stopped_epoch: int | None = None
    history = []

    print(f"\nTraining on {device} | {config.num_epochs} epochs | LR: {config.lr}")
    print("=" * 80)

    for epoch in range(config.num_epochs):
        print(f"\n[Epoch {epoch + 1}/{config.num_epochs}]")

        # Training
        train_loss = train_fn(
            config=config,
            model=model,
            model_ema=model_ema,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
        )

        # Validation
        if val_dataloader is not None:
            eval_model = model_ema.module if model_ema is not None else model

            val_output = valid_fn(
                config=config,
                model=eval_model,
                dataloader=val_dataloader,
                device=device,
            )
            val_loss = val_output['loss']
            val_score = -val_loss

            improvement = val_score - best_score
            is_best = improvement > min_delta
            old_best = best_score

            if is_best:
                best_score = val_score
                best_epoch = epoch + 1
                epochs_without_improve = 0
                if model_ema is not None:
                    torch.save(model_ema.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
            else:
                epochs_without_improve += 1

            model_type = " (EMA)" if model_ema is not None else ""
            if is_best:
                delta_for_print = 0.0 if old_best == -float("inf") else val_score - old_best
                print(
                    f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                    f"val score{model_type} {val_score:.4f} | "
                    f"BEST! ↑{delta_for_print:.4f}"
                )
            else:
                print(
                    f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | val score{model_type} {val_score:.4f}"
                )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_score": val_score,
                    "is_best": is_best,
                    "best_score": best_score,
                    "no_improve_count": epochs_without_improve,
                    "model_type": "ema" if model_ema is not None else "normal",
                }
            )

            if patience > 0 and (epoch + 1) >= max(1, min_epochs) and epochs_without_improve >= patience:
                stopped_epoch = epoch + 1
                print(f"Early stopping triggered (no improvement in {patience} epochs).")
                break

        else:
            # No validation, just save the model
            print(f'train loss {train_loss:.4f}')
            if model_ema is not None:
                torch.save(model.state_dict(), best_model_path)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": None,
                    "val_score": None,
                    "is_best": False,
                    "best_score": None,
                    "no_improve_count": None,
                    "model_type": "ema" if model_ema is not None else "normal",
                }
            )


    print('=' * 80)
    
    if val_dataloader is not None:
        model_type_str = " (EMA)" if model_ema is not None else ""
        if best_score == -float("inf"):
            print(f"\nNo validation improvements recorded{model_type_str}.\n")
        else:
            print(f"\nBest Score{model_type_str}: {best_score:.6f} (epoch {best_epoch})\n")
        if stopped_epoch is not None:
            print(f"Early stopping at epoch {stopped_epoch} (patience {patience}).\n")
        history_df = pl.DataFrame(history)
        print(history_df.sort("val_score", descending=True).head(5))
    else:
        print("\nTraining completed (no validation)\n")
        history_df = pl.DataFrame(history)
        print(history_df)

    return history_df


class InferenceEnv:
    def __init__(
            self,
            config: Config,
            preprocessor: Preprocessor,
            models: list[nn.Module],
            device: torch.device,
            train_df: pl.DataFrame,
            feature_cols: list[str],
            position_params: dict,
            latest_date_id: int = 8980
    ):
        self.preprocessor = preprocessor
        self.config = config
        self.models = models
        self.device = device
        self.position_params = position_params

        print(f'Position params: {position_params}')

        self.target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
        self.label_cols = ['lagged_forward_returns', 'lagged_risk_free_rate', 'lagged_market_forward_excess_returns']

        self.latest_df = train_df.filter(pl.col('date_id') <= latest_date_id)

        self.dtype_map  = {col: dtype for col, dtype in zip(self.latest_df.columns, self.latest_df.dtypes, strict=True)}
        self.feature_cols = feature_cols
        self.seq_length = self.config.sliding_window_view_size

        # Online training parameters
        self.online_training_frequency = getattr(self.config, "online_training_frequency", 0)
        self.online_training_epochs = getattr(self.config, "online_training_epochs", 1)
        self.online_training_lr = getattr(self.config, "online_training_lr", self.config.lr)
        self.online_training_batch_size = getattr(self.config, "online_training_batch_size", self.config.batch_size)

        # Label history for online training: list of (date_id, labels_array) tuples
        self.label_history = []
        self.pred_list = []

    def _run_online_training(self) -> None:
        """Run online with accumulated labels."""
        if len(self.label_history) == 0:
            return

        print(f"\n[Online Training] Using {len(self.label_history)} samples")

        # Get feature data for the accumulated labels
        label_date_ids = np.array([date_id for date_id, _ in self.label_history])

        # Transform features
        feature_df = self.preprocessor.transform(self.latest_df)
        all_date_ids = feature_df['date_id'].to_numpy()
        all_features = feature_df. select(pl.col(self.feature_cols)).to_numpy()

        # Create sliding windows
        X_seq_all = create_sliding_window_x(X=all_features, seq_length=self.seq_length)
        seq_date_ids = all_date_ids[self.seq_length - 1 :]

        # Filter to only include dates we have labels for
        mask = np.isin(seq_date_ids, label_date_ids)
        X_seq = X_seq_all[mask]

        # Get corresponding labels
        label_dict = {date_id: labels for date_id, labels in self.label_history}
        y = np.array([label_dict[int(date_id)] for date_id in seq_date_ids[mask]])

        if len(X_seq) == 0:
            print(" No valid training samples, skipping")
            return

        # Create dataset and dataloader
        train_dataset = TrainDataset(X_seq=X_seq, y=y)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=min(self.online_training_batch_size, len(train_dataset)),
            shuffle=True,
        )

        # Train each model
        for model_idx, model in enumerate(self.models):
            optimizer = RAdamScheduleFree(
                model.parameters(),
                lr=self.online_training_lr,
                betas=self.config.betas,
            )

            for epoch in range(self.config.online_training_epochs):
                loss = train_fn(
                    config=self.config,
                    model=model,
                    dataloader=train_dataloader,
                    optimizer=optimizer,
                    device=self.device,
                    model_ema=None,
                )
                print(
                    f"  Model {model_idx + 1}/{len(self.models)} | "
                    f"Epoch {epoch + 1}/{self.online_training_epochs} | "
                    f"Loss: {loss:.4f}"
                )

            # Set back to eval mode
            model.eval()

        print("[Online Training] Completed\n")

    def predict(self, test: pl.DataFrame) -> float:
        assert (test['date_id'].item() - 1) == self.latest_df['date_id'].max(), 'Test data_id should be continuous'

        test_date_id = test['date_id'].item()

        # Online training: accumulate lagged labels
        # lagged_* columns in test contain the true labels for date_id - 1
        if self.online_training_frequency > 0 and all(col in test.columns for col in self.label_cols):
            lagged_labels = test.select(self.label_cols).to_numpy()[0]
            label_date_id = test_date_id - 1
            self.label_history.append((label_date_id, lagged_labels))

            # Run online training if we have enough samples
            if len(self.label_history) >= self.online_training_frequency:
                self._run_online_training()
                self.label_history = []  # Reset after training

        # add target test record to latest_df
        self.latest_df = (
            pl.concat(
                [
                    self.latest_df,
                    test.drop(self.label_cols + ['is_scored']),
                ],
                how='diagonal_relaxed',
            )
            .sort('date_id')
            .cast(self.dtype_map)
        )

        # preprocess
        latest_feature_df = self.preprocessor.transform(self.latest_df)

        latest_X_seq = create_sliding_window_x(
            X=latest_feature_df.select(pl.col(self.feature_cols)).to_numpy(),
            seq_length=self.seq_length,
        )[[-1]]

        # Dataset
        latest_dataset = TestDataset(X_seq=latest_X_seq)
        latest_dataloader = DataLoader(latest_dataset, batch_size=1)

        test_preds = inference_fn(
            config=self.config,
            models=self.models,
            dataloader=latest_dataloader,
            device=self.device,
            verbose=False,
        )
        test_position = float(
            predictions_to_position(
                predictions=test_preds[:, 0],
                prediction_vars=test_preds[:, 1],
                risk_aversion=self.position_params['risk_aversion'],
                scale_factor=self.position_params['scale_factor'],
                base_position=self.position_params['base_position'],
            )[0]
        )
        self.pred_list.append(test.select(pl.col('date_id')).with_columns(pl.lit(test_position).alias('prediction')))

        return test_position


if __name__ == '__main__':
    import rootutils

    rootutils.setup_root('.', cwd=True)

    # %%
    # Settings
    config = Config()
    settings = DirectorySettings(exp_name=config.exp_name)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # save config
    with open(settings.OUTPUT_DIR / 'config.json', 'w') as f:
        json.dump(config.model_dump(), f, indent=4)

    # %%
    # Data Load
    raw_train_df = pl.read_csv(settings.COMP_DATASET_DIR / 'train.csv')
    raw_test_df = pl.read_csv(settings.COMP_DATASET_DIR / 'test.csv')

    target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
    train_labels_df = raw_train_df.select(['date_id'] + target_cols)

    # Create lagged_* columns (shift target columns by 1)
    train_df = raw_train_df.with_columns(
        [
            pl.col("forward_returns").shift(1).alias("lagged_forward_returns"),
            pl.col("risk_free_rate").shift(1).alias("lagged_risk_free_rate"),
            pl.col("market_forward_excess_returns").shift(1).alias("lagged_market_forward_excess_returns"),
        ]
    ).drop(target_cols)

    preprocessor = Preprocessor(config=config)
    train_feature_df = preprocessor.fit_transform(train_df)
    preprocessor.save(dirpath=settings.OUTPUT_DIR)
    print(train_df)

    # %%
    # Preprocess
    feature_cols = [x for x in train_feature_df.columns if x.startswith(config.feature_prefix)]
    target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']

    # save feature_cols and target_cols
    joblib.dump(feature_cols, settings.OUTPUT_DIR / 'feature_cols.pkl')
    joblib.dump(target_cols, settings.OUTPUT_DIR / 'target_cols.pkl')

    Xt_seq, yt = create_sliding_window_x(
        X=train_feature_df.select(pl.col('date_id')).to_numpy(),
        y=train_labels_df.select(pl.col('date_id')).to_numpy(),
        seq_length=config.sliding_window_view_size,
    )
    X_seq, y = create_sliding_window_x(
        X=train_feature_df.select(pl.col(feature_cols)).to_numpy(),
        y=train_labels_df.select(pl.col(target_cols)).to_numpy(),
        seq_length=config.sliding_window_view_size,
    )

    # %%
    # [Optional] Full Training
    if config.full_train:
        train_dataset = TrainDataset(X_seq=X_seq, y=y)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        best_model_paths = []
        for seed in config.seeds:
            print(f"Seed: {seed}")
            out_dir = settings.OUTPUT_DIR / "full_training" / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed)

            best_model_path = out_dir / "best_model.pth"
            history_df = train_loop(
                config=config,
                train_dataloader=train_dataloader,
                val_dataloader=None,
                best_model_path=best_model_path,
                device=device,
            )
            history_df.write_csv(out_dir / "history_full.csv")
            plot_training_history(
                history_df=history_df,
                filepath=out_dir / "training_history_full.png",
                figsize=(10, 6),
            )
            best_model_paths.append(best_model_path)

        print("Full training completed.")

        preprocessor = Preprocessor(config=config)
        preprocessor.load(settings.OUTPUT_DIR)

        best_models = [
            load_best_model(
                model=CustomModel(
                    config=config,
                    num_input_channels=X_seq.shape[2],
                    num_output_channels=y.shape[1],
                ),
                model_path=settings.OUTPUT_DIR / "full_training" / f"seed_{seed}" / "best_model.pth",
                device=device,
            )
            for seed in config.seeds
        ]

        position_params = load_position_params(settings.OUTPUT_DIR, config)
        inference_env = InferenceEnv(
            config=config,
            preprocessor=preprocessor,
            models=best_models,
            device=device,
            train_df=train_df,
            feature_cols=feature_cols,
            position_params=position_params,
        )

        inference_server = default_inference_server.DefaultInferenceServer(inference_env.predict)
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            inference_server.serve()
        else:
            inference_server.run_local_gateway((settings.COMP_DATASET_DIR,))

        test_pred_df = pl.concat(inference_env.pred_list).sort("date_id")
        print(test_pred_df)
        print("Inference server demo completed.")
        exit(0)

    # %%
    # Train/Validation Split
    val_mask = (
            (yt >= config.test_start_date_id) & (yt <= config.test_start_date_id + config.test_window_size)
    ).reshape(-1)
    tr_mask = (yt < config.test_start_date_id).reshape(-1)  # train only on data before the holdout range
    tr_x, tr_y, tr_t = X_seq[tr_mask], y[tr_mask], yt[tr_mask].reshape(-1)
    val_x, val_y, val_t = X_seq[val_mask], y[val_mask], yt[val_mask].reshape(-1)
    print(f"tr_t: {tr_t.min()} - {tr_t.max()} | {len(tr_t)} samples")
    print(f"val_t: {val_t.min()} - {val_t.max()} | {len(val_t)} samples")

    # Dataset
    train_dataset = TrainDataset(X_seq=tr_x, y=tr_y)
    val_dataset = TrainDataset(X_seq=val_x, y=val_y)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=4)

    # %%
    # Training
    if not config.only_test:
        best_model_paths = []
        for seed in config.seeds:
            print(f"Seed: {seed}")
            seed_everything(seed)
            out_dir = settings.OUTPUT_DIR / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)

            best_model_path = out_dir / "best_model.pth"
            history_df = train_loop(
                config=config,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                best_model_path=best_model_path,
                device=device,
            )
            history_df.write_csv(out_dir / "history.csv")
            plot_training_history(
                history_df=history_df,
                filepath=out_dir / "training_history.png",
                figsize=(10, 6),
            )
            best_model_paths.append(best_model_path)

    # %%
    # Validation
    best_models = [
        load_best_model(
            model=CustomModel(
                config=config,
                num_input_channels=X_seq.shape[2],
                num_output_channels=2,
            ),
            model_path=settings.OUTPUT_DIR / f"seed_{seed}" / "best_model.pth",
            device=device,
        )
        for seed in config.seeds
    ]

    val_preds = inference_fn(
        config=config,
        models=best_models,
        dataloader=val_dataloader,
        device=device,
    )

    optimized_params = optimize_position_params(
        predictions=val_preds[:, 0],
        prediction_vars=val_preds[:, 1],
        forward_returns=val_y[:, 0],
        risk_free_rate=val_y[:, 1],
    )
    with open(settings.OUTPUT_DIR / "position_params.json", "w") as f:
        json.dump(optimized_params, f, indent=4)
    print(f"Optimized params: {optimized_params}\n")

    position_params = load_position_params(settings.OUTPUT_DIR, config)
    val_positions = predictions_to_position(
        predictions=val_preds[:, 0],
        prediction_vars=val_preds[:, 1],
        risk_aversion=position_params["risk_aversion"],
        scale_factor=position_params["scale_factor"],
        base_position=position_params["base_position"],
    )
    val_pred_df = pl.DataFrame(val_preds, schema=["prediction", "prediction_var"]).with_columns(
        pl.Series(val_t).alias("date_id").cast(pl.Int32),
        pl.Series(val_y[:, 0]).alias("forward_returns"),
        pl.Series(val_y[:, 1]).alias("risk_free_rate"),
        pl.Series(val_y[:, 2]).alias("market_forward_excess_returns"),
        pl.Series(val_positions).alias("position"),
    )

    # score
    out_dir = settings.OUTPUT_DIR / "val_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_metrics = score_fn_v2(
        predictions=val_positions,
        forward_returns=val_pred_df["forward_returns"].to_numpy(),
        risk_free_rate=val_pred_df["risk_free_rate"].to_numpy(),
    )
    print(f"\nFinal Validation Score: {val_metrics['adjusted_sharpe']:.6f}\n")
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=4)
    val_pred_df.write_csv(out_dir / "val_predictions.csv")

    # Plot validation results
    plot_prediction_distribution(
        predictions=val_preds[:, 0],
        targets=val_y[:, 2],  # market_forward_excess_returns
        filepath=out_dir / "prediction_distribution.png",
    )

    plot_prediction_variance(
        predictions=val_preds[:, 0],
        prediction_vars=val_preds[:, 1],
        filepath=out_dir / "prediction_variance.png",
    )

    plot_position_distribution(
        positions=val_positions,
        filepath=out_dir / "position_distribution.png",
    )

    plot_prediction_timeseries(
        date_ids=val_t,
        predictions=val_preds[:, 0],
        targets=val_y[:, 2],  # market_forward_excess_returns
        filepath=out_dir / "prediction_timeseries.png",
    )

    plot_correlation_heatmap(
        predictions=val_preds[:, 0],
        forward_returns=val_y[:, 0],
        risk_free_rate=val_y[:, 1],
        market_forward_excess_returns=val_y[:, 2],
        positions=val_positions,
        filepath=out_dir / "correlation_heatmap.png",
    )

    # %%
    # Inference Server Demo
    preprocessor = Preprocessor(config=config)
    preprocessor.load(settings.OUTPUT_DIR)
    position_params = load_position_params(settings.OUTPUT_DIR, config)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
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

    test_pred_df = (
        pl.concat(inference_env.pred_list)
        .join(raw_test_df.select(pl.col(["date_id", "is_scored"])), on="date_id")
        .sort("date_id")
    )
    print(test_pred_df)

    test_score_df = test_pred_df.filter(pl.col("is_scored") == 1)
    test_label_df = train_labels_df.join(test_score_df, on="date_id", how="inner")
    print(test_score_df)

    score = score_fn_v2(
        predictions=test_score_df["prediction"].to_numpy(),
        forward_returns=test_label_df["forward_returns"].to_numpy(),
        risk_free_rate=test_label_df["risk_free_rate"].to_numpy(),
    )
    print(f"\nTest Score on public labels: {score['adjusted_sharpe']:.6f}\n")

# %%
