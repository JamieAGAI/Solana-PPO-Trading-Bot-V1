import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from collections import deque
import requests
from binance.client import Client as BinanceClient
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair
from solders.transaction import Transaction, VersionedTransaction
from solders.pubkey import Pubkey
from solders.instruction import Instruction
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import get_associated_token_address, create_associated_token_account
from solders.system_program import create_account, CreateAccountParams
import base58
import base64
from solana.rpc.commitment import Confirmed
from solders.message import to_bytes_versioned
from sklearn.preprocessing import RobustScaler
import csv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from joblib import dump, load
from sklearn.model_selection import TimeSeriesSplit

# Load .env if exists
if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()
ALCHEMY_RPC = os.getenv('ALCHEMY_RPC', 'https://solana-mainnet.g.alchemy.com/v2/your_key')
WALLET_PRIVATE_KEY = os.getenv('WALLET_PRIVATE_KEY', 'your_base58_key')

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Constants aligned with opty2, updated for robustness
USDC_MINT = Pubkey.from_string('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
SOL_MINT = Pubkey.from_string('So11111111111111111111111111111111111111112')
JUPITER_API_BASE = "https://quote-api.jup.ag/v6"
SPL_TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
SLIPPAGE = 0.005
BUY_PERCENTAGE = 1.0  
SELL_PERCENTAGE = 1.0
MIN_TRADE_AMOUNT = 10.0
TRADE_FEE = 0.003
MIN_PROFIT_THRESHOLD = 10.0
STOP_LOSS_THRESHOLD = -0.04  
SEQ_LEN = 40  # Increased for better context
MAX_DD_THRESHOLD = 0.15  # Increased to 20% based on trial avg
EPISODES = 10  
DAYS_PER_EPISODE = 300  
rsi_len = 6
rsi_sell_thresh = 60
atr_len = 14
DATA_CACHE = 'sol_usdt_730_days_15min.csv'  
MIN_SOL_RESERVE = 0.1  
lookback_left = 6
lookback_right = 4
atr_mult = 2.9290654217059315
rsi_buy_thresh = 53
learning_rate = 0.00033865894170859325
ent_coef = 0.05
gamma = 0.9654873600594873
batch_size = 218
n_epochs = 19
clip_range = 0.2544930437430099
action_threshold = 0.28
HOLDOUT_DAYS = 100
HOLD_PENALTY = -0.05 
SHARPE_WEIGHT = 0.15295612919604964
SORTINO_WEIGHT = 0.36051224239221835
UNREALIZED_PENALTY = 0.18636762174589672
DD_PENALTY_WEIGHT = 0.11234515070233216
ACTIVITY_INCENTIVE = 0.05

# Get balances
def get_balances(wallet_pubkey, solana_client):
    sol = solana_client.get_balance(wallet_pubkey).value / 1e9
    usdc_ata = get_associated_token_address(wallet_pubkey, USDC_MINT)
    usdc = solana_client.get_token_account_balance(usdc_ata).value.ui_amount or 0.0
    return sol, usdc

# Initialize ATA
def initialize_ata(mint, wallet_keypair, solana_client):
    ata = get_associated_token_address(wallet_keypair.pubkey(), mint)
    try:
        solana_client.get_token_account_balance(ata)
    except:
        tx = Transaction.new_signed(
            wallet_keypair,
            [create_associated_token_account(wallet_keypair.pubkey(), wallet_keypair.pubkey(), mint)],
            solana_client.get_latest_blockhash().value.blockhash
        )
        solana_client.send_transaction(tx)
    return ata

# Jupiter swap 
def jupiter_swap(wallet_keypair, input_mint, output_mint, amount, solana_client, retries=3):
    if amount < MIN_TRADE_AMOUNT:
        print(f"Skipping small swap: {amount} below min {MIN_TRADE_AMOUNT}")
        return False
    decimals_in = 6 if input_mint == USDC_MINT else 9
    amount_lamports = int(amount * 10**decimals_in)
    if amount_lamports < 10**(decimals_in - 2):
        return False
    
    wallet_pubkey = wallet_keypair.pubkey()
    sol_bal, usdc_bal = get_balances(wallet_pubkey, solana_client)
    if (input_mint == USDC_MINT and usdc_bal < amount) or (input_mint == SOL_MINT and sol_bal < amount):
        return False

    input_ata = initialize_ata(input_mint, wallet_keypair, solana_client)
    output_ata = initialize_ata(output_mint, wallet_keypair, solana_client)
    if not input_ata or not output_ata:
        return False

    quote_url = f"{JUPITER_API_BASE}/quote?inputMint={str(input_mint)}&outputMint={str(output_mint)}&amount={amount_lamports}&slippageBps={int(SLIPPAGE*10000)}"
    for attempt in range(retries):
        try:
            quote = requests.get(quote_url).json()
            if 'error' in quote:
                print(f"Quote error: {quote['error']}")
                time.sleep(5)
                continue
            decimals_out = 9 if output_mint == SOL_MINT else 6
            out_amount = int(quote['outAmount']) / 10**decimals_out  # Approx out_amount from quote
            
            swap_url = f"{JUPITER_API_BASE}/swap"
            swap_request = {"quoteResponse": quote, "userPublicKey": str(wallet_pubkey), "wrapAndUnwrapSol": True}
            swap_tx = requests.post(swap_url, json=swap_request).json()['swapTransaction']
            raw_tx = base64.b64decode(swap_tx)
            tx = VersionedTransaction.from_bytes(raw_tx)
            signed_tx = wallet_keypair.sign_transaction(tx)
            tx_sig = solana_client.send_transaction(signed_tx, opts=Confirmed).value
            print(f"Swap TX: {tx_sig}")
            return True, out_amount
        except Exception as e:
            print(f"Swap attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return False

# Fetch historical data 
def fetch_historical_data(days=3, force_fetch=False):
    data_cache = f'sol_usdt_{days}_days_15min.csv'
    if not force_fetch and os.path.exists(data_cache):
        print(f"Loading cached data for {days} days...")
        try:
            return pd.read_csv(data_cache, index_col='timestamp', parse_dates=True)
        except Exception as e:
            print(f"Cache load failed: {e}. Fetching new data...")
    
    print(f"Fetching new data for {days} days...")
    client = BinanceClient('', '')
    interval = BinanceClient.KLINE_INTERVAL_15MINUTE
    start_time = int((time.time() - days * 86400) * 1000)
    klines = client.get_historical_klines('SOLUSDT', interval, start_str=start_time)
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.to_csv(data_cache)  # Save/overwrite cache
    return df

# Compute indicators 
def compute_indicators(df, lookback_left, lookback_right):
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma100'] = df['close'].rolling(100).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_len).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['vol_ma'] = df['volume'].rolling(5).mean()
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = tr.rolling(atr_len).mean()
    
    pivot_window = lookback_left + lookback_right + 1
    df['pivot_high'] = df['high'].rolling(window=pivot_window, center=False).max()
    df['pivot_low'] = df['low'].rolling(window=pivot_window, center=False).min()
    
    lookback_fib = lookback_left * 2
    df['swing_high'] = df['high'].rolling(lookback_fib).max()
    df['swing_low'] = df['low'].rolling(lookback_fib).min()
    fib_range = df['swing_high'] - df['swing_low']
    df['fib382'] = df['swing_high'] - fib_range * 0.382
    df['fib50'] = df['swing_high'] - fib_range * 0.5
    df['fib618'] = df['swing_high'] - fib_range * 0.618
    
    # ADX, DI+, DI-
    dm_plus = ((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low'])) * (df['high'] - df['high'].shift())
    dm_minus = ((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift())) * (df['low'].shift() - df['low'])
    dm_plus = dm_plus.rolling(atr_len).mean()
    dm_minus = dm_minus.rolling(atr_len).mean()
    di_plus = 100 * dm_plus / df['atr']
    di_minus = 100 * dm_minus / df['atr']
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    df['adx'] = dx.rolling(atr_len).mean()
    df['di_plus'] = di_plus
    df['di_minus'] = di_minus

    df['price_change'] = df['close'].pct_change().fillna(0)
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)  # Replace inf with 0
    
    return df

# TradingCallback (same as opty2)
class TradingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_rewards = None
        with open('rewards.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'env_id', 'reward'])

    def _on_training_start(self) -> None:
        if self.current_episode_rewards is None:
            num_envs = self.training_env.num_envs
            self.current_episode_rewards = [0.0] * num_envs

    def _on_step(self) -> bool:
        for i, r in enumerate(self.locals['rewards']):
            self.current_episode_rewards[i] += r
        
        for i, d in enumerate(self.locals['dones']):
            if d:
                episode_num = len(self.episode_rewards) + 1
                self.episode_rewards.append(self.current_episode_rewards[i])
                with open('rewards.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode_num, i, self.current_episode_rewards[i]])
                self.current_episode_rewards[i] = 0.0
        return True

# CustomLSTMExtractor (updated for 18 features with portfolio state)
class CustomLSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.lstm = torch.nn.LSTM(input_size=18, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(128, features_dim)
    
    def forward(self, observations):
        obs = observations.view(-1, SEQ_LEN, 18)  # Reshape to (batch, seq, features)
        lstm_out, _ = self.lstm(obs)
        return self.linear(lstm_out[:, -1, :])  # Last hidden state

# TradingEnv class with fixed parameters, no safe mode, portfolio state added, opty2 rewards
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=60000, seq_len=SEQ_LEN, scaler=None, lookback_left=6, lookback_right=6, atr_mult=2.9290654217059315, rsi_buy_thresh=53, action_threshold=0.309719234200141):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.current_step = seq_len - 1
        self.balance = initial_balance
        self.position = 0
        self.entry_price = None
        self.net_worth_history = [initial_balance]
        self.max_net_worth = initial_balance
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.seq_len = seq_len
        self.state_history = deque(maxlen=seq_len)
        self.scaler = scaler if scaler is not None else RobustScaler()
        self.lookback_left = lookback_left
        self.lookback_right = lookback_right
        self.atr_mult = atr_mult
        self.rsi_buy_thresh = rsi_buy_thresh
        self.action_threshold = action_threshold
        self.sharpe_weight = SHARPE_WEIGHT
        self.sortino_weight = SORTINO_WEIGHT
        self.unrealized_penalty = UNREALIZED_PENALTY
        self.dd_penalty_weight = DD_PENALTY_WEIGHT
        self.activity_incentive = ACTIVITY_INCENTIVE
        self.hold_penalty = HOLD_PENALTY
        self._fit_scaler(df)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))  # Continuous: -1 (full sell) to 1 (full buy)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.seq_len * 18,), dtype=np.float32)  # +3 portfolio features

    def _fit_scaler(self, df):
        scaler_columns = ['close', 'rsi', 'atr', 'ma200', 'pivot_high', 'fib618', 'adx', 'di_plus', 'di_minus']
        features_df = df[scaler_columns].replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        features = features_df.values
        if len(features) > 1:  # Require >1 for std
            self.scaler.fit(features)
        else:
            print("Warning: Insufficient data for scaler fit - using identity")
        return self.scaler

    def reset(self, *, seed=None, options=None):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.net_worth_history = [self.initial_balance]
        self.max_net_worth = self.initial_balance
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.state_history = deque(maxlen=self.seq_len)
        for i in range(self.seq_len):
            self.state_history.append(self._get_single_state(i))
        self.current_step = self.seq_len - 1
        return np.array(self.state_history).flatten(), {}

    def _get_single_state(self, step):
        row = self.df.iloc[step]
        prev_close = self.df.iloc[step-1]['close'] if step > 0 else row['close']
        price_change = (row['close'] - prev_close) / prev_close if step > 0 else 0
        
        dyn_tolerance = row['atr'] * self.atr_mult
        near_support = abs(row['close'] - row.get('pivot_low', row['close'])) < dyn_tolerance
        buy_signal = near_support and row['close'] > row.get('ma50', row['close']) and row['rsi'] < self.rsi_buy_thresh and row['close'] > row['ma200'] and row['volume'] > row.get('vol_ma', row['volume'])
        near_resistance = abs(row['close'] - row.get('pivot_high', row['close'])) < dyn_tolerance
        sell_signal = near_resistance and row['rsi'] > rsi_sell_thresh and row['close'] > row['ma200']
        indicator_action = 1 if buy_signal else (2 if sell_signal else 0)
        
        raw_state = np.array([
            row['close'], row['rsi'], row.get('atr', 1), row['ma200'],
            row.get('pivot_high', row['close']) - row['close'],
            row['close'] - row.get('fib618', row['close']),
            row.get('adx', 25), row.get('di_plus', 0), row.get('di_minus', 0)
        ])
        normalized_features = self.scaler.transform(raw_state.reshape(1, -1)).flatten()
        
        # Portfolio state features
        unrealized = (row['close'] - self.entry_price) * self.position if self.position > 0 and self.entry_price is not None else 0
        net_worth = self.balance + (self.position * row['close'] if self.position > 0 else 0)
        position_norm = self.position / (self.balance / row['close']) if self.balance > 0 and row['close'] > 0 else 0
        unrealized_pct = unrealized / self.balance if self.balance > 0 else 0
        net_worth_ratio = net_worth / self.initial_balance
        
        state = np.concatenate([
            normalized_features,
            np.array([
                1 if row['close'] > row['ma200'] else 0,
                1 if row['volume'] > row['vol_ma'] else 0,
                1 if row['rsi'] < self.rsi_buy_thresh else 0,
                row['close'] - row.get('pivot_low', row['close']),
                price_change,
                indicator_action,
                position_norm,
                unrealized_pct,
                net_worth_ratio
            ])
        ]).astype(np.float32)
        state = np.nan_to_num(state, nan=0.0)
        return state

    def get_state(self):
        return np.array(self.state_history)

    def step(self, action):
        print(f"Action: {action}")
        action = action[0]  # Unpack continuous action
        row = self.df.iloc[self.current_step]
        reward = 0
        unrealized = (row['close'] - self.entry_price) * self.position if self.position > 0 and self.entry_price is not None else 0
        
        # Fixed parameters, no safe mode
        stop_loss_threshold = STOP_LOSS_THRESHOLD
        
        if self.position > 0 and unrealized < stop_loss_threshold * (self.position * row['close']):
            action = -1  # Force full sell
            reward -= 0.5
        
        # Continuous action interpretation
        if action > self.action_threshold:  # Buy signal
            buy_value = self.balance * 1.0
            if buy_value >= MIN_TRADE_AMOUNT:
                buy_amount = buy_value / row['close']
                self.position += buy_amount
                self.entry_price = (self.entry_price * (self.position - buy_amount) + row['close'] * buy_amount) / self.position if self.position > buy_amount else row['close']
                self.balance -= buy_value
                self.balance *= (1 - TRADE_FEE)
                reward += 0.2 if row['rsi'] < self.rsi_buy_thresh and row.get('adx', 0) > 25 else 0.1
                reward += self.activity_incentive  # Small incentive for activity
                self.trade_count += 1
        elif action < -self.action_threshold:  # Sell signal
            sell_amount = self.position * 1.0
            if sell_amount * row['close'] >= MIN_TRADE_AMOUNT and sell_amount > 0:
                sell_value = sell_amount * row['close']
                profit = sell_value - (sell_amount * self.entry_price) if self.entry_price is not None else 0
                self.balance += sell_value * (1 - TRADE_FEE)
                self.position -= sell_amount
                if self.position <= 0:
                    self.entry_price = None  # Reset if fully closed
                reward += (profit / 60000 * 5) if profit > 0 else -0.1
                reward += self.activity_incentive  # Small incentive for activity
                if profit > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.trade_count += 1
        # Else: Hold
        else:
            if self.position > 0:
                reward += self.hold_penalty  
        
        self.current_step += 1
        net_worth = self.balance + (self.position * row['close'] if self.position > 0 else 0)
        current_dd = max(0, 1 - net_worth / self.max_net_worth) if self.max_net_worth > 0 else 0
        terminated = current_dd > MAX_DD_THRESHOLD
        truncated = self.current_step >= len(self.df) - 1
        self.net_worth_history.append(net_worth)
        self.max_net_worth = max(self.max_net_worth, net_worth)
        
        # Opty2-style reward with Sortino/Sharpe
        if len(self.net_worth_history) > 20:
            recent_history = self.net_worth_history[-21:]
            returns = np.diff(recent_history) / recent_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 96)
            reward += self.sharpe_weight * sharpe
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 1:
                sortino_std = np.std(downside_returns)
                sortino = np.mean(returns) / (sortino_std + 1e-6) * np.sqrt(252 * 96)
            else:
                sortino = 0
            reward += self.sortino_weight * sortino
        
        if unrealized < -0.05 * self.balance:
            reward -= self.unrealized_penalty
        
        reward -= self.dd_penalty_weight * current_dd if current_dd > 0.1 else 0
        
        if self.current_step < len(self.df):
            self.state_history.append(self._get_single_state(self.current_step))
        
        return self.get_state().flatten(), reward, terminated, truncated, {}

def evaluate_model(model, env):
    vec_env = DummyVecEnv([lambda: env])
    obs = vec_env.reset()
    dones = [False]
    while not dones[0]:
        action, _ = model.predict(obs)
        obs, rew, dones, _ = vec_env.step(action)
    env = vec_env.envs[0]
    net_history = env.net_worth_history
    final_nw = net_history[-1]
    profit = (final_nw - 60000) / 60000 * 100
    dd_list = [1 - net_history[i] / max(net_history[:i+1]) for i in range(1, len(net_history))]
    max_dd = max(dd_list) * 100 if dd_list else 0
    trades = env.trade_count
    win_rate = (env.wins / trades * 100) if trades > 0 else 0
    return {'profit': profit, 'max_dd': max_dd, 'trades': trades, 'win_rate': win_rate}

def main(backtest=False):
    if backtest:
        # Backtest/train mode with rolling windows on 365 days
        df_full = fetch_historical_data(days=730, force_fetch=True)
        df_full = compute_indicators(df_full, lookback_left, lookback_right)
        print(f"Full data length: {len(df_full)} rows")

        holdout_rows = HOLDOUT_DAYS * 96
        train_rows = 365 * 96
        if len(df_full) < train_rows + holdout_rows:
            print("Insufficient data for training and holdout")
            return

        train_df = df_full.iloc[-(train_rows + holdout_rows):-holdout_rows].reset_index(drop=True)
        holdout_df = df_full.iloc[-holdout_rows:].reset_index(drop=True)
        print(f"Training data length: {len(train_df)} rows")
        print(f"Holdout data length: {len(holdout_df)} rows")

        # 3-fold TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=3)
        metrics_list = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(train_df)):
            fold_train_df = train_df.iloc[train_idx]
            fold_test_df = train_df.iloc[test_idx]

            scaler = RobustScaler()
            scaler_columns = ['close', 'rsi', 'atr', 'ma200', 'pivot_high', 'fib618', 'adx', 'di_plus', 'di_minus']
            features = fold_train_df[scaler_columns].dropna().values
            if len(features) > 0:
                scaler.fit(features)

            def make_env():
                return TradingEnv(fold_train_df, scaler=scaler)

            env = DummyVecEnv([make_env for _ in range(4)])

            policy_kwargs = {'net_arch': [dict(pi=[256, 256], vf=[256, 256])], 'activation_fn': torch.nn.ReLU, 'features_extractor_class': CustomLSTMExtractor}

            total_timesteps = EPISODES * (len(fold_train_df) - SEQ_LEN) // 4  # Adjust for envs
            model = PPO("MlpPolicy", env, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma, batch_size=batch_size, n_epochs=n_epochs, clip_range=clip_range, policy_kwargs=policy_kwargs, verbose=1, device=device)
            model.learn(total_timesteps=total_timesteps)
            print(f"Trained on fold {fold + 1}")

            test_env = TradingEnv(fold_test_df, scaler=scaler)
            metrics = evaluate_model(model, test_env)
            metrics_list.append(metrics)
            print(f"Fold {fold + 1} metrics: {metrics}")

        avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        print(f"Average metrics: {avg_metrics}")

        # Full train on train_df
        scaler = RobustScaler()
        features = train_df[scaler_columns].dropna().values
        if len(features) > 0:
            scaler.fit(features)
        dump(scaler, 'scaler.joblib')

        def make_env():
            return TradingEnv(train_df, scaler=scaler)

        env = DummyVecEnv([make_env for _ in range(4)])

        total_timesteps = EPISODES * (len(train_df) - SEQ_LEN) // 4
        callback = TradingCallback()
        model = PPO("MlpPolicy", env, learning_rate=learning_rate, ent_coef=ent_coef, gamma=gamma, batch_size=batch_size, n_epochs=n_epochs, clip_range=clip_range, policy_kwargs=policy_kwargs, verbose=1, device=device)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        print("Trained on full data")

        # Holdout evaluation
        holdout_env = TradingEnv(holdout_df, scaler=scaler)
        holdout_metrics = evaluate_model(model, holdout_env)
        print(f"Holdout metrics: {holdout_metrics}")

        model.save('ppo_model')
        print("Model saved as ppo_model.zip")
    else:
        # Live trading with fixed parameters
        solana_client = SolanaClient(ALCHEMY_RPC)
        private_key_bytes = base58.b58decode(WALLET_PRIVATE_KEY)
        wallet_keypair = Keypair.from_bytes(private_key_bytes)
        wallet_pubkey = wallet_keypair.pubkey()
        
        # Fetch data for dummy env and scaler
        df_full = fetch_historical_data(days=3, force_fetch=True)
        df_full = compute_indicators(df_full, lookback_left=6, lookback_right=4)
        
        if os.path.exists('scaler.joblib'):
            scaler = load('scaler.joblib')
        else:
            scaler = RobustScaler()
            scaler_columns = ['close', 'rsi', 'atr', 'ma200', 'pivot_high', 'fib618', 'adx', 'di_plus', 'di_minus']
            features = df_full[scaler_columns].dropna().values
            if len(features) > 0:
                scaler.fit(features)
            dump(scaler, 'scaler.joblib')

        dummy_env = DummyVecEnv([lambda: TradingEnv(df_full, scaler=scaler)])

        model = PPO.load('ppo_model', env=dummy_env)

        state_history = deque(maxlen=SEQ_LEN)
        position = 0  # SOL amount
        entry_price = None

        while True:
            latest_df = fetch_historical_data(days=3, force_fetch=True)
            latest_df = compute_indicators(latest_df, lookback_left=6, lookback_right=4)
            if len(latest_df) < SEQ_LEN:
                time.sleep(300)
                continue
            
            # Build state
            for i in range(-SEQ_LEN, 0):
                state = TradingEnv(latest_df, scaler=scaler)._get_single_state(len(latest_df) + i)  # Use class method
                state_history.append(state)
            state = np.array(state_history).flatten()

            sol_bal, usdc_bal = get_balances(wallet_pubkey, solana_client)
            latest = latest_df.iloc[-1]
            current_price = latest['close']
            position_value = position * current_price
            net_worth = usdc_bal + position_value
            stop_loss_threshold = STOP_LOSS_THRESHOLD

            unrealized = (current_price - entry_price) * position if position > 0 and entry_price else 0
            if position > 0 and unrealized < stop_loss_threshold * position_value:
                print("SL triggered - force sell")
                sell_amount = position * 1.0
                if sell_amount * current_price > MIN_TRADE_AMOUNT:
                    success = jupiter_swap(wallet_keypair, SOL_MINT, USDC_MINT, sell_amount, solana_client)
                    if success:
                        position = 0
                        entry_price = None
            else:
                action, _ = model.predict(state)
                action = action[0]

                if action > 0.309719234200141:
                    buy_value = usdc_bal * 1.0
                    if buy_value >= MIN_TRADE_AMOUNT:
                        success = jupiter_swap(wallet_keypair, USDC_MINT, SOL_MINT, buy_value, solana_client)
                        if success:
                            position += buy_value / current_price  # Approx
                            entry_price = (entry_price * (position - (buy_value / current_price)) + current_price * (buy_value / current_price)) / position if position > (buy_value / current_price) else current_price
                elif action < -0.309719234200141:
                    sell_amount = position * 1.0
                    if sell_amount * current_price >= MIN_TRADE_AMOUNT and sell_amount > MIN_SOL_RESERVE:
                        success = jupiter_swap(wallet_keypair, SOL_MINT, USDC_MINT, sell_amount, solana_client)
                        if success:
                            position -= sell_amount
                            if position <= 0:
                                position = 0
                                entry_price = None

            time.sleep(300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', action='store_true')
    args = parser.parse_args()
    main(args.backtest)