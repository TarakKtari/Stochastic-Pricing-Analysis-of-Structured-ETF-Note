# 🧠 NVIDIA Structured Note Evaluation (2020–2025)

This Jupyter notebook performs a financial analysis of NVIDIA stock (NVDA) to evaluate a structured financial product. It combines traditional financial modeling, machine learning (Random Forest), and deep learning (TensorFlow/Keras) to simulate stock paths and estimate structured note payoffs.

---

## 1. Key Components of the Analysis

### 1.1 📈 Data Collection & Preprocessing

- **Data Source**: `yfinance` — NVIDIA (NVDA) stock data from 2020 to 2025.  
- **Features Used**: Open, High, Low, Close (OHLC), Volume.  
- **Processing**: Log returns computed for drift and volatility estimation.

---

### 1.2 ⚙️ Parameter Estimation

Three methods are used to model stock behavior:

#### 🔹 Historical Parameters (`compute_historical_params`)
- Computes annualized drift (`mu_hist`) and volatility (`sigma_hist`) from log returns.  
- Based on 252 trading days/year.

#### 🔹 Machine Learning Prediction (`predict_ml_params`)
- Model: `RandomForestRegressor` from `sklearn.ensemble`.  
- **Features**: Open, High, Low, Volume.  
- **Targets**: Rolling mean (drift) and standard deviation (volatility).  
- **Validation**: 80-20 train-test split.

#### 🔹 Parameter Blending (`optimize_blend`)
- Combines historical and ML estimates using KFold cross-validation.  
- Optimizes blending weights (α, β) to minimize Mean Squared Error (MSE).

---

### 1.3 🧪 Monte Carlo Simulation (`simulate_stock_paths`)

Simulates 10,000 stock price paths using **Geometric Brownian Motion (GBM)**:

- **Initial Price**: `S0 = 124.91`  
- **Inputs**: Blended drift (`mu_final`), volatility (`sigma_final`)  
- **Time Horizon**: `T = 1.083` years (≈ 263 trading days)  
- **Time Step**: `dt = 1/263`

**GBM Formula**:

```math
S_t = S_{t-1} \cdot \exp\left[\left(\mu - \frac{1}{2} \sigma^2\right) dt + \sigma \sqrt{dt} \cdot Z\right], \quad Z \sim \mathcal{N}(0, 1)
### 1.4 💰 Structured Note Payoff Calculation (`structured_note_payoff`)

Evaluates a structured financial product with:

- 📅 **Coupon payments** at fixed dates.  
- ⏩ **Early redemption** if the stock hits a specified threshold.  
- 🛡️ **Principal protection or penalty** at maturity.

**Key Outputs**:

- ✅ **Probability of full principal repayment**: **89.3%**  
- 📉 **Principal recovery rate** (if penalized).  
- 🔁 **Early redemption probability** (varies by simulation).  
- 💵 **Coupon payment probability** (varies by simulation).

**Visualization**:  
- Uses `matplotlib` to plot the **discounted payout distribution**.

---

### 1.5 🤖 Neural Network for Payoff Prediction (`train_neural_network`)

A 3-layer **neural network** (built using TensorFlow/Keras) is trained to predict structured note payoffs directly from the **terminal stock price**.

#### Neural Network Architecture

```python
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(1,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="linear")  # Regression output
])
