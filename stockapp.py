import tkinter as tk
from tkinter import messagebox
import numpy as np
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model

LOOKBACK_DAYS = 30  

# Define percentage categories
PERCENTAGE_MOVES = {
    0: "Neutral",
    1: "+3%",
    2: "+7%",
    3: "+10%",
    4: "+15%",
    5: "+30%",
    6: "-3%",
    7: "-7%",
    8: "-10%",
    9: "-15%",
    10: "-30%"
}

def get_stock_data(ticker):
    """Fetch stock data from Yahoo Finance with error handling."""
    try:
        data = yf.download(ticker, period="3mo")
        if data.empty:
            raise ValueError("No data found for the given ticker.")
        data.dropna(inplace=True)
        return data[['Close', 'Volume']]
    except Exception as e:
        messagebox.showerror("Data Fetch Error", f"Failed to fetch data: {e}")
        return None

def calculate_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI) with error handling."""
    try:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = np.where(loss == 0, 0, gain / loss)  # Prevent division errors
        return 100 - (100 / (1 + rs))
    except Exception as e:
        messagebox.showerror("RSI Calculation Error", f"Error calculating RSI: {e}")
        return np.zeros(len(data))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate the MACD indicator with error handling."""
    try:
        short_ema = data.ewm(span=short_window, adjust=False).mean()
        long_ema = data.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd - signal  
    except Exception as e:
        messagebox.showerror("MACD Calculation Error", f"Error calculating MACD: {e}")
        return np.zeros(len(data))

def predict_stock_trend():
    """Handles stock trend prediction with comprehensive error handling."""
    ticker = ticker_entry.get().upper().strip()
    if not ticker:
        messagebox.showwarning("Input Error", "Please enter a stock ticker.")
        return

    predict_button.config(state=tk.DISABLED)  
    result_label.config(text="Processing...", fg="blue")
    root.update()

    data = get_stock_data(ticker)
    if data is None or data.empty:
        result_label.config(text="Failed to retrieve stock data.", fg="red")
        predict_button.config(state=tk.NORMAL)
        return

    try:
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'] = calculate_macd(data['Close'])
        data.dropna(inplace=True)
    except Exception as e:
        messagebox.showerror("Indicator Error", f"Error calculating indicators: {e}")
        predict_button.config(state=tk.NORMAL)
        return

    if len(data) < LOOKBACK_DAYS:
        messagebox.showwarning("Insufficient Data", "Not enough data for prediction.")
        result_label.config(text="Insufficient data.", fg="red")
        predict_button.config(state=tk.NORMAL)
        return

    try:
        model = load_model("stock_trend_model.h5")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        messagebox.showerror("File Error", "Model or scaler file not found.")
        predict_button.config(state=tk.NORMAL)
        return
    except Exception as e:
        messagebox.showerror("Loading Error", f"Error loading model or scaler: {e}")
        predict_button.config(state=tk.NORMAL)
        return

    try:
        scaled_data = scaler.transform(data[['Close', 'Volume', 'RSI', 'MACD']])
        X_input = scaled_data[-LOOKBACK_DAYS:].reshape(1, LOOKBACK_DAYS, 4)
        prediction = model.predict(X_input)[0]
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction: {e}")
        predict_button.config(state=tk.NORMAL)
        return

    best_class = np.argmax(prediction)
    confidence = prediction[best_class] * 100
    percentage_move = PERCENTAGE_MOVES.get(best_class, "Unknown")

    if best_class == 0:
        result_text = f"{ticker}: 🔍 Neutral trend."
        result_color = "black"
    else:
        direction = "up📈" if best_class <= 5 else "down📉"
        result_text = f"{ticker} is likely to go {direction} {percentage_move} ({confidence:.1f}% confidence)."
        result_color = "green" if best_class <= 5 else "red"

    result_label.config(text=result_text, fg=result_color)
    predict_button.config(state=tk.NORMAL)




# UI Setup
root = tk.Tk()
root.title("Stock Trend Predictor")
root.geometry("400x220")
root.resizable(False, False)

# Widgets
tk.Label(root, text="Enter Stock Ticker:", font=("Arial", 12)).pack(pady=10)
ticker_entry = tk.Entry(root, font=("Arial", 12), width=20)
ticker_entry.pack()

predict_button = tk.Button(root, text="Predict Stock", font=("Arial", 12), command=predict_stock_trend)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=350, justify="center")
result_label.pack(pady=10)

root.mainloop()
