import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from src.stock_prediction import StockPrediction
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction App")

      
        self.symbol_label = ttk.Label(root, text="Stock Symbol:")
        self.symbol_entry = ttk.Entry(root, width=10)

        
        self.predict_button = ttk.Button(root, text="Predict", command=self.predict_stock)
        
      
        self.result_label = ttk.Label(root, text="")

    
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Stock Prices")
        self.ax.set_xlabel("Timestamp")
        self.ax.set_ylabel("Prices")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=3, column=0, columnspan=2, pady=10)

      
        self.symbol_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.symbol_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        self.predict_button.grid(row=1, column=0, columnspan=2, pady=10)
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

       
        self.stock_prediction = StockPrediction(api_key='JV9QWYGGRA2MM9ST')  # Replace with your actual API key

    def predict_stock(self):
        self.result_label.config(text="")
        
        try:
            symbol = self.symbol_entry.get().upper()  

            stock_data = self.stock_prediction.get_stock_data(symbol, interval='1min', output_size='compact')

            if stock_data is not None:
                result_text = f"Stock Data for {symbol}:\n{stock_data}"
                self.result_label.config(text=result_text)

                if not hasattr(self, 'trained_model') or self.trained_model is None:
                   
                    historical_data = self.stock_prediction.get_historical_data(symbol, interval='daily', output_size='full')
                    if historical_data is not None and '4. close' in historical_data.columns:
                        engineered_data = self.stock_prediction.engineer_features(historical_data, close_column='4. close')
                        self.trained_model = self.stock_prediction.train_model(engineered_data)
                    else:
                        raise ValueError("Error fetching or processing historical data for training.")

                if self.trained_model is not None:
                    last_3_days_data = stock_data.head(3)
                    predictions = self.stock_prediction.make_predictions(self.trained_model, last_3_days_data)

                    # Create a new figure for plotting
                    fig, ax = plt.subplots(figsize=(8, 6))

                    # Plot actual closing prices
                    ax.plot(last_3_days_data.index, last_3_days_data['4. close'], label='Actual Closing Prices', color='blue', marker='o')

                    # Plot predicted closing prices
                    prediction_dates = last_3_days_data.index[-len(predictions):]  # Use only the dates corresponding to predictions
                    ax.plot(prediction_dates, predictions, label='Predicted Closing Prices', color='red', linestyle='--', marker='o')

                    
                    ax.legend()
                    ax.set_title(f"Stock Prices for {symbol}")
                    ax.set_xlabel("Timestamp")
                    ax.set_ylabel("Prices")

                    
                    plt.tight_layout()
                    plt.show()


                    predictions_text = f"Predictions for the last 3 days:\n{pd.DataFrame({'Predicted Closing Prices': predictions}, index=last_3_days_data.index)}"
                    self.result_label.config(text=f"{result_text}\n\n{predictions_text}")
                else:
                    raise ValueError("Model is not trained. Please check the training process.")


            else:
                raise ValueError("Error fetching stock data. Please check the symbol and try again.")
        except Exception as e:
            self.result_label.config(text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()
