## 🚧 Future Improvements

### **1. Full Capital Backtesting**
- Currently, performance is only measured during open positions.  
- To do: Add a continuous capital simulation (even when not in a trade) to allow accurate calculation of **Sharpe Ratio**, **CAGR**, **Drawdowns**, etc.

### **2. Benchmark Comparison (Buy & Hold Strategy)**
- Currently, there is no benchmark to compare against.  
- To do: Compare the strategy to a passive investment in **AAPL** or an index like the **S&P 500**, to assess whether it truly outperforms the market.

### **3. Generalization to Other Tickers and Time Periods**
- Currently, the code only works for AAPL and a fixed date range (2021-2025).  
- To do: Make the code **modular** and flexible to test different stocks and historical ranges, in order to validate the **robustness** and **adaptability** of the strategy.
