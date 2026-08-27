"""
Example showing the CORRECT way to save a threshold curve using save_chart
"""

# CORRECT ORDER:
# 1. Create figure and plot data
# 2. Save the figure (while it's still in memory)
# 3. Optionally show the figure
# 4. Close the figure to free memory

# Import the save_chart function
from notebook_helper import save_chart
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example data (similar to what you'd have in your threshold curve calculation)
# In your actual code, you'd compute df_curve from your Splink predictions
thresholds = np.linspace(0.0, 1.0, 101)
precision = 0.5 + 0.4 * (1 - np.exp(-5 * thresholds))  # Example precision curve
recall = np.exp(-3 * thresholds)  # Example recall curve
f1_score = 2 * precision * recall / (precision + recall + 1e-8)  # Example F1 curve

# Create a DataFrame like your df_curve
df_curve = pd.DataFrame({
    'threshold': thresholds,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score
})

# NOW CREATE AND SAVE THE FIGURE IN THE CORRECT ORDER:

# 1. Create the figure and plot the data
plt.figure(figsize=(10, 6))
plt.plot(df_curve["threshold"], df_curve["precision"], label="Precision", color="blue", lw=2)
plt.plot(df_curve["threshold"], df_curve["recall"], label="Recall", color="orange", lw=2)
plt.plot(df_curve["threshold"], df_curve["f1_score"], label="F1-Score", color="green", linestyle="--")

plt.title("Custom Model Evaluation Curve (By Match Probability)")
plt.xlabel("Match Probability Threshold")
plt.ylabel("Score")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)

# 2. SAVE THE FIGURE FIRST (while it's still in memory and properly rendered)
# This is the key step - save BEFORE calling plt.show()
save_chart(plt.gcf(), "splink_local_demo", "threshold_curve")
print("Figure saved successfully!")

# 3. Optionally show the figure (after saving)
plt.show()

# 4. Close the figure to free memory
plt.close()

# Alternative: If you don't need to show the figure, you can skip plt.show()
# Just create, save, and close:
"""
plt.figure(figsize=(10, 6))
# ... plotting code ...
save_chart(plt.gcf(), "splink_local_demo", "threshold_curve")
plt.close()
"""

# Key points:
# - NEVER call plt.show() before saving if you want to save the figure
# - plt.show() can render and clear the figure in some backends
# - Always save the figure immediately after creating it, before showing it
# - plt.close() is important to free memory, especially in loops