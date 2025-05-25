import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
model = "GRU" # name the file you wish to display (ANY OF: SRNN, LSTM, GRU)
loss_data = pd.read_csv(f'{model} loss.csv') 

loss_data['Epoch'] = range(1, len(loss_data) + 1)

# Plot Loss
plt.figure(figsize=(12, 6))
plt.plot(loss_data['Epoch'], loss_data['loss'], label='Training Loss')
plt.plot(loss_data['Epoch'], loss_data['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss over Epochs for {model}')
plt.legend()
plt.grid(True)
plt.show()

# Plot MAPE
plt.figure(figsize=(12, 6))
plt.plot(loss_data['Epoch'], loss_data['mape'], label='Training MAPE')
plt.plot(loss_data['Epoch'], loss_data['val_mape'], label='Validation MAPE')
plt.xlabel('Epoch')
plt.ylabel('MAPE')
plt.title('Training and Validation MAPE over Epochs')
plt.legend()
plt.grid(True)
plt.show()
