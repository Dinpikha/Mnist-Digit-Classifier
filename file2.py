import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

import pickle


with open("history.pkl", "rb") as f:
    history_dict = pickle.load(f)

# print(history_dict)
print(history_dict['accuracy'],history_dict['val_accuracy'])
plt.figure(figsize=(7,7))
plt.plot(history_dict['accuracy'],label='Training accuracy',color='pink',linewidth=2)
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
plt.legend()
plt.show()
