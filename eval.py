import numpy as np
from tensorflow.keras.models import load_model

print("Loading data...",flush=True)
X = np.load('processed_data_euroc/X.npy', mmap_mode='r')
y = np.load('processed_data_euroc/y.npy', mmap_mode='r')
print("Data loaded",flush=True)

print("Loading model...",flush=True)
model = load_model('models/lstm_euroc_spec.h5')
print("Model loaded",flush=True)

batch_size = 5000
total = len(X)
print("Total samples:", total)

acc_list = []
loss_list = []

for i in range(0, total, batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]

    loss, acc = model.evaluate(X_batch, y_batch, verbose=0)

    acc_list.append(acc)
    loss_list.append(loss)

    print(f"Processed {min(i+batch_size, total)}/{total}")

print("Final Accuracy:", np.mean(acc_list))
print("Final Loss:", np.mean(loss_list))
