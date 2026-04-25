#Import library
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
#-------------load data----------------#
df=pd.read_csv(r"C:\Users\rafaa\Downloads\heart_model\heart (1).csv")
print(df.head())
print(df.shape)
print(df.describe())

"""After we understand our data we need to prepare Layers and the feaure of our model"""
#--------------split data-----------------#
X=df.drop(columns='target')
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print()
print(y_train.shape)
print(y_test.shape)

#Scale the data for better prediction
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.fit_transform(X_test)

#Build model
"""Regularization: Add a Dropout Layer (e.g., rate = 0.2).
This randomly "turns off" 20% of neurons during training,
forcing the model to be more robust and prevent memorization of the training data."""
model=models.Sequential([
    # Input + Hidden Layer 1
    
    layers.Dense(32, activation='relu', input_shape=(X_train_sc.shape[1],)),
    layers.Dropout(0.2),
    
    # Hidden Layer 2
    layers.Dense(16, activation='relu'),
    
    # Output Layer
    layers.Dense(1, activation='sigmoid')
])

# 3. Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train with Early Stopping
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) #Prevent overfitting

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

#Model summary
model.summary()

#Plot DL model processing
hist=history.history
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(1,2,1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()