import turicreate as tc

# 1. Load the data
data = tc.SFrame('turi.sframe')

# 2. Split to train and test data
train_data, test_data = data.random_split(0.8)

# 3. Create model
model = tc.image_classifier.create(train_data, target='sneaker_name')

# 4. Predictions
predictions = model.predict(test_data)

# 5. Evaluate the model and show metrics
metrics = model.evaluate(test_data)
print(metrics['accuracy'])

# 6. Save the model
model.save('turiExample.model')

# 7. Export to CoreML format
model.export_coreml('model/TuriCreate.mlmodel')