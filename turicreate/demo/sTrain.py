import turicreate as tc

# Load the data
data =  tc.SFrame('turi.sframe')

# Make a train-test split
train_data, test_data = data.random_split(0.8)

# Create the model
model = tc.image_classifier.create(train_data, target='sneaker_name')

# Save predictions to an SArray
predictions = model.predict(test_data)

# Evaluate the model and print the results
metrics = model.evaluate(test_data)
print(metrics['accuracy'])

# Save the model for later use in Turi Create
model.save('sneaker.model')


# new_cats_dogs['predictions'] = model.predict(new_cats_dogs)
new_shoes = tc.image_analysis.load_images('dataset', with_path=True)

new_shoes['predictions'] = model.predict(new_shoes)
new_shoes.explore()
# Export for use in Core ML
# model.export_coreml('MyCustomImageClassifier.mlmodel')