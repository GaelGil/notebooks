import turicreate as tc


def trainAndSaveModel():
    """
    """
    # 1. Load the data
    data = tc.SFrame('sneaker.sframe')

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
    model.save('sneaker.model')


trainAndSaveModel()