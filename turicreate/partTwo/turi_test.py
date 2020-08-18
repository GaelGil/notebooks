import turicreate as tc

model = tc.load_model('turi.model')

data = tc.SFrame('sneakers.sframe')

predictions = model.predict(data)

print(data)
print(predictions)
metrics = model.evaluate(data)
print(metrics['accuracy'])

