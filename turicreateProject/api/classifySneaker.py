import turicreate as tc
import os


def classify(model='moreSneaker.Model', sneakerImg):
    """
    This function uses a model to predict
    """
    model = tc.load_model(model)
    img = tc.Image(sneakerImg)
    prediction = model.predict(img)


    return prediction




