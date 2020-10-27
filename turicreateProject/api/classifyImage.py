import turicreate as tc
import os


def imgClassifier(model='moreSneaker.Model', sneakerImg:str):
    """
    This function uses a model to predict
    """
    model = tc.load_model(model)
    img = tc.Image(sneakerImg)
    prediction = model.predict(img)
    # print(f'{sneakerImg}: {prediction}')
    # print(" ")

    return prediction




