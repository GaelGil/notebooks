import turicreate as tc
import os


def sneakerClassifier(model:str, sneakerImg:str):
    """
    This function 
    """
    model = tc.load_model(model)
    img = tc.Image(sneakerImg)
    prediction = model.predict(img)
    print(f'{sneakerImg}: {prediction}')
    print(" ")

    return prediction



model = 'sneaker.model'




for file in os.listdir("./testImages"):
    img = (f'./testImages/{file}')
    sneakerClassifier(model, img)

