import turicreate as tc


def toSframe(dataFolder:str):
    """
    This function 
    """
    # Load images from folder
    data = tc.image_analysis.load_images(dataFolder, with_path=True)

    # Create label column based on folder name
    data['sneaker_name'] = data['path'].apply(lambda path: os.path.basename(os.path.dirname(path)))

    # Save as .sframe
    data.save('img.sframe')



def sneakerClassifier(model:str, sneakerImg:str):
    """
    This function 
    """
    model = tc.load_model(model)
    img = tc.Image(sneakerImg)
    prediction = model.predict(img)

    return prediction



model = 'turi.model'
img = './sneakers/jordan_eleven/shoe.png'

print(sneakerClassifier(model, img))


