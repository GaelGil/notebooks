import turicreate as tc



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

# img1 = './sneakers/jordan_eleven/shoe1.png'
# img2 = './sneakers/jordan_eleven/shoe2.png'
# img3 = './sneakers/jordan_eleven/shoe3.png'
# img4 = './sneakers/jordan_one/shoe1.png'
# img5 = './sneakers/jordan_one/shoe2.png'

jordan_one_blue = './jordan_one_blue.png'
jordan_one_brown = './jordan_one_brown.jpg'
jordan_one_pink = './jordan_one_pink.jpg'

# sneakerClassifier(model, img1)
# sneakerClassifier(model, img2)
# sneakerClassifier(model, img3)
# sneakerClassifier(model, img4)
# sneakerClassifier(model, img5)
sneakerClassifier(model, jordan_one_blue)
sneakerClassifier(model, jordan_one_pink)
sneakerClassifier(model,jordan_one_brown)




