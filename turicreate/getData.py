import turicreate as tc
import os


def loadDataInSframe():
    """
    """
    # 1. Load images
    data = tc.image_analysis.load_images('dataset', with_path=True)

    # 2. Create label column based on folder name
    data['sneaker_name'] = data['path'].apply(lambda path: os.path.basename(os.path.dirname(path)))

    # 3. Save as .sframe
    data.save('sneaker.sframe')



loadDataInSframe()