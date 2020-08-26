import turicreate as tc
import os

# 1. Load images
data = tc.image_analysis.load_images('sneakers', with_path=True)

# 2. Create label column based on folder name
data['sneaker_name'] = data['path'].apply(lambda path: os.path.basename(os.path.dirname(path)))

# 3. Save as .sframe
data.save('sneakers.sframe')

# 4. Explore
data.explore()