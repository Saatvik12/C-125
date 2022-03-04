import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
X,Y = fetch_openml('mnist_784')
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=7500,test_size=2500,random_state=9)
X_train_scale=X_train\255
X_test_scale=X_test\255
clf=LogisticRegression(solver='saga',multi_class='multinomial')
clf.fit(X_train_scale,Y_train)
def get_prediction(image):
    im_pil=Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]