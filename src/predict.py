import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

labels = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

model = load_model("../rice_cnn.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = labels[np.argmax(predictions)]
    return predicted_class, predictions

if __name__ == "__main__":
    test_img = "../data/Rice_Image_Dataset/Basmati/1.jpg"  # update with your test image
    label, probs = predict_image(test_img)
    print(f"Prediction: {label}")
    print(f"Probabilities: {probs}")
