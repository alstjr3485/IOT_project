import cv2
import numpy as np
from keras.models import load_model

list1 = [0]
i = 0

list2 = [0]
j = 0

sumlist = 0

# Load the model
model = load_model('keras_model.h5')

# CAMERA can be 0 or 1 based on default camera of your computer.
camera = cv2.VideoCapture(2)

# Grab the labels from the labels.txt file. This will be used later.
labels = open('labels.txt', 'r').readlines()

while True:
    # Grab the webcameras image.
    ret, image = camera.read()
    
    # Resize the raw image into (224-height,224-width) pixels.
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Show the image in a window
    cv2.imshow('Webcam Image', image)
    
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    
    # Normalize the image array
    image = (image / 127.5) - 1
    
    # Have the model predict what the current image is. Model.predict
    # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
    # it is the first label and 80% sure its the second label.
    probabilities = model.predict(image)
    
    # Print what the highest value probabilitie label
    idx = np.argmax(probabilities)
    print(labels[idx])
    
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break
    elif keyboard_input == 48:
        list1.insert(i, labels[idx][:-1])
        i += 1
        if labels[idx] == '0 Garlic\n':
            list2.insert(j, 5980)
        elif labels[idx] == '1 Potato\n':
            list2.insert(j, 490)
        elif labels[idx] == '2 Egg\n':
            list2.insert(j, 3750)
        elif labels[idx] == '3 Onion\n':
            list2.insert(j, 2490)
        elif labels[idx] == '4 Chicken\n':
            list2.insert(j, 6900)
        elif labels[idx] == '5 Pork\n':
            list2.insert(j, 8700)
        elif labels[idx] == '6 Welsh_onion\n':
            list2.insert(j, 1590)
        elif labels[idx] == '7 Been_sprouts\n':
            list2.insert(j, 1500)
        j += 1

camera.release()
cv2.destroyAllWindows()

list1.pop()
list2.pop()

for k in range(0,len(list2)):
    sumlist += list2[k]
    
print("구매한 항목 :", list1)
print("개당 가격 :",list2)
print("합계 : " + str(sumlist) + "원 입니다.")
