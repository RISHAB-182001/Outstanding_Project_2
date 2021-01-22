import cv2

# Our Image
img_file = 'CarImage.jpg'

# Our pre-trained car classifier
classifier_file = 'cars.xml'


# create opencv image
# Reads in the pixel data of the image file into a multi-dimensional array,so that every pixel has its own data
img = cv2.imread(img_file)

#Convert to Grayscale(Needed for HAAR Cascade,to make codes simpler and detection easier)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create a Car Classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect Cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw Rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)



#Display the image with the cars
cv2.imshow("Rishab's Car Detector Model", black_n_white)

#Displays the output until a key is pressed to exit
cv2.waitKey()

print("Code Completed")
