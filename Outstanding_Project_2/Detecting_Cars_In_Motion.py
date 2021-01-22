import cv2


# Our Image
img_file = 'CarImage.jpg'
video = cv2.VideoCapture('Tesla Autopilot Dashcam.mp4')

# Our pre-trained car classifier
classifier_file = 'cars.xml'

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#Infinite loop,until while made false
while True:

    #Read the current frame
    (read_successful, frame) = video.read()

    #Safe Coding
    if read_successful:
        #Must convert to greyscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect Cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    # Draw Rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #Display the image with the cars
    cv2.imshow('Car Detector Model', frame)

    #Adding Gaussian Blur 
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    #Displays the output until a key is pressed to exit
    cv2.waitKey(1)

print("Code Completed")

