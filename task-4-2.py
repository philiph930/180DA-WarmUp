# References:
# - https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# This code applies the KMeans algorithm to a specific rectangle in the webcam frame, and displays the dominant color (swatch and text) in the upper left corner

import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

# Run KMeans with 5 clusters to find dominant color
def get_dominant_color(frame, k_val = 3):
    pixels = np.float32(frame.reshape(-1, 3))
    
    # Define KMeans model and run it on the image frame
    model = KMeans(n_clusters=k_val, n_init=10)
    model.fit(pixels)
    
    counts = np.bincount(model.labels_)
    index = np.argmax(counts)
    color = model.cluster_centers_[index]

    return color.astype(int)

def report_dom_color_in_rect():
    cap = cv.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        h, w, _ = frame.shape 
        cx, cy = w // 2, h // 2

        size = 100
        rect = frame[cy - size:cy + size, cx - size:cx + size]

        dom_color = get_dominant_color(rect)

        # draw center box
        cv.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 255, 0), 2)

        # swatch in top corner
        color_patch = tuple(map(int, dom_color))
        cv.rectangle(frame, (15, 15), (115, 60), color_patch, -1)

        # label with BGR values
        label = f"BGR: {dom_color[0]}, {dom_color[1]}, {dom_color[2]}"
        cv.putText(frame, label, (15, 80), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # show it
        cv.imshow('color check', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    report_dom_color_in_rect()

    
