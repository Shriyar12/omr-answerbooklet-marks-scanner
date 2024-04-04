import cv2
import numpy as np

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow('edges',edges)
    return edges

def find_grid_corners(edges):
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Approximate contours to polygons
    polygons = [cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True) for cnt in contours]
    
    # Find the largest polygon (presumably the grid)
    grid_polygon = max(polygons, key=cv2.contourArea)
    
    # Find the corners of the grid
    corners = np.squeeze(grid_polygon)
    
    # Sort corners by their x+y coordinates
    corners = corners[np.argsort(corners.sum(axis=1))]
    
    # Reorder corners to [top-left, top-right, bottom-right, bottom-left]
    if corners[0][1] > corners[1][1]:
        corners[0], corners[1] = corners[1], corners[0]
    if corners[2][1] < corners[3][1]:
        corners[2], corners[3] = corners[3], corners[2]
    
    return corners

def transform_perspective(img, corners):
    # Define the destination points for perspective transformation
    width = max(np.linalg.norm(corners[0] - corners[1]), np.linalg.norm(corners[2] - corners[3]))
    height = max(np.linalg.norm(corners[0] - corners[3]), np.linalg.norm(corners[1] - corners[2]))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    
    # Convert corners and dst to float32
    corners = corners.astype(np.float32)
    dst = dst.astype(np.float32)
    
    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(corners, dst)
    cv2.imshow('affjgfkgfjg',M)
    # Apply the perspective transformation
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))
    
    return warped

def main(image_path):
    # Preprocess the image
    edges = preprocess_image(image_path)
    
    # Find corners of the grid
    corners = find_grid_corners(edges)
    
    # Load original image
    img = cv2.imread(image_path)
    
    # Transform perspective to bird's-eye view
    warped = transform_perspective(img, corners)
    
    # Resize the transformed image to a smaller size
    scale_percent = 50  # Adjust this value to change the scale (50% smaller in this example)
    width = int(warped.shape[1] * scale_percent / 100)
    height = int(warped.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_warped = cv2.resize(warped, dim, interpolation=cv2.INTER_AREA)
    
    # Display original and transformed images
    cv2.imshow("Original Image", img)
    cv2.imshow("Bird's-eye View (Resized)", resized_warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "gat2.jpg"
# Test the main function with an image
if __name__ == "__main__":
    
    main(image_path)
