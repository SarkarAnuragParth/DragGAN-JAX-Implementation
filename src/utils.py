import cv2


def get_drag_points(image_input):
    """
    Displays an image and allows the user to select a handle point and a target point.
    
    Args:
        image_input (str or numpy.ndarray): Path to the image file, or an already loaded image array.
        
    Returns:
        tuple: ((handle_x, handle_y), (target_x, target_y)) or None if selection was incomplete.
    """
    
    # Load the image if a path is provided, otherwise copy the array
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not load image from {image_input}")
    else:
        img = image_input.copy()

    # We use a clone so we can draw on it without modifying the original data
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    display_img = img.copy()
    selected_points = []
    
    # Define the mouse callback function
    def select_point(event, x, y, flags, param):
        # Trigger on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_points) == 0:
                # First click: Handle Point (Red)
                selected_points.append((x, y))
                cv2.circle(display_img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.putText(display_img, "Handle", (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Select Points", display_img)
                
            elif len(selected_points) == 1:
                # Second click: Target Point (Green)
                selected_points.append((x, y))
                handle_pt = selected_points[0]
                target_pt = selected_points[1]
                
                # Draw the target point and an arrow connecting them
                cv2.circle(display_img, target_pt, radius=5, color=(0, 255, 0), thickness=-1)
                cv2.putText(display_img, "Target", (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.arrowedLine(display_img, handle_pt, target_pt, color=(255, 0, 0), thickness=2)
                cv2.imshow("Select Points", display_img)

    # Setup the GUI window
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", select_point)

    print("GUI Opened.")
    print("1. Click once to select the Handle point.")
    print("2. Click again to select the Target point.")
    print("3. Press ANY KEY on your keyboard to close the window and return the points.")

    # Show the image and wait for a keyboard press to exit
    cv2.imshow("Select Points", display_img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    # Return the points if both were selected
    if len(selected_points) == 2:
        print(f"Recorded - Handle: {selected_points[0]}, Target: {selected_points[1]}")
        return selected_points[0], selected_points[1]
    else:
        print("Selection incomplete. Returning None.")
        return None

