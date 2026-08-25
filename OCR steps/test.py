import cv2
import numpy as np

def segment_exam_ink(image_path):
    # 1. Load the original exam paper image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Exam image file could not be loaded.")

    # 2. Convert from BGR to HSV color space for accurate color tracking
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Define the HSV color range for Red Ink (Teacher's Marks)
    # Red wraps around the HSV spectrum, so we need two ranges combined
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 4. Extract Isolate Teacher Marks (Red Layer Only)
    teacher_marks = cv2.bitwise_and(img, img, mask=red_mask)
    # Turn the background from black to clean white for the OCR engine
    teacher_marks[red_mask == 0] = [255, 255, 255]

    # 5. Extract Student Text (Remove Red Ink)
    # Invert the red mask to select everything EXCEPT red ink
    not_red_mask = cv2.bitwise_not(red_mask)
    student_text = cv2.bitwise_and(img, img, mask=not_red_mask)
    # Turn the removed red ink areas into white pixels so it looks like blank paper
    student_text[red_mask > 0] = [255, 255, 255]

    # 6. Save the output files for the next AI stage
    cv2.imwrite("isolated_teacher_marks.jpg", teacher_marks)
    cv2.imwrite("clean_student_text.jpg", student_text)
    print("Success: Image split into 'clean_student_text.jpg' and 'isolated_teacher_marks.jpg'")

# Run the color pipeline
segment_exam_ink(r"d:/Coded files/python/Webscrapping/image/images.jpeg")
