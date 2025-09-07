import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image

# ----------------- CONFIG -----------------
API_KEY = "AIzaSyDJW0OgqgRdxsN5d1Aeg2PKQkQ63qGI5b4"  # Your Gemini API key
WIDTH, HEIGHT = 1280, 720

# ----------------- FUNCTIONS -----------------
def find_working_camera(max_index=5):
    """Find the first working camera index."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

def getHandsInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if not paused:  # Only draw if not paused
        if fingers == [0, 1, 0, 0, 0]:  # Index finger up
            current_pos = lmList[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, prev_pos, current_pos, (0, 255, 0), 10)
            prev_pos = current_pos
        elif fingers == [1, 1, 1, 1, 1]:  # All fingers up
            canvas = np.zeros_like(canvas)
    return current_pos, canvas

def sendToAI(model, canvas, fingers, task):
    # AI should only work if NOT paused
    if not paused and fingers == [0, 0, 0, 0, 1]:  # Only pinky up
        pil_image = Image.fromarray(canvas)
        response = model.generate_content([task, pil_image])
        return response.text
    return None

# ----------------- SETUP -----------------
# Find camera
camera_index = find_working_camera()
if camera_index is None:
    print("No working camera found!")
    exit()

# AI setup
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Camera
cap = cv2.VideoCapture(camera_index)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1,
                        detectionCon=0.7, minTrackCon=0.5)

# State variables
prev_pos = None
canvas = None
output_text = ""
task = "Guess the drawing"  # Default task
paused = False  # New flag for pause/resume

print("Press '1' for Guess the Drawing, '2' for Solve the Math Problem")
print("Raise index finger to draw, all fingers to clear, pinky to send to AI")
print("Two fingers up = Pause | Three fingers up = Resume")
print("Press 'q' to quit")

# ----------------- MAIN LOOP -----------------
while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        break

    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandsInfo(img)
    if info:
        fingers, lmList = info

        # Check for pause/resume gestures
        if fingers == [0, 1, 1, 0, 0]:  # Two fingers up
            paused = True
        elif fingers == [1, 1, 1, 0, 0]:  # Three fingers up
            paused = False

        prev_pos, canvas = draw(info, prev_pos, canvas)
        ai_output = sendToAI(model, canvas, fingers, task)
        if ai_output:
            output_text = ai_output

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    # Show task, pause state, and AI output
    cv2.putText(image_combined, f"Task: {task}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if paused:
        cv2.putText(image_combined, "PAUSED", (WIDTH - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    if output_text:
        cv2.putText(image_combined, output_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Virtual Drawing", image_combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        task = "Guess the drawing"
    elif key == ord('2'):
        task = "Solve the math problem"

cap.release()
cv2.destroyAllWindows()
