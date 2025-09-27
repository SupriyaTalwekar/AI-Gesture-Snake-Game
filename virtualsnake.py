import cv2
import mediapipe as mp
import time
import random
import numpy as np

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

# Set up the game window
w, h = 640, 480
win = np.zeros((h, w, 3), dtype=np.uint8)

# Snake properties
snake = [(300, 240)]
snake_dir = (20, 0)
snake_len = 1

# Food position
food = (random.randint(20, w - 20), random.randint(20, h - 20))

# Score
score = 0

# Detect direction from finger position
def get_direction(x, y, center_x, center_y):
    dx = x - center_x
    dy = y - center_y
    if abs(dx) > abs(dy):
        return (20, 0) if dx > 0 else (-20, 0)
    else:
        return (0, 20) if dy > 0 else (0, -20)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            lm_list = lm.landmark

            # Get tip of index finger
            x = int(lm_list[8].x * w)
            y = int(lm_list[8].y * h)
            center_x, center_y = w // 2, h // 2
            snake_dir = get_direction(x, y, center_x, center_y)

    # Move the snake
    new_head = (snake[-1][0] + snake_dir[0], snake[-1][1] + snake_dir[1])
    snake.append(new_head)
    if len(snake) > snake_len:
        del snake[0]

    # Check food collision
    if abs(new_head[0] - food[0]) < 20 and abs(new_head[1] - food[1]) < 20:
        snake_len += 5
        score += 1
        food = (random.randint(20, w - 20), random.randint(20, h - 20))

    # Draw game window
    win[:] = (0, 0, 0)
    for pt in snake:
        cv2.rectangle(win, (pt[0], pt[1]), (pt[0]+10, pt[1]+10), (0, 255, 0), -1)
    cv2.circle(win, food, 10, (0, 0, 255), -1)
    cv2.putText(win, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Snake Game", win)
    if cv2.waitKey(200) & 0xFF == ord('q'):  # Slowed down the snake (200ms per frame)
        break

cap.release()
cv2.destroyAllWindows()
