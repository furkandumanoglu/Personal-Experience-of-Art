import cv2
import json

image = cv2.imread("/Users/furkandumanoglu/Desktop/PEA/babylon.jpg")

if image is None:
    print("❌ Image not found")
    exit()
    
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"📍 ({x}, {y})")

        points.append({"x": x, "y": y})

        # Görsel üzerine nokta çiz
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

print("👉 Ellerin olduğu noktalara tıkla, çıkmak için ESC")

while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

# JSON olarak kaydet
with open("hand_points.json", "w") as f:
    json.dump(points, f, indent=4)

print("✅ Koordinatlar kaydedildi: hand_points.json")