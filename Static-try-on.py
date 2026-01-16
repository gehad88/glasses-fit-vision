import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import os

# ==========================================
# 1. إعداد الموديل (ResNet50)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceShapeResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(FaceShapeResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, 512),
            nn.ReLU(), nn.BatchNorm1d(512),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.resnet(x)

# تحميل الموديل
MODEL_PATH = r"C:\Users\Gehad Omar\Desktop\CV Project\Face-Shape\best_face_shape_model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
model = FaceShapeResNet(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==========================================
# 2. معايير التحجيم لكل نوع نظارة
# ==========================================
GLASSES_SCALE_PARAMS = {
    "aviator.png": {"width_multiplier": 1.9, "vertical_offset": 0},
    "wayfarer.png": {"width_multiplier": 1.7, "vertical_offset": -5},
    "round.png": {"width_multiplier": 1.7, "vertical_offset": 0},
    "rectangle.png": {"width_multiplier": 1, "vertical_offset": -3},
    "cat.png": {"width_multiplier": 1, "vertical_offset": -2}
}

# ==========================================
# 3. وظيفة التركيب المحسّنة (تتعامل مع الصور RGB و RGBA)
# ==========================================
def overlay_transparent(background, overlay, x, y, size=None):
    if size:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    
    # التأكد من وجود قناة الشفافية
    if overlay.shape[2] < 4:
        return background

    h, w, _ = overlay.shape
    rows, cols, _ = background.shape
    
    if y + h > rows or x + w > cols or x < 0 or y < 0: 
        return background

    overlay_img, mask = overlay[:, :, :3], overlay[:, :, 3] / 255.0
    
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (1.0 - mask) * background[y:y+h, x:x+w, c] + mask * overlay_img[:, :, c]
    return background

# ==========================================
# 4. معالجة الصورة مع التحجيم الديناميكي
# ==========================================
def process_static_image(person_img_path, glasses_dir):
    if not os.path.exists(person_img_path):
        print(f"Error: Image not found at {person_img_path}")
        return

    # أ. التوقع باستخدام الموديل
    img_raw = Image.open(person_img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_raw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_shape = class_names[torch.argmax(model(input_tensor)).item()]
    
    # ب. اختيار النظارة الموصى بها
    recommendations = {
        "Heart": "aviator.png", "Oval": "wayfarer.png", 
        "Round": "rectangle.png", "Square": "round.png", "Oblong": "round.png"
    }
    target_glasses = recommendations.get(predicted_shape, "wayfarer.png")
    
    img_cv = cv2.imread(person_img_path)
    glasses_path = os.path.join(glasses_dir, target_glasses)
    img_gl = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
    
    if img_gl is None:
        print(f"Warning: Glasses image {target_glasses} not found. Using first available.")
        img_gl = cv2.imread(os.path.join(glasses_dir, os.listdir(glasses_dir)[0]), cv2.IMREAD_UNCHANGED)

    scale_params = GLASSES_SCALE_PARAMS.get(target_glasses, {"width_multiplier": 2.2, "vertical_offset": 0})
    
    # ج. تحديد الملامح والتركيب
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = img_cv.shape[:2]
            
            lx, ly = landmarks.landmark[33].x * w, landmarks.landmark[33].y * h
            rx, ry = landmarks.landmark[263].x * w, landmarks.landmark[263].y * h
            cx, cy = int(landmarks.landmark[6].x * w), int(landmarks.landmark[6].y * h)
            
            eye_dist = np.sqrt((rx - lx)**2 + (ry - ly)**2)
            glasses_width = int(eye_dist * scale_params["width_multiplier"])
            aspect_ratio = img_gl.shape[0] / img_gl.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)
            
            cy_adjusted = cy + scale_params["vertical_offset"]
            
            final_img = overlay_transparent(
                img_cv.copy(), img_gl, 
                cx - (glasses_width // 2), cy_adjusted - (glasses_height // 2), 
                (glasses_width, glasses_height)
            )
            
            # إضافة المعلومات النصية (بحجم يتناسب مع الصورة)
            font_scale = max(1, h // 1000)
            thickness = max(2, h // 500)
            cv2.putText(final_img, f"Detected: {predicted_shape}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

            # د. حل مشكلة حجم النافذة (Resizing for display)
            window_name = "Static Image Try-On"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # تسمح بتغيير حجم النافذة
            
            # حساب مقاس العرض ليكون مناسباً للشاشة (بحد أقصى 1000 بكسل عرض)
            display_w = 1000
            display_h = int(h * (display_w / w))
            cv2.resizeWindow(window_name, display_w, display_h)

            cv2.imshow(window_name, final_img)
            cv2.imwrite(f"Result_{predicted_shape}.jpg", final_img)
            
            print(f"Success! Prediction: {predicted_shape}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No face detected!")

# ==========================================
# 5. التشغيل
# ==========================================
BASE = r"C:\Users\Gehad Omar\Desktop\CV Project\Face-Shape"
process_static_image(
    person_img_path=os.path.join(BASE, "person.jpg"),
    glasses_dir=os.path.join(BASE, "glasses_library")
)