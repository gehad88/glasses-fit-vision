import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
import os

# ==========================================
# 1. إعداد الموديل والتحميل
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

MODEL_PATH = r"C:\Users\Gehad Omar\Desktop\CV Project\Face-Shape\best_face_shape_model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
model = FaceShapeResNet(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==========================================
# 2. وظائف التلوين ومعالجة الصور
# ==========================================

def apply_color_tint(img, color_bgr):
    """تغيير لون النظارة مع الحفاظ على الشفافية والتفاصيل"""
    if img is None: return None
    # فصل القنوات
    b, g, r, a = cv2.split(img)
    # حساب شدة الإضاءة للحفاظ على اللمعان (Grayscale)
    gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
    # تحويل اللون المختار ليكون متناسباً مع الإضاءة
    target_b = (gray / 255.0) * color_bgr[0]
    target_g = (gray / 255.0) * color_bgr[1]
    target_r = (gray / 255.0) * color_bgr[2]
    
    return cv2.merge([target_b.astype(np.uint8), target_g.astype(np.uint8), target_r.astype(np.uint8), a])

def crop_to_content(img):
    if img is None: return None
    if img.shape[2] == 3: # إضافة قناة شفافية لو مش موجودة
        b, g, r = cv2.split(img)
        a = np.ones(b.shape, dtype=b.dtype) * 255
        img = cv2.merge([b, g, r, a])
    alpha = img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None: return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def overlay_transparent_v3(background, overlay, x, y, target_width, angle=0):
    overlay = crop_to_content(overlay)
    if overlay is None: return background
    aspect_ratio = overlay.shape[0] / overlay.shape[1]
    target_height = int(target_width * aspect_ratio)
    overlay = cv2.resize(overlay, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if angle != 0:
        h, w = overlay.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        overlay = cv2.warpAffine(overlay, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    h, w, _ = overlay.shape
    rows, cols, _ = background.shape
    start_x, start_y = x - (w // 2), y - (h // 2)
    if start_y + h > rows or start_x + w > cols or start_x < 0 or start_y < 0: return background
    overlay_img, mask = overlay[:, :, :3], overlay[:, :, 3] / 255.0
    roi = background[start_y:start_y+h, start_x:start_x+w]
    for c in range(3):
        roi[:, :, c] = (1.0 - mask) * roi[:, :, c] + mask * overlay_img[:, :, c]
    background[start_y:start_y+h, start_x:start_x+w] = roi
    return background

# ==========================================
# 3. التحكم بالماوس والألوان
# ==========================================
current_idx = 0
# الألوان المتاحة (BGR): أحمر، أخضر، أزرق، أسود، ذهبي
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (30, 30, 30), (0, 215, 255)]
selected_color = (30, 30, 30) # الافتراضي أسود
btn_clicked = False
click_type = ""

def handle_mouse(event, x, y, flags, param):
    global current_idx, btn_clicked, click_type, selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        # فحص الضغط على شريط الألوان (Color Bar) في أسفل الشاشة
        if 40 <= y <= 80:
            for i, col in enumerate(colors):
                if 20 + i*60 <= x <= 70 + i*60:
                    selected_color = col
                    btn_clicked = True
                    click_type = "color"
        # أزرار التنقل
        if 200 <= y <= 260:
            if 20 <= x <= 80: # زر يسار
                click_type = "prev"; btn_clicked = True
            elif 560 <= x <= 620: # زر يمين
                click_type = "next"; btn_clicked = True

# ==========================================
# 4. تشغيل النظام
# ==========================================
cap = cv2.VideoCapture(0)
cv2.namedWindow('AI Smart Try-On')
cv2.setMouseCallback('AI Smart Try-On', handle_mouse)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

predicted_shape, original_glasses = None, None
glasses_folder = r"C:\Users\Gehad Omar\Desktop\CV Project\Face-Shape\glasses_library"
all_glasses = [f for f in os.listdir(glasses_folder) if f.lower().endswith(('.png', '.jpg'))]

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h_f, w_f = frame.shape[:2]

    if predicted_shape is None:
        rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            output = model(transform(rgb_pil).unsqueeze(0).to(device))
            predicted_shape = class_names[torch.argmax(output).item()]
        
        current_idx = 0
        original_glasses = cv2.imread(os.path.join(glasses_folder, all_glasses[current_idx]), cv2.IMREAD_UNCHANGED)

    # معالجة تغيير النظارة أو اللون
    if btn_clicked:
        if click_type != "color":
            if click_type == "next": current_idx = (current_idx + 1) % len(all_glasses)
            else: current_idx = (current_idx - 1) % len(all_glasses)
            original_glasses = cv2.imread(os.path.join(glasses_folder, all_glasses[current_idx]), cv2.IMREAD_UNCHANGED)
        
        # تطبيق اللون فوراً
        current_glasses = apply_color_tint(original_glasses, selected_color)
        btn_clicked = False
    else:
        current_glasses = apply_color_tint(original_glasses, selected_color)

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            lx, ly = landmarks.landmark[33].x * w_f, landmarks.landmark[33].y * h_f
            rx, ry = landmarks.landmark[263].x * w_f, landmarks.landmark[263].y * h_f
            angle = np.degrees(np.arctan2(ry - ly, rx - lx))
            eye_dist = np.sqrt((rx - lx)**2 + (ry - ly)**2)
            cx, cy = int(landmarks.landmark[6].x * w_f), int(landmarks.landmark[6].y * h_f)
            frame = overlay_transparent_v3(frame, current_glasses, cx, cy, int(eye_dist*2.1), angle)

    # --- واجهة المستخدم (UI) ---
    # رسم شريط الألوان
    for i, col in enumerate(colors):
        cv2.rectangle(frame, (20 + i*60, 40), (70 + i*60, 80), col, -1)
        if selected_color == col: # تمييز اللون المختار
            cv2.rectangle(frame, (20 + i*60, 40), (70 + i*60, 80), (255,255,255), 2)

    # أزرار التنقل
    cv2.rectangle(frame, (20, 200), (80, 260), (50,50,50), -1)
    cv2.putText(frame, "<", (35, 245), 2, 1.2, (255,255,255), 2)
    cv2.rectangle(frame, (w_f-80, 200), (w_f-20, 260), (50,50,50), -1)
    cv2.putText(frame, ">", (w_f-65, 245), 2, 1.2, (255,255,255), 2)

    cv2.putText(frame, f"Face: {predicted_shape}", (w_f//2 - 80, 30), 1, 1.2, (0, 255, 0), 2)
    cv2.imshow('AI Smart Try-On', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()