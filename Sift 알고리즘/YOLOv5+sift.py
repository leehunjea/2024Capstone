import cv2
import torch
import torchvision.transforms as transforms

# 사전 훈련된 객체 탐지 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지 전처리 함수
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(640),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 두 이미지 로드
img1 = cv2.imread('Test5.jpg')
img2 = cv2.imread('Test4.jpg')

# 객체 탐지 및 변화 영역 마스크 생성
results1 = model(preprocess(img1).unsqueeze(0))
results2 = model(preprocess(img2).unsqueeze(0))

masks1 = results1.render()[0]
masks2 = results2.render()[0]

# 변화 영역 시각화
diff = cv2.absdiff(img1, img2)
diff[masks1 == 0] = 0  # 이전 이미지의 객체 영역 제외
diff[masks2 != 0] = 0  # 새로운 이미지의 객체 영역만 유지

# 결과 출력
cv2.imshow('Change Detection', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()