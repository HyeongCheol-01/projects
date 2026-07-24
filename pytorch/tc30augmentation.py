# 데이터 증강(Data Augmentation)
# 데이터 증강은 기존 학습 데이터의 양과 다양성을 인위적으로 늘려
# 딥러닝 모델의 과적합(Overfitting)을 방지하고 일반화 성능을 향상시키는 기법이다.

# Keras: ImageDataGenerator 사용
# PyTorch: torchvision.transforms 사용
#       → transform(image)를 호출할 때마다 랜덤 증강 적용

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

# 이미지 로딩
# OpenCV로 이미지 읽기. OpenCV는 이미지를 BGR 형식으로 읽음
image = cv2.imread('test_aug.jpeg')

# 이미지 파일이 없는 경우 오류 처리
if image is None:
    raise FileNotFoundError("test_aug.jpeg 파일을 찾을 수 없습니다.")

# OpenCV BGR 이미지를 RGB로 변환 : matplotlib과 PIL은 RGB 기준으로 이미지를 다룸
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# numpy 배열 이미지를 PIL 이미지로 변환
# torchvision.transforms는 보통 PIL Image를 입력으로 사용함
pil_image = Image.fromarray(image)

# 3. 원본 이미지 출력
plt.figure(figsize=(4, 4))
plt.imshow(pil_image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# 4. 데이터 증강 결과 출력 함수
def show_aug_func(pil_image, transform, n_images=4, title='Augmented Images'):
    """
    PyTorch 데이터 증강 결과를 시각화하는 함수
    pil_image : PIL.Image 원본 이미지
    transform : torchvision.transforms.Compose 적용할 데이터 증강 파이프라인
    n_images : int   생성해서 보여줄 증강 이미지 개수
    title : str      그래프 제목

    PyTorch의 transform은 호출할 때마다 랜덤 증강을 새로 적용한다.
    예: aug_image = transform(pil_image)
    위 코드를 여러 번 실행하면 같은 원본 이미지라도 매번 다른 형태의 증강 이미지가 생성될 수 있다.
    """

    fig, axs = plt.subplots(
        nrows=1, ncols=n_images, figsize=(24, 8)
    )
    fig.suptitle(title, fontsize=18)

    for i in range(n_images):
        # transform을 호출할 때마다 랜덤 증강 적용
        aug_image = transform(pil_image)

        # transform 결과가 torch.Tensor인 경우
        if isinstance(aug_image, torch.Tensor):
            # PyTorch Tensor 이미지 형식: (C, H, W)
            # matplotlib 출력 형식: (H, W, C)
            aug_image = aug_image.permute(1, 2, 0).numpy()

            aug_image = np.clip(aug_image, 0, 1)  

        axs[i].imshow(aug_image)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

# 5. 좌우 반전
horizontal_flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0)
])

show_aug_func(
    pil_image,
    horizontal_flip_transform,
    n_images=4,
    title='Random Horizontal Flip'
)

# 6. 상하 반전
vertical_flip_transform = transforms.Compose([
    transforms.RandomVerticalFlip(p=1.0)
])

show_aug_func(
    pil_image,
    vertical_flip_transform,
    n_images=4,
    title='Random Vertical Flip'
)

# 7. 회전
rotation_transform = transforms.Compose([
    transforms.RandomRotation(degrees=45)
])

show_aug_func(
    pil_image,
    rotation_transform,
    n_images=4,
    title='Random Rotation'
)

# 8. 좌우 랜덤 이동
width_shift_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=0, translate=(0.3, 0.0), fill=0
    )
])

show_aug_func(
    pil_image,
    width_shift_transform,
    n_images=4,
    title='Random Width Shift'
)

# 9. 색상 변화
color_jitter_transform = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
    )
])

show_aug_func(
    pil_image,
    color_jitter_transform,
    n_images=4,
    title='Color Jitter'
)

# 10. 여러 증강을 한 번에 적용
combined_transform = transforms.Compose([
    transforms.RandomRotation(degrees=50), # -50도 ~ +50도 범위에서 랜덤 회전

    transforms.RandomAffine( # 이동 + 확대/축소
        degrees=0, translate=(0.2, 0.0), scale=(0.5, 1.5), fill=0
    ),

    transforms.ColorJitter(brightness=(0.7, 1.3)),  # 밝기 변화
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.RandomVerticalFlip(p=0.5)  # 상하 반전
])

show_aug_func(
    pil_image,
    combined_transform,
    n_images=4,
    title='Combined Augmentation'
)

# 11. Tensor 변환까지 포함한 증강 예제
# 실제 PyTorch 학습에서는 마지막에 transforms.ToTensor()를 자주 사용한다.
# ToTensor() 역할: PIL Image 또는 numpy 이미지를 torch.Tensor로 변환
#   이미지 shape: (H, W, C) -> (C, H, W)
#   픽셀값: 0~255 -> 0~1
# 이 예제는 학습용 transform 구조를 보여주기 위한 코드이다.
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2
    ),
    transforms.ToTensor()
])

show_aug_func(
    pil_image,
    train_transform,
    n_images=4,
    title='Train Transform with ToTensor'
)

# 12. 테스트용 transform 예제
# 테스트 데이터에는 랜덤 증강을 넣지 않는다.
# 이유: 테스트 데이터까지 랜덤하게 변형하면 평가 결과가 매번 달라질 수 있고,
#   모델의 실제 성능을 안정적으로 확인하기 어렵다.
# 따라서 test_transform은 보통 다음처럼 구성한다.
#   Resize   ToTensor
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

test_tensor = test_transform(pil_image)

print('test_tensor shape:', test_tensor.shape)
print('test_tensor min:', test_tensor.min().item())
print('test_tensor max:', test_tensor.max().item())

# 13. Dataset에서 transform을 사용하는 예제
# 실제 학습에서는 이미지를 미리 모두 증강해서 저장하지 않고,
# Dataset의 __getitem__()에서 transform을 적용하는 경우가 많다.
# 이렇게 하면 학습 epoch마다 같은 이미지라도 조금씩 다른 형태로 모델에 입력될 수 있다.
class CustomImageDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset에서 transform을 적용하는 예제 클래스
    image_paths: 이미지 파일 경로 리스트
    labels: 정답 라벨 리스트
    transform: torchvision.transforms로 만든 변환/증강 파이프라인
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path) # 이미지 읽기
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)  # numpy -> PIL

        # transform 적용
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long) # label도 tensor로 변환 가능
        return image, label

# 14. Dataset 사용 예시
# 아래 코드는 실제 image_paths, labels가 있을 때 사용한다.
# 예: image_paths = ['img1.jpg', 'img2.jpg', ...]
#     labels = [0, 1, ...]
# train_dataset = CustomImageDataset(
#     image_paths=image_paths,
#     labels=labels,
#     transform=train_transform
# )
#
# test_dataset = CustomImageDataset(
#     image_paths=test_paths,
#     labels=test_labels,
#     transform=test_transform
# )

"""
from torch.utils.data import DataLoader

train_dataset = CustomImageDataset(
    image_paths=image_paths,
    labels=labels,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

for images, labels in train_loader:
    print(images.shape)  # torch.Size([32, 3, 64, 64])
    print(labels.shape)  # torch.Size([32])
    break
"""

# 15. 정리
# Keras ImageDataGenerator와 PyTorch transforms 비교  ----
# Keras: ImageDataGenerator(horizontal_flip=True)
# PyTorch: transforms.RandomHorizontalFlip()

# Keras: ImageDataGenerator(rotation_range=45)
# PyTorch: transforms.RandomRotation(degrees=45)

# Keras: ImageDataGenerator(width_shift_range=0.3)
# PyTorch: transforms.RandomAffine(translate=(0.3, 0.0))

# Keras: ImageDataGenerator(brightness_range=(0.7, 1.3))
# PyTorch: transforms.ColorJitter(brightness=(0.7, 1.3))

# Keras: ImageDataGenerator(zoom_range=0.5)
# PyTorch: transforms.RandomAffine(scale=(0.5, 1.5))
