#96.10

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Add, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
from tensorflow.keras.layers import GlobalAveragePooling2D

# 시드 값 고정
SEED = 77
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Pillow의 이미지 크기 제한 해제
Image.MAX_IMAGE_PIXELS = None

# 경로 설정
train_disease_path = 'C:/finalproject_dataset/disease/train'
test_disease_path = 'C:/finalproject_dataset/disease/test'
train_healthy_path = 'C:/finalproject_dataset/healthy/train'
test_healthy_path = 'C:/finalproject_dataset/healthy/test'

# 데이터 불러오기 및 전처리 함수
def load_data(folder, label, target_size=(255, 255)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.endswith(('.jpg')):
            try:
                img = load_img(img_path, target_size=target_size)  # 이미지 불러오기 및 리사이즈
                img = img_to_array(img) / 255.0  # 정규화
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"이미지 로드 실패: {img_path}, 오류: {e}")
    return np.array(images), np.array(labels)

# 각 폴더에서 데이터 불러오기
x_train_disease, y_train_disease = load_data(train_disease_path, label=1)
x_test_disease, y_test_disease = load_data(test_disease_path, label=1)
x_train_healthy, y_train_healthy = load_data(train_healthy_path, label=0)
x_test_healthy, y_test_healthy = load_data(test_healthy_path, label=0)

# 훈련 및 테스트 세트 병합
x_train = np.concatenate([x_train_disease, x_train_healthy], axis=0)
y_train = np.concatenate([y_train_disease, y_train_healthy], axis=0)
x_test = np.concatenate([x_test_disease, x_test_healthy], axis=0)
y_test = np.concatenate([y_test_disease, y_test_healthy], axis=0)

# SMOTE 적용
smote = SMOTE(random_state=SEED)
x_train_reshaped = x_train.reshape(len(x_train), -1)  # 이미지를 2D로 변환
x_train_smote, y_train_smote = smote.fit_resample(x_train_reshaped, y_train)  # SMOTE 적용
x_train_smote = x_train_smote.reshape(-1, 255, 255, 3)  # 원래 이미지 형태로 복원

# Train/Validation 분리
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_smote, y_train_smote, test_size=0.2, random_state=SEED, stratify=y_train_smote
)

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 데이터 증강 생성기
train_generator = datagen.flow(
    x_train_split,
    y_train_split,
    batch_size=32,
    seed=SEED
)

def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:  # 입력 채널과 필터 수가 다를 경우
        shortcut = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut)

    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])  # Skip connection
    return x

# 모델 설계
input_layer = Input(shape=(255, 255, 3))

# 첫 번째 Conv 블록
x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Residual Block 추가 (총 5개 추가)
x = residual_block(x, 32)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 64)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 128)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 256)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 512)
x = MaxPooling2D((2, 2))(x)

x = residual_block(x, 1024)
x = MaxPooling2D((2, 2))(x)

# 전역 평균 풀링 적용
x = GlobalAveragePooling2D()(x)

# Dense 레이어
x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)
# 출력 레이어
output_layer = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(x)

# 모델 정의
model = Model(inputs=input_layer, outputs=output_layer)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=(x_val_split, y_val_split),
    epochs=40,
    callbacks=[early_stopping]
)

# 모델 평가
val_loss, val_accuracy = model.evaluate(x_val_split, y_val_split, verbose=0)
print(f"Validation Set - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Set - Loss: {test_loss:.4f}, Accuracy: {test_accuracy * 100:.2f}%")

# 학습 기록에서 정확도와 손실 데이터 가져오기
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# 정확도 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ROC 커브 및 성능 지표
y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Disease"], yticklabels=["Healthy", "Disease"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
roc_auc = auc(*roc_curve(y_test, y_pred_probs)[0:2])

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")


# 특징 추출을 위한 모델 생성 (Flatten 레이어 직전까지)
feature_extractor = Model(inputs=model.input, outputs=model.get_layer(index=-4).output)

# 모델을 사용해 학습 데이터와 테스트 데이터에서 Feature 추출
train_features = feature_extractor.predict(x_train)
test_features = feature_extractor.predict(x_test)

# t-SNE로 차원 축소 후 시각화
def plot_tsne(features, labels, title):
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    tsne_result = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for label, color, marker in zip([0, 1], ['blue', 'red'], ['o', 'x']):
        plt.scatter(
            tsne_result[labels == label, 0],
            tsne_result[labels == label, 1],
            c=color,
            label=f"Class {label}",
            alpha=0.5,
            edgecolor='k',
            marker=marker
        )
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# t-SNE 시각화: 학습 데이터
plot_tsne(train_features, y_train, "t-SNE Visualization (Train Data)")

# t-SNE 시각화: 테스트 데이터
plot_tsne(test_features, y_test, "t-SNE Visualization (Test Data)")