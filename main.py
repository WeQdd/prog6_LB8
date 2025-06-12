import cv2
import argparse
import tkinter as tk
from tkinter import filedialog

# функция определения лиц
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


# Загрузка моделей
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Загружаем нейросети
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)


# Обработка одного изображения
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]): min(faceBox[3], frame.shape[0]-1),
                     max(0, faceBox[0]): min(faceBox[2], frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Вывод на экран
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Показать и сохранить
    cv2.imshow("Результат", resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    output_path = image_path.replace('.', '_out.')
    cv2.imwrite(output_path, resultImg)
    print(f"Сохранено: {output_path}")


# Работа с камерой
def process_camera():
    video = cv2.VideoCapture(0)
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]): min(faceBox[3], frame.shape[0]-1),
                         max(0, faceBox[0]): min(faceBox[2], frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detecting age and gender", resultImg)


# Выбор файла через GUI
def select_file_and_process():
    root = tk.Tk()
    root.withdraw()  # скрываем главное окно
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        process_image(file_path)
    else:
        print("Файл не выбран. Переход к камере.")
        process_camera()


# === Точка входа ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Определение пола и возраста')
    parser.add_argument('--image', help='Путь к изображению', type=str)
    args = parser.parse_args()

    if args.image:
        process_image(args.image)
    else:
        use_gui = input("Использовать графический интерфейс для выбора файла? (y/n): ").lower() == 'y'
        if use_gui:
            select_file_and_process()
        else:
            process_camera()