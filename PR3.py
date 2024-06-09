import cv2
import os

# Относительный путь к файлу каскада Хаара
haarcascade_path = 'haarcascade_frontalface_alt2.xml'

# Проверка наличия файла каскада Хаара
if not os.path.exists(haarcascade_path):
    print(f"Ошибка: {haarcascade_path} не найден.")
else:
    # Загрузка изображения
    image_path = 'chelsea.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Ошибка: изображение '{image_path}' не найдено или не может быть загружено.")
    else:
        # Преобразование изображения в оттенки серого
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Создание объекта CascadeClassifier
        faceCascade = cv2.CascadeClassifier(haarcascade_path)

        # Обнаружение лиц на изображении
        faces = faceCascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=5)

        # Отрисовка прямоугольника вокруг каждого лица
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (199, 21, 133), 2)

        # Отображение результата с выделенными лицами
        cv2.imshow('Обнаруженные лицау', image)
        cv2.waitKey(0)

        # Закрытие окна
        cv2.destroyAllWindows()
