from abc import ABC, abstractmethod


class ImageHasher(ABC):
    @abstractmethod
    def encode(self, pil_image):
        """
        Абстрактный метод для кодирования PIL изображения.

        :param pil_image: PIL.Image объект
        :return: хеш изображения
        """
        pass

    @abstractmethod
    def encode_file(self, image_path):
        """
        Абстрактный метод для кодирования изображения из файла.

        :param image_path: путь к файлу изображения
        :return: хеш изображения
        """
        pass