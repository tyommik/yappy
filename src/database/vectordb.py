from pymilvus import (
    connections,
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    IndexType
)

import numpy as np


class VectorDB:
    def __init__(self, db_path, collection_name, dimension=64, recreate_collection=False):
        self.client = MilvusClient(db_path)  # Инициализация клиента
        self.collection_name = collection_name
        self.dimension = dimension  # Ожидаемая длина векторов - 64
        self.connections = connections.connect(uri=db_path)

        # Определяем поля схемы
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),  # Поле id
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),  # Вектор длины 64
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=32),  # Поле для хэша
            FieldSchema(name="frame_index", dtype=DataType.INT64),  # Поле для индекса кадра
            FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=255)  # Поле для имени видео
        ]

        # Создаем схему коллекции
        schema = CollectionSchema(fields=fields, description="Схема для коллекции видеофреймов")

        # TODO Возможно сделать её неудаляемой
        # Дропаем коллекцию, если существует
        if recreate_collection and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name=collection_name)

        # self.client.create_collection(
        #     collection_name=self.collection_name,
        #     schema=schema,
        #     auto_id=True,
        #     enable_dynamic_field=True,
        #     metric_type="L2",
        # )


        # Create collection schema
        schema = CollectionSchema(fields=fields, description="Schema for video frames")

        # Create collection

        self.collection = Collection(name=self.collection_name, schema=schema)

        """Build an index for the collection for faster search."""
        index_params = {
            "index_type": "IVF_FLAT",  # Use IVF_FLAT for example
            "metric_type": "L2",  # Metric type for float vectors
            "params": {"nlist": 100}  # nlist is a hyperparameter for the index
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        print(f"Index built successfully for collection {self.collection_name}")

        """Load the collection into memory."""
        self.collection.load()
        print(f"Collection {self.collection_name} loaded into memory.")

    def add_frame_hash(self, vector, frame_index, video_id, id_value=None):
        try:
            # vector = self.phash_to_vector(hash_string)

            # Если id не передан, генерируем автоматически (например, используя автоинкремент)
            if id_value is None:
                id_value = self.get_next_id()

            data = [{
                "id": id_value,  # Добавляем id явным образом
                "vector": vector,
                "hash": 0x0000000001,
                "frame_index": int(frame_index),
                "video_name": video_id
            }]
            return self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
        except Exception as e:
            print(f"Ошибка при добавлении хэша: {e}")
            return None

    def get_next_id(self):
        # Реализуйте логику для автоинкремента id, например, можно найти последний id
        # и увеличить его на 1. Это лишь пример логики, используйте то, что подходит для вас.
        max_id_query = self.client.query(
            collection_name=self.collection_name,
            filter=None,
            output_fields=["id"],
            order_by="id DESC",
            limit=1
        )
        if max_id_query:
            return max_id_query[0]['id'] + 1
        else:
            return 1  # Начать с 1, если коллекция пуста

    def search_similar_frames(self, query_vector, filter_expr=None, limit=2):
        try:
            # query_vector = self.phash_to_vector(query_hash)

            return self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                filter=filter_expr,
                search_params={"metric_type": "L2"},
                output_fields=["hash", "frame_index", "video_id"]
            )
        except Exception as e:
            print(f"Ошибка при поиске похожих кадров: {e}")
            return None

    def query_frames(self, filter_expr):
        try:
            return self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["hash", "frame_index", "video_id"]
            )
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {e}")
            return None

    def delete_frames(self, filter_expr):
        try:
            return self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
        except Exception as e:
            print(f"Ошибка при удалении кадров: {e}")
            return None

    @staticmethod
    def phash_to_vector(phash_string):
        # Преобразуем шестнадцатеричную строку в бинарный вектор
        binary_list = [int(char, 16) for char in phash_string]
        binary_vector = []
        for num in binary_list:
            binary_vector.extend([int(bit) for bit in format(num, '04b')])

        # Убедимся, что длина вектора совпадает с ожидаемой (например, 64)
        if len(binary_vector) != 64:
            raise ValueError("Неправильная длина pHash, ожидался 64-битный хэш")

        float_vector = np.array(binary_vector, dtype=np.float32)
        return float_vector

    def delete_collection(self):
        # create collection
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)