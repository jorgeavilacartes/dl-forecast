from typing import Callable, Optional, List
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    """Generador para keras
    Por cada llamada retorna X e y como batch
    """
    VERSION=1
    def __init__(self, 
                list_id_ts,
                labels,
                idx_feature_y,
                file_loader: Callable, # from file to array
                preprocessing: Optional[Callable] = None, 
                preprocessing_y: Optional[Callable] = None, 
                shuffle=True,
                batch_size=32,
                ):
        self.list_id_ts = list_id_ts
        self.labels = labels
        self.idx_feature_y = idx_feature_y
        self.preprocessing = preprocessing
        self.preprocessing_y = preprocessing_y
        self.file_loader = file_loader
        self.shuffle=shuffle
        self.batch_size=batch_size

        # Initialize first batch
        self.on_epoch_end()

    def on_epoch_end(self,):
        "Actualiza y aleatoriza (si shuffle=True) los indices al final de cada epoca"
        self.indexes = np.arange(len(self.list_id_ts))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # shuffle indexes in place

    def __len__(self):
        "Numero de batches por epoca"
        return int(np.floor(len(self.list_id_ts) / self.batch_size))

    def __getitem__(self, index):
        "Para alimentar al modelo. Genera un batch"
        # Generar indices para el batch actual 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Seleccionar la lista de series de tiempo a cargar
        list_id_ts_temp = [self.list_id_ts[k] for k in indexes]

        # Generar input y output
        X, y = self.input_output_generation(list_id_ts_temp)

        return X, y

    def input_output_generation(self, list_id_ts_temp: List[str]):
        """Generates and augment data containing batch_size samples

        Args:
            list_path_temp (List[str]): sublist of list_path

        Returns:
            X : numpy.array
            y : numpy.array hot-encoding
        """        
        # Extraer informacion para cargar cada serie de tiempo (path_time_serie, pos, idx_feature_y)
        dict_id_ts_temp = {} # diccionario para guardar inputs requeridos para cargar cada serie de tiempo por su id_ts
        for id_ts in list_id_ts_temp:
            info_id_ts = self.labels.get(id_ts)
            dict_id_ts_temp[id_ts] = dict(
                path_time_serie=info_id_ts.get("path_time_serie"),
                pos=info_id_ts.get("pos"),
                idx_feature_y=self.idx_feature_y
            )
            
        # cargar series de tiempo (X,y)
        list_ts = [self.file_loader(**inputs_ts_loader) for id_ts, inputs_ts_loader in dict_id_ts_temp.items()]
        
        # Separar input X de output y
        list_X = []
        list_y = []
        
        for ts in list_ts: 
            list_X.append(ts[0])
            list_y.append(ts[1])

        # Preprocessing
        if callable(self.preprocessing):
            list_X = [self.preprocessing(x) for x in list_X]
        if callable(self.preprocessing_y):
            list_y = [self.preprocessing_y(y) for y in list_y]

        # Create batch for the selected model
        X = self.batch_creator(list_X)
        y = self.batch_creator(list_y)
        
        return X, y

    @staticmethod
    def batch_creator(list_numpy):
        "Crear un batch a partir de una lista de numpys"
        # Agregar dimension extra al inicio
        numpy_as_batch = [np.expand_dims(x,axis=0) for x in list_numpy]

        # Devolver el batch concatenado
        return np.concatenate(numpy_as_batch, axis=0)