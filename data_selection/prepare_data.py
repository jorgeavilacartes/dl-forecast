
import os
import time
import datetime 
import json
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

class PrepareData:
    """
    Limpiar y extraer las series de tiempo de una tabla CSV
    
    + Asegurar fechas continuas, completar con 0 las no registradas
    + Separar series de tiempo de acuerdo al identificador que se eliga (Ej: id_producto, id_producto + cadena)
    + Guardar todas las series de tiempo generadas con el nombre de su identificador en formato numpy. Además, guardar
    un archivo json con la lista de timesteps y los nombres de las features de cada serie de tiempo

    """

    def __init__(self, path_data: str, colname_datetime: str, colname_features: List[str], colname_id_time_series: str = None,):
        """
        + Los datos son cargados desde 'path_data'.
        + 'colname_datetime' corresponde a la columna que contiene las fechas.
        + Se crea una serie de tiempo por cada valor distinto en la columna 'colname_id_time_series'. Si esta es None, 
        se considera que los datos corresponden a una sola serie de tiempo.
        """
        self.path_data = path_data
        self.colname_datetime = colname_datetime
        self.colname_features = colname_features
        self.colname_id_time_series = colname_id_time_series 
        
        self.time_series = {} # Diccionario para guardar series de tiempo por su id


    def __call__(self,):
        "Cargar los datos y generar las series de tiempo"
        self.load_data() # Cargar datos
        self.get_id_time_series() # Obtener id de cada serie de tiempo
        self.get_timesteps() # Obtener rango de fechas
        self.get_minmax() # Obtener minimos y maximos por feature
        self.get_mean_std() # Obtener promedio y desv std por feature
        print("Generando series de tiempo")
        time.sleep(1)
        for id_time_serie in tqdm(self.id_time_series):
            self.get_time_serie(id_time_serie)


    def load_data(self,):
        "Cargar datos"
        ALLOWED_FILES = [".csv"] # En caso de agregar mas formas de cargar. Ej: xlsx, pickle.
        
        # Extension del archivo proporcionado
        extension = os.path.splitext(self.path_data)[-1]

        # Verificar si es uno de los archivos que podemos cargar
        assert extension in set(ALLOWED_FILES), "Archivo debe ser uno de estos {}. El suyo '{}'".format(ALLOWED_FILES, extension)

        # Cargar el archivo
        if self._file_exists(filename = self.path_data):
            self.data = pd.read_csv(self.path_data)
            print("Archivo cargado desde {}".format(self.path_data))


    def get_id_time_series(self,):
        "Definir el identificador de cada serie de tiempo a generar"
        self.colname_id = "ID_ts"
        self.data[self.colname_id] = self.data[self.colname_id_time_series].apply(lambda row: 
                                                            "_".join([ str(c) + "-" + str(r) 
                                                            for c,r in 
                                                            zip(self.colname_id_time_series,row) ]), axis=1)
        
        # Total de series de tiempo que se van a extraer
        self.id_time_series = list(set(self.data[self.colname_id].tolist()))
        total_id = len(self.id_time_series)

        print("Se encontraron {} series de tiempo con id {}.".format(total_id, self.colname_id))


    def get_time_serie(self, id_time_serie):
        """Obtener serie de tiempo para un id, en el rango total de fechas.
        Guardar la serie de tiempo generada en el atributo .time_series
        """
        
        # Extraer datos de la serie de tiempo solicitada
        cols = [self.colname_datetime]
        cols.extend(self.colname_features)
        time_serie = self.data.query("`ID_ts` == '{}'".format(id_time_serie))[cols].copy()
        time_serie_by_date = {d.get(self.colname_datetime): [d.get(feature) for feature in self.colname_features] for d in time_serie.to_dict("records")} 

        # Extraer las fechas
        dates_time_serie = list(time_serie_by_date.keys())

        # Construir la serie de tiempo en el rango total de fechas
        rows = [] 
        for date in self.timesteps:
            str_date = self.date_to_str(date)
            if str_date in dates_time_serie:
                date_values = time_serie_by_date.get(str_date) 
                #info_date = time_serie_by_date.get(str_date)
                #date_values = info_date#[info_date for feature in self.colname_features]
            else:
                date_values = [0 for _ in self.colname_features]
            
            rows.append(date_values)
        
        self.time_series[id_time_serie] = np.array(rows)
        

    def get_timesteps(self,):
        "Obtener rango de fechas"
        # Obtener la columna con todas las fechas        
        dates = self.data[self.colname_datetime].tolist()

        # Transformar a datetime
        dates = [self.str_to_date(date) for date in dates]

        # Calcular fecha minima y maxima
        self.min_date = min(dates)
        self.max_date = max(dates)

        # Obtener el listado de timesteps
        n_days = (self.max_date-self.min_date).days + 1 # todos los dias incluidos inicial y final
        self.timesteps = [ self.add_days(self.min_date, days) for days in range(n_days)]
        
        print(f"Datos desde {self.date_to_str(self.min_date)} hasta {self.date_to_str(self.max_date)}, ({n_days} dias) ")

    def get_minmax(self,):
        self.list_min = self.data[self.colname_features].min(axis=0).tolist()
        self.list_max = self.data[self.colname_features].max(axis=0).tolist()

    def get_mean_std(self,):
        self.list_mean = self.data[self.colname_features].mean(axis=0).tolist()
        self.list_std = self.data[self.colname_features].std(axis=0).tolist()
        
    def save(self,):
        """Guardar series de tiempo generadas como numpy y un archivo de 
        configuracion con los timesteps, features y paths a los numpy"""
        
        folder = Path("time_series")
        folder.mkdir(exist_ok=True)
        folder.joinpath("numpy").mkdir(exist_ok=True)
        
        print("Guardando series de tiempo")
        time.sleep(1)
        for name_ts, ts_array in tqdm(self.time_series.items()):
            path_save = str(folder.joinpath("numpy/{}.npy".format(name_ts)))    
            np.save(path_save, ts_array)

        time_series_config = dict(
            features=self.colname_features,
            timesteps=[self.date_to_str(ts) for ts in self.timesteps],
            id_time_series=list(self.time_series.keys()),
            basepath_time_series=str(folder.joinpath("numpy").absolute()),
            list_min=self.list_min,
            list_max=self.list_max,
            list_mean=self.list_mean,
            list_std=self.list_std
        )
        
        path_save_config = str(folder.joinpath("time_series_config.json"))
        with open(path_save_config, "w", encoding="utf8") as fp:
            json.dump(time_series_config, fp, ensure_ascii=False, indent=4)

        print("Series de tiempo guardadas en {}".format(str(folder.absolute())))

    @staticmethod
    def _file_exists(filename):
        "Verificar si el archivo proporcionado existe en memoria"
        if os.path.exists(filename):
            return True
        else:
            print("El archivo no existe. Revise si el directorio '{}' es correcto.".format(filename))

    @staticmethod
    def str_to_date(date):
        "Transformar una fecha en formato str a date. Formato 'YYYY-MM-dd'"
        
        if isinstance(date, str):
            return datetime.date.fromisoformat(date)
        else:
            # TODO Comprobar correcto uso de raise y Exception
            raise Exception("'date' debe ser un string de fecha con formato 'YYYY-MM-dd'")

    @staticmethod   
    def date_to_str(date):
        "Transformar un string de la forma 'YYYY-MM-dd' a objeto del tipo datetime.date(year, month, day)"
        return date.isoformat()

    @staticmethod
    def add_days(date, days = 1):
        "Agregar/quitar dias a una fecha en formato date"
        assert isinstance(date, datetime.date), "'date' debe ser un objeto datetime.date"
        return date + datetime.timedelta(days)