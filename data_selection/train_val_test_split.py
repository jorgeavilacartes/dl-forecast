import datetime
import json
import time

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Union

class TrainValTestSplit: 
    """
    Entregar conjuntos de train, val y test dado un conjunto de series de tiempo
    paralelas temporalmente. 
    Se entregan las posiciones en la serie original de cada muestra generada. 

    """

    def __init__(self, id_time_series: List[str], timesteps: List[str], features: List[str], basepath_time_series: str ):
        """
        Todas las series de tiempo ingresadas se asumen del mismo largo y con la misma fecha inicial.
            id_time_series:  lista con los id a cada serie de tiempo generada con PrepareData
            timesteps: lista con las fechas asociadas a las series de tiempo
            features: lista con los nombres de las columnas de las series de tiempo
            basepath_time_series: directorio donde estan guardadas las series de tiempo como numpy
        """
        self.id_time_series = id_time_series
        self.timesteps = timesteps
        self.features = features
        self.basepath_time_series = Path(basepath_time_series)
        
    def __call__(self, perc_train=0.7, len_input = 30, len_output = 1):
        """Generar conjuntos de train, val y test con referencias a inputs y etiquetas

        Args:
            perc_train (float, optional): porcion de la serie de tiempo para entrenamiento. Defaults to 0.7.
            len_input (int, optional): largo (profundidad) de la serie para el input. Defaults to 30.
            len_output (int, optional): largo (profundidad) de la serie para el output. Defaults to 1.
        """        

        # Obtener referencias a los indices para X e y en cada conjunto
        subsets = self.split_subsets(perc_train, len_input, len_output)

        # Agregar representantes de cada serie de tiempo a cada conjunto
        datasets=defaultdict(dict)
        
    
        # Generar los conjuntos train, val y test
        for subset in subsets:
            
            # Agregar cada serie de tiempo
            for id_ts in self.id_time_series:
                
                # Etiquetas de la serie de tiempo en el conjunto seleccionado
                ts_labels = {
                    subset + "_" + id_ts + "_" + str(k).zfill(3) : {
                        "pos": v,
                        "path_time_serie": str(self.basepath_time_series.joinpath(id_ts+".npy"))
                        }
                    for k,v in subsets[subset].items()
                            }
                
                # Agregamos las etiquetas al diccionario comun del conjunto
                datasets[subset].update(ts_labels)
        
        # largo de la serie para el input y el output
        self.len_input = len_input
        self.len_output = len_output

        self.datasets = dict(datasets)

        #Comprobar que ningun conjunto de vacio
        all_subset_non_empty = [len(self.datasets[subset])>0 for subset in ["train","test","val"] ]
        assert all(all_subset_non_empty) ,"Al menos un conjunto no tiene elementos. Elija un 'len_output' o 'perc_train' menor"
        print("Conjuntos de train, val y test creados. Ver atributos .datasets")

    def save(self,):
        "Guardar listas de train, val y test, diccionario labels"
        
        folder = Path("data")
        folder.mkdir(exist_ok=True)
        
        print("Guardando conjuntos train, val, test y labels")
        time.sleep(1)
        
        # Diccionario para guardar toda la informacion
        labels = {}
        
        # Guardar listas para train, val y test con los id de cada muestra
        for subset in self.datasets:
            
            # Seleccionar conjunto
            dict_subset = self.datasets[subset]
            
            # Agregar a labels
            labels.update(dict_subset)

            # Extraer sus id
            list_subset = list(dict_subset.keys())
            
            path_save = folder.joinpath("list_{}.json".format(subset))
            with open(str(path_save), "w", encoding="utf8") as fp:
                json.dump(list_subset, fp, ensure_ascii=False, indent=4)

        # Guardar labels
        path_save = folder.joinpath("labels.json")
        with open(str(path_save), "w", encoding="utf8") as fp:
                json.dump(labels, fp, ensure_ascii=False, indent=4)

        #  Guardar configuracion del split
        split_config = dict(
            len_input=self.len_input,
            len_output=self.len_output,
            timesteps=self.timesteps,
            features=self.features,
            base_path_time_series=str(self.basepath_time_series),
        )

        path_save = folder.joinpath("split_config.json")
        with open(str(path_save), "w", encoding="utf8") as fp:
                json.dump(split_config, fp, ensure_ascii=False, indent=4)

        # Guardar 
        print("Datos para entrenamiento guardados en {}".format(str(folder.absolute())))

    def split_subsets(self, perc_train, len_input, len_output):
        "Generar posiciones para X e y en cada conjunto train, val, test"
        # Fin entrenamiento y validacion para separar conjuntos
        self.end_train = int(len(self.timesteps) // (1/perc_train))
        self.end_val   = int(self.end_train + (len(self.timesteps) - self.end_train)//2)

        # Entrenamiento
        split_train = self.get_all_ranges(
            len_input, len_output, initial_pos=0, last_pos=self.end_train
            )

        # Validacion
        split_val   = self.get_all_ranges(
            len_input, len_output, initial_pos=self.end_train, last_pos=self.end_val
            )

        # Test
        split_test = self.get_all_ranges(
            len_input, len_output, initial_pos=self.end_val, last_pos=len(self.timesteps)
        )

        return dict(
            train=split_train, 
            val=split_val, 
            test=split_test
        )

    def get_all_ranges(self, len_input, len_output, initial_pos, last_pos):
        "Obtener las combinaciones de X,y referenciados por posicion y fecha para los largos definidos"
        
        # Largo total que necesito para generar X e y
        total_len = len_input + len_output
        
        # Largo de los timesteps que dispongo
        len_timesteps = last_pos - initial_pos
        
        # Generar las subseries 
        #split = []
        split = {}
        current_pos = 0
        while (initial_pos + current_pos + total_len) <= (initial_pos + len_timesteps):
            # Generar posiciones para X e y
            X_from = initial_pos + current_pos 
            X_to   = X_from + len_input
            y_to   = X_to + len_output

            idx_X = [X_from, X_to]
            idx_y = [X_to, y_to]

            # Avanzar una posicion en la serie de tiempo y repetir
            current_pos +=1
            
            # Guardar referencias 
            #split.append({"X": idx_X, "y": idx_y})
            split[current_pos] = {"X": idx_X, "y": idx_y}

        return split
