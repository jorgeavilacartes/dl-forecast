import numpy as np

class LoadTimeSerie:
    """
    Cargar una serie de tiempo para el flujo de entrenamiento

    Devuelve X,y
    """
    def __call__(self, path_time_serie, pos, idx_feature_y):
        """Cargar serie de tiempo en base a su id de referencia.
        Obtener X,y en base a sus posiciones de labels.json
        """
        X_pos = pos.get("X")
        y_pos = pos.get("y")
        
        # Cargar serie de tiempo
        ts = self.load_time_serie(path_time_serie)

        # Extraer la parte requerida
        X = ts[X_pos[0]:X_pos[1],:]
        y = ts[y_pos[0]:y_pos[1], idx_feature_y]
        return X,y

    def load_time_serie(self, path_time_serie):
        "Cargar serie de tiempo desde numpy"
        return np.load(path_time_serie)