import numpy as np

class TimeSeriesAnalyzer:
    """
    Decidir si una serie de tiempo satisface algunos requerimientos
    minimos para ser considerada en el entrenamiento
    - timesteps en las filas
    - features en las columnas
    """
    def __init__(self, max_zero_timesteps=0.5, non_zero_features=True, eps_norm=0.00001, features_with_unique_values = False):
        # TODO usar estos inputs
        self.max_zero_timesteps = max_zero_timesteps # para timesteps
        self.non_zero_features = non_zero_features
        self.eps_norm = eps_norm
        self.features_with_unique_values = features_with_unique_values

    def __call__(self, time_serie):
        "Aplicar filtros para aceptar (True) o rechazar (False) una serie de tiempo"
        filter_by_timesteps = self.under_max_zero_by_timesteps(time_serie),
        filter_by_feature = self.under_max_zero_by_features(time_serie)
        filter_by_eps_norm = self.over_eps_norm(time_serie)
        filter_unique_values = self.unique_value_by_features(time_serie)
        filter_nan_values = self.ommit_nan(time_serie)
        
        # Agregar filtros a una lista
        filters = [
            filter_by_timesteps,
            filter_by_feature,
            filter_by_eps_norm,
            filter_unique_values,
            filter_nan_values
        ]

        # Todos los filtros deben cumplirse para ser aceptada
        return all(filters)

    def under_max_zero_by_timesteps(self, time_serie,):
        "Verificar si la serie de tiempo tiene menos valores/timesteps nulos de los requeridos"
        len_time_serie = time_serie.shape[0]

        # Verificar para cada timesteps (file)
        is_zero_value = np.apply_along_axis(
            func1d= lambda row: all([x==0 for x in row]), 
            axis=1 , # evaluar por filas
            arr=time_serie    # en el array time_serie
            )

        # Numero total de timesteps nulos en la serie de tiempo
        n_zero_values = sum(is_zero_value)
        
        # Retornar True si la serie de tiempo cumple con tener menos nulos del maximo permitido por timesteps
        return True if n_zero_values/len_time_serie < self.max_zero_timesteps else False

    def under_max_zero_by_features(self, time_serie,):
        "Verificar que ningun feature sea completamente nulo"

        # Verificar para cada timesteps (file)
        is_zero_value = np.apply_along_axis(
            func1d= lambda timestep: all([x==0 for x in timestep]), 
            axis=0 , # evaluar por columnas
            arr=time_serie    # en el array time_serie
            )

        # comprobar si hay features nulos en la serie de tiempo
        zero_values = any(is_zero_value)
        
        # Retornar True si la serie de tiempo no tiene features nulas en todos los timesteps
        return not zero_values

    def over_eps_norm(self, time_serie):
        "Verificar que ningun feature tenga una norma < eps_norm"
        # Verificar para cada timesteps (file)
        feature_norms = np.apply_along_axis(
            func1d= lambda feature: np.linalg.norm(feature), 
            axis=0 , # evaluar por columnas
            arr=time_serie    # en el array time_serie
            )

        # Rechazar serie si alguna de las features tiene norma inferior a 'eps_norm'
        any_norm_over_eps = any([x<self.eps_norm for x in feature_norms])
        return False if any_norm_over_eps else True

    def unique_value_by_features(self, time_serie,):
        "Verificar que ningun feature sea completamente nulo"

        # Verificar para cada feature (columna) si tiene un unico valor 
        has_unique_value = np.apply_along_axis(
            func1d= lambda feature: True if len(set(feature))==1 else False, 
            axis=0 , # evaluar por columnas
            arr=time_serie    # en el array time_serie
            )
            
        # comprobar si hay features con un solo valor en la serie de tiempo
        unique_value = any(has_unique_value)
        
        # Retornar True si la serie de tiempo no tiene features con un unico valor
        return not unique_value

    def ommit_nan(self, time_serie):
        "Omitir una serie de tiempo si tiene valores NaN"
        has_nan_values = any(list(np.isnan(time_serie).flatten()))
        return False if has_nan_values else True

    def template_new_filter(self, time_serie):
        """Para aplicar otro filtro, el unico requisito es procesar la serie de tiempo
        y retornar un booleano:
                > True: si la serie de tiempo pasa el filtro
                > False: si la serie de tiempo debe ser descartada 
    """
        pass