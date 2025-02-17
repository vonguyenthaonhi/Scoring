import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor



class Preprocessing:
    
    def __init__(self):
        self.modes = {}  # Stockage les modes de chaque colonne pour application au test
        self.models = {}  # Stockage les modèles de régression par variable pour application au test
    
    def imputation_par_mode_train(self, data, categorical_cols):
        """
        Imputation sur les données d'entraînement : stocke la mode pour chaque colonne.
        """
        for col in categorical_cols:
            mode = data[col].mode()[0]
            self.modes[col] = mode
            data[col] = data[col].fillna(mode)
        return data
    
    def imputation_par_mode_test(self, data, categorical_cols):
        """
        Imputation sur les données de test : utilise la mode calculée sur les données d'entraînement.
        """
        for col in categorical_cols:
            if col in self.modes:
                data[col] = data[col].fillna(self.modes[col])
        return data
    
    def imputation_par_regression_train(self, data, variables_for_imputation, categorical_cols):
        """
        Entraînement du modèle de régression pour chaque variable avec des valeurs manquantes en utilisant Random Forest.
        """
        # One hot encoding des variables catégorielles
        df_encoded = pd.get_dummies(data, drop_first=False)
        bool_columns = df_encoded.select_dtypes(include='bool').columns
        df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)
        df_imputed = df_encoded.copy()

        #Imputation des NA pour chaque variable
        for variable in variables_for_imputation:
            predictors = df_encoded.columns.difference([variable]).tolist()
            df_without_na = df_imputed.dropna(subset=[variable])
            df_with_na = df_imputed[df_imputed[variable].isna()]

            if not df_with_na.empty:
                X_train = df_without_na[predictors]
                y_train = df_without_na[variable]
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                self.models[variable] = model
                
                X_missing = df_with_na[predictors]
                df_imputed.loc[df_imputed[variable].isna(), variable] = model.predict(X_missing)

        # transformation inverse des variables encodées
        data = self.reverse_ohe(df_imputed, categorical_cols)
        return data
    
    def imputation_par_regression_test(self, data, variables_for_imputation, categorical_cols):
        """
        Imputation sur les données de test avec les modèles entraînés.
        """
        df_encoded = pd.get_dummies(data, drop_first=False)
        bool_columns = df_encoded.select_dtypes(include='bool').columns
        df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)
        df_imputed = df_encoded.copy()

        for target in variables_for_imputation:
            if target in self.models:
                predictors = df_encoded.columns.difference([target]).tolist()
                df_with_na = df_imputed[df_imputed[target].isna()]

                if not df_with_na.empty:
                    X_missing = df_with_na[predictors]
                    df_imputed.loc[df_imputed[target].isna(), target] = self.models[target].predict(X_missing)

        data = self.reverse_ohe(df_imputed, categorical_cols)
        return data
    
    def reverse_ohe(self, df_encoded, original_categorical_cols):
        """
        Reconvertir les colonnes One-Hot Encoded en leur forme originale.
        """

        df_reversed = df_encoded.copy()
        for col in original_categorical_cols:
            ohe_columns = [col_name for col_name in df_encoded.columns if col in col_name]
            if ohe_columns:
                df_reversed[col] = df_encoded[ohe_columns].idxmax(axis=1).str.replace(f'{col}_', '')
                df_reversed.drop(columns=ohe_columns, inplace=True)
        return df_reversed
    

    def arondir_les_variables_prédites (self,data, list_col_int):
        """
        Arrodir des variables integer.
        """
        for col in list_col_int:
            data[col] = round(data[col], 0)
        return data


    


