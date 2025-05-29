import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import glob
import os
from datetime import datetime, timedelta
import warnings
import requests
import json
import time
import schedule

warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 10
plt.rcParams['figure.max_open_warning'] = 0
sns.set_style("whitegrid")

import matplotlib

matplotlib.use('TkAgg')


class PM10Predictor20Min:
    def __init__(self, data_folder_path, api_endpoint="http://localhost:8080/api/pm10/predictions"):
        """
        Inicjalizacja modelu predykcyjnego PM10 na 20 minut

        Args:
            data_folder_path (str): Ścieżka do folderu z plikami CSV
            api_endpoint (str): Endpoint Spring Boot do wysyłania predykcji
        """
        self.data_folder_path = data_folder_path
        self.api_endpoint = api_endpoint
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

    def load_and_combine_data(self):
        """Wczytuje i łączy wszystkie pliki CSV z folderu"""
        print("[INFO] Wczytywanie plików CSV...")

        csv_files = glob.glob(os.path.join(self.data_folder_path, "*.csv"))

        if not csv_files:
            print(f"[ERROR] Nie znaleziono plików CSV w folderze: {self.data_folder_path}")
            return False

        print(f"[INFO] Znaleziono {len(csv_files)} plików CSV")

        dataframes = []

        for file in csv_files:
            try:
                df_temp = None
                separators = [',', ';', '\t', '|']

                for sep in separators:
                    try:
                        df_temp = pd.read_csv(file, sep=sep)
                        if len(df_temp.columns) > 1:
                            break
                    except:
                        continue

                if df_temp is None or len(df_temp.columns) <= 1:
                    print(f"[WARNING] Nie można odczytać struktury pliku: {os.path.basename(file)}")
                    continue

                dataframes.append(df_temp)
                print(f"[INFO] Wczytano: {os.path.basename(file)} - {len(df_temp)} rekordów")

            except Exception as e:
                print(f"[ERROR] Błąd wczytywania {file}: {e}")

        if not dataframes:
            print("[ERROR] Nie udało się wczytać żadnych danych")
            return False

        self.df = pd.concat(dataframes, ignore_index=True)
        print(f"[INFO] Łącznie wczytano {len(self.df)} rekordów")

        return True

    def preprocess_data(self):
        """Przetwarzanie i czyszczenie danych"""
        print("[INFO] Przetwarzanie danych...")

        if 'Date' not in self.df.columns or 'Time' not in self.df.columns:
            print("[ERROR] Brak kolumn Date lub Time!")
            return False

        try:
            self.df['datetime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'])
            print("[INFO] Utworzono kolumnę datetime")
        except Exception as e:
            print(f"[ERROR] Błąd tworzenia datetime: {e}")
            return False

        if 'PM10' not in self.df.columns:
            print("[ERROR] Nie znaleziono kolumny PM10!")
            return False

        print("[INFO] Sprawdzanie i naprawianie dat...")

        current_time = datetime.now()
        future_mask = self.df['datetime'] > current_time
        future_count = future_mask.sum()

        if future_count > 0:
            print(f"[WARNING] Znaleziono {future_count} rekordów z przyszłymi datami - usuwam")
            self.df = self.df[~future_mask]

        self.df = self.df.sort_values('datetime').reset_index(drop=True)

        before_duplicates = len(self.df)
        self.df = self.df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
        duplicates_removed = before_duplicates - len(self.df)
        if duplicates_removed > 0:
            print(f"[INFO] Usunięto {duplicates_removed} duplikatów czasowych")

        print("[INFO] Czyszczenie danych PM10...")

        self.df['PM10'] = pd.to_numeric(self.df['PM10'], errors='coerce')

        nan_count = self.df['PM10'].isna().sum()
        if nan_count > 0:
            print(f"[INFO] Usuwam {nan_count} wartości NaN PM10")
            self.df = self.df.dropna(subset=['PM10'])

        negative_count = (self.df['PM10'] < 0).sum()
        if negative_count > 0:
            print(f"[INFO] Usuwam {negative_count} ujemnych wartości PM10")
            self.df = self.df[self.df['PM10'] >= 0]

        q99 = self.df['PM10'].quantile(0.99)
        outliers_mask = self.df['PM10'] > q99
        outliers_count = outliers_mask.sum()
        if outliers_count > 0:
            print(f"[INFO] Usuwam {outliers_count} outlierów PM10 (>{q99:.1f})")
            self.df = self.df[~outliers_mask]

        print(f"[INFO] Finalne statystyki PM10: min={self.df['PM10'].min():.2f}, max={self.df['PM10'].max():.2f}")

        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute
        self.df['second'] = self.df['datetime'].dt.second
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        self.df['day_of_year'] = self.df['datetime'].dt.dayofyear


        for lag in [1, 2, 3, 5, 10, 30, 60, 120, 300, 600]:
            self.df[f'PM10_lag_{lag}'] = self.df['PM10'].shift(lag)

        for window in [5, 10, 30, 60, 300, 600]:
            self.df[f'PM10_ma_{window}'] = self.df['PM10'].rolling(window=window, min_periods=1).mean()

        available_features = []
        potential_features = ['PM25', 'IAQ', 'HCHO', 'CO2', 'TIN', 'TOUT', 'RHIN', 'RHOUT',
                              'P', 'NO2', 'NO', 'SO2', 'H2S', 'CO', 'HCN', 'HCL', 'NH3']

        for feature in potential_features:
            if feature in self.df.columns:
                available_features.append(feature)

        print(f"[INFO] Dostępne cechy środowiskowe: {available_features}")

        self.feature_columns = available_features + [
            'hour', 'minute', 'second', 'day_of_week', 'day_of_year'
        ]

        lag_features = [col for col in self.df.columns if 'PM10_lag_' in col or 'PM10_ma_' in col]
        self.feature_columns.extend(lag_features)

        for col in self.feature_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        before_cleanup = len(self.df)
        self.df = self.df.dropna(subset=self.feature_columns + ['PM10']).reset_index(drop=True)
        cleanup_removed = before_cleanup - len(self.df)

        if cleanup_removed > 0:
            print(f"[INFO] Usunięto {cleanup_removed} wierszy z brakującymi danymi")

        print(f"[INFO] Przygotowano {len(self.df)} rekordów z {len(self.feature_columns)} cechami")
        print(f"[INFO] Okres danych: {self.df['datetime'].min()} do {self.df['datetime'].max()}")

        return True

    def create_features_and_target(self, target_seconds_ahead=1200):
        """Przygotowanie cech i zmiennej docelowej dla predykcji na 20 minut"""
        print(f"[INFO] Przygotowywanie cech dla predykcji na {target_seconds_ahead} sekund...")

        X = self.df[self.feature_columns].copy()

        y = self.df['PM10'].shift(-target_seconds_ahead)

        X = X[:-target_seconds_ahead]
        y = y[:-target_seconds_ahead]

        print(f"[INFO] Wymiary danych: X{X.shape}, y{y.shape}")

        return X, y

    def train_model(self, X, y):
        """Trenowanie modelu Random Forest"""
        print("[INFO] Trenowanie modelu...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=False
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"[INFO] Wyniki modelu:")
        print(f"   MAE: {mae:.3f}")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   R²: {r2:.3f}")

        return X_train, X_test, y_train, y_test, y_pred

    def predict_next_20_minutes(self):
        """Predykcja PM10 na następne 20 minut (1200 sekund)"""
        print("[INFO] Predykcja PM10 na następne 20 minut...")

        last_row = self.df.iloc[-1].copy()
        predictions = []
        timestamps = []

        current_time = datetime.now()
        print(f"[INFO] Bazowy czas predykcji: {current_time}")
        print(f"[INFO] Ostatni pomiar z danych: {last_row['datetime']}")

        for second in range(1, 1201):
            future_time = current_time + timedelta(seconds=second)

            last_row['hour'] = future_time.hour
            last_row['minute'] = future_time.minute
            last_row['second'] = future_time.second
            last_row['day_of_week'] = future_time.weekday()
            last_row['day_of_year'] = int(future_time.strftime('%j'))

            features = last_row[self.feature_columns].values.reshape(1, -1)
            features_scaled = self.scaler.transform(features)

            pred = self.model.predict(features_scaled)[0]
            predictions.append(max(0, pred))
            timestamps.append(future_time)

            if 'PM10_lag_1' in self.feature_columns:
                last_row['PM10_lag_1'] = pred

        return timestamps, predictions

    def send_predictions_to_api(self, timestamps, predictions):
        """Wysyłanie predykcji do Spring Boot API"""
        try:
            current_pm10 = float(self.df['PM10'].iloc[-1])
            last_measurement_time = self.df['datetime'].iloc[-1].isoformat()

            prediction_points = []
            for i in range(0, len(predictions), 60):
                prediction_points.append({
                    "timestamp": timestamps[i].isoformat(),
                    "predictedPM10": round(float(predictions[i]), 2)
                })

            payload = {
                "generatedAt": datetime.now().isoformat(),
                "lastMeasuredPM10": round(current_pm10, 2),
                "lastMeasurementTime": last_measurement_time,
                "predictionHorizonMinutes": 20,
                "predictions": prediction_points,
                "summary": {
                    "avgPredictedPM10": round(float(np.mean(predictions)), 2),
                    "minPredictedPM10": round(float(np.min(predictions)), 2),
                    "maxPredictedPM10": round(float(np.max(predictions)), 2),
                    "trend": "ROSNĄCY" if predictions[-1] > predictions[0] else "MALEJĄCY" if predictions[-1] <
                                                                                              predictions[
                                                                                                  0] else "STABILNY"
                }
            }

            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            print(f"[INFO] Wysyłanie predykcji do API: {self.api_endpoint}")

            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200 or response.status_code == 201:
                print(f"[SUCCESS] Predykcje wysłane pomyślnie! Status: {response.status_code}")
                return True
            else:
                print(f"[ERROR] Błąd wysyłania predykcji. Status: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except requests.exceptions.ConnectionError:
            print("[ERROR] Nie można połączyć się z API. Sprawdź czy Spring Boot działa.")
            return False
        except requests.exceptions.Timeout:
            print("[ERROR] Timeout podczas wysyłania do API.")
            return False
        except Exception as e:
            print(f"[ERROR] Błąd wysyłania do API: {e}")
            return False

    def run_prediction_cycle(self):
        """Uruchomienie jednego cyklu predykcji i wysłania do API"""
        print("\n" + "=" * 60)
        print(f"NOWY CYKL PREDYKCJI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        try:
            if not self.load_and_combine_data():
                print("[ERROR] Błąd przeładowywania danych")
                return False

            if not self.preprocess_data():
                print("[ERROR] Błąd przetwarzania danych")
                return False

            if len(self.df) < 2000:
                print(f"[ERROR] Za mało danych ({len(self.df)} < 2000)")
                return False

            X, y = self.create_features_and_target(target_seconds_ahead=1200)

            if len(X) == 0:
                print("[ERROR] Brak danych do trenowania")
                return False

            self.train_model(X, y)

            timestamps, predictions = self.predict_next_20_minutes()

            success = self.send_predictions_to_api(timestamps, predictions)

            if success:
                print("[INFO] Cykl predykcji zakończony pomyślnie!")
            else:
                print("[WARNING] Cykl predykcji zakończony, ale wystąpiły problemy z API")

            return success

        except Exception as e:
            print(f"[ERROR] Błąd w cyklu predykcji: {e}")
            return False

    def start_scheduled_predictions(self):
        """Uruchomienie scheduled predykcji co 20 minut"""
        print("=" * 60)
        print("URUCHAMIANIE SCHEDULED PREDYKCJI PM10")
        print("Predykcje będą wysyłane co 20 minut do Spring Boot API")
        print(f"API Endpoint: {self.api_endpoint}")
        print("=" * 60)

        print("[INFO] Uruchamianie pierwszej predykcji...")
        self.run_prediction_cycle()

        schedule.every(20).minutes.do(self.run_prediction_cycle)

        print(f"[INFO] Zaplanowano predykcje co 20 minut")
        print("[INFO] Naciśnij Ctrl+C aby zatrzymać")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n[INFO] Zatrzymywanie scheduled predykcji...")
            print("[INFO] Program zakończony przez użytkownika")


def main():
    """Główna funkcja uruchamiająca scheduled predykcje"""
    print("PM10 PREDICTOR - SCHEDULED API MODE")
    print("=" * 60)

    data_folder = "folder"
    api_endpoint = "http://localhost:8080/api/pm10/predictions"

    predictor = PM10Predictor20Min(data_folder, api_endpoint)

    predictor.start_scheduled_predictions()


if __name__ == "__main__":
    main()