import numpy as np
import pandas as pd
import json

class SimpleModel:
    def __init__(self, model_type="linear"):
        """
        Initialiseer het model.
        
        Parameters
        ----------
        model_type : str
            Type model (bijv. 'linear', 'polynomial', 'neural' — uitbreidbaar).
        
        Deze functie zet standaardwaarden klaar voor de parameters (gewichten en bias)
        en markeert het model als 'niet getraind'.
        """
        self.model_type = model_type
        self.w = None
        self.b = None
        self.trained = False

    def _init_params(self):
        """Initialiseer willekeurige parameters"""
        self.w = np.random.randn()
        self.b = np.random.randn()

    def predict(self, x):
        """Voorspel y op basis van x"""
        if not self.trained:
            raise ValueError("Model is niet getraind! Roep eerst .fit() aan.")
        x = np.array(x)
        return self.w * x + self.b
        
    def check_prediction(self, x_value, description=""):
        """
        Controleer en print een voorspelling netjes.
        """
        try:
            y_pred = self.predict(x_value)
            print(f"\n📊 {description} → {y_pred:.2f}")
        except ValueError as e:
            print(e)

    def fit(self, x, y, lr=0.01, epochs=1000, verbose=True):
        """Train het model met gradient descent"""
        x = np.array(x)
        y = np.array(y)
        self._init_params()

        for epoch in range(epochs):
            y_pred = self.w * x + self.b
            loss = np.mean((y_pred - y)**2)
            # Gradiënten
            dw = np.mean(2 * (y_pred - y) * x)
            db = np.mean(2 * (y_pred - y))

            # Update parameters
            self.w -= lr * dw
            self.b -= lr * db

            if verbose and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | w: {self.w:.3f} | b: {self.b:.3f}")

        self.trained = True
        print("✅ Training klaar!")
        return self

    def save(self, filename="model.json"):
        """Sla modelparameters op naar bestand"""
        if not self.trained:
            raise ValueError("Kan niet opslaan: model is niet getraind.")
        data = {
            "model_type": self.model_type,
            "w": float(self.w),
            "b": float(self.b),
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"💾 Model opgeslagen als '{filename}'")

    def load(self, filename="model.json"):
        """Laad modelparameters uit bestand"""
        with open(filename, "r") as f:
            data = json.load(f)
        self.model_type = data["model_type"]
        self.w = data["w"]
        self.b = data["b"]
        self.trained = True
        print(f"📂 Model geladen uit '{filename}'")

# --- Gebruik voorbeeld ---
if __name__ == "__main__":pass
    # # Trainingsdata: leer y = x + 1
    # x = np.arange(1, 11)
    # y = x + 1

    # model = SimpleModel(model_type="linear")
    # model.fit(x, y, lr=0.01, epochs=6000)
    
    # # Test
    # print("\nVoorspelling voor 10 →", model.predict(10))

    # # Opslaan
    # model.save("linear_model.json")

    # # Nieuw model aanmaken en laden
    # nieuw_model = SimpleModel()
    # nieuw_model.load("linear_model.json")
    # print("Voorspelling na laden →", nieuw_model.predict(10))

