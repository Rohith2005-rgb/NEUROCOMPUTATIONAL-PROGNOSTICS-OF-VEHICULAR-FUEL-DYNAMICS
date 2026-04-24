import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

class FuelConsumptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A Machine Learning Model for Average Fuel Consumption in Heavy Vehicles")
        self.root.geometry("800x600")
        
        self.dataset = None
        self.model = None
        self.scaler = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main title
        title_label = tk.Label(self.root, 
                             text="A Machine Learning Model for Average Fuel Consumption in Heavy Vehicles",
                             font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Upload section
        upload_frame = tk.Frame(self.root)
        upload_frame.pack(pady=10)
        
        self.upload_label = tk.Label(upload_frame, text="No dataset loaded", font=("Arial", 10))
        self.upload_label.pack(side=tk.LEFT, padx=10)
        
        upload_btn = tk.Button(upload_frame, text="Upload Dataset", command=self.upload_dataset)
        upload_btn.pack(side=tk.LEFT)
        
        # Options frame
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=30)
        
        # Option buttons
        btn1 = tk.Button(options_frame, text="Red Dataset & Generate Model", 
                        command=self.process_dataset, width=30, height=2)
        btn1.grid(row=0, column=0, padx=10, pady=10)
        
        btn2 = tk.Button(options_frame, text="Run ANN Algorithm", 
                        command=self.run_ann, width=30, height=2)
        btn2.grid(row=0, column=1, padx=10, pady=10)
        
        btn3 = tk.Button(options_frame, text="Fuel Consumption Graph", 
                        command=self.show_graph, width=30, height=2)
        btn3.grid(row=1, column=0, padx=10, pady=10)
        
        btn4 = tk.Button(options_frame, text="Predict Average Fuel Consumption", 
                        command=self.predict_consumption, width=30, height=2)
        btn4.grid(row=1, column=1, padx=10, pady=10)
        
        btn5 = tk.Button(options_frame, text="Exit", 
                        command=self.root.quit, width=30, height=2)
        btn5.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def upload_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                self.upload_label.config(text=f"Dataset loaded: {os.path.basename(file_path)}")
                self.status_var.set(f"Dataset loaded successfully with {len(self.dataset)} records")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
                self.status_var.set("Error loading dataset")
    
    def process_dataset(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please upload a dataset first")
            return
            
        try:
            # Filter for heavy vehicles (SUVs, trucks, vans)
            heavy_classes = ['SUV - STANDARD', 'PICKUP TRUCK - STANDARD', 'VAN - PASSENGER', 
                           'VAN - CARGO', 'MINIVAN', 'SPECIAL PURPOSE VEHICLE']
            self.heavy_vehicles = self.dataset[self.dataset['VEHICLECLASS'].isin(heavy_classes)]
            
            # Prepare features and target
            features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']
            target = 'FUELCONSUMPTION_COMB'
            
            # Remove rows with missing values
            self.heavy_vehicles = self.heavy_vehicles[features + [target]].dropna()
            
            self.X = self.heavy_vehicles[features].values
            self.y = self.heavy_vehicles[target].values
            
            # Scale the features
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)
            
            messagebox.showinfo("Success", "Dataset processed successfully for heavy vehicles")
            self.status_var.set("Dataset processed. Ready for ANN training.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dataset: {str(e)}")
            self.status_var.set("Error processing dataset")
    
    def run_ann(self):
        if not hasattr(self, 'X_scaled'):
            messagebox.showwarning("Warning", "Please process the dataset first")
            return
            
        try:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y, test_size=0.2, random_state=42)
            
            # Create and train the ANN model
            self.model = MLPRegressor(hidden_layer_sizes=(100, 50), 
                                    activation='relu', 
                                    solver='adam', 
                                    max_iter=1000, 
                                    random_state=42)
            
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            messagebox.showinfo("ANN Results", 
                              f"ANN Model Trained Successfully\n\n"
                              f"Mean Squared Error: {mse:.2f}\n"
                              f"R-squared Score: {r2:.2f}")
            
            self.status_var.set("ANN model trained successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train ANN model: {str(e)}")
            self.status_var.set("Error training ANN model")
    
    def show_graph(self):
        if not hasattr(self, 'heavy_vehicles'):
            messagebox.showwarning("Warning", "Please process the dataset first")
            return
            
        try:
            plt.figure(figsize=(10, 6))
            
            # Group by vehicle class and calculate average fuel consumption
            avg_consumption = self.heavy_vehicles.groupby('VEHICLECLASS')['FUELCONSUMPTION_COMB'].mean()
            
            # Plot
            avg_consumption.plot(kind='bar', color='red')
            plt.title('Average Fuel Consumption by Heavy Vehicle Class')
            plt.xlabel('Vehicle Class')
            plt.ylabel('Average Combined Fuel Consumption (L/100km)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
            self.status_var.set("Fuel consumption graph displayed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display graph: {str(e)}")
            self.status_var.set("Error displaying graph")
    
    def predict_consumption(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the ANN model first")
            return
            
        try:
            # Create prediction window
            pred_window = tk.Toplevel(self.root)
            pred_window.title("Predict Average Fuel Consumption")
            
            # Labels and entries for each feature
            tk.Label(pred_window, text="Engine Size (L):").grid(row=0, column=0, padx=10, pady=5)
            engine_entry = tk.Entry(pred_window)
            engine_entry.grid(row=0, column=1, padx=10, pady=5)
            
            tk.Label(pred_window, text="Number of Cylinders:").grid(row=1, column=0, padx=10, pady=5)
            cyl_entry = tk.Entry(pred_window)
            cyl_entry.grid(row=1, column=1, padx=10, pady=5)
            
            tk.Label(pred_window, text="City Fuel Consumption (L/100km):").grid(row=2, column=0, padx=10, pady=5)
            city_entry = tk.Entry(pred_window)
            city_entry.grid(row=2, column=1, padx=10, pady=5)
            
            tk.Label(pred_window, text="Highway Fuel Consumption (L/100km):").grid(row=3, column=0, padx=10, pady=5)
            hwy_entry = tk.Entry(pred_window)
            hwy_entry.grid(row=3, column=1, padx=10, pady=5)
            
            # Prediction button
            def make_prediction():
                try:
                    features = np.array([
                        float(engine_entry.get()),
                        float(cyl_entry.get()),
                        float(city_entry.get()),
                        float(hwy_entry.get())
                    ]).reshape(1, -1)
                    
                    # Scale the features
                    features_scaled = self.scaler.transform(features)
                    
                    # Make prediction
                    prediction = self.model.predict(features_scaled)
                    
                    messagebox.showinfo("Prediction Result", 
                                      f"Predicted Combined Fuel Consumption: {prediction[0]:.2f} L/100km")
                    
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numerical values")
                except Exception as e:
                    messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
            predict_btn = tk.Button(pred_window, text="Predict", command=make_prediction)
            predict_btn.grid(row=4, column=0, columnspan=2, pady=10)
            
            self.status_var.set("Ready for prediction input")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create prediction window: {str(e)}")
            self.status_var.set("Error in prediction")

if __name__ == "__main__":
    root = tk.Tk()
    app = FuelConsumptionApp(root)
    root.mainloop()