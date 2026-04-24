import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
import seaborn as sns
from PIL import Image, ImageTk
import numpy as np

class FuelConsumptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Model for Average Fuel Consumption in Heavy Vehicles")
        self.root.geometry("900x700")
        
        # Load and set background image
        try:
            self.bg_image = Image.open("background.jpg")
            self.bg_image = self.bg_image.resize((900, 700), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            
            self.background_label = tk.Label(root, image=self.bg_photo)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        except:
            # Fallback if background image not found
            self.background_label = tk.Label(root, bg="#2c3e50")
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.dataset = None
        self.model = None
        self.ann_model = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        self.main_frame = tk.Frame(self.root, bg="white", bd=5, relief=tk.RIDGE)
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", width=800, height=600)
        
        # Title
        title_label = tk.Label(self.main_frame, 
                             text="A Machine Learning Model for Average Fuel Consumption in Heavy Vehicles",
                             font=("Helvetica", 16, "bold"), 
                             bg="white", fg="#2c3e50")
        title_label.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(self.main_frame, 
                             text="Upload Heavy Vehicles Fuel Dataset", 
                             command=self.upload_dataset,
                             font=("Helvetica", 12),
                             bg="#3498db", fg="white",
                             padx=20, pady=10)
        upload_btn.pack(pady=20)
        
        # Options frame
        self.options_frame = tk.Frame(self.main_frame, bg="white")
        self.options_frame.pack(pady=20)
        
        # Option buttons (initially disabled)
        self.red_button = tk.Button(self.options_frame, 
                                  text="Red Dataset & Generate Model", 
                                  command=self.generate_model,
                                  state=tk.DISABLED,
                                  font=("Helvetica", 10),
                                  bg="#e74c3c", fg="white",
                                  padx=10, pady=5)
        self.red_button.grid(row=0, column=0, padx=10, pady=5)
        
        self.ann_button = tk.Button(self.options_frame, 
                                  text="Run ANN Algorithm", 
                                  command=self.run_ann,
                                  state=tk.DISABLED,
                                  font=("Helvetica", 10),
                                  bg="#2ecc71", fg="white",
                                  padx=10, pady=5)
        self.ann_button.grid(row=0, column=1, padx=10, pady=5)
        
        self.graph_button = tk.Button(self.options_frame, 
                                    text="Fuel Consumption Graph", 
                                    command=self.show_graph,
                                    state=tk.DISABLED,
                                    font=("Helvetica", 10),
                                    bg="#f39c12", fg="white",
                                    padx=10, pady=5)
        self.graph_button.grid(row=0, column=2, padx=10, pady=5)
        
        self.exit_button = tk.Button(self.options_frame, 
                                   text="Exit", 
                                   command=self.root.quit,
                                   font=("Helvetica", 10),
                                   bg="#95a5a6", fg="white",
                                   padx=10, pady=5)
        self.exit_button.grid(row=0, column=3, padx=10, pady=5)
        
        # Prediction frame
        self.prediction_frame = tk.Frame(self.main_frame, bg="white")
        self.prediction_frame.pack(pady=20)
        
        prediction_label = tk.Label(self.prediction_frame, 
                                  text="Predict Average Fuel Consumption",
                                  font=("Helvetica", 12, "bold"),
                                  bg="white", fg="#2c3e50")
        prediction_label.pack()
        
        # Input fields for prediction
        self.input_fields = {}
        self.create_input_fields()
        
        # Predict button
        self.predict_btn = tk.Button(self.prediction_frame, 
                                   text="Predict", 
                                   command=self.predict_consumption,
                                   state=tk.DISABLED,
                                   font=("Helvetica", 10),
                                   bg="#9b59b6", fg="white",
                                   padx=10, pady=5)
        self.predict_btn.pack(pady=10)
        
        # Result label
        self.result_label = tk.Label(self.prediction_frame, 
                                   text="",
                                   font=("Helvetica", 12),
                                   bg="white", fg="#e74c3c")
        self.result_label.pack()
        
        # Status bar
        self.status_bar = tk.Label(self.root, 
                                  text="Ready", 
                                  bd=1, relief=tk.SUNKEN, 
                                  anchor=tk.W,
                                  bg="#2c3e50", fg="white")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_input_fields(self):
        fields = [
            ("Engine Size (L)", "engine_size"),
            ("Cylinders", "cylinders"),
            ("Vehicle Class (SUV: Small=0, Standard=1)", "vehicle_class"),
            ("Transmission (AS8=0, AS10=1, etc.)", "transmission"),
            ("Fuel Type (Z=0, X=1, D=2, E=3)", "fuel_type")
        ]
        
        for i, (label_text, field_name) in enumerate(fields):
            label = tk.Label(self.prediction_frame, text=label_text, bg="white")
            label.pack()
            
            entry = tk.Entry(self.prediction_frame)
            entry.pack()
            
            self.input_fields[field_name] = entry
    
    def upload_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                self.status_bar.config(text=f"Dataset loaded: {file_path}")
                
                # Filter for heavy vehicles (SUVs and trucks)
                heavy_classes = ["SUV: Standard", "Pickup truck: Standard", "SUV: Small"]
                self.dataset = self.dataset[self.dataset["Vehicle Class"].isin(heavy_classes)]
                
                # Enable buttons
                self.red_button.config(state=tk.NORMAL)
                self.ann_button.config(state=tk.NORMAL)
                self.graph_button.config(state=tk.NORMAL)
                self.predict_btn.config(state=tk.NORMAL)
                
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def generate_model(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please upload a dataset first")
            return
            
        try:
            # Prepare data
            df = self.dataset.copy()
            
            # Convert categorical variables to numerical
            df['Vehicle Class'] = df['Vehicle Class'].map({
                'SUV: Small': 0,
                'SUV: Standard': 1,
                'Pickup truck: Standard': 2
            })
            
            df['Fuel Type'] = df['Fuel Type'].map({'Z': 0, 'X': 1, 'D': 2, 'E': 3})
            
            # Extract transmission type (simplified)
            df['Transmission'] = df['Transmission'].str.extract('(\d+)').astype(float)
            
            # Select features and target
            features = ['Engine Size(L)', 'Cylinders', 'Vehicle Class', 'Transmission', 'Fuel Type']
            X = df[features]
            y = df['Fuel Consumption(Comb (L/100 km))']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            messagebox.showinfo("Success", 
                              f"Model generated successfully!\nMean Absolute Error: {mae:.2f} L/100km")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate model: {str(e)}")
    
    def run_ann(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please upload a dataset first")
            return
            
        try:
            # Prepare data (same as in generate_model)
            df = self.dataset.copy()
            df['Vehicle Class'] = df['Vehicle Class'].map({
                'SUV: Small': 0,
                'SUV: Standard': 1,
                'Pickup truck: Standard': 2
            })
            df['Fuel Type'] = df['Fuel Type'].map({'Z': 0, 'X': 1, 'D': 2, 'E': 3})
            df['Transmission'] = df['Transmission'].str.extract('(\d+)').astype(float)
            
            features = ['Engine Size(L)', 'Cylinders', 'Vehicle Class', 'Transmission', 'Fuel Type']
            X = df[features]
            y = df['Fuel Consumption(Comb (L/100 km))']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale data
            X_mean = X_train.mean()
            X_std = X_train.std()
            X_train = (X_train - X_mean) / X_std
            X_test = (X_test - X_mean) / X_std
            
            # Train ANN
            self.ann_model = MLPRegressor(hidden_layer_sizes=(50, 30), 
                                        activation='relu', 
                                        solver='adam', 
                                        max_iter=1000, 
                                        random_state=42)
            self.ann_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.ann_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            messagebox.showinfo("Success", 
                              f"ANN trained successfully!\nMean Absolute Error: {mae:.2f} L/100km")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train ANN: {str(e)}")
    
    def show_graph(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please upload a dataset first")
            return
            
        try:
            # Create a new window for the graph
            graph_window = tk.Toplevel(self.root)
            graph_window.title("Fuel Consumption Analysis")
            graph_window.geometry("800x600")
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            
            # Plot 1: Fuel consumption by vehicle class
            sns.boxplot(data=self.dataset, 
                       x='Vehicle Class', 
                       y='Fuel Consumption(Comb (L/100 km))', 
                       ax=ax1)
            ax1.set_title("Fuel Consumption by Vehicle Class")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            
            # Plot 2: Fuel consumption vs engine size
            sns.scatterplot(data=self.dataset, 
                           x='Engine Size(L)', 
                           y='Fuel Consumption(Comb (L/100 km))', 
                           hue='Vehicle Class',
                           ax=ax2)
            ax2.set_title("Fuel Consumption vs Engine Size")
            
            plt.tight_layout()
            
            # Embed plot in Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create graph: {str(e)}")
    
    def predict_consumption(self):
        if self.model is None and self.ann_model is None:
            messagebox.showerror("Error", "Please generate at least one model first")
            return
            
        try:
            # Get input values
            input_data = {}
            for field, entry in self.input_fields.items():
                value = entry.get()
                if not value:
                    messagebox.showerror("Error", f"Please enter a value for {field}")
                    return
                input_data[field] = float(value)
            
            # Create DataFrame for prediction
            X_pred = pd.DataFrame([[
                input_data['engine_size'],
                input_data['cylinders'],
                input_data['vehicle_class'],
                input_data['transmission'],
                input_data['fuel_type']
            ]], columns=['Engine Size(L)', 'Cylinders', 'Vehicle Class', 'Transmission', 'Fuel Type'])
            
            # Make prediction
            if self.model:
                pred = self.model.predict(X_pred)[0]
                model_type = "Random Forest"
            else:
                # Scale data for ANN
                X_pred_scaled = (X_pred - self.X_mean) / self.X_std
                pred = self.ann_model.predict(X_pred_scaled)[0]
                model_type = "ANN"
            
            self.result_label.config(text=f"Predicted Fuel Consumption: {pred:.2f} L/100km (using {model_type})")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FuelConsumptionApp(root)
    root.mainloop()