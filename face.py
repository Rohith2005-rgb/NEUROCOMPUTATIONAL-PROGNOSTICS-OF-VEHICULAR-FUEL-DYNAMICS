import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FuelConsumptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Model for Average Fuel Consumption in Heavy Vehicles")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.dataset = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = StandardScaler()
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', padx=10, pady=10)
        header_frame.pack(fill='x')
        
        self.header_label = tk.Label(
            header_frame,
            text="A Machine Learning Model for Average Fuel Consumption in Heavy Vehicles",
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        self.header_label.pack()
        
        # Upload section
        upload_frame = tk.LabelFrame(
            self.root,
            text="Upload Heavy Vehicles Fuel Dataset",
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=15,
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        upload_frame.pack(pady=10, padx=20, fill='x')
        
        self.upload_status = tk.Label(
            upload_frame,
            text="No dataset loaded",
            fg='red',
            bg='#f0f0f0',
            font=('Arial', 9)
        )
        self.upload_status.pack(side='left', expand=True, anchor='w')
        
        upload_btn = tk.Button(
            upload_frame,
            text="Browse Dataset",
            command=self.upload_dataset,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=10,
            pady=5
        )
        upload_btn.pack(side='right')
        
        # Options frame
        options_frame = tk.LabelFrame(
            self.root,
            text="Options",
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=15,
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        options_frame.pack(pady=10, padx=20, fill='x')
        
        # Buttons
        btn_style = {
            'font': ('Arial', 10, 'bold'),
            'width': 25,
            'pady': 8,
            'bd': 0
        }
        
        self.btn_generate = tk.Button(
            options_frame,
            text="Generate Model",
            command=self.generate_model,
            bg='#2ecc71',
            fg='white',
            **btn_style
        )
        self.btn_generate.pack(pady=5)
        
        self.btn_ann = tk.Button(
            options_frame,
            text="Run ANN Algorithm",
            command=self.run_ann,
            bg='#e74c3c',
            fg='white',
            **btn_style
        )
        self.btn_ann.pack(pady=5)
        
        self.btn_graphs = tk.Button(
            options_frame,
            text="Fuel Consumption Graph",
            command=self.show_graphs,
            bg='#9b59b6',
            fg='white',
            **btn_style
        )
        self.btn_graphs.pack(pady=5)
        
        self.btn_exit = tk.Button(
            options_frame,
            text="Exit",
            command=self.root.quit,
            bg='#34495e',
            fg='white',
            **btn_style
        )
        self.btn_exit.pack(pady=5)
        
        # Results frame
        self.results_frame = tk.LabelFrame(
            self.root,
            text="Results",
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=15,
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        self.results_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Text widget with scrollbar
        self.results_text = tk.Text(
            self.results_frame,
            height=15,
            wrap='word',
            font=('Consolas', 10),
            bg='white',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        self.results_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(
            self.results_frame,
            orient='vertical',
            command=self.results_text.yview
        )
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief='sunken',
            anchor='w',
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    def upload_dataset(self):
        file_types = [
            ('CSV files', '*.csv'),
            ('Text files', '*.txt'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            try:
                # Read the file
                if file_path.endswith('.csv'):
                    self.dataset = pd.read_csv(file_path)
                else:
                    self.dataset = pd.read_csv(file_path, sep='\t')
                
                # Clean column names and data
                self.dataset.columns = [col.strip().replace('<p>', '').replace('</p>', '') 
                                      for col in self.dataset.columns]
                self.dataset = self.dataset.applymap(
                    lambda x: x.strip().replace('<p>', '').replace('</p>', '') 
                    if isinstance(x, str) else x
                )
                
                # Update UI
                self.upload_status.config(
                    text=f"Dataset loaded: {file_path.split('/')[-1]}",
                    fg='green'
                )
                self.log_message(f"✅ Dataset successfully loaded from: {file_path}")
                self.log_message(f"\n📊 Dataset preview (first 5 rows):\n")
                self.log_message(self.dataset.head().to_string())
                self.status_bar.config(text="Dataset loaded successfully")
                
                # Enable buttons
                self.btn_generate.config(state='normal')
                self.btn_ann.config(state='disabled')
                self.btn_graphs.config(state='normal')
                
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to load dataset:\n{str(e)}"
                )
                self.status_bar.config(text="Error loading dataset")
    
    def generate_model(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please upload a dataset first!")
            return
            
        try:
            # Prepare data
            X = self.dataset.drop('class', axis=1)
            y = self.dataset['class']
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            self.log_message("\n🔧 Model Generation Complete!")
            self.log_message(f"  - Training set size: {len(self.X_train)} samples")
            self.log_message(f"  - Test set size: {len(self.X_test)} samples")
            self.log_message(f"  - Features used: {list(self.dataset.columns[:-1])}")
            
            # Enable ANN button
            self.btn_ann.config(state='normal')
            self.status_bar.config(text="Model generated successfully")
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to generate model:\n{str(e)}"
            )
            self.status_bar.config(text="Error generating model")
    
    def run_ann(self):
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please generate the model first!")
            return
            
        try:
            # Show loading message
            self.log_message("\n🧠 Training ANN Model... (Please wait)")
            self.root.update()
            
            # Create and train ANN model
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                verbose=False
            )
            
            self.model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            # Display results
            self.log_message("\n📈 ANN Algorithm Results:")
            self.log_message(f"  - Mean Squared Error (MSE): {mse:.4f}")
            self.log_message(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
            self.log_message(f"  - R-squared Score: {r2:.4f}")
            
            # Show sample predictions
            sample_results = pd.DataFrame({
                'Actual': self.y_test[:10],
                'Predicted': y_pred[:10].round(2)
            })
            self.log_message("\n🔍 Sample Predictions:\n" + sample_results.to_string(index=False))
            
            self.status_bar.config(text="ANN training completed successfully")
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to run ANN algorithm:\n{str(e)}"
            )
            self.status_bar.config(text="Error in ANN training")
    
    def show_graphs(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please upload a dataset first!")
            return
            
        try:
            # Create a new window for graphs
            graph_window = tk.Toplevel(self.root)
            graph_window.title("Fuel Consumption Analysis Graphs")
            graph_window.geometry("1000x800")
            graph_window.configure(bg='#f0f0f0')
            
            # Create notebook for multiple tabs
            notebook = ttk.Notebook(graph_window)
            notebook.pack(fill='both', expand=True)
            
            # Tab 1: Feature Distributions
            dist_frame = ttk.Frame(notebook)
            notebook.add(dist_frame, text="Feature Distributions")
            
            fig1, axes1 = plt.subplots(3, 3, figsize=(12, 10))
            fig1.suptitle("Feature Distributions", fontsize=14)
            
            features = self.dataset.columns[:-1]  # Exclude target
            for i, col in enumerate(features):
                ax = axes1[i//3, i%3]
                self.dataset[col].plot(kind='hist', ax=ax, bins=20, color='#3498db')
                ax.set_title(col, fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            
            canvas1 = FigureCanvasTkAgg(fig1, master=dist_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill='both', expand=True)
            
            # Tab 2: Correlation Matrix
            corr_frame = ttk.Frame(notebook)
            notebook.add(corr_frame, text="Correlation Matrix")
            
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            corr = self.dataset.corr()
            cax = ax2.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            fig2.colorbar(cax)
            
            ax2.set_xticks(range(len(corr.columns)))
            ax2.set_yticks(range(len(corr.columns)))
            ax2.set_xticklabels(corr.columns, rotation=45, ha='left')
            ax2.set_yticklabels(corr.columns)
            ax2.set_title("Feature Correlation Matrix", pad=20)
            
            canvas2 = FigureCanvasTkAgg(fig2, master=corr_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill='both', expand=True)
            
            # Tab 3: Actual vs Predicted (if model exists)
            if self.model is not None:
                pred_frame = ttk.Frame(notebook)
                notebook.add(pred_frame, text="Actual vs Predicted")
                
                y_pred = self.model.predict(self.X_test)
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.scatter(self.y_test, y_pred, alpha=0.6, color='#e74c3c')
                ax3.plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 
                        'k--', lw=2)
                ax3.set_xlabel('Actual Fuel Consumption', fontsize=10)
                ax3.set_ylabel('Predicted Fuel Consumption', fontsize=10)
                ax3.set_title('Actual vs Predicted Fuel Consumption', fontsize=12)
                ax3.grid(True, linestyle='--', alpha=0.6)
                
                canvas3 = FigureCanvasTkAgg(fig3, master=pred_frame)
                canvas3.draw()
                canvas3.get_tk_widget().pack(fill='both', expand=True)
            
            self.status_bar.config(text="Graphs generated successfully")
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to generate graphs:\n{str(e)}"
            )
            self.status_bar.config(text="Error generating graphs")
    
    def log_message(self, message):
        self.results_text.insert("end", message + "\n")
        self.results_text.see("end")
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = FuelConsumptionApp(root)
    root.mainloop()