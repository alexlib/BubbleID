import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from BubbleID import BubbleID
import pandas as pd

# Set appearance and theme
ctk.set_appearance_mode("Dark")  # "Dark", "Light", or "System"
ctk.set_default_color_theme("dark-blue")  # Options: "blue", "green", "dark-blue"
import cv2
import glob
import os

class ImageSequenceCapture:
    def __init__(self, folder, ext="jpg"):
        self.files = sorted(glob.glob(os.path.join(folder, f"*.{ext}")))
        self.index = 0
        self.total = len(self.files)

    def isOpened(self):
        return self.total > 0

    def read(self):
        if self.index >= self.total:
            return False, None
        frame = cv2.imread(self.files[self.index])
        self.index += 1
        return True, frame

    def release(self):
        self.files = []
        self.index = 0
        self.total = 0

    def get(self, prop_id):
        # Mimic cv2.VideoCapture.get
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self.index
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30  # arbitrary default, adjust if you know the fps
        else:
            return 0

    def set(self, prop_id, value):
        # Mimic cv2.VideoCapture.set
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            if 0 <= int(value) < self.total:
                self.index = int(value)
                return True
        return False


# Create main app window
class VideoGui(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=3)  # Top row = 3 parts
        self.grid_rowconfigure(1, weight=1)  # Bottom row = 1 part

        self.grid_columnconfigure(0, weight=1)  # Left column = 1 part
        self.grid_columnconfigure(1, weight=2)  # Right column = 2 parts

        # Section 1 (Top Left)
        self.section1 = ctk.CTkFrame(self, border_width=2)
        self.section1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Main label

        # --- 1. Image Folder ---
        line1_frame = ctk.CTkFrame(self.section1)
        line1_frame.pack(padx=10, pady=(5, 10), fill="x")

        folder1_label = ctk.CTkLabel(line1_frame, text="Select Image Folder:", font=ctk.CTkFont(size=14))
        folder1_label.pack(anchor="w", pady=(0, 5))

        entry_button_frame1 = ctk.CTkFrame(line1_frame)
        entry_button_frame1.pack(fill="x")

        self.image_folder_var = ctk.StringVar()
        folder1_entry = ctk.CTkEntry(entry_button_frame1, textvariable=self.image_folder_var, state="readonly")
        folder1_entry.pack(side="left", fill="x", expand=True)

        folder1_button = ctk.CTkButton(entry_button_frame1, text="Browse", command=self.browse_image_folder, width=80)
        folder1_button.pack(side="left", padx=(10, 0))

        # --- 2. AVI File ---
        line2_frame = ctk.CTkFrame(self.section1)
        line2_frame.pack(padx=10, pady=(5, 10), fill="x")

        file_label = ctk.CTkLabel(line2_frame, text="Select .AVI File:", font=ctk.CTkFont(size=14))
        file_label.pack(anchor="w", pady=(0, 5))

        entry_button_frame2 = ctk.CTkFrame(line2_frame)
        entry_button_frame2.pack(fill="x")

        self.file_path_var = ctk.StringVar()
        file_entry = ctk.CTkEntry(entry_button_frame2, textvariable=self.file_path_var, state="readonly")
        file_entry.pack(side="left", fill="x", expand=True)

        browse_button = ctk.CTkButton(entry_button_frame2, text="Browse", command=self.browse_file, width=80)
        browse_button.pack(side="left", padx=(10, 0))

        # --- 3. Save Folder ---
        line3_frame = ctk.CTkFrame(self.section1)
        line3_frame.pack(padx=10, pady=(5, 10), fill="x")

        folder2_label = ctk.CTkLabel(line3_frame, text="Select Save Folder:", font=ctk.CTkFont(size=14))
        folder2_label.pack(anchor="w", pady=(0, 5))

        entry_button_frame3 = ctk.CTkFrame(line3_frame)
        entry_button_frame3.pack(fill="x")

        self.save_folder_var = ctk.StringVar()
        folder2_entry = ctk.CTkEntry(entry_button_frame3, textvariable=self.save_folder_var)
        folder2_entry.pack(side="left", fill="x", expand=True)

        folder2_button = ctk.CTkButton(entry_button_frame3, text="Browse", command=self.browse_save_folder, width=80)
        folder2_button.pack(side="left", padx=(10, 0))
        # extension
        line4_frame = ctk.CTkFrame(self.section1)
        line4_frame.pack(padx=10, pady=(5, 10), fill="x")

        self.entry=ctk.CTkEntry(line4_frame, placeholder_text="Enter Extension Name")
        self.entry.pack(pady=10)

        # Buttons frame at the bottom of section1
        buttons_frame = ctk.CTkFrame(self.section1)
        buttons_frame.pack(fill="x", padx=10, pady=(20, 10))

        # Generate Data button
        generate_button = ctk.CTkButton(buttons_frame, text="Generate Data", command=self.generate_data)
        generate_button.pack(side="left", expand=True, fill="x", padx=(0, 5))

        # Load Data button
        load_button = ctk.CTkButton(buttons_frame, text="Load Data", command=self.load_data)
        load_button.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # Section 2 (Top Right)
        self.bbox_data = {}  # frame_num (int) -> list of (x1, y1, x2, y2, conf, cls)

        self.section2 = ctk.CTkFrame(self, border_width=2)
        self.section2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Initialize video capture to None, no video loaded yet
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0

        self.video_label = ctk.CTkLabel(self.section2, text="")
        self.video_label.pack(padx=10, pady=(10, 0))

        self.slider = ctk.CTkSlider(
            self.section2,
            from_=0,
            to=0,
            command=self.slider_callback,
            state="disabled"
        )
        self.slider.pack(fill="x", padx=10, pady=10)

        # Checkbox for showing bounding boxes
        # Create a container frame for checkbox + button
        bb_frame = ctk.CTkFrame(self.section2, fg_color="transparent")
        bb_frame.pack(anchor="w", padx=10, pady=(0, 10))

        # Checkbox
        self.show_bb_var = ctk.BooleanVar(value=False)
        bb_checkbox = ctk.CTkCheckBox(
            bb_frame,
            text="bb",
            variable=self.show_bb_var,
            command=self.update_frame_with_bb,
            corner_radius=20
        )
        bb_checkbox.pack(side="left")

        # Button next to it
        bb_button = ctk.CTkButton(
            bb_frame,
            text="Generate Masks",
            command=self.generate_mask
        )
        bb_button.pack(side="left", padx=(5, 0))  # Small gap

        # Section 3 (Bottom Left)
        self.section3 = ctk.CTkFrame(self, border_width=2)
        self.section3.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # === Section 3 Content ===

        # --- 1. Model Weights File ---
        weights_line = ctk.CTkFrame(self.section3)
        weights_line.pack(padx=10, pady=(10, 5), fill="x")

        weights_label = ctk.CTkLabel(weights_line, text="Model Weights File:", font=ctk.CTkFont(size=14))
        weights_label.pack(anchor="w", pady=(0, 5))

        weights_entry_button_frame = ctk.CTkFrame(weights_line)
        weights_entry_button_frame.pack(fill="x")

        self.weights_path_var = ctk.StringVar()
        weights_entry = ctk.CTkEntry(weights_entry_button_frame, textvariable=self.weights_path_var, state="readonly")
        weights_entry.pack(side="left", fill="x", expand=True)

        weights_button = ctk.CTkButton(weights_entry_button_frame, text="Browse", command=self.browse_weights_file,
                                       width=80)
        weights_button.pack(side="left", padx=(10, 0))

        # --- 2. Device Selection Dropdown ---
        device_frame = ctk.CTkFrame(self.section3)
        device_frame.pack(padx=10, pady=(15, 10), fill="x")

        device_label = ctk.CTkLabel(device_frame, text="Device:", font=ctk.CTkFont(size=14))
        device_label.pack(anchor="w", pady=(0, 5))

        self.device_option = ctk.StringVar(value="cpu")  # Default to CPU
        device_dropdown = ctk.CTkOptionMenu(device_frame, variable=self.device_option, values=["cpu", "cuda"])
        device_dropdown.pack(fill="x")

        # Section 4 (Bottom Right)
        self.section4 = ctk.CTkFrame(self, border_width=2)
        self.section4.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Section 4 - Tabbed plots
        self.plot_tabs = ctk.CTkTabview(self.section4)
        self.plot_tabs.pack(fill="both", expand=True)

        self.vapor_tab = self.plot_tabs.add("Vapor Fraction")
        self.count_tab = self.plot_tabs.add("Bubble Count")
        self.size_tab = self.plot_tabs.add("Bubble Size")

        # Prepare empty dicts to store figures, axes, canvases, and vertical lines
        self.figures = {}
        self.axes = {}
        self.canvases = {}
        self.vlines = {}

        # Create empty plots for each tab ONCE
        for tab_name, tab in zip(["vapor", "count", "size"],
                                 [self.vapor_tab, self.count_tab, self.size_tab]):
            fig = Figure(figsize=(5, 2), dpi=100)
            ax = fig.add_subplot(111)
            # Empty initial plot
            line, = ax.plot([], [], color="orange", marker="o")
            vline = ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_title(tab_name.replace("_", " ").title())
            ax.set_xlabel("Frame")
            ax.set_ylabel("Value")

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            self.figures[tab_name] = fig
            self.axes[tab_name] = ax
            self.canvases[tab_name] = canvas
            self.vlines[tab_name] = vline

        self.plot_data = {"vapor": [], "count": [], "size": []}



    def show_frame(self, frame_num):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = self.cap.read()

        if ret:
            frame_number = int(frame_num) + 1  # Adjust for 1-based indexing in your txt

            if self.show_bb_var.get():
                if frame_number in self.bbox_data:
                    for (x1, y1, x2, y2, conf, cls) in self.bbox_data[frame_number]:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"Cls {cls} ({conf:.2f})"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                                    1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((320, 240))  # Resize as needed
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk  # Prevent garbage collection

            if hasattr(self, "frame_line") and self.frame_line is not None:
                self.frame_line.set_xdata([frame_number])  # move the line
                self.plot_canvas.draw()  # redraw canvas

    def slider_callback(self, value):
        self.current_frame = int(value)
        self.show_frame(self.current_frame)
        self.update_plots_with_frame(self.current_frame)

    def browse_image_folder(self):
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if folder_path:
            self.image_folder_var.set(folder_path)

    def browse_save_folder(self):
        folder_path = filedialog.askdirectory(title="Select Save Folder")
        if folder_path:
            self.save_folder_var.set(folder_path)

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="Select .AVI File", filetypes=[("AVI files", "*.avi")])
        if file_path:
            self.file_path_var.set(file_path)

    def generate_data(self):
        images=self.image_folder_var.get()
        videopath=self.file_path_var.get()
        savefolder=self.save_folder_var.get()
        extension=self.entry.get()
        modelweights_loc=self.weights_path_var.get()
        device=self.device_option.get()
        Exp=BubbleID.DataAnalysis(images, videopath, savefolder, extension, modelweights_loc, device, 1, 1, 1, 1)
        Exp.GenerateData()

    def browse_weights_file(self):
        file_path = filedialog.askopenfilename(title="Select Model Weights File", filetypes=[("All files", "*.*")])
        if file_path:
            self.weights_path_var.set(file_path)

    def update_frame_with_bb(self):
        self.show_frame(self.current_frame)

    def load_data(self):
        # pick the first image in the folder
        for file in os.listdir(self.image_folder_var.get()):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                img_path = os.path.join(self.image_folder_var.get(), file)
                break  # just take the first image found

        # open and get size
        with Image.open(img_path) as img:
            self.width, self.height = img.size


        print("Load Data clicked!")
        avi_path = self.file_path_var.get()
        img_fol=self.image_folder_var.get()

        if avi_path:
            self.cap = cv2.VideoCapture(avi_path)
        else:
            self.cap=ImageSequenceCapture(img_fol, ext="jpg")

        # Open the video
        #self.cap = cv2.VideoCapture(avi_path)
        if not self.cap.isOpened():
            print("Error: Could not open selected video.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.configure(to=self.total_frames - 1, state="normal")
        self.current_frame = 0

        self.show_frame(self.current_frame)

        # --- Load bounding boxes ---
        bbox_file=self.save_folder_var.get()+"/bb-Boiling-"+self.entry.get()+".txt"
        self.bbox_data = {}  # Reset existing boxes

        try:
            with open(bbox_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 7:
                        continue  # skip malformed lines

                    try:
                        frame_num = int(parts[0])
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        conf = float(parts[5])
                        cls = int(parts[6])

                        if frame_num not in self.bbox_data:
                            self.bbox_data[frame_num] = []
                        self.bbox_data[frame_num].append((x1, y1, x2, y2, conf, cls))
                    except ValueError as e:
                        print(f"Skipping line due to parse error: {line.strip()} | Error: {e}")
            print("Loaded bounding boxes.")
        except FileNotFoundError:
            print("Bounding box file not found.")
        #self.plot_npy_file2()
        self.plot_npy_file()  # or your full path

    def plot_npy_file(self):
        # Load your npy data only once for each category (update these paths accordingly)
        vapor_data=np.load(self.save_folder_var.get()+"/vapor_"+self.entry.get()+".npy")
        count_data=np.load(self.save_folder_var.get()+"/bubble_size_bt-"+self.entry.get()+".npy", allow_pickle=True)
        #vapor_data = np.load("C:/Users/cldunlap/Downloads/SaveData_B-126/SaveData_B-126/vapor_Test1.npy")
        #count_data = np.load("C:/Users/cldunlap/Downloads/SaveData_B-126/SaveData_B-126/bubble_size_bt-Test1.npy", allow_pickle=True)  # example

        self.plot_data["vapor"] = vapor_data/(self.width*self.height)
        bc=[]
        bs=[]
        for i in range(len(count_data)):
            bc.append(len(count_data[i]))
            bs.append(max(count_data[i]))
        self.plot_data["count"] = bc
        self.plot_data["size"] = np.array(bs)/(self.width*self.height)

        # Update plots with new data without recreating canvases
        for tab_name in ["vapor", "count", "size"]:
            ax = self.axes[tab_name]
            ax.clear()
            ax.plot(self.plot_data[tab_name], color="lightblue")
            ax.set_title(tab_name.replace("_", " ").title())
            ax.set_xlabel("Frame")
            ax.set_ylabel("Value")

            # redraw vertical line at frame 0
            self.vlines[tab_name] = ax.axvline(x=0, color='red', linestyle='--', linewidth=2)

            self.canvases[tab_name].draw()


    def update_plots_with_frame(self, frame_number):
        for tab_name in ["vapor", "count", "size"]:
            vline = self.vlines.get(tab_name)
            if vline:
                vline.set_xdata([frame_number])
                self.canvases[tab_name].draw()



    def generate_mask(self):
        if self.cap is None:
            print("No video loaded.")
            return

        # Move to the current frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read frame.")
            return

        model_weights=self.weights_path_var.get()
        cfg = get_cfg()
        cfg.OUTPUT_DIR = "./"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.DEVICE = self.device_option.get()
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_weights)  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold

        predictor = DefaultPredictor(cfg)
        outputs=predictor(frame)
        mask=np.any(outputs['instances'].pred_masks.cpu().numpy(), axis=0)

        # --- Optional: overlay mask in red with transparency ---
        mask_colored = np.zeros_like(frame)
        mask_colored[:, :] = (0, 0, 255)  # red mask
        alpha = 0.4  # transparency

        # Blend only where mask is nonzero
        frame = np.where(mask[..., None] > 0,
                         (frame * (1 - alpha) + mask_colored * alpha).astype(np.uint8),
                         frame)

        # Convert BGR → RGB for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((320, 240))
        imgtk = ImageTk.PhotoImage(image=img)

        # Show in same label
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk


import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd

# ======================
# COLUMN 1: Experiment Manager
# ======================
class ExperimentManager(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Dropdown for Steady State / Transient
        self.mode_var = ctk.StringVar(value="Steady State")
        self.dropdown = ctk.CTkOptionMenu(self, values=["Steady State", "Transient"], variable=self.mode_var)
        self.dropdown.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Experiment List
        self.exp_frame = ctk.CTkScrollableFrame(self, label_text="Experiments")
        self.exp_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Add Experiment Button
        self.add_btn = ctk.CTkButton(self, text="+ Add Experiment", command=self.add_experiment_popup)
        self.add_btn.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.experiments = []  # store experiment data

        # Make frame expand properly
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def add_experiment_popup(self):
        popup = ctk.CTkToplevel(self)
        popup.title("Add Experiment")
        popup.geometry("400x250")

        # Modal behavior
        popup.transient(self.winfo_toplevel())
        popup.grab_set()
        popup.focus()

        folder_var = ctk.StringVar()
        csv_var = ctk.StringVar()

        def select_folder():
            folder = filedialog.askdirectory()
            if folder:
                folder_var.set(folder)

        def select_csv():
            file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if file:
                csv_var.set(file)

        def save_experiment():
            folder = folder_var.get()
            csv_path = csv_var.get()
            if not folder or not csv_path:
                messagebox.showerror("Error", "Please select both folder and CSV file")
                return
            self.add_experiment(folder, csv_path)
            popup.destroy()

        ctk.CTkLabel(popup, text="Data Folder:").pack(pady=(10, 0))
        ctk.CTkEntry(popup, textvariable=folder_var).pack(fill="x", padx=10)
        ctk.CTkButton(popup, text="Browse", command=select_folder).pack(pady=5)

        ctk.CTkLabel(popup, text="CSV File:").pack(pady=(10, 0))
        ctk.CTkEntry(popup, textvariable=csv_var).pack(fill="x", padx=10)
        ctk.CTkButton(popup, text="Browse", command=select_csv).pack(pady=5)

        ctk.CTkButton(popup, text="Save", command=save_experiment).pack(pady=10)

    def add_experiment(self, folder, csv_path):
        exp_data = {"folder": folder, "csv": csv_path}
        self.experiments.append(exp_data)

        row_index = len(self.experiments) - 1
        exp_label = ctk.CTkLabel(self.exp_frame, text=f"{os.path.basename(folder)} | {os.path.basename(csv_path)}")
        exp_label.grid(row=row_index, column=0, sticky="w", padx=5, pady=2)

        remove_btn = ctk.CTkButton(self.exp_frame, text="Remove", width=80,
                                   command=lambda idx=row_index: self.remove_experiment(idx))
        remove_btn.grid(row=row_index, column=1, padx=5, pady=2)

    def remove_experiment(self, index):
        if 0 <= index < len(self.experiments):
            self.experiments.pop(index)
            for widget in self.exp_frame.winfo_children():
                widget.destroy()
            for i, exp in enumerate(self.experiments):
                exp_label = ctk.CTkLabel(self.exp_frame, text=f"{os.path.basename(exp['folder'])} | {os.path.basename(exp['csv'])}")
                exp_label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
                remove_btn = ctk.CTkButton(self.exp_frame, text="Remove", width=80,
                                           command=lambda idx=i: self.remove_experiment(idx))
                remove_btn.grid(row=i, column=1, padx=5, pady=2)


# ======================
# COLUMN 3: Axis Selector
# ======================
class AxisSelector(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # ===== X Axis =====
        ctk.CTkLabel(self, text="X Axis:").pack(anchor="w", pady=(5, 0))
        self.x_options = ["heat flux", "time", "frame"]
        self.x_vars = {}
        for opt in self.x_options:
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self, text=opt, variable=var,
                                 command=lambda o=opt: self.select_only_one(self.x_vars, o))
            cb.pack(anchor="w", padx=10,pady=5)
            self.x_vars[opt] = var

        # ===== Y Axis =====
        ctk.CTkLabel(self, text="Y Axis:").pack(anchor="w", pady=(10, 0))
        self.y_options = ["vapor fraction", "bubble count", "bubble size", "heat flux", "temperature"]
        self.y_vars = {}
        for opt in self.y_options:
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self, text=opt, variable=var,
                                 command=lambda o=opt: self.select_only_one(self.y_vars, o))
            cb.pack(anchor="w", padx=10,pady=5)
            self.y_vars[opt] = var

        self.display_avg_var = ctk.BooleanVar(value=False)
        avg_checkbox = ctk.CTkCheckBox(
            self,
            text="average",
            variable=self.display_avg_var,
            command=self.display_averages,
            corner_radius=20
        )
        avg_checkbox.pack(side="left")

    def select_only_one(self, var_dict, selected_key):
        """Uncheck all except the selected one."""
        for key, var in var_dict.items():
            var.set(key == selected_key)

    def get_selected_axes(self):
        """Return currently selected X and Y axis values."""
        x_axis = next((k for k, v in self.x_vars.items() if v.get()), None)
        y_axis = next((k for k, v in self.y_vars.items() if v.get()), None)
        return x_axis, y_axis

    def display_averages(self):
        print(self.display_avg_var.get())


# ======================
# COLUMN 2: Plot Frame
# ======================
'''
class PlotFrame(ctk.CTkFrame):
    def __init__(self, master, exp_manager, axis_selector, **kwargs):
        super().__init__(master, **kwargs)
        self.exp_manager = exp_manager
        self.axis_selector = axis_selector

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack()

        # Plot button
        frame2=ctk.CTkFrame(self)
        frame2.pack(padx=10, pady=(5, 10), fill="x")
        ctk.CTkButton(frame2, text="Load & Plot Data", command=self.load_and_plot_data).pack(pady=10)
'''
class PlotFrame(ctk.CTkFrame):
    def __init__(self, master, exp_manager, axis_selector, **kwargs):
        super().__init__(master, **kwargs)
        self.exp_manager = exp_manager
        self.axis_selector = axis_selector

        # --- Frame for the plot ---
        plot_frame = ctk.CTkFrame(self)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.toolbar.pack()

        # --- Frame for buttons ---
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))  # below plot

        ctk.CTkButton(button_frame, text="Load & Plot Data",
                      command=self.load_and_plot_data).pack(pady=10)

    def load_and_plot_data(self):
        # Get experiments
        exps = self.exp_manager.experiments
        x_axis, y_axis = self.axis_selector.get_selected_axes()

        if not exps:
            messagebox.showwarning("No Experiments", "Please add at least one experiment.")
            return
        if not x_axis or not y_axis:
            messagebox.showwarning("Missing Axes", "Please select both X and Y axes.")
            return

        self.ax.clear()
        print(self.exp_manager.mode_var.get())
        if self.exp_manager.mode_var.get()=="Transient":
            for i, exp in enumerate(exps):
                lib=self.load_data_transient(exp)
                y_data=lib[y_axis]
                x_data=lib[x_axis]
                if x_axis == "heat flux":
                    ind=np.argmax(x_data)
                    name = exp['csv'].split(".")[0].split("/")[-1]
                    sort_ind=np.argsort(np.array(x_data[:ind]))
                    x_sort=np.array(x_data)[sort_ind]
                    y_sort=np.array(y_data)[sort_ind]
                    if self.axis_selector.display_avg_var.get():
                        window=300
                        y_avg=pd.Series(y_sort).rolling(window).mean()
                        line, = self.ax.plot(x_sort,y_sort,alpha=0.1)
                        color=line.get_color()
                        self.ax.plot(x_sort,y_avg,label=name, color=color)
                    else:
                        self.ax.plot(x_sort,y_sort, alpha=0.5, label=name)
                else:
                    name = exp['csv'].split(".")[0].split("/")[-1]
                    if self.axis_selector.display_avg_var.get():
                        window=300
                        y_avg=pd.Series(y_data).rolling(window).mean()
                        line, =self.ax.plot(x_data, y_data, alpha=0.1)
                        color=line.get_color()
                        self.ax.plot(x_data,y_avg,label=name, color=color)
                    else:
                        self.ax.plot(x_data, y_data, label=name,alpha=0.5)

            self.ax.set_xlabel(x_axis.title())
            self.ax.set_ylabel(y_axis.title())
            self.ax.legend()
            self.canvas.draw()
        elif self.exp_manager.mode_var.get()=="Steady State":
            for i, exp in enumerate(exps):
                lib = self.load_data_ss(exp)
                if x_axis == "heat flux":
                    y_data = lib[y_axis]
                    x_data = lib[x_axis]
                    name=exp['csv'].split(".")[0].split("/")[-1]
                    self.ax.errorbar(x_data[0],y_data[0], xerr=x_data[1],yerr=y_data[1], fmt='s', markersize=8, capsize=2, label=name)

            self.ax.set_xlabel(x_axis.title())
            self.ax.set_ylabel(y_axis.title())
            self.ax.legend()
            self.canvas.draw()

    def load_data_transient(self,exp):
        try:
            df=pd.read_csv(exp['csv'])
            vf_file=exp['folder']+"/vapor_"+df['exp_id'][0]+".npy"
            vf=np.load(vf_file)/(df["img_w"][0]*df["img_h"][0])
            count_file=exp['folder']+"/bubble_size_bt-" + df['exp_id'][0]+".npy"
            count=np.load(count_file, allow_pickle=True)
            bc=[]
            bs=[]
            for j in range(len(count)):
                bc.append(len(count[j]))
                bs.append(max(count[j]))
            bs=np.array(bs)*df["pix_area"][0]
            frame=[i+1 for i in range(len(vf))]
            time = [(i+1)*(1/df['fps'][0]) for i in range(len(vf))]
            # Load temp
            temp_data=np.loadtxt(df['temp_path'][0], skiprows=23)
            #temp=tempdata[:,1]
            temp=np.interp(time+df['vid_start'][0], temp_data[:,0]+df['temp_start'][0], temp_data[:,1])
            # calculate heat flux

            # Heat flux time
            index_ = np.argmin(temp_data[:, 1]) + (np.argmax(temp_data[:, 1]) - np.argmin(temp_data[:, 1])) // 2
            referencetemp = [temp_data[index_, i + 1] for i in range(4)]
            sorted = np.argsort(referencetemp)[::-1]

            temp1 = temp_data[:, sorted[0] + 1]
            temp2 = temp_data[:, sorted[1] + 1]
            temp3 = temp_data[:, sorted[2] + 1]
            temp4 = temp_data[:, sorted[3] + 1]

            temp1 = np.transpose(np.array([temp1, temp2, temp3, temp4]))

            # Calculate heat flux
            tc_loc = np.array([0, 2.54, 5.08, 7.62])
            tc_loc = tc_loc * .001
            n = 4
            k = 392
            slope_d = n * np.sum(np.power(tc_loc, 2)) - np.sum(tc_loc) ** 2
            slope = (n * np.dot(temp1, tc_loc) - np.sum(tc_loc) * np.sum(temp1, axis=1)) / slope_d
            hf_og = -k * slope / 10000
            # add start time to both temp and vid
            hf=np.interp(time+df['vid_start'][0], temp_data[:,0]+df['temp_start'][0], hf_og)
            # interpolate to find hf to vid time
            # interpolate to find temp at vid time

        except Exception as e:
            print("issue")
            messagebox.showerror("Error Reading CSV")


        return {"vapor fraction": vf,"bubble count": bc, "bubble size": bs,"frame": frame,"time": time, "temperature":temp, "heat flux": hf}


    def load_data_ss(self,exp):
        try:
            df=pd.read_csv(exp['csv'])
            vf=[]
            vf_std=[]
            bc=[]
            bc_std=[]
            bs=[]
            bs_std=[]
            hf=[]
            hf_std=[]
            for i in range(len(df['exp_id'])):
                hf.append(df["heat_flux"][i])
                hf_std.append(df["hf_error"][i])
                vf_file=exp['folder']+"/vapor_"+df['exp_id'][i]+".npy"
                vf1=np.load(vf_file)/(df["img_w"][i]*df["img_h"][i])
                vf.append(np.mean(vf1))
                vf_std.append(np.std(vf1))

                count_file=exp['folder']+"/bubble_size_bt-" + df['exp_id'][i]+".npy"
                count=np.load(count_file, allow_pickle=True)
                bc1=[]
                bs1=[]
                for j in range(len(count)):
                    bc1.append(len(count[j]))
                    bs1.append(max(count[j]))
                bs1=np.array(bs1)*df['pix_area'][i]
                bc.append(np.mean(bc1))
                bc_std.append(np.std(bc1))
                bs.append(np.mean(bs1))
                bs_std.append(np.std(bs1))


        except Exception as e:
            print("issue")
            messagebox.showerror("Error Reading CSV")


        return {"heat flux": (hf,hf_std), "bubble count": (bc,bc_std), "bubble size": (bs,bs_std),"vapor fraction": (vf,vf_std)}
# ======================
# MAIN APP
# ======================
class PlottingGui(ctk.CTkFrame):
    def __init__(self,parent):
        super().__init__(parent)


        self.columnconfigure((0, 1, 2), weight=1)
        self.rowconfigure(0, weight=1)

        # Column 1 → Experiment manager
        col1 = ExperimentManager(self)
        col1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.exp_manager = col1

        # Column 3 → Axis selector
        col3 = AxisSelector(self)
        col3.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.axis_selector = col3

        # Column 2 → Plot
        col2 = PlotFrame(self, self.exp_manager, self.axis_selector)
        col2.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)


import os
import sys
import tkinter as tk

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    try:
        base_path = sys._MEIPASS  # PyInstaller temporary folder
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import customtkinter as ctk
from tkinter import filedialog
import sys
import io
import labelme2coco

# Redirect stdout to a Textbox
class TextRedirector(io.StringIO):
    def __init__(self, textbox):
        super().__init__()
        self.textbox = textbox

    def write(self, s):
        self.textbox.insert("end", s)
        self.textbox.see("end")  # auto-scroll
        self.textbox.update_idletasks()

    def flush(self):
        pass


class InputFrame(ctk.CTkFrame):
    def __init__(self, master, label, is_file=False, editable=False, **kwargs):
        super().__init__(master, **kwargs)

        self.is_file = is_file
        self.var = ctk.StringVar()

        # Label
        self.label = ctk.CTkLabel(self, text=label, width=120, anchor="w")
        self.label.grid(row=0, column=0, padx=5, pady=5)

        # Entry
        self.entry = ctk.CTkEntry(self, textvariable=self.var, width=300)
        if not editable:
            self.entry.configure(state="readonly")
        self.entry.grid(row=0, column=1, padx=5, pady=5)

        # Browse button
        self.button = ctk.CTkButton(self, text="Browse", command=self.browse)
        self.button.grid(row=0, column=2, padx=5, pady=5)

    def browse(self):
        if self.is_file:
            path = filedialog.askopenfilename()
        else:
            path = filedialog.askdirectory()
        if path:
            self.var.set(path)


class ModelTrainingFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Inputs
        self.img_folder = InputFrame(self, "Image Folder:")
        self.img_folder.pack(fill="x")
        self.lma = InputFrame(self, "Labelme Annotations:")
        self.lma.pack(fill="x")

        # Convert button
        self.convert_btn = ctk.CTkButton(self, text="Convert to COCO", command=self.convert_to_coco)
        self.convert_btn.pack(pady=5)

        self.coco_path = InputFrame(self, "Path to COCO Json:")
        self.coco_path.pack(fill="x")
        self.save_folder = InputFrame(self, "Save Folder:", editable=True)
        self.save_folder.pack(fill="x")
        self.weights = InputFrame(self, "Pretrained Weights:", is_file=True)
        self.weights.pack(fill="x")

        # Side-by-side buttons
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.pack(pady=10)
        self.finetune_btn = ctk.CTkButton(self.btn_frame, text="Fine Tune Model", command=self.fine_tune_model)
        self.finetune_btn.grid(row=0, column=0, padx=5)
        self.train_btn = ctk.CTkButton(self.btn_frame, text="Train Model", command=self.train_model)
        self.train_btn.grid(row=0, column=1, padx=5)

        # Console output box
        self.console = ctk.CTkTextbox(self, height=200)
        self.console.pack(fill="both", padx=10, pady=10, expand=True)

        # Redirect stdout
        sys.stdout = TextRedirector(self.console)
        print("ModelTrainingFrame ready.")

    # --- Button Actions ---
    def convert_to_coco(self):
        labelme2coco.convert(self.lma.var.get(),".tests/data/",0.85)
    def fine_tune_model(self):
        BubbleID.FineTuneModel(" ",self.weights.var.get(),".tests/data/train.json","tests/data/val.json")
    def train_model(self):
        BubbleID.TrainSegmentationModel("./tests/data/train.json", "")

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BubbleID")
        self.geometry("1200x700")
        #icon=tk.PhotoImage(file="/home/ishraq1235/Documents/BubbleIDAnalysis/icon.png")
        #self.iconphoto(True, icon)
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True)

        tab3 = self.tabview.add("Training")
        tab1 = self.tabview.add("Video")
        tab2 = self.tabview.add("Plotting")  # placeholder

        self.training = ModelTrainingFrame(tab3)
        self.training.pack(fill="both", expand=True)

        self.bubble_gui = VideoGui(tab1)
        self.bubble_gui.pack(fill="both", expand=True)

        # Later you can add the other GUI similarly
        self.plotting_gui = PlottingGui(tab2)
        self.plotting_gui.pack(fill="both", expand=True)

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = MainApp()
    app.mainloop()
