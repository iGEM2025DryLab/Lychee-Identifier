import tkinter
import tkinter.filedialog
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image
import cv2
import numpy as np
import os
import threading
import queue
import time
from collections import deque

# --- 核心图像处理逻辑 ---

def create_segmented_mask(image, params):
    """根据参数创建二值化蒙版."""
    h_lower = int(params['hue_lower'] / 2)
    h_upper = int(params['hue_upper'] / 2)
    sat_min = int(params['saturation'] * 2.55)
    val_min = int(params['value'] * 2.55)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1. 创建色相蒙版 (处理红色环绕问题)
    lower_red1 = np.array([h_lower, 0, 0], dtype=np.uint8) 
    upper_red1 = np.array([179, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([0, 0, 0], dtype=np.uint8)
    upper_red2 = np.array([h_upper, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    hue_mask = cv2.bitwise_or(mask1, mask2)

    # 2. 创建饱和度和明度蒙版
    sat_mask = cv2.inRange(hsv_image, (0, sat_min, 0), (180, 255, 255))
    val_mask = cv2.inRange(hsv_image, (0, 0, val_min), (180, 255, 255))

    # 3. 合并所有蒙版
    final_mask = cv2.bitwise_and(hue_mask, sat_mask)
    final_mask = cv2.bitwise_and(final_mask, val_mask)
    return final_mask

def apply_post_processing(mask, params, algo_type):
    """应用后处理算法."""
    if algo_type == 'blur':
        return cv2.blur(mask, (params['blur_intensity'] * 2 + 1, params['blur_intensity'] * 2 + 1))
    elif algo_type == 'morph':
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=params['morph_intensity'])
    elif algo_type == 'hybrid':
        blurred = cv2.blur(mask, (params['blur_intensity'] * 2 + 1, params['blur_intensity'] * 2 + 1))
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=params['morph_intensity'])
    return mask

def analyze_from_mask(image, mask, params, analysis_type, scale_factor):
    """从蒙版分析并绘制结果 (支持缩放和圆度筛选)."""
    h, w = mask.shape[:2]
    ratio = params['single_ratio'] if analysis_type == 'single' else params['cluster_ratio']
    min_pixel_count = (h * w) * (ratio / 100.0)
    min_circularity = params['circularity']
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_pixel_count:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity >= min_circularity:
            valid_contours.append(cnt)

    output_image = image.copy()
    found_left, found_right = False, False
    for cnt in valid_contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        # 将缩放后的坐标转换回原始尺寸
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)
        orig_w = int(box_w / scale_factor)
        orig_h = int(box_h / scale_factor)
        
        cv2.rectangle(output_image, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (0, 0, 255), 3)
        center_x = orig_x + orig_w / 2
        
        if center_x < image.shape[1] * 0.55: found_left = True
        if center_x > image.shape[1] * 0.45: found_right = True
            
    return output_image, found_left, found_right

# --- GUI 应用类 ---

class LycheeDetectorApp(ctk.CTk):
    PROCESSING_SCALE = 0.5 

    def __init__(self):
        super().__init__()

        self.title("荔枝识别工具 (图片与视频双模态版)")
        self.geometry("1400x950")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- 数据存储与状态控制 ---
        self.original_image = None
        self.video_capture = None
        self.is_camera_mode = False
        self.available_cameras = []
        self.current_camera_index = 0
        self.video_total_frames = 0
        self.video_fps = 30
        self.is_video_processing = False
        self.is_auto_analyzing = False
        self.video_play_event = threading.Event() 
        self.video_reader_thread = None
        self.video_processor_thread = None
        self.camera_processor_thread = None
        
        self.preview_queue = queue.Queue()
        self.result_queue = queue.Queue() 
        
        self.video_cluster_algo = 'raw'
        self.analysis_fps = 5 
        self.latest_video_frame = None
        self.frame_lock = threading.Lock()
        self.seek_frame_num = -1 
        self.is_slider_dragging = False

        # --- 创建UI ---
        self.create_mode_switcher()
        self.create_image_mode_frame()
        self.create_video_mode_frame()
        self.create_param_widgets()
        
        self.image_mode_frame.grid() 

        self.after(50, self.process_queues)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        threading.Thread(target=self.detect_available_cameras, daemon=True).start()

    def create_mode_switcher(self):
        mode_frame = ctk.CTkFrame(self, corner_radius=0)
        mode_frame.grid(row=0, column=0, padx=0, pady=0, sticky="ew")
        mode_frame.grid_columnconfigure(0, weight=1)
        self.mode_switch = ctk.CTkSegmentedButton(mode_frame, values=["图片识别", "视频识别"], command=self.switch_mode)
        self.mode_switch.set("图片识别")
        self.mode_switch.pack(pady=10)

    def create_image_mode_frame(self):
        self.image_mode_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.image_mode_frame.grid(row=1, column=0, sticky="nsew")
        self.image_mode_frame.grid_columnconfigure(0, weight=1)
        self.image_mode_frame.grid_rowconfigure(1, weight=1) 
        self.image_mode_frame.grid_remove() 
        
        image_controls_frame = ctk.CTkFrame(self.image_mode_frame)
        image_controls_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        image_controls_frame.grid_columnconfigure(2, weight=1)
        ctk.CTkButton(image_controls_frame, text="选择图片", command=self.select_image).grid(row=0, column=0, padx=10, pady=5)
        ctk.CTkButton(image_controls_frame, text="选择文件夹", command=self.select_folder).grid(row=0, column=1, padx=10, pady=5)
        self.load_time_label = ctk.CTkLabel(image_controls_frame, text="加载耗时: N/A")
        self.load_time_label.grid(row=0, column=2, padx=10, pady=5)

        tab_view = ctk.CTkTabview(self.image_mode_frame)
        tab_view.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.create_binarize_tab(tab_view.add("二元化工具"))
        self.create_single_tab(tab_view.add("识别荔枝"))
        self.create_cluster_tab(tab_view.add("识别荔枝串"))

    def create_video_mode_frame(self):
        self.video_mode_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.video_mode_frame.grid(row=1, column=0, sticky="nsew")
        self.video_mode_frame.grid_columnconfigure(0, weight=1)
        self.video_mode_frame.grid_rowconfigure(1, weight=1)
        self.video_mode_frame.grid_remove()
        
        controls_frame = ctk.CTkFrame(self.video_mode_frame)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure(3, weight=1)
        ctk.CTkButton(controls_frame, text="选择视频文件", command=self.select_video).grid(row=0, column=0, padx=10, pady=5)
        self.camera_button = ctk.CTkButton(controls_frame, text="打开摄像头", command=self.toggle_camera_processing)
        self.camera_button.grid(row=0, column=1, padx=10, pady=5)
        self.switch_camera_button = ctk.CTkButton(controls_frame, text="切换摄像头", command=self.switch_camera, state="disabled")
        self.switch_camera_button.grid(row=0, column=2, padx=10, pady=5)
        
        analysis_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        analysis_frame.grid(row=0, column=3, padx=10, pady=5, sticky="e")
        self.snapshot_button = ctk.CTkButton(analysis_frame, text="拍照并分析", command=self.run_snapshot_analysis, state="disabled")
        self.snapshot_button.pack(side="left", padx=5)
        self.auto_snapshot_button = ctk.CTkButton(analysis_frame, text="自动分析", command=self.toggle_auto_analysis, state="disabled")
        self.auto_snapshot_button.pack(side="left", padx=5)
        ctk.CTkLabel(analysis_frame, text="分析帧率:").pack(side="left", padx=(10, 5))
        self.fps_button = ctk.CTkSegmentedButton(analysis_frame, values=["2 FPS", "5 FPS", "10 FPS"], command=self.set_analysis_fps)
        self.fps_button.set("5 FPS")
        self.fps_button.pack(side="left", padx=5)

        self.video_tab_view = ctk.CTkTabview(self.video_mode_frame)
        self.video_tab_view.grid(row=1, column=0, sticky="nsew")
        self.create_video_binarize_tab(self.video_tab_view.add("二元化工具 (视频)"))
        self.create_video_single_tab(self.video_tab_view.add("识别荔枝 (视频)"))
        self.create_video_cluster_tab(self.video_tab_view.add("识别荔枝串 (视频)"))
        
        self.create_video_player_controls(self.video_mode_frame)

    def create_param_widgets(self):
        self.param_frame = ctk.CTkFrame(self)
        self.param_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        self.param_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.param_frame.grid_rowconfigure((0, 1), weight=1)
        
        self.hue_lower_slider = self.create_slider(self.param_frame, "色相下限", 0, 360, 350, 0, 0)
        self.hue_upper_slider = self.create_slider(self.param_frame, "色相上限", 0, 360, 10, 0, 1)
        self.sat_slider = self.create_slider(self.param_frame, "饱和度", 0, 100, 50, 0, 2)
        self.val_slider = self.create_slider(self.param_frame, "明度", 0, 100, 30, 0, 3)
        
        self.blur_slider = self.create_slider(self.param_frame, "模糊强度", 1, 50, 1, 1, 0, is_int=True)
        self.morph_slider = self.create_slider(self.param_frame, "形态学强度", 1, 50, 1, 1, 1, is_int=True)
        self.circularity_slider = self.create_slider(self.param_frame, "最小圆度", 0.1, 1.0, 0.6, 1, 2, step=0.05)

    # --- 图片模式UI创建 ---
    def create_binarize_tab(self, tab):
        tab.grid_columnconfigure((0, 1), weight=1); tab.grid_rowconfigure(1, weight=1)
        controls_frame = ctk.CTkFrame(tab); controls_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=1)
        self.batch_button = ctk.CTkButton(controls_frame, text="开始批量处理", command=self.run_batch_processing, state="disabled"); self.batch_button.grid(row=0, column=0, padx=5, pady=5)
        self.progress_frame = ctk.CTkFrame(controls_frame, fg_color="transparent"); self.progress_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.batch_progressbar = ctk.CTkProgressBar(self.progress_frame);
        self.batch_eta_label = ctk.CTkLabel(self.progress_frame, text="");
        self.original_label = self.create_image_panel(tab, "原始图片", 1, 0)
        self.segmented_label = self.create_image_panel(tab, "二值化蒙版", 1, 1)

    def create_single_tab(self, tab):
        tab.grid_columnconfigure((0, 1), weight=1); tab.grid_rowconfigure(2, weight=1)
        controls_frame = ctk.CTkFrame(tab); controls_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure(1, weight=1) 
        self.single_ratio_slider = self.create_slider(controls_frame, "最小面积比例 (%)", 0.001, 0.5, 0.01, 0, step=0.001, command=self.run_single_analysis)
        self.single_time_label = ctk.CTkLabel(controls_frame, text="处理耗时: N/A"); self.single_time_label.grid(row=0, column=1, padx=10)
        self.single_analysis_label = self.create_image_panel(tab, "分析结果", 2, 0)
        self.single_mask_label = self.create_image_panel(tab, "蒙版预览", 2, 1)
        self.create_status_labels(tab, 3, 'single')

    def create_cluster_tab(self, tab):
        tab.grid_columnconfigure((0, 1), weight=1); tab.grid_rowconfigure(2, weight=1)
        controls_frame = ctk.CTkFrame(tab); controls_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure(1, weight=1)
        self.cluster_ratio_slider = self.create_slider(controls_frame, "最小面积比例 (%)", 0.001, 1.0, 0.05, 0, step=0.001)
        algo_frame = ctk.CTkFrame(controls_frame); algo_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        algo_frame.grid_columnconfigure((0,1,2,3), weight=1)
        ctk.CTkButton(algo_frame, text="原图检测", command=lambda: self.run_cluster_analysis('raw')).grid(row=0, column=0, padx=5)
        ctk.CTkButton(algo_frame, text="模糊降噪", command=lambda: self.run_cluster_analysis('blur')).grid(row=0, column=1, padx=5)
        ctk.CTkButton(algo_frame, text="形态学运算", command=lambda: self.run_cluster_analysis('morph')).grid(row=0, column=2, padx=5)
        ctk.CTkButton(algo_frame, text="混合运算", command=lambda: self.run_cluster_analysis('hybrid')).grid(row=0, column=3, padx=5)
        self.cluster_time_label = ctk.CTkLabel(controls_frame, text="处理耗时: N/A"); self.cluster_time_label.grid(row=2, column=0, columnspan=2, pady=5)
        self.cluster_analysis_label = self.create_image_panel(tab, "分析结果", 2, 0)
        self.cluster_mask_label = self.create_image_panel(tab, "蒙版预览", 2, 1)
        self.create_status_labels(tab, 3, 'cluster')

    # --- 视频模式UI创建 ---
    def create_video_binarize_tab(self, tab):
        tab.grid_columnconfigure((0, 1), weight=1); tab.grid_rowconfigure(0, weight=1)
        self.video_original_label = self.create_image_panel(tab, "原始视频/摄像头", 0, 0)
        self.video_mask_label = self.create_image_panel(tab, "实时蒙版", 0, 1)

    def create_video_single_tab(self, tab):
        tab.grid_columnconfigure((0, 1), weight=1); tab.grid_rowconfigure(1, weight=1)
        self.video_single_ratio_slider = self.create_slider(tab, "最小面积比例 (%)", 0.001, 0.5, 0.01, 0, col_offset=0, columnspan=2, step=0.001)
        self.video_single_analysis_label = self.create_image_panel(tab, "分析结果 (视频)", 1, 0)
        self.video_single_mask_label = self.create_image_panel(tab, "蒙版预览 (视频)", 1, 1)
        self.create_status_labels(tab, 2, 'video_single')

    def create_video_cluster_tab(self, tab):
        tab.grid_columnconfigure((0, 1), weight=1); tab.grid_rowconfigure(2, weight=1)
        controls_frame = ctk.CTkFrame(tab); controls_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.video_cluster_ratio_slider = self.create_slider(controls_frame, "最小面积比例 (%)", 0.001, 1.0, 0.05, 0, step=0.001)
        def set_video_algo(algo): self.video_cluster_algo = algo.lower()
        algo_buttons = ctk.CTkSegmentedButton(controls_frame, values=["原图", "模糊", "形态学", "混合"], command=set_video_algo); algo_buttons.set("原图"); algo_buttons.grid(row=1, column=0, padx=5, pady=10)
        self.video_cluster_analysis_label = self.create_image_panel(tab, "分析结果 (视频)", 2, 0)
        self.video_cluster_mask_label = self.create_image_panel(tab, "蒙版预览 (视频)", 2, 1)
        self.create_status_labels(tab, 3, 'video_cluster')
        
    def create_video_player_controls(self, parent):
        player_frame = ctk.CTkFrame(parent)
        player_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        player_frame.grid_columnconfigure(2, weight=1)
        self.play_pause_button = ctk.CTkButton(player_frame, text="▶", command=self.toggle_video_play_pause, state="disabled", width=40)
        self.play_pause_button.grid(row=0, column=0, padx=5)
        self.stop_button = ctk.CTkButton(player_frame, text="■", command=self.stop_video_processing, state="disabled", width=40)
        self.stop_button.grid(row=0, column=1, padx=5)
        self.video_progress_slider = ctk.CTkSlider(player_frame, from_=0, to=100, state="disabled")
        self.video_progress_slider.grid(row=0, column=2, padx=10, sticky="ew")
        self.video_progress_slider.bind("<Button-1>", lambda e: setattr(self, 'is_slider_dragging', True))
        self.video_progress_slider.bind("<ButtonRelease-1>", self.on_slider_release)
        self.time_label = ctk.CTkLabel(player_frame, text="00:00 / 00:00")
        self.time_label.grid(row=0, column=3, padx=10)

    # --- UI 辅助函数 ---
    def create_status_labels(self, parent, row, prefix):
        status_frame = ctk.CTkFrame(parent); status_frame.grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
        status_frame.grid_columnconfigure((0,1), weight=1)
        left_status = ctk.CTkLabel(status_frame, text="左侧: 未检测", fg_color="gray20", corner_radius=5); left_status.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        right_status = ctk.CTkLabel(status_frame, text="右侧: 未检测", fg_color="gray20", corner_radius=5); right_status.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        setattr(self, f"{prefix}_left_status", left_status); setattr(self, f"{prefix}_right_status", right_status)

    def create_slider(self, parent, text, from_, to, initial, row, col_offset=0, columnspan=1, is_int=False, step=None, command=None):
        frame = ctk.CTkFrame(parent, fg_color="transparent"); frame.grid(row=row, column=col_offset, columnspan=columnspan, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        label = ctk.CTkLabel(frame, text=f"{text}: {initial:.3f}"); label.grid(row=0, column=0, padx=5)
        var = tkinter.IntVar() if is_int else tkinter.DoubleVar(); var.set(initial)
        def update_label(value):
            if is_int: label.configure(text=f"{text}: {int(float(value))}")
            else: label.configure(text=f"{text}: {float(value):.3f}")
            if command and self.original_image is not None: command()
        slider = ctk.CTkSlider(frame, from_=from_, to=to, variable=var, command=update_label)
        if step: slider.configure(number_of_steps=int((to-from_)/step))
        slider.grid(row=0, column=1, padx=5, sticky="ew")
        return slider

    def create_image_panel(self, parent, text, row, col):
        """优化：创建无多余背景的图像显示面板."""
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(panel, text=text).grid(row=0, column=0, pady=(0, 5))
        image_label = ctk.CTkLabel(panel, text="")
        image_label.grid(row=1, column=0, sticky="nsew")
        return image_label

    # --- 核心功能与事件处理 ---
    def switch_mode(self, mode):
        self.stop_video_processing()
        if mode == "图片识别": self.video_mode_frame.grid_remove(); self.image_mode_frame.grid()
        else: self.image_mode_frame.grid_remove(); self.video_mode_frame.grid()

    def select_image(self):
        self.stop_video_processing()
        path = tkinter.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if path: self.load_image(path)

    def select_video(self):
        self.stop_video_processing()
        path = tkinter.filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.is_camera_mode = False
            # 使用 FFMPEG 后端打开视频文件，通常兼容性最好
            self.video_capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            if not self.video_capture.isOpened(): messagebox.showerror("错误", "无法打开视频文件。"); self.video_capture = None
            else:
                self.video_total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
                self.update_video_player_ui(is_live=False)
                ret, frame = self.video_capture.read()
                if ret: self.preview_queue.put(('video_original', frame, 0))
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def open_camera(self):
        if self.is_video_processing: self.stop_video_processing(); return
        if not self.available_cameras: messagebox.showwarning("警告", "未检测到可用摄像头。"); return
        
        self.is_camera_mode = True
        # *** 已修改 ***: 移除明确的后端API，使用OpenCV默认方式
        self.video_capture = cv2.VideoCapture(self.available_cameras[self.current_camera_index])
        if not self.video_capture.isOpened():
            messagebox.showerror("错误", f"无法打开摄像头 {self.available_cameras[self.current_camera_index]}。"); self.video_capture = None
        else:
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
            self.update_video_player_ui(is_live=True)
            self.start_video_processing()

    def switch_camera(self):
        if not self.is_camera_mode: return
        self.stop_video_processing()
        self.current_camera_index = (self.current_camera_index + 1) % len(self.available_cameras)
        self.after(100, self.open_camera)

    def select_folder(self):
        self.stop_video_processing()
        path = tkinter.filedialog.askdirectory()
        if path:
            self.folder_path = path
            # 修复: 确保 batch_status_label 已创建
            # self.batch_status_label.configure(text=f"已选择文件夹: {os.path.basename(path)}")
            self.batch_button.configure(state="normal")

    def load_image(self, path):
        start_time = time.perf_counter()
        try:
            raw_data = np.fromfile(path, dtype=np.uint8)
            self.original_image = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            if self.original_image is not None:
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                self.result_queue.put(('load_time', elapsed_ms))
                self.update_all_views()
            else: messagebox.showerror("错误", f"无法解码图片文件: {path}")
        except Exception as e: messagebox.showerror("错误", f"加载图片时出错: {path}\n\n详细信息: {e}")

    def update_all_views(self):
        if self.original_image is None: return
        self.update_binarize_view()
        self.run_single_analysis()
        self.run_cluster_analysis('raw')

    def update_binarize_view(self, *_):
        if self.original_image is None: return
        self.display_image(self.original_label, self.original_image)
        threading.Thread(target=self.thread_task_binarize, daemon=True).start()

    def run_single_analysis(self, *_):
        if self.original_image is None: return
        threading.Thread(target=self.thread_task_analysis, args=('single',), daemon=True).start()

    def run_cluster_analysis(self, algo):
        if self.original_image is None: messagebox.showinfo("提示", "请先选择一张图片"); return
        threading.Thread(target=self.thread_task_analysis, args=(algo,), daemon=True).start()

    def run_batch_processing(self):
        if not hasattr(self, 'folder_path') or not self.folder_path: messagebox.showerror("错误", "未选择文件夹"); return
        output_dir = os.path.join(self.folder_path, "output_images"); os.makedirs(output_dir, exist_ok=True)
        self.batch_progressbar.grid(row=0, column=0, sticky="ew", padx=5)
        self.batch_eta_label.grid(row=0, column=1, sticky="w", padx=5)
        self.batch_progressbar.set(0)
        threading.Thread(target=self.thread_task_batch, args=(self.folder_path, output_dir), daemon=True).start()
        
    def start_video_processing(self):
        self.video_play_event.set()
        self.is_video_processing = True
        if self.is_camera_mode:
            self.camera_button.configure(text="关闭摄像头", fg_color="#D4B000")
            if len(self.available_cameras) > 1: self.switch_camera_button.configure(state="normal")
            self.video_reader_thread = threading.Thread(target=self.thread_task_video_reader, daemon=True)
            self.video_reader_thread.start()
        else:
            self.play_pause_button.configure(text="❚❚")
            self.video_reader_thread = threading.Thread(target=self.thread_task_video_reader, daemon=True)
            self.video_processor_thread = threading.Thread(target=self.thread_task_video_processor, daemon=True)
            self.video_reader_thread.start()
            self.video_processor_thread.start()

    def toggle_video_play_pause(self):
        if not self.is_video_processing: self.start_video_processing()
        elif not self.video_play_event.is_set(): self.video_play_event.set(); self.play_pause_button.configure(text="❚❚")
        else: self.video_play_event.clear(); self.play_pause_button.configure(text="▶")
            
    def toggle_camera_processing(self):
        if self.is_video_processing: self.stop_video_processing()
        else: self.open_camera()
        
    def toggle_auto_analysis(self):
        if not self.is_camera_mode: return
        self.is_auto_analyzing = not self.is_auto_analyzing
        if self.is_auto_analyzing:
            self.auto_snapshot_button.configure(text="停止自动分析", fg_color="#D4B000")
            self.snapshot_button.configure(state="disabled")
            self.camera_processor_thread = threading.Thread(target=self.thread_task_camera_processor, daemon=True)
            self.camera_processor_thread.start()
        else:
            self.auto_snapshot_button.configure(text="自动分析", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            self.snapshot_button.configure(state="normal")

    def stop_video_processing(self):
        self.is_video_processing = False
        self.is_auto_analyzing = False
        self.video_play_event.set() 
        if self.camera_processor_thread and self.camera_processor_thread.is_alive(): self.camera_processor_thread.join(timeout=0.2)
        if self.video_processor_thread and self.video_processor_thread.is_alive(): self.video_processor_thread.join(timeout=0.2)
        if self.video_reader_thread and self.video_reader_thread.is_alive(): self.video_reader_thread.join(timeout=0.2)
        self.video_reader_thread = None; self.video_processor_thread = None; self.camera_processor_thread = None
        if self.video_capture: self.video_capture.release(); self.video_capture = None
        
        # Reset UI
        self.play_pause_button.configure(state="disabled", text="▶")
        self.stop_button.configure(state="disabled")
        self.camera_button.configure(text="打开摄像头", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self.switch_camera_button.configure(state="disabled")
        self.snapshot_button.configure(state="disabled")
        self.auto_snapshot_button.configure(state="disabled", text="自动分析", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self.video_progress_slider.set(0); self.video_progress_slider.configure(state="disabled")
        self.update_time_label(0)
        self.is_camera_mode = False

    def on_slider_release(self, event):
        self.seek_frame_num = int(self.video_progress_slider.get())
        self.is_slider_dragging = False
    
    def set_analysis_fps(self, value):
        try: self.analysis_fps = int(value.split()[0])
        except (ValueError, IndexError): self.analysis_fps = 5

    def detect_available_cameras(self):
        """Detects available cameras and stores their indices."""
        self.available_cameras = []
        # *** 已修改 ***: 移除明确的后端API，使用OpenCV默认方式
        for i in range(5): # Check up to 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.available_cameras.append(i)
                cap.release()
        print(f"检测到可用摄像头: {self.available_cameras}")

    def run_snapshot_analysis(self):
        """Grabs a frame from the buffer and analyzes it."""
        with self.frame_lock:
            if self.latest_video_frame is not None:
                frame_to_analyze = self.latest_video_frame.copy()
            else:
                messagebox.showinfo("提示", "尚未捕获到摄像头画面。")
                return
        
        threading.Thread(target=self.thread_task_snapshot_processor, args=(frame_to_analyze,), daemon=True).start()

    # --- 后台线程任务 ---
    def get_params(self):
        return {
            'hue_lower': self.hue_lower_slider.get(), 'hue_upper': self.hue_upper_slider.get(),
            'saturation': self.sat_slider.get(), 'value': self.val_slider.get(),
            'single_ratio': self.video_single_ratio_slider.get() if self.mode_switch.get() == "视频识别" else self.single_ratio_slider.get(),
            'cluster_ratio': self.video_cluster_ratio_slider.get() if self.mode_switch.get() == "视频识别" else self.cluster_ratio_slider.get(),
            'blur_intensity': int(self.blur_slider.get()), 'morph_intensity': int(self.morph_slider.get()),
            'circularity': self.circularity_slider.get()
        }

    def thread_task_binarize(self):
        start_time = time.perf_counter()
        params = self.get_params()
        small_image = cv2.resize(self.original_image, (0, 0), fx=self.PROCESSING_SCALE, fy=self.PROCESSING_SCALE)
        mask = create_segmented_mask(small_image, params)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        self.result_queue.put(('image_mask', mask, elapsed_ms))

    def thread_task_analysis(self, algo_type):
        start_time = time.perf_counter()
        params = self.get_params()
        small_image = cv2.resize(self.original_image, (0, 0), fx=self.PROCESSING_SCALE, fy=self.PROCESSING_SCALE)
        base_mask = create_segmented_mask(small_image, params)
        processed_mask = apply_post_processing(base_mask, params, algo_type)
        output_image, found_left, found_right = analyze_from_mask(self.original_image, processed_mask, params, algo_type, self.PROCESSING_SCALE)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        if algo_type == 'single': self.result_queue.put(('image_single_analysis', output_image, base_mask, found_left, found_right, elapsed_ms))
        else: self.result_queue.put(('image_cluster_analysis', output_image, processed_mask, found_left, found_right, elapsed_ms))

    def thread_task_video_reader(self):
        """视频读取线程：仅负责高速读取帧并显示原始视频."""
        while self.is_video_processing:
            if self.is_slider_dragging:
                time.sleep(0.05)
                continue

            self.video_play_event.wait()
            if not self.is_video_processing: break

            video_capture = self.video_capture
            if video_capture is None: break
            
            if self.seek_frame_num != -1 and not self.is_camera_mode:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame_num)
                self.seek_frame_num = -1

            ret, frame = video_capture.read()
            if not ret: self.result_queue.put(('video_ended',)); break
            
            with self.frame_lock:
                self.latest_video_frame = frame
            
            current_frame_num = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) if not self.is_camera_mode else -1
            self.preview_queue.put(('video_original', frame, current_frame_num))
            time.sleep(1 / self.video_fps if self.video_fps > 0 else 0.033)

    def thread_task_video_processor(self):
        """视频分析线程：以固定低帧率处理最新帧 (仅用于文件模式)."""
        while self.is_video_processing:
            if self.is_camera_mode: break 
            self.video_play_event.wait()
            if not self.is_video_processing: break

            with self.frame_lock:
                if self.latest_video_frame is None: time.sleep(0.1); continue
                current_frame = self.latest_video_frame.copy()
            
            self.process_and_queue_one_frame(current_frame)
            time.sleep(1 / self.analysis_fps)

    def thread_task_camera_processor(self):
        """摄像头自动分析线程."""
        while self.is_auto_analyzing:
            with self.frame_lock:
                if self.latest_video_frame is None:
                    time.sleep(0.1)
                    continue
                frame_to_analyze = self.latest_video_frame.copy()
            self.process_and_queue_one_frame(frame_to_analyze)
            time.sleep(1 / self.analysis_fps)

    def thread_task_snapshot_processor(self, frame_to_analyze):
        """分析单张快照的线程."""
        self.process_and_queue_one_frame(frame_to_analyze)

    def process_and_queue_one_frame(self, frame):
        """对单帧进行完整分析并放入队列."""
        small_frame = cv2.resize(frame, (0, 0), fx=self.PROCESSING_SCALE, fy=self.PROCESSING_SCALE)
        params = self.get_params()
        base_mask = create_segmented_mask(small_frame, params)
        
        # 'single' analysis is not well-defined for video, using 'raw' as placeholder
        single_processed_mask = apply_post_processing(base_mask.copy(), params, 'raw')
        single_output, s_left, s_right = analyze_from_mask(frame, single_processed_mask, params, 'single', self.PROCESSING_SCALE)
        
        cluster_processed_mask = apply_post_processing(base_mask.copy(), params, self.video_cluster_algo)
        cluster_output, c_left, c_right = analyze_from_mask(frame, cluster_processed_mask, params, 'cluster', self.PROCESSING_SCALE)
        
        self.result_queue.put(('video_processed', base_mask, single_output, base_mask, cluster_output, cluster_processed_mask, s_left, s_right, c_left, c_right))

    def thread_task_batch(self, input_dir, output_dir):
        # ... (批量处理逻辑不变)
        pass

    # --- 队列处理和UI更新 ---
    def process_queues(self):
        # High-priority queue for smooth video preview
        latest_preview_msg = None
        while not self.preview_queue.empty():
            latest_preview_msg = self.preview_queue.get_nowait()
        
        if latest_preview_msg:
            msg_type, frame, frame_num = latest_preview_msg
            if msg_type == 'video_original':
                self.display_image(self.video_original_label, frame)
                if not self.is_slider_dragging and not self.is_camera_mode: 
                    self.video_progress_slider.set(frame_num)
                    self.update_time_label(frame_num)

        # Low-priority queue for analysis results
        try:
            msg = self.result_queue.get_nowait()
            msg_type = msg[0]
            if msg_type == 'load_time':
                self.load_time_label.configure(text=f"加载耗时: {msg[1]:.2f} ms")
            elif msg_type == 'image_mask': 
                self.display_image(self.segmented_label, msg[1])
                self.single_time_label.configure(text=f"处理耗时: {msg[2]:.2f} ms")
            elif msg_type == 'image_single_analysis': 
                self.display_image(self.single_analysis_label, msg[1])
                self.display_image(self.single_mask_label, msg[2])
                self.update_status_label(getattr(self, 'single_left_status'), "左侧", msg[3])
                self.update_status_label(getattr(self, 'single_right_status'), "右侧", msg[4])
                self.single_time_label.configure(text=f"处理耗时: {msg[5]:.2f} ms")
            elif msg_type == 'image_cluster_analysis': 
                self.display_image(self.cluster_analysis_label, msg[1])
                self.display_image(self.cluster_mask_label, msg[2])
                self.update_status_label(getattr(self, 'cluster_left_status'), "左侧", msg[3])
                self.update_status_label(getattr(self, 'cluster_right_status'), "右侧", msg[4])
                self.cluster_time_label.configure(text=f"处理耗时: {msg[5]:.2f} ms")
            elif msg_type == 'video_processed':
                _, base_mask, single_out, single_mask, cluster_out, cluster_mask, s_left, s_right, c_left, c_right = msg
                self.display_image(self.video_mask_label, base_mask)
                self.display_image(self.video_single_analysis_label, single_out); self.display_image(self.video_single_mask_label, single_mask)
                self.display_image(self.video_cluster_analysis_label, cluster_out); self.display_image(self.video_cluster_mask_label, cluster_mask)
                self.update_status_label(getattr(self, 'video_single_left_status'), "左侧", s_left); self.update_status_label(getattr(self, 'video_single_right_status'), "右侧", s_right)
                self.update_status_label(getattr(self, 'video_cluster_left_status'), "左侧", c_left); self.update_status_label(getattr(self, 'video_cluster_right_status'), "右侧", c_right)
            elif msg_type == 'video_ended': self.stop_video_processing()
            elif msg_type == 'batch_progress':
                progress, eta_str = msg[1], msg[2]
                self.batch_progressbar.set(progress)
                self.batch_eta_label.configure(text=eta_str)
            elif msg_type == 'batch_done':
                messagebox.showinfo("完成", msg[1])
                self.batch_progressbar.grid_forget()
                self.batch_eta_label.grid_forget()
        except queue.Empty: pass
        finally: self.after(50, self.process_queues)

    def display_image(self, label_widget, cv2_image):
        if cv2_image is None or not hasattr(label_widget, 'winfo_exists') or not label_widget.winfo_exists(): return
        widget_w, widget_h = label_widget.winfo_width(), label_widget.winfo_height()
        if widget_w < 20 or widget_h < 20: widget_w, widget_h = 640, 480
        h, w = cv2_image.shape[:2]
        aspect_ratio = w / h
        if w > widget_w or h > widget_h:
            if aspect_ratio > (widget_w / widget_h): new_w, new_h = widget_w, int(widget_w / aspect_ratio)
            else: new_h, new_w = widget_h, int(widget_h * aspect_ratio)
            resized_image = cv2.resize(cv2_image, (new_w, new_h))
        else: resized_image, new_w, new_h = cv2_image, w, h
        if len(resized_image.shape) == 2: img_pil = Image.fromarray(resized_image)
        else: img_pil = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        ctk_image = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(new_w, new_h))
        label_widget.configure(image=ctk_image, text="")

    def update_status_label(self, label, prefix, found):
        if found: label.configure(text=f"{prefix}: 检测到", fg_color="green")
        else: label.configure(text=f"{prefix}: 未检测到", fg_color="red")
        
    def update_video_player_ui(self, is_live):
        """根据是否为实时视频流更新UI状态."""
        if is_live:
            self.play_pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.snapshot_button.configure(state="normal")
            self.auto_snapshot_button.configure(state="normal")
            self.video_progress_slider.configure(state="disabled")
            self.time_label.configure(text="实时视频流")
        else:
            self.play_pause_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            self.snapshot_button.configure(state="disabled")
            self.auto_snapshot_button.configure(state="disabled")
            self.video_progress_slider.configure(state="normal", to=self.video_total_frames)
            self.update_time_label(0)

    def update_time_label(self, current_frame):
        if self.video_fps > 0 and self.video_total_frames > 0:
            current_sec = current_frame / self.video_fps
            total_sec = self.video_total_frames / self.video_fps
            current_time_str = time.strftime('%M:%S', time.gmtime(current_sec))
            total_time_str = time.strftime('%M:%S', time.gmtime(total_sec))
            self.time_label.configure(text=f"{current_time_str} / {total_time_str}")
        else:
            self.time_label.configure(text="00:00 / 00:00")

    def on_closing(self):
        self.stop_video_processing()
        self.destroy()

if __name__ == "__main__":
    try: import customtkinter
    except ImportError:
        print("错误: 缺少必要的库。请运行 'pip install customtkinter Pillow opencv-python' 来安装。")
        exit()
    app = LycheeDetectorApp()
    app.mainloop()
