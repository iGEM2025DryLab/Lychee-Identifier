import tkinter
import tkinter.filedialog
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
import queue

# --- 核心图像处理逻辑 (从之前的脚本移植) ---

def create_segmented_mask(image, hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2, sat_min, val_min):
    """根据颜色范围创建二值化蒙版."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_image, hsv_lower1, hsv_upper1)
    mask2 = cv2.inRange(hsv_image, hsv_lower2, hsv_upper2)
    hue_mask = cv2.bitwise_or(mask1, mask2)
    sat_mask = cv2.inRange(hsv_image, (0, sat_min, 0), (180, 255, 255))
    val_mask = cv2.inRange(hsv_image, (0, 0, val_min), (180, 255, 255))
    final_mask = cv2.bitwise_and(hue_mask, sat_mask)
    final_mask = cv2.bitwise_and(final_mask, val_mask)
    return final_mask

def apply_box_blur(mask_data, intensity):
    """应用均值模糊."""
    return cv2.blur(mask_data, (intensity * 2 + 1, intensity * 2 + 1))

def apply_closing(mask_data, intensity):
    """应用形态学闭运算."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask_data, cv2.MORPH_CLOSE, kernel, iterations=intensity)

def find_clusters(mask_data, min_pixel_count):
    """寻找并筛选轮廓."""
    contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_pixel_count]

# --- GUI 应用类 ---

class LycheeDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 窗口和主题设置 ---
        self.title("荔枝识别工具")
        self.geometry("1400x900")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 数据存储 ---
        self.original_image = None # 原始 OpenCV 图像
        self.params = {} # 存储所有滑块的值
        self.image_queue = queue.Queue() # 用于线程间图像数据传递

        # --- 创建主框架和标签页 ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.tab_view = ctk.CTkTabview(self.main_frame)
        self.tab_view.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.tab_view.add("二元化工具")
        self.tab_view.add("识别荔枝")
        self.tab_view.add("识别荔枝串")
        
        # --- 创建通用参数框架 ---
        self.create_param_widgets()

        # --- 创建每个标签页的内容 ---
        self.create_binarize_tab(self.tab_view.tab("二元化工具"))
        self.create_single_tab(self.tab_view.tab("识别荔枝"))
        self.create_cluster_tab(self.tab_view.tab("识别荔枝串"))
        
        # --- 启动UI更新循环 ---
        self.after(100, self.process_queue)

    def create_param_widgets(self):
        """创建通用的参数滑块."""
        param_frame = ctk.CTkFrame(self.main_frame)
        param_frame.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
        param_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # 色相
        self.hue_lower_slider = self.create_slider(param_frame, "色相下限", 0, 360, 350, 0)
        self.hue_upper_slider = self.create_slider(param_frame, "色相上限", 0, 360, 10, 1)
        # 饱和度
        self.sat_slider = self.create_slider(param_frame, "饱和度", 0, 100, 50, 2)
        # 明度
        self.val_slider = self.create_slider(param_frame, "明度", 0, 100, 30, 3)

    def create_binarize_tab(self, tab):
        """创建二元化工具标签页的UI."""
        tab.grid_columnconfigure((0, 1), weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # 控制按钮
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        ctk.CTkButton(controls_frame, text="选择图片", command=self.select_image).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkButton(controls_frame, text="选择文件夹 (批量处理)", command=self.select_folder).grid(row=0, column=1, padx=5, pady=5)
        self.batch_button = ctk.CTkButton(controls_frame, text="开始批量处理", command=self.run_batch_processing, state="disabled")
        self.batch_button.grid(row=0, column=2, padx=5, pady=5)
        self.batch_status_label = ctk.CTkLabel(controls_frame, text="请先选择文件夹")
        self.batch_status_label.grid(row=1, column=0, columnspan=3, pady=5)

        # 图像显示
        self.original_label = self.create_image_label(tab, "原始图片", 1, 0)
        self.segmented_label = self.create_image_label(tab, "二值化蒙版", 1, 1)

    def create_single_tab(self, tab):
        """创建识别荔枝标签页的UI."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        # 控制
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(controls_frame, text="选择图片进行分析", command=self.select_image).grid(row=0, column=0, padx=5, pady=5)
        self.single_ratio_slider = self.create_slider(controls_frame, "最小面积比例 (%)", 0.001, 0.5, 0.01, 1, step=0.001, command=self.run_single_analysis)

        # 结果显示
        self.single_analysis_label = self.create_image_label(tab, "分析结果", 2, 0)
        
        # 状态指示
        status_frame = ctk.CTkFrame(tab)
        status_frame.grid(row=3, column=0, pady=10, sticky="ew")
        status_frame.grid_columnconfigure((0,1), weight=1)
        self.single_left_status = ctk.CTkLabel(status_frame, text="左侧: 未检测", fg_color="gray20", corner_radius=5)
        self.single_left_status.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.single_right_status = ctk.CTkLabel(status_frame, text="右侧: 未检测", fg_color="gray20", corner_radius=5)
        self.single_right_status.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

    def create_cluster_tab(self, tab):
        """创建识别荔枝串标签页的UI."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        # 控制
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(controls_frame, text="选择图片进行分析", command=self.select_image).grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        self.cluster_ratio_slider = self.create_slider(controls_frame, "最小面积比例 (%)", 0.001, 1.0, 0.05, 1, step=0.001)
        
        algo_frame = ctk.CTkFrame(controls_frame)
        algo_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
        algo_frame.grid_columnconfigure((0,1,2,3), weight=1)
        ctk.CTkButton(algo_frame, text="原图检测", command=lambda: self.run_cluster_analysis('raw')).grid(row=0, column=0, padx=5)
        ctk.CTkButton(algo_frame, text="模糊降噪", command=lambda: self.run_cluster_analysis('blur')).grid(row=0, column=1, padx=5)
        ctk.CTkButton(algo_frame, text="形态学运算", command=lambda: self.run_cluster_analysis('morph')).grid(row=0, column=2, padx=5)
        ctk.CTkButton(algo_frame, text="混合运算", command=lambda: self.run_cluster_analysis('hybrid')).grid(row=0, column=3, padx=5)

        self.blur_slider = self.create_slider(controls_frame, "模糊强度", 1, 50, 1, 3, is_int=True)
        self.morph_slider = self.create_slider(controls_frame, "形态学强度", 1, 50, 1, 4, is_int=True)

        # 结果显示
        self.cluster_analysis_label = self.create_image_label(tab, "分析结果", 2, 0)
        
        # 状态指示
        status_frame = ctk.CTkFrame(tab)
        status_frame.grid(row=3, column=0, pady=10, sticky="ew")
        status_frame.grid_columnconfigure((0,1), weight=1)
        self.cluster_left_status = ctk.CTkLabel(status_frame, text="左侧: 未检测", fg_color="gray20", corner_radius=5)
        self.cluster_left_status.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.cluster_right_status = ctk.CTkLabel(status_frame, text="右侧: 未检测", fg_color="gray20", corner_radius=5)
        self.cluster_right_status.grid(row=0, column=1, padx=10, pady=5, sticky="ew")


    # --- UI 辅助函数 ---
    def create_slider(self, parent, text, from_, to, initial, row, col_offset=0, is_int=False, step=None, command=None):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=col_offset, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        label = ctk.CTkLabel(frame, text=f"{text}: {initial}")
        label.grid(row=0, column=0, padx=5)
        
        var = tkinter.IntVar() if is_int else tkinter.DoubleVar()
        var.set(initial)
        
        def update_label(value):
            if is_int:
                label.configure(text=f"{text}: {int(float(value))}")
            else:
                label.configure(text=f"{text}: {float(value):.3f}")
            if command:
                command()

        slider = ctk.CTkSlider(frame, from_=from_, to=to, variable=var, command=update_label)
        if step:
            slider.configure(number_of_steps=int((to-from_)/step))
        slider.grid(row=0, column=1, padx=5, sticky="ew")
        return slider

    def create_image_label(self, parent, text, row, col):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text=text).grid(row=0, column=0, pady=5)
        label = ctk.CTkLabel(frame, text="")
        label.grid(row=1, column=0, sticky="nsew")
        return label

    # --- 核心功能函数 ---
    def select_image(self):
        """打开文件对话框选择单个图片."""
        path = tkinter.filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if path:
            self.load_image(path)

    def select_folder(self):
        """选择文件夹以进行批量处理."""
        path = tkinter.filedialog.askdirectory()
        if path:
            self.folder_path = path
            self.batch_status_label.configure(text=f"已选择文件夹: {os.path.basename(path)}")
            self.batch_button.configure(state="normal")

    def load_image(self, path):
        """加载图片并在后台更新UI (修复中文路径问题)."""
        try:
            # 使用可以处理Unicode路径的方式读取文件
            raw_data = np.fromfile(path, dtype=np.uint8)
            self.original_image = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            
            if self.original_image is not None:
                self.update_all_views()
            else:
                messagebox.showerror("错误", f"无法解码图片文件: {path}")
        except Exception as e:
            messagebox.showerror("错误", f"加载图片时出错: {path}\n\n详细信息: {e}")

    def update_all_views(self):
        """当图片或参数改变时，更新所有视图."""
        if self.original_image is None:
            return
        self.update_binarize_view()
        self.run_single_analysis()
        # 集群分析需要手动触发
        self.display_image(self.original_image, self.cluster_analysis_label)


    def update_binarize_view(self, *_):
        """只更新二元化视图."""
        if self.original_image is None:
            return
        self.display_image(self.original_image, self.original_label)
        
        # 启动后台线程处理
        threading.Thread(target=self.thread_task_binarize, daemon=True).start()

    def run_single_analysis(self, *_):
        """运行单荔枝检测."""
        if self.original_image is None:
            return
        threading.Thread(target=self.thread_task_analysis, 
                         args=('single', self.single_analysis_label, self.single_left_status, self.single_right_status), 
                         daemon=True).start()

    def run_cluster_analysis(self, algo):
        """运行荔枝串检测."""
        if self.original_image is None:
            messagebox.showinfo("提示", "请先选择一张图片")
            return
        threading.Thread(target=self.thread_task_analysis, 
                         args=(algo, self.cluster_analysis_label, self.cluster_left_status, self.cluster_right_status), 
                         daemon=True).start()

    def run_batch_processing(self):
        """运行批量处理."""
        if not hasattr(self, 'folder_path') or not self.folder_path:
            messagebox.showerror("错误", "未选择文件夹")
            return
        
        output_dir = os.path.join(self.folder_path, "output_images")
        os.makedirs(output_dir, exist_ok=True)
        
        threading.Thread(target=self.thread_task_batch, args=(self.folder_path, output_dir), daemon=True).start()
        messagebox.showinfo("提示", f"批量处理已开始，结果将保存到\n{output_dir}")


    # --- 后台线程任务 ---
    def get_params(self):
        """从UI获取所有参数."""
        return {
            'hue_lower': self.hue_lower_slider.get(),
            'hue_upper': self.hue_upper_slider.get(),
            'saturation': self.sat_slider.get(),
            'value': self.val_slider.get(),
            'single_ratio': self.single_ratio_slider.get(),
            'cluster_ratio': self.cluster_ratio_slider.get(),
            'blur_intensity': int(self.blur_slider.get()),
            'morph_intensity': int(self.morph_slider.get())
        }

    def _create_mask_from_params(self, image, params):
        """根据参数创建蒙版的辅助函数 (已修复)."""
        # 修正: 确保所有数组元素在转换为uint8前都是有效的数值
        h_lower = int(params['hue_lower'] / 2)
        h_upper = int(params['hue_upper'] / 2)
        sat_min = int(params['saturation'] * 2.55)
        val_min = int(params['value'] * 2.55)

        # 修正: 明确指定dtype=np.uint8来解决类型不匹配的错误
        hsv_lower1 = np.array([h_lower, 0, 0], dtype=np.uint8)
        hsv_upper1 = np.array([179, 255, 255], dtype=np.uint8)
        hsv_lower2 = np.array([0, 0, 0], dtype=np.uint8)
        hsv_upper2 = np.array([h_upper, 255, 255], dtype=np.uint8)
        
        return create_segmented_mask(image, hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2, sat_min, val_min)

    def thread_task_binarize(self):
        """二值化后台任务."""
        params = self.get_params()
        mask = self._create_mask_from_params(self.original_image, params)
        self.image_queue.put(('mask', mask))

    def thread_task_analysis(self, algo_type, target_label, left_status_label, right_status_label):
        """分析任务的通用后台线程."""
        params = self.get_params()
        image = self.original_image.copy()
        h, w, _ = image.shape
        
        mask = self._create_mask_from_params(image, params)
        
        # 后处理
        if algo_type == 'blur':
            mask = apply_box_blur(mask, params['blur_intensity'])
        elif algo_type == 'morph':
            mask = apply_closing(mask, params['morph_intensity'])
        elif algo_type == 'hybrid':
            mask = apply_box_blur(mask, params['blur_intensity'])
            mask = apply_closing(mask, params['morph_intensity'])

        # 寻找轮廓
        ratio = params['single_ratio'] if algo_type == 'single' else params['cluster_ratio']
        min_pixel_count = (h * w) * (ratio / 100.0)
        contours = find_clusters(mask, min_pixel_count)
        
        # 绘制
        found_left, found_right = False, False
        for cnt in contours:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 0, 255), 3)
            center_x = x + box_w / 2
            if center_x < w * 0.55: found_left = True
            if center_x > w * 0.45: found_right = True
            
        self.image_queue.put(('analysis', target_label, image, left_status_label, found_left, right_status_label, found_right))

    def thread_task_batch(self, input_dir, output_dir):
        """批量处理的后台任务."""
        try:
            params = self.get_params()
            supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
            total_files = len(image_files)
            if total_files == 0:
                self.batch_status_label.configure(text="输入文件夹中没有图片")
                return

            self.batch_status_label.configure(text=f"开始处理 {total_files} 张图片...")

            for i, filename in enumerate(image_files):
                self.batch_status_label.configure(text=f"正在处理: {i+1}/{total_files} - {filename}")
                
                image_path = os.path.join(input_dir, filename)
                
                raw_data = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

                if image is None:
                    print(f"跳过无法读取的文件: {filename}")
                    continue

                h, w, _ = image.shape
                
                mask = self._create_mask_from_params(image, params)
                
                mask = apply_box_blur(mask, params['blur_intensity'])
                mask = apply_closing(mask, params['morph_intensity'])

                ratio = params['cluster_ratio']
                min_pixel_count = (h * w) * (ratio / 100.0)
                contours = find_clusters(mask, min_pixel_count)
                
                output_image = image.copy()
                for cnt in contours:
                    x, y, box_w, box_h = cv2.boundingRect(cnt)
                    cv2.rectangle(output_image, (x, y), (x + box_w, y + box_h), (0, 0, 255), 3)

                output_path = os.path.join(output_dir, f"processed_{filename}")
                is_success, im_buf_arr = cv2.imencode(f".{filename.split('.')[-1]}", output_image)
                if is_success:
                    im_buf_arr.tofile(output_path)

            self.batch_status_label.configure(text=f"批量处理完成！共处理 {total_files} 张图片。")
        except Exception as e:
            self.batch_status_label.configure(text=f"批量处理出错: {e}")


    # --- 队列处理和UI更新 ---
    def process_queue(self):
        """处理来自后台线程的数据并更新UI."""
        try:
            msg = self.image_queue.get_nowait()
            if msg[0] == 'mask':
                self.display_image(msg[1], self.segmented_label)
            elif msg[0] == 'analysis':
                _, label, image, left_label, found_left, right_label, found_right = msg
                self.display_image(image, label)
                self.update_status_label(left_label, "左侧", found_left)
                self.update_status_label(right_label, "右侧", found_right)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def display_image(self, cv2_image, label_widget):
        """将OpenCV图像显示在Tkinter标签上."""
        if cv2_image is None: return
        
        widget_w, widget_h = label_widget.winfo_width(), label_widget.winfo_height()
        if widget_w < 20 or widget_h < 20: 
            widget_w, widget_h = 640, 480

        h, w = cv2_image.shape[:2]
        aspect_ratio = w / h
        
        if w > widget_w or h > widget_h:
            if aspect_ratio > (widget_w / widget_h):
                new_w = widget_w
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = widget_h
                new_w = int(new_h * aspect_ratio)
            resized_image = cv2.resize(cv2_image, (new_w, new_h))
        else:
            resized_image = cv2_image

        if len(resized_image.shape) == 2: 
            img = Image.fromarray(resized_image)
        else: 
            img = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        
        img_tk = ImageTk.PhotoImage(image=img)
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk 

    def update_status_label(self, label, prefix, found):
        """更新状态标签的文本和颜色."""
        if found:
            label.configure(text=f"{prefix}: 检测到", fg_color="green")
        else:
            label.configure(text=f"{prefix}: 未检测到", fg_color="red")


if __name__ == "__main__":
    # --- 运行前检查和安装依赖 ---
    try:
        import customtkinter
        from PIL import Image, ImageTk
    except ImportError:
        print("错误: 缺少必要的库。请运行 'pip install customtkinter Pillow' 来安装。")
        exit()
        
    app = LycheeDetectorApp()
    app.mainloop()
