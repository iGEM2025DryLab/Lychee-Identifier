import cv2
import numpy as np
import os
import argparse

def create_segmented_mask(image, hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2, sat_min, val_min):
    """
    根据颜色范围创建二值化蒙版.
    由于红色在HSV空间中环绕0/360度，我们需要处理两个范围。
    """
    # 将图像从 BGR 转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 创建第一个红色范围的蒙版
    mask1 = cv2.inRange(hsv_image, hsv_lower1, hsv_upper1)
    # 创建第二个红色范围的蒙版
    mask2 = cv2.inRange(hsv_image, hsv_lower2, hsv_upper2)
    
    # 合并两个蒙版
    hue_mask = cv2.bitwise_or(mask1, mask2)
    
    # 创建饱和度和明度的蒙版
    sat_mask = cv2.inRange(hsv_image, (0, sat_min, 0), (180, 255, 255))
    val_mask = cv2.inRange(hsv_image, (0, 0, val_min), (180, 255, 255))

    # 将所有蒙版合并，得到最终的二值化图像
    final_mask = cv2.bitwise_and(hue_mask, sat_mask)
    final_mask = cv2.bitwise_and(final_mask, val_mask)
    
    return final_mask

def process_image(image_path, output_dir, params):
    """
    处理单个图像：分割、分析、绘制边界框并保存.
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    h, w, _ = image.shape
    total_pixels = h * w
    
    # 1. 创建二值化蒙版
    # 注意: OpenCV 的 H 范围是 0-179, S 和 V 是 0-255
    # 我们需要将网页版的值 (H:0-360, S:0-100, V:0-100) 转换过来
    hue_lower = params['hue_lower']
    hue_upper = params['hue_upper']
    
    # 将网页的 (350-360, 0-10) 范围转换为OpenCV的两个范围
    # 范围1: 350/2 -> 175 到 179
    # 范围2: 0/2 -> 0 到 10/2 -> 5
    hsv_lower1 = np.array([hue_lower / 2, 0, 0]) 
    hsv_upper1 = np.array([179, 255, 255])
    hsv_lower2 = np.array([0, 0, 0])
    hsv_upper2 = np.array([hue_upper / 2, 255, 255])
    
    sat_min = int(params['saturation'] * 2.55)
    val_min = int(params['value'] * 2.55)

    mask = create_segmented_mask(image, hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2, sat_min, val_min)

    # 2. 应用后处理算法
    algo = params['algorithm']
    processed_mask = mask.copy() # 复制蒙版以进行处理

    if algo == 'blur':
        intensity = params['blur_intensity']
        # 使用均值模糊
        processed_mask = cv2.blur(processed_mask, (intensity * 2 + 1, intensity * 2 + 1))
    elif algo == 'morph':
        intensity = params['morph_intensity']
        # 使用形态学闭运算
        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=intensity)
    elif algo == 'hybrid':
        blur_intensity = params['blur_intensity']
        morph_intensity = params['morph_intensity']
        # 先模糊
        blurred_mask = cv2.blur(processed_mask, (blur_intensity * 2 + 1, blur_intensity * 2 + 1))
        # 再进行形态学闭运算
        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_intensity)

    # 3. 寻找轮廓 (荔枝串)
    # cv2.RETR_EXTERNAL 只检测最外层的轮廓
    # cv2.CHAIN_APPROX_SIMPLE 压缩水平、垂直和对角线段，只留下它们的端点
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 筛选轮廓并绘制结果
    min_pixel_count = total_pixels * (params['min_area_ratio'] / 100.0)
    output_image = image.copy()
    found_left = False
    found_right = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_pixel_count:
            # 获取边界框
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            # 在输出图像上绘制红色矩形
            cv2.rectangle(output_image, (x, y), (x + box_w, y + box_h), (0, 0, 255), 3)
            
            # 判断荔枝串位置
            center_x = x + box_w / 2
            if center_x < w * 0.55: # 左侧区域 (包含10%重叠)
                found_left = True
            if center_x > w * 0.45: # 右侧区域 (包含10%重叠)
                found_right = True

    # 5. 保存结果图片
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"processed_{filename}")
    cv2.imwrite(output_path, output_image)
    
    # 打印状态
    print(f"--- 处理完成: {filename} ---")
    print(f"保存结果至: {output_path}")
    print(f"左侧区域: {'检测到' if found_left else '未检测到'}")
    print(f"右侧区域: {'检测到' if found_right else '未检测到'}\n")


def main():
    # --- 自动获取路径 ---
    # 获取脚本所在的绝对路径目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 定义默认的输入和输出文件夹名称
    default_input_folder = "input_images"
    default_output_folder = "output_images"
    # 构建输入和输出文件夹的完整路径
    input_dir = os.path.join(script_dir, default_input_folder)
    output_dir = os.path.join(script_dir, default_output_folder)

    # --- 解析命令行参数（保留算法自定义功能） ---
    parser = argparse.ArgumentParser(
        description="荔枝检测命令行工具。自动检测脚本目录下的 'input_images' 文件夹，并输出到 'output_images' 文件夹。",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助信息格式
    )
    
    # 添加与网页版对应的所有参数
    parser.add_argument('--hue_lower', type=float, default=350.0, help="色相下限 (0-360)，默认为 350")
    parser.add_argument('--hue_upper', type=float, default=10.0, help="色相上限 (0-360)，默认为 10")
    parser.add_argument('--saturation', type=float, default=50.0, help="饱和度下限 (0-100)，默认为 50")
    parser.add_argument('--value', type=float, default=30.0, help="明度下限 (0-100)，默认为 30")
    parser.add_argument('--min_area_ratio', type=float, default=0.05, help="荔枝串/荔枝的最小面积比例 (%%)，默认为 0.05")
    
    parser.add_argument('--algorithm', type=str, default='raw', choices=['raw', 'blur', 'morph', 'hybrid'], help="选择后处理算法: raw, blur, morph, hybrid。默认为 raw")
    parser.add_argument('--blur_intensity', type=int, default=1, help="模糊强度 (用于 blur 和 hybrid 算法)，默认为 1")
    parser.add_argument('--morph_intensity', type=int, default=1, help="形态学运算强度 (用于 morph 和 hybrid 算法)，默认为 1")

    args = parser.parse_args()
    params = vars(args)

    # --- 检查路径并执行处理 ---
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入文件夹 '{input_dir}' 不存在。")
        print(f"请在脚本所在目录下创建一个名为 '{default_input_folder}' 的文件夹，并放入待处理的图片。")
        # 自动创建一个空的输入文件夹以作提示
        os.makedirs(input_dir, exist_ok=True)
        print(f"已为您创建示例输入文件夹: {input_dir}")
        return

    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # 遍历输入文件夹中的所有图片
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"在文件夹 '{input_dir}' 中未找到支持的图片文件。")
        return
        
    print(f"开始处理 '{input_dir}' 中的 {len(image_files)} 张图片...")
    print(f"使用参数: {params}")
    print("-" * 30)

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        process_image(image_path, output_dir, params)

    print("所有处理已完成。")


if __name__ == '__main__':
    main()
