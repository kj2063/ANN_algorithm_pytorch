import random
import numpy as np
import cv2

def set_bounding_boxes (image_data : np.ndarray, 
                        bbox_info_list : np.ndarray,
                        bbox_info_format : str,
                        category_list : np.ndarray, 
                        category_dict : dict,
                        color_dict : dict) -> np.ndarray:
    """
    Parameters
    ----------
    image_data (numpy.ndarray)
        image data in numpy.ndarray type
    bbox_info_list (numpy.ndarray)
        bounding box infomation in numpy.ndarray type with (n,4) shape
    bbox_info_format (str)
        'yolo' | 'coco' | ...
    category_list (numpy.ndarray)
        key of target class in numpy.ndarray type with (n,) shape
    category_dict (dict)
        dictionary created from class number(key : int) and class names(value : str)
    color_dict (dict)
        dictionary created from class names(key : int) and RGB color tuple(value : tuple)
        
    Returns
    --------
    img (numpy.ndarray)
        image where object is displayed as a square label
    """
    
    img = image_data.copy()
    
    for bbox_info, category in zip(bbox_info_list, category_list):
        set_bounding_box(img, bbox_info, category_dict[category], bbox_info_format, color_dict)
    
    return img

def set_bounding_box (image_data, 
                      bounding_box_info, 
                      category_name, 
                      bbox_info_format, 
                      color_dict):
    
    h, w, _ = image_data.shape
    
    if(bbox_info_format == 'yolo'):
        x_center_ratio = bounding_box_info[0]
        y_center_ratio = bounding_box_info[1]
        width_ratio = bounding_box_info[2]
        height_ratio = bounding_box_info[3]

        x_min = int(w * x_center_ratio - (w * width_ratio / 2))
        x_max = int(w * x_center_ratio + (w * width_ratio / 2))
        y_min = int(h * y_center_ratio - (h * height_ratio / 2))
        y_max = int(h * y_center_ratio + (h * height_ratio / 2))
    
    cv2.rectangle(image_data, (x_min, y_min), (x_max, y_max), color_dict[category_name], 3)
    
    ((text_width, text_height), _) = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image_data, (x_min, y_min - int(1.3 * text_height)), (x_min + int(1.1 * text_width), y_min), color_dict[category_name], -1)
    cv2.putText(
        image_data,
        text=category_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.55, 
        color=(255,255,255), 
        lineType=cv2.LINE_AA,
    )
    
    return image_data

def get_random_color_dict (category_dict):
    color_list = []
    target_list = list(category_dict.values())

    for _ in range(len(category_dict)):
        color_list.append((random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)))

    color_dict = dict(zip(target_list, color_list))
    
    return color_dict