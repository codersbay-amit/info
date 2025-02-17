from image_utils import remove_text_with_easyocr,draw_multiline_text_in_bbox_center
from PIL import Image
from yolo_prediction import getBoxes
def refiner(image_path:str,title:str=None,subtitle:str=None,cta_text:str=None,primary_color:tuple=(0,0,2),secondory_color:tuple=(35,35,35)):
    blank_image=remove_text_with_easyocr(Image.open(image_path))
    boxes=getBoxes(image_path=image_path)
    if "title" in boxes.keys():
        blank_image=draw_multiline_text_in_bbox_center(image=blank_image,text=title,bbox=boxes['title'][0],gradient_start=primary_color,gradient_end=secondory_color)
    if "subheading" in boxes.keys():
        blank_image=draw_multiline_text_in_bbox_center(image=blank_image,text=subtitle,bbox=boxes['subheading'][0],gradient_start=primary_color,gradient_end=secondory_color)
    blank_image.show()
    return blank_image



