
import os
import xml.etree.ElementTree as ET
import argparse

def convert_pascal_voc_to_yolo(input_dir, output_dir, classes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.xml'):
            continue

        xml_path = os.path.join(input_dir, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)
        
        yolo_annotations = []

        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            class_id = classes.index(cls)
            
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            
            x_center = (x_min + x_max) / 2.0 / image_width
            y_center = (y_min + y_max) / 2.0 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        yolo_filename = os.path.splitext(filename)[0] + '.txt'
        yolo_path = os.path.join(output_dir, yolo_filename)
        
        with open(yolo_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Pascal VOC annotations to YOLO format')
    parser.add_argument('input_dir', type=str, help='/home/rosh/Desktop/ASSIGNMENT/Preprocessed_data/labels_new')
    parser.add_argument('output_dir', type=str, help='/home/rosh/Desktop/ASSIGNMENT/pascalVOC_to_yolo_data')
    parser.add_argument('--classes', type=str, nargs='+', help='List of class names in the dataset', required=True)
    
    args = parser.parse_args()
    
    convert_pascal_voc_to_yolo(args.input_dir, args.output_dir, args.classes)


#Command Line Script to run
#python pascalVOC_to_yolo.py /path/to/input_dir /path/to/output_dir --classes person hard-hat gloves glasses boots vest ppe-suit ear-protector safety-harness
#python /home/rosh/Desktop/ASSIGNMENT/pascalVOC_to_yolo.py /home/rosh/Desktop/ASSIGNMENT/Preprocessed_data/labels_new /home/rosh/Desktop/ASSIGNMENT/pascalVOC_to_yolo_data --classes person hard-hat gloves glasses boots vest ppe-suit ear-protector safety-harness