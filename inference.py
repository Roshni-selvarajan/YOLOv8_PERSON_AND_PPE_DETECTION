import os
import cv2
import argparse
from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def draw_boxes(image, boxes, labels, confidences, color=(0, 255, 0)):
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{label} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def perform_inference(input_dir, output_dir, person_model_path, ppe_model_path):
    os.makedirs(output_dir, exist_ok=True)

    person_model = load_model(person_model_path)
    ppe_model = load_model(ppe_model_path)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            original_image = image.copy()

            # Person detection
            person_results = person_model.predict(image)
            person_bboxes = person_results[0].boxes.xyxy.cpu().numpy()
            person_confs = person_results[0].boxes.conf.cpu().numpy()
            person_labels = ["person"] * len(person_bboxes)

            if person_bboxes.size > 0:
                for idx, person_box in enumerate(person_bboxes):
                    x1, y1, x2, y2 = map(int, person_box[:4])
                    cropped_image = original_image[y1:y2, x1:x2]

                    # PPE detection on the cropped image
                    ppe_results = ppe_model.predict(cropped_image)
                    ppe_bboxes = ppe_results[0].boxes.xyxy.cpu().numpy()
                    ppe_confs = ppe_results[0].boxes.conf.cpu().numpy()
                    ppe_labels = [ppe_results[0].names[int(cls)] for cls in ppe_results[0].boxes.cls.cpu().numpy()]

                    # Adjust PPE bounding boxes to the original image coordinates
                    for ppe_box in ppe_bboxes:
                        ppe_box[0] += x1
                        ppe_box[1] += y1
                        ppe_box[2] += x1
                        ppe_box[3] += y1

                    # Draw bounding boxes on the original image
                    image = draw_boxes(image, [person_box], [person_labels[idx]], [person_confs[idx]], color=(0, 255, 0))
                    image = draw_boxes(image, ppe_bboxes, ppe_labels, ppe_confs, color=(255, 0, 0))

            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using person and PPE detection models.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images.")
    parser.add_argument("--person_det_model", type=str, required=True, help="Path to the person detection model.")
    parser.add_argument("--ppe_detection_model", type=str, required=True, help="Path to the PPE detection model.")
    
    args = parser.parse_args()
    perform_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)


    #python inference.py --input_dir /path/to/input/images --output_dir /path/to/output/images --person_det_model /path/to/person_model.pt --ppe_detection_model /path/to/ppe_model.pt
