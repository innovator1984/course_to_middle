import cv2

labels_dict = {0: 'car', 1: 'plate'}

def box_label(img, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), line_width=2):
    # Add one xyxy box to image with label
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    p1, p2 = (x1, y1), (x2, y2)
    cv2.rectangle(img, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_width - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, line_width / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_warning(image, warning_text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.5):
    x, y, w, h = 0, 0, 375, 50
    cv2.rectangle(image, (0, 0), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(img=image, text=warning_text, org=(x + int(w / 10), y + int(h / font_scale)),
                fontFace=font,
                fontScale=0.6, color=(0, 0, 255), thickness=2)
    return image


