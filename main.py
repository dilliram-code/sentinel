from config.settings import *
from services.camera_service import *
from services.detection_service import *
from utils.drawing_utils import *

import cv2


def main():

    cap = initialize_camera(
        WEBCAM_ID
    )

    while True:

        frame = get_frame(cap)

        if frame is None:
            break

        detections = detect_people(
            frame
        )

        for box in detections:

            draw_box(
                frame,
                box,
                "Person"
            )

        cv2.imshow(
            "Campus AI",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

    release_camera(cap)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()