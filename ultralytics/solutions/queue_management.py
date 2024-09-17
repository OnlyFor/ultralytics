# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import yaml_load, DEFAULT_SOL_CFG_PATH, DEFAULT_CFG_DICT
check_requirements("shapely>=2.0.0")

from shapely.geometry import Point, Polygon


class QueueManager:
    """A class to manage the queue in a real-time video stream based on object tracks."""

    def __init__(self, **args):
        self.CFG = yaml_load(DEFAULT_SOL_CFG_PATH)
        self.CFG.update(args)
        DEFAULT_CFG_DICT.update(args)
        print(self.CFG)
        print(type(self.CFG["region"]))
        print(type([(20, 60), (20, 680), (1120, 680), (1120, 60)]))
        # Initialize queue region
        self.reg_pts = self.CFG["region"] if self.CFG["region"] is not None else list([(20, 60), (20, 680), (1120, 680), (1120, 60)])
        print(self.reg_pts)
        self.counting_region = (
            Polygon(self.reg_pts) if len(self.reg_pts) >= 3 else Polygon([(20, 60), (20, 680), (1120, 680), (1120, 60)])
        )

        self.counts = 0                         # Queue counts Information
        self.track_history = defaultdict(list)  # Tracks history dictionary
        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow

    def extract_and_process_tracks(self, tracks, im0):
        """Extracts and processes tracks for queue management in a video stream."""
        # Initialize annotator and draw the queue region
        annotator = Annotator(im0, DEFAULT_CFG_DICT["line_width"], self.CFG["names"])
        self.counts = 0  # Reset counts every frame
        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                annotator.box_label(box, label=self.CFG["names"][cls], color=colors(int(track_id), True))

                # Update track history
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                annotator.draw_centroid_and_tracks(
                    track_line,
                    color=colors(int(track_id), True),
                    track_thickness=DEFAULT_CFG_DICT["line_width"],
                )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Check if the object is inside the counting region
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        self.counts += 1

        # Display queue counts
        label = f"Queue Counts : {str(self.counts)}"
        if label is not None:
            annotator.queue_counts_display(
                label,
                points=self.reg_pts,
                region_color=(255, 0, 255),
                txt_color=(104, 31, 17),
            )

        if self.env_check and DEFAULT_CFG_DICT["show"]:
            annotator.draw_region(reg_pts=self.reg_pts, thickness=DEFAULT_CFG_DICT["line_width"] * 2, color=(255, 0, 255))
            cv2.imshow("Ultralytics YOLOv8 Queue Manager", im0)
            # Close window on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def process_queue(self, im0, tracks):
        """
        Main function to start the queue management process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.extract_and_process_tracks(tracks, im0)  # Extract and process tracks
        return im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    queue_manager = QueueManager(classes_names)
