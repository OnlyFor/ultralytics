# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import yaml_load, DEFAULT_SOL_CFG_PATH, DEFAULT_CFG_DICT

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self, **args):
        self.CFG = yaml_load(DEFAULT_SOL_CFG_PATH)
        self.CFG.update(args)
        DEFAULT_CFG_DICT.update(args)
        print(self.CFG)

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)] if self.CFG["region"] is None else self.CFG["region"]

        self.im0 = None             # Image (ndarray)
        self.in_counts = 0          # variable to store in_counts
        self.out_counts = 0         # variable to store out_counts
        self.count_ids = []         # ID's that are already counted
        self.class_wise_count = {}  # class_wise counting dictionary
        self.track_history = defaultdict(list)  # Tracks history dictionary
        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow

        # Initialize counting region
        self.counting_region = Polygon(self.CFG["region"]) if len(self.CFG["region"]) >= 3 else LineString(CFG["reg_pts"])
        print("Ultralytics Counter Initiated...!!!")

        # Define the counting line segment
        self.counting_line_segment = LineString(
            [
                (self.reg_pts[0][0], self.reg_pts[0][1]),
                (self.reg_pts[1][0], self.reg_pts[1][1]),
            ]
        )

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        # Annotator Init and region drawing
        annotator = Annotator(self.im0, 2, self.CFG["names"])

        # Draw region or line
        annotator.draw_region(reg_pts=self.CFG["region"], color=(104, 0, 123), thickness=2 * 2)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                annotator.box_label(box, label=self.CFG["names"][cls], color=colors(int(track_id), True))

                # Store class info
                if self.CFG["names"][cls] not in self.class_wise_count:
                    self.class_wise_count[self.CFG["names"][cls]] = {"IN": 0, "OUT": 0}

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if DEFAULT_CFG_DICT["show"]:
                    annotator.draw_centroid_and_tracks(
                        track_line,
                        color=colors(int(track_id), True),
                        track_thickness=DEFAULT_CFG_DICT["line_width"],
                    )

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # Count objects in any polygon
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))

                    if prev_position is not None and is_inside and track_id not in self.count_ids:
                        self.count_ids.append(track_id)

                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.CFG["names"][cls]]["IN"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.CFG["names"][cls]]["OUT"] += 1

                # Count objects using line
                elif len(self.reg_pts) == 2:
                    if prev_position is not None and track_id not in self.count_ids:
                        # Check if the object's movement segment intersects the counting line
                        if LineString([(prev_position[0], prev_position[1]), (box[0], box[1])]).intersects(
                            self.counting_line_segment
                        ):
                            self.count_ids.append(track_id)

                            # Determine the direction of movement (IN or OUT)
                            dx = (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0])
                            dy = (box[1] - prev_position[1]) * (self.counting_region.centroid.y - prev_position[1])
                            if dx > 0 and dy > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.CFG["names"][cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.CFG["names"][cls]]["OUT"] += 1

        labels_dict = {}

        labels_dict.update({str.capitalize(
            key): f"{'IN ' + str(value['IN']) if self.CFG['show_in'] else ''} {'OUT ' + str(value['OUT']) if self.CFG['show_out'] else ''}".strip()
                            for key, value in self.class_wise_count.items() if
                            value["IN"] != 0 or value["OUT"] != 0 and (
                                        self.CFG['show_in'] or self.
                                        CFG['show_in'])})

        if labels_dict:
            annotator.display_analytics(self.im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        if DEFAULT_CFG_DICT["show"] and self.env_check:
            cv2.imshow("Ultralytics YOLOv8 Object Counter", self.im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    ObjectCounter(classes_names)
