# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(self, **args):
        """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""

        self.CFG = yaml_load(DEFAULT_SOL_CFG_PATH)
        self.CFG.update(args)
        DEFAULT_CFG_DICT.update(args)

        self.initialized = False
        self.model = YOLO(DEFAULT_CFG_DICT["model"])  # Load the Ultralytics YOLO Model
        self.initialized = False

        self.colormap = colormap    # Heatmap colormap
        self.track_history = defaultdict(list)  # Tracks history dictionary
        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow

        # Region & Line Information
        self.counting_region = None
        self.line_dist_thresh = line_dist_thresh
        self.region_thickness = region_thickness
        self.region_color = count_reg_color

        # Object Counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color

        # Region and line selection
        self.count_reg_pts = count_reg_pts
        print(self.count_reg_pts)
        if self.count_reg_pts is not None:
            if len(self.count_reg_pts) == 2:
                print("Line Counter Initiated.")
                self.counting_region = LineString(self.count_reg_pts)
            elif len(self.count_reg_pts) >= 3:
                print("Polygon Counter Initiated.")
                self.counting_region = Polygon(self.count_reg_pts)
            else:
                print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
                print("Using Line Counter Now")
                self.counting_region = LineString(self.count_reg_pts)

    def generate_heatmap(self, im0):
        """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
        """
        # Object tracking
        tracks = self.model.track(
            source=im0,
            persist=True,
            tracker=DEFAULT_CFG_DICT["tracker"],
            classes=DEFAULT_CFG_DICT["classes"],
            iou=DEFAULT_CFG_DICT["iou"],
            conf=DEFAULT_CFG_DICT["conf"], )

        # Initialize heatmap only once
        if not self.initialized:
            heatmap = np.zeros((int(im0.shape[0]), int(im0.shape[1])), dtype=np.float32)
            self.initialized = True

        heatmap *= 0.99  # heatmap decay factor

        annotator = Annotator(im0, DEFAULT_CFG_DICT["line_width"], None)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Draw counting region
            if self.count_reg_pts is not None:
                self.annotator.draw_region(
                    reg_pts=self.count_reg_pts, color=self.region_color, thickness=self.region_thickness
                )

            for box, cls, track_id in zip(self.boxes, self.clss, self.track_ids):
                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

                center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                y, x = np.ogrid[0: self.heatmap.shape[0], 0: self.heatmap.shape[1]]
                mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

                self.heatmap[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += (
                        2 * mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                )

                # Store tracking hist
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                if self.count_reg_pts is not None:
                    # Count objects in any polygon
                    if len(self.count_reg_pts) >= 3:
                        is_inside = self.counting_region.contains(Point(track_line[-1]))

                        if prev_position is not None and is_inside and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["OUT"] += 1

                    # Count objects using line
                    elif len(self.count_reg_pts) == 2:
                        if prev_position is not None and track_id not in self.count_ids:
                            distance = Point(track_line[-1]).distance(self.counting_region)
                            if distance < self.line_dist_thresh and track_id not in self.count_ids:
                                self.count_ids.append(track_id)

                                if (box[0] - prev_position[0]) * (
                                        self.counting_region.centroid.x - prev_position[0]
                                ) > 0:
                                    self.in_counts += 1
                                    self.class_wise_count[self.names[cls]]["IN"] += 1
                                else:
                                    self.out_counts += 1
                                    self.class_wise_count[self.names[cls]]["OUT"] += 1

        else:
            for box, cls in zip(self.boxes, self.clss):
                center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                y, x = np.ogrid[0: self.heatmap.shape[0], 0: self.heatmap.shape[1]]
                mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

                self.heatmap[int(box[1]): int(box[3]), int(box[0]): int(box[2])] += (
                        2 * mask[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                )

        if self.count_reg_pts is not None:
            labels_dict = {}

            for key, value in self.class_wise_count.items():
                if value["IN"] != 0 or value["OUT"] != 0:
                    if not self.view_in_counts and not self.view_out_counts:
                        continue
                    elif not self.view_in_counts:
                        labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                    elif not self.view_out_counts:
                        labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                    else:
                        labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

            if labels_dict is not None:
                self.annotator.display_analytics(self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10)

        # Normalize, apply colormap to heatmap and combine with original image
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), self.colormap)
        self.im0 = cv2.addWeighted(self.im0, 0.5, heatmap_colored, 0.5, 0)

        if self.env_check and self.view_img:
            self.display_frames()

        return self.im0

    def display_frames(self):
        """Display frame."""
        cv2.imshow("Ultralytics Heatmap", self.im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    heatmap = Heatmap(classes_names)
