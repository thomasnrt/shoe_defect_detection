import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO


class ShoeDetector(Node):
    def __init__(self):
        super().__init__('shoe_detector')

        # === HARD CODED MODEL PATH ===
        model_path = "/home/yassine/ros2_ws/src/shoe_detector/models/best_kaagle_10mars.pt"
        self.conf = 0.5
        image_topic = '/camera/camera/color/image_raw'

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # Subscribers / Publishers
        self.sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        self.pub_annotated = self.create_publisher(
            Image, '/shoe_detector/image_annotated', 10
        )

        self.pub_detections = self.create_publisher(
            Detection2DArray, '/shoe_detector/detections', 10
        )

        self.get_logger().info(f"Subscribed to {image_topic}")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # === YOLO inference ===
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        # === Annotated image ===
        annotated = results.plot()
        out_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out_img.header = msg.header
        self.pub_annotated.publish(out_img)

        # === Structured detections ===
        det_array = Detection2DArray()
        det_array.header = msg.header
        names = self.model.names

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                d = Detection2D()
                d.bbox.center.position.x = (x1 + x2) / 2.0
                d.bbox.center.position.y = (y1 + y2) / 2.0
                d.bbox.size_x = x2 - x1
                d.bbox.size_y = y2 - y1

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = names.get(cls_id, str(cls_id))
                hyp.hypothesis.score = score

                d.results.append(hyp)
                det_array.detections.append(d)

        self.pub_detections.publish(det_array)


def main():
    rclpy.init()
    node = ShoeDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
