import asyncio
import threading
import numpy as np
import cv2
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from ultralytics import YOLO

# Gazebo imports
from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage


CAMERA_TOPIC       = "/downward_camera/image"
CALIBRATION_YAML   = "sim.yaml"
YOLO_WEIGHTS       = "yolov8n-visdrone.pt"    # get from Ultralytics VisDrone release
ARUCO_DICT_ID      = cv2.aruco.DICT_6X6_250
MARKER_LENGTH_M    = 0.3
MAVSDK_ADDRESS     = "udpin://0.0.0.0:14540"

ALTITUDE           = 10

#Sciipted flight. 
#North = Green Line
#East = Red Line
NORTH_SPEED        = 1.0
NORTH_SECONDS      = 4
EAST_SPEED         = 1.0
EAST_SECONDS       = 60

# Tracking mode control
HOVER_GAIN_XY          = 0.009    # correction speed based on pixel distance to object
MAX_HOVER_SPEED_XY     = 1
HOVER_GAIN_TVEC        = 0.4      # correction speed based on tvec distance to marker
MAX_HOVER_SPEED_TVEC   = 1.0      # cap correction speed
DECEND_GAIN            = 0.6      # decend speed based on distance too marker
MAX_DECEND_SPEED       = 1      # cap decend speed



#Shared target class that can update/return tvec and XY values
# Vision runs on separate threads, flight loop runs on asyncio, lock keeps reads/writes safe
class SharedTarget:
    def __init__(self):
        self._lock = threading.Lock()
        self.tvec = None
        self.x = None
        self.y = None

    def updateTvec(self, tvec):
        with self._lock:
            self.tvec = tvec.flatten()
    

    def readTvec(self):
        with self._lock:
            if self.tvec is None:
                return None
            return self.tvec.copy()
    
    def updateXY(self, x, y):
        with self._lock:
            self.x = x
            self.y = y

    def readXY(self):
        with self._lock:
            if self.x is None or self.y is None:
                return None, None
            return self.x, self.y

class GazeboCamera:
    #Setup and subscribe to camera 
    # gz-transport runs its own thread, so we buffer the latest frame and hand it out on request
    def __init__(self, topic):
        self._lock = threading.Lock()
        self._latest = None
        self._node = GzNode()

        ok = self._node.subscribe(GzImage, topic, self._on_image)
        if not ok:
            raise RuntimeError(f"Failed to subscribe to {topic}")
        print(f"[cam] subscribed to {topic}")

    #Get image from data stream
    def _on_image(self, msg):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            arr = arr.reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            with self._lock:
                self._latest = img
        except Exception as e:
            print(f"[cam] decode error: {e}")
    
    #returns latest image
    def get(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

#Callibration setup
def load_calibration(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    mtx = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()

    if mtx is None:
        raise RuntimeError(f"Could not read camera_matrix from {yaml_path}")
    return mtx, dist

#ArucuDetector class that creates a detector object which stores mtx, dist, ideal marker corners, and detector.
class ArucoDetector:
    def __init__(self, mtx, dist, dict_id, aruco_marker_side_length):
        self.mtx = mtx
        self.dist = dist

        #Creates ideal aruco marker
        # solvePnP needs the real-world corner positions to compare against the detected pixel corners
        half_size = aruco_marker_side_length / 2
        self.marker_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ], dtype=np.float32)

        #Setup detetctor parameters
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        arucoParameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary, arucoParameters)

    #Get Tvec and Rvec values if found using solvepnp formula
    # IPPE_SQUARE is the fast solver made for planar square markers like ArUco
    def _distance_estimate(self, marker_corners):
        image_points = marker_corners.reshape(-1, 2).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(
            self.marker_points, image_points, self.mtx, self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if success:
            return (rvec, tvec)
        else:
            return None

    #Return list of markers and their tvec values
    def detect(self, img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(imgGray)

        poses = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                result = self._distance_estimate(corners[i])
                if result is not None:
                    rvec, tvec = result
                    poses.append({
                        "id": int(marker_id),
                        "rvec": rvec,
                        "tvec": tvec,
                        "corners": corners[i],
                    })
        return poses

#Setup Drone connection
async def connect():
    drone = System()
    await drone.connect(system_address=MAVSDK_ADDRESS)

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    return drone

# Draw the YOLO detection box, image center, object center, and label
def draw_detection(img, d, img_centreX, img_centreY):
    x1, y1, x2, y2 = d["bbox"]
    ox, oy         = d["center"]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(img, (img_centreX, img_centreY), 30, (0, 0, 255), 2)
    cv2.circle(img, (ox, oy), 5, (255, 0, 0), -1)

    label = f"{d['name']} {d['conf']:.2f}"
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Run YOLO on the frame and return detections of the target class only (default 3 = van in VisDrone)
def yolo_detect(model, img, img_centreX, img_centreY, target_class=3):
    results = model(img, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) != target_class:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ox, oy, w, h  = map(int, box.xywh[0])
            class_id = int(box.cls.item())
            conf     = box.conf.item()

            if conf < 0.2:
                continue
            detections.append({
                "class_id": class_id,
                "name":     model.names[class_id],
                "conf":     conf,
                "bbox":     (x1, y1, x2, y2),
                "center":   (ox, oy),
                "x_dist":   ox - img_centreX,
                "y_dist":   oy - img_centreY,
            })
    return detections


#Runs both YOLO and ArUco on every frame and updates the shared target
async def detection_loop(camera, detector, target, model):
    print("[det] detection loop started")
    last_print = 0.0
    print_interval = 1
    img_centreX = 640 // 2
    img_centreY = 480 // 2

    while True:
        img = camera.get()
        if img is None:
            await asyncio.sleep(0.05)
            continue

        img = cv2.resize(img, (640, 480))

        # Run YOLO on a background thread so it doesn't block the asyncio event loop
        yolo_detections = await asyncio.to_thread(
            yolo_detect, model, img, img_centreX, img_centreY
        )
        # ArUco is fast (~1-2ms) so running it inline is fine
        poses = detector.detect(img)
        now = asyncio.get_running_loop().time()

        if yolo_detections:
            d = yolo_detections[0]
            draw_detection(img, d, img_centreX, img_centreY)
            target.updateXY(d["x_dist"], d["y_dist"])
            if now - last_print >= print_interval:
                print(f"X: {d['x_dist']} | Y: {d['y_dist']}")
                last_print = now

        for p in poses:
            t = p["tvec"].flatten()

            if now - last_print >= print_interval:
                print(f"id={p['id']}  x={t[0]:+.2f}  y={t[1]:+.2f}  z={t[2]:+.2f}")
                last_print = now

            target.updateTvec(p["tvec"])

            cv2.aruco.drawDetectedMarkers(img, [p["corners"]], np.array([[p["id"]]]))
            cv2.drawFrameAxes(img, detector.mtx, detector.dist,
                      p["rvec"], p["tvec"], MARKER_LENGTH_M * 0.5)

        cv2.imshow("Video", img)
        cv2.waitKey(1)
        await asyncio.sleep(0.01)

# Convert marker translation into body-frame velocity
def tvec_to_body_velocity(tvec):
   
    #   marker in front of drone (green/north direction) -> camera y is negative
    #   marker to right of drone (red/east direction)    -> camera x is positive
    #
    # So:
    #   forward = -y  (negative y means marker is ahead, drone flies forward)
    #   right   = +x  (positive x means marker is to right, drone flies right)
    x, y, z = tvec[0], tvec[1], tvec[2]

    forward = -y * HOVER_GAIN_TVEC
    right   =  x * HOVER_GAIN_TVEC
    down = z * DECEND_GAIN
    # Clamp so a big offset doesn't produce an aggressive command
    forward = max(-MAX_HOVER_SPEED_TVEC, min(MAX_HOVER_SPEED_TVEC, forward))
    right   = max(-MAX_HOVER_SPEED_TVEC, min(MAX_HOVER_SPEED_TVEC, right))
    down   = max(-MAX_DECEND_SPEED, min(MAX_DECEND_SPEED, down))
    
    return forward, right, down

# Same idea as tvec_to_body_velocity but uses YOLO pixel offsets instead of a 3D pose
def xy_to_body_velocity(x, y):
   
    #   object in front of drone (green/north direction) -> pixel y is negative
    #   object to right of drone (red/east direction)    -> pixel x is positive
    #
    # So:
    #   forward = -y  (negative y means object is ahead, drone flies forward)
    #   right   = +x  (positive x means object is to right, drone flies right)
    


    forward = -y * HOVER_GAIN_XY
    right   =  x * HOVER_GAIN_XY
    
    forward = max(-MAX_HOVER_SPEED_XY, min(MAX_HOVER_SPEED_XY, forward))
    right   = max(-MAX_HOVER_SPEED_XY, min(MAX_HOVER_SPEED_XY, right))
    
    return forward, right

# Scripted search pattern, runs until the watcher cancels it
async def scripted_flight(drone):

    print(f"-- Flying NORTH at {NORTH_SPEED} m/s for {NORTH_SECONDS} s (body forward, green axis)")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(NORTH_SPEED, 0.0, 0.0, 0.0))
    await asyncio.sleep(NORTH_SECONDS)

    print("-- Hovering for 6 s")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(6)

    print(f"-- Flying EAST at {EAST_SPEED} m/s for {EAST_SECONDS} s (body right, red axis)")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, EAST_SPEED, 0.0, 0.0))
    await asyncio.sleep(EAST_SECONDS)

    print("-- Hovering for 6 s")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(6)

# Watcher cancels scripted flight when marker OR object appears
# Returns True if either was spotted, False if flight finished naturally.
async def marker_watcher(target, flight_task):
    while True:
        if flight_task.done():
            return False

        tvec_check = target.readTvec()
        if tvec_check is not None:
            print("-- Marker spotted! Cancelling scripted flight.")
            flight_task.cancel()
            return True

        x, y = target.readXY()
        if x is not None and y is not None:
            print("-- Object spotted! Cancelling scripted flight.")
            flight_task.cancel()
            return True

        await asyncio.sleep(0.1)

# Marker takes priority over YOLO here because it gives a full 3D pose, YOLO only gives pixel position
async def tracking_mode(drone, target):

    print("Tracking")

    while True:
        
        tvec = target.readTvec()
        x, y = target.readXY()
        
        #Marker takes priority
        if tvec is not None:
            #If drone is centered within tolerance and decended close to marker return and allow it to land via main
            if abs(tvec[0]) < 0.2 and abs(tvec[1]) < 0.2 and tvec[2] < 1:
                print("Drone centered within tolerance")
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
                return   

            #If drone is centered within tolerance start decending
            elif abs(tvec[0]) < 0.2 and abs(tvec[1]) < 0.2:
                print("-- xy Target reached within tolerance, descending")
                forward, right, down = tvec_to_body_velocity(tvec)

            #If drone is not centred apply corrections to centre
            else:
                forward, right, down = tvec_to_body_velocity(tvec)
                down = 0.0   # not centered yet — don't descend
        
        #Fall back to YOLO object if no marker
        elif x is not None and y is not None:
            forward, right = xy_to_body_velocity(x, y)
            down = 0.0

        else:
            forward, right, down = 0.0, 0.0, 0.0



        #Apply velocity based on scenario
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(forward, right, down, 0.0))
        await asyncio.sleep(0.1)

# Watches the in-air flag, cancels background tasks once the drone has touched down
async def observe_is_in_air(drone, running_tasks):
    was_in_air = False

    async for is_in_air in drone.telemetry.in_air():
        if is_in_air:
            was_in_air = is_in_air

        if was_in_air and not is_in_air:
            for task in running_tasks:
                task.cancel()
            print("Landed Properly")
            return

# Check GPS, battery, and armed state before allowing takeoff
async def pre_flight(drone):


    flight_check = False

    async for gps in drone.telemetry.gps_info():
        gps_count = gps.num_satellites
        gps_fixtype = gps.fix_type
        break

    async for battery in drone.telemetry.battery():
        battery_per = battery.remaining_percent
        battery_voltage = battery.voltage_v
        break

    async for arm in drone.telemetry.armed():
        armed_check = arm
        break

    
    # Pre flight Report
    print("\n========= PRE-FLIGHT REPORT =========")

    print("\n  GPS:")
    print(f"    Satellites : {gps_count}  "
          f"{'[PASS]' if gps_count >= 6 else '[FAIL] need 6+'}")
    print(f"    Fix Type   : {gps_fixtype}  "
          f"{'[PASS]' if gps_fixtype.value >= 3 else '[FAIL] need FIX_3D+'}")

    print("\n  Battery:")
    print(f"    Level      : {battery_per:.1f}%  "
          f"{'[PASS]' if battery_per > 30 else '[FAIL] need 30%+'}")
    print(f"    Voltage    : {battery_voltage:.2f}V")

    print("\n  State:")
    print(f"    Armed      : {armed_check}  "
          f"{'[FAIL] should not be armed' if armed_check else '[PASS]'}")

    print("\n=====================================")

    if (
        gps_count >= 6
        and gps_fixtype.value >= 3
        and battery_per > 30
        and not armed_check
       
    ):
        flight_check = True
        print("  STATUS: READY TO FLY [PASS]")
    else:
        flight_check = False
        print("  STATUS: NOT READY TO FLY [FAIL]")

    print("=====================================\n")

    return flight_check

async def main():
    
    #Setup camera object
    camera = GazeboCamera(CAMERA_TOPIC)
    #Get callibration values
    mtx, dist = load_calibration(CALIBRATION_YAML)
    #Setup aruco detector object
    detector = ArucoDetector(mtx, dist, ARUCO_DICT_ID, MARKER_LENGTH_M)

    #Load YOLO model up front so the blocking load doesn't happen mid-flight
    print("-- Loading YOLO model")
    model = YOLO(YOLO_WEIGHTS)
    #Await Drone connection
    drone = await connect()
    
    # Wait for MAVLink streams to settle before reading telemetry
    await asyncio.sleep(10)
    flight_check = await pre_flight(drone)

    if not flight_check:
        return
    
    target = SharedTarget()

    #Create detection loop task, runs YOLO + ArUco on every frame and updates target
    detection_task = asyncio.create_task(detection_loop(camera, detector, target, model))
    running_tasks = [detection_task]
    
    #End all running tasks if drone has landed
    # Created before takeoff so it sees the full flight cycle, not just what happens after arming
    termination_task = asyncio.create_task(observe_is_in_air(drone, running_tasks))

    #Let Drone Arm and Takeoff before both scripted flight and tracking mode can be active
    print("-- Arming")
    await drone.action.arm()

    print("-- Take Off")
    await drone.action.set_takeoff_altitude(8)
    await drone.action.takeoff()
    await asyncio.sleep(30)

    # Send a zero setpoint first so offboard mode has something valid to hold before we start it
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()

    #Create flight task to allow scripted flight to happen
    flight_task = asyncio.create_task(scripted_flight(drone))
    #Create watcher task to see if marker or object is found
    watcher_task = asyncio.create_task(marker_watcher(target, flight_task))

    #Await watcher task allows for the scripted flight to complete unless a marker/object is found
    #If a marker/object is found it ends scripted flight

    marker_seen = await watcher_task
    #This is an addtional check to make sure flight check has cancceled
    #FLIGHT CHECK SHOULD ALREADY BE CANCCLED VIA THE WATCHER FUNCTION ITSELF
    if marker_seen:
        try:
            await flight_task
        except asyncio.CancelledError:
            pass
        #Switch to tracking mode
        await tracking_mode(drone, target)
    else:
        print("-- Never saw a marker")

    # Both branches converge here: stop offboard and land
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
    await drone.action.land()

    # Wait for supervisor to detect actual landing and end tasks
    await termination_task
    print("Mission Complete")


if __name__ == "__main__":
    asyncio.run(main())
