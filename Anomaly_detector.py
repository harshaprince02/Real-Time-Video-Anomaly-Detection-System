import cv2
import numpy as np
import argparse
import time
import os
import pandas as pd
from collections import OrderedDict
from scipy.spatial import distance as dist
import platform
import tkinter as tk

# -------------------------
# Cross-platform Beep
# -------------------------
def beep():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 300)
        else:
            print("\a", end="", flush=True)
    except:
        pass

# -------------------------
# Centroid Tracker
# -------------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=40, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.rects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, rect):
        self.objects[self.nextObjectID] = centroid
        self.rects[self.nextObjectID] = rect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        for d in [self.objects, self.rects, self.disappeared]:
            if objectID in d:
                del d[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.rects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.rects[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])

        return self.objects, self.rects

# -------------------------
# Trackable Object
# -------------------------
class TrackableObject:
    def __init__(self, objectID, centroid, timestamp, bbox):
        self.objectID = objectID
        self.centroids = [centroid]
        self.bboxes = [bbox]
        self.in_zone = False
        self.enter_time = None
        self.last_seen = timestamp
        self.behavior = "unknown"

    def update(self, centroid, timestamp, bbox):
        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.last_seen = timestamp

# -------------------------
# Helper Functions
# -------------------------
def centroid_in_rect(centroid, rect):
    (cx, cy) = centroid
    x, y, w, h = rect
    return x <= cx <= x + w and y <= cy <= y + h

def classify_behavior(tobj: TrackableObject):
    if len(tobj.centroids) < 2:
        return "unknown"
    (x1, y1), (x2, y2) = tobj.centroids[-2], tobj.centroids[-1]
    speed = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
    (sx, sy, ex, ey) = tobj.bboxes[-1]
    h = ey - sy

    # Improved fall detection
    if len(tobj.bboxes) >= 3:
        _, sy_old, _, ey_old = tobj.bboxes[-3]
        height_change = (ey_old - sy_old) - h
    else:
        height_change = 0

    if speed > 20: return "running"
    elif speed > 5: return "walking"
    elif height_change > 50 and speed > 10: return "falling"
    elif h < 80: return "sitting"
    return "idle"

def detect_fight(objects, obj_rects, trackableObjects):
    ids = list(objects.keys())
    if len(ids) < 2: return None
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            id1, id2 = ids[i], ids[j]
            if id1 not in trackableObjects or id2 not in trackableObjects:
                continue
            c1, c2 = objects[id1], objects[id2]
            if np.linalg.norm(np.array(c1)-np.array(c2)) < 120:
                b1 = trackableObjects[id1].behavior
                b2 = trackableObjects[id2].behavior
                if b1 in ("running","walking") and b2 in ("running","walking"):
                    return (id1,id2)
    return None

# -------------------------
# Main Function
# -------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "events_log.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(columns=["timestamp","event_type","object_id","frame","bbox","behavior"]).to_csv(log_path,index=False)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    backSub = cv2.createBackgroundSubtractorMOG2()
    ct = CentroidTracker(maxDisappeared=40, maxDistance=70)
    trackableObjects = {}
    log_buffer = []

    if args.video:
        vs = cv2.VideoCapture(args.video)
    else:
        from imutils.video import VideoStream
        vs = VideoStream(src=0).start()
        time.sleep(1.0)

    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    cv2.namedWindow("Anomaly Detector", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Anomaly Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    zone_enabled = False
    family_mode = args.family_mode
    panic_triggered = False
    frame_idx = 0

    print("[INFO] Controls: 'f'=Family Mode, 'p'=Panic Mode, 'r'=Restricted Zone, 'q'=Quit")

    while True:
        frame_idx += 1
        frame = vs.read() if args.video is None else vs.read()[1]
        if frame is None: break

        # Keep camera horizontal (no rotation)
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        orig = frame.copy()
        (H,W) = frame.shape[:2]
        timestamp = time.time()

        fgMask = backSub.apply(frame)
        _, fgMask = cv2.threshold(fgMask,127,255,cv2.THRESH_BINARY)
        motion_percent = (np.count_nonzero(fgMask)/(W*H))*100
        motion_flag = motion_percent > args.motion_thresh_percent

        if frame_idx % 2 == 0:
            rects,_ = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
            refined = [(x,y,x+w,y+h) for (x,y,w,h) in rects if w>30 and h>60]
            objects, obj_rects = ct.update(refined)
        else:
            objects, obj_rects = ct.objects, ct.rects

        events = []

        zx, zy, zw, zh = int(W*0.6), int(H*0.25), int(W*0.25), int(H*0.35)
        zone_color = (0, 255, 255)
        someone_in_zone = False

        for objectID in list(objects.keys()):
            centroid = tuple(objects[objectID])
            bbox = obj_rects[objectID]

            if objectID not in trackableObjects:
                trackableObjects[objectID] = TrackableObject(objectID, centroid, timestamp, bbox)
            else:
                trackableObjects[objectID].update(centroid, timestamp, bbox)
            tobj = trackableObjects[objectID]
            tobj.behavior = classify_behavior(tobj)

            # Restricted Zone
            if zone_enabled and centroid_in_rect(centroid,(zx,zy,zw,zh)):
                someone_in_zone = True
                if not tobj.in_zone:
                    tobj.in_zone = True
                    tobj.enter_time = timestamp
                    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                    pd.DataFrame([{"timestamp":ts_str,"event_type":"zone_entry","object_id":str(objectID),
                                   "frame":frame_idx,"bbox":str(bbox),"behavior":tobj.behavior}]).to_csv(log_path,mode='a',header=False,index=False)
                    snap = orig.copy()
                    cv2.putText(snap, f"ZONE ALERT ID {objectID} {tobj.behavior}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.imwrite(os.path.join(args.out_dir, f"zone_entry_frame{frame_idx}.jpg"), snap)
                    beep()
            elif tobj.in_zone:
                tobj.in_zone = False
                events.append(("zone_exit", objectID, bbox, tobj.behavior))

            if motion_flag:
                events.append(("motion_alert", objectID, bbox, tobj.behavior))

        if someone_in_zone:
            zone_color = (0,0,255)

        fight_pair = detect_fight(objects,obj_rects,trackableObjects)
        if fight_pair:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            pd.DataFrame([{"timestamp":ts_str,"event_type":"fight_alert","object_id":str(fight_pair),
                           "frame":frame_idx,"bbox":"(0,0,0,0)","behavior":"fighting"}]).to_csv(log_path,mode='a',header=False,index=False)
            snap = orig.copy()
            cv2.putText(snap, f"FIGHT ALERT IDs {fight_pair}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.imwrite(os.path.join(args.out_dir, f"fight_alert_frame{frame_idx}.jpg"), snap)
            beep()

        if panic_triggered:
            for objectID in objects.keys():
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                pd.DataFrame([{"timestamp":ts_str,"event_type":"panic_alert","object_id":str(objectID),
                               "frame":frame_idx,"bbox":str(obj_rects[objectID]),
                               "behavior":trackableObjects[objectID].behavior}]).to_csv(log_path,mode='a',header=False,index=False)
                snap = orig.copy()
                cv2.putText(snap, f"PANIC ALERT ID {objectID}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.imwrite(os.path.join(args.out_dir, f"panic_alert_frame{frame_idx}_ID{objectID}.jpg"), snap)
            beep()
            panic_triggered = False

        if family_mode:
            events = [e for e in events if e[0] in ("panic_alert","fight_alert")]

        for ev_type,objid,bbox,behavior in events:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            log_buffer.append({"timestamp":ts_str,"event_type":ev_type,"object_id":str(objid),
                               "frame":frame_idx,"bbox":str(bbox),"behavior":behavior})

        if len(log_buffer) >= 10:
            pd.DataFrame(log_buffer).to_csv(log_path,mode='a',header=False,index=False)
            log_buffer = []

        if zone_enabled:
            cv2.rectangle(frame,(zx,zy),(zx+zw,zy+zh),zone_color,2)
            cv2.putText(frame,"RESTRICTED ZONE",(zx,zy-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,zone_color,2)

        # Draw objects with behavior-based colors
        for objectID in objects.keys():
            (startX,startY,endX,endY) = obj_rects[objectID]
            behavior = trackableObjects[objectID].behavior
            if fight_pair and objectID in fight_pair:
                color=(0,0,255)
            elif behavior=="walking":
                color=(255,0,0)
            elif behavior=="running":
                color=(0,165,255)
            elif behavior=="sitting":
                color=(128,0,128)
            elif behavior=="falling":
                color=(0,0,139)
            else:
                color=(0,255,0)
            cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
            cv2.putText(frame,f"ID {objectID} {behavior}",
                        (startX,startY-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        # Add live timestamp
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        mode_text = f"Family: {'ON' if family_mode else 'OFF'} | Panic: {'ON' if panic_triggered else 'OFF'} | Zone: {'ON' if zone_enabled else 'OFF'}"
        cv2.putText(frame,mode_text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.imshow("Anomaly Detector",frame)

        key = cv2.waitKey(1) & 0xFF
        if key==ord("q"): break
        elif key==ord("f"):
            family_mode = not family_mode
            print(f"[INFO] Family mode {'ENABLED' if family_mode else 'DISABLED'}")
        elif key==ord("p"):
            panic_triggered = True
            print("[PANIC] Panic event triggered!")
        elif key==ord("r"):
            zone_enabled = not zone_enabled
            print(f"[INFO] Restricted Zone {'ENABLED' if zone_enabled else 'DISABLED'}")

    if log_buffer:
        pd.DataFrame(log_buffer).to_csv(log_path,mode='a',header=False,index=False)

    cv2.destroyAllWindows()
    if args.video:
        vs.release()
    else:
        vs.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",type=str,default=None,help="Path to input video file.")
    parser.add_argument("--out-dir",type=str,default="output",help="Directory for logs and snapshots.")
    parser.add_argument("--motion-thresh-percent",type=float,default=0.5,help="Motion threshold percent.")
    parser.add_argument("--family-mode",action="store_true",help="Start with family mode enabled.")
    args = parser.parse_args()
    main(args)
