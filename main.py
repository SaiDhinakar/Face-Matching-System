import dlib
import cv2
import numpy as np
import os
from datetime import datetime
from collections import defaultdict
import pickle

class FaceTracker:
    def __init__(self, max_disappear=30, min_confidence=0.6):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappear = max_disappear
        self.min_confidence = min_confidence
        
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappear:
                    self.deregister(object_id)
            return self.objects
            
        centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            centroids[i] = (cX, cY)
            
        if len(self.objects) == 0:
            for i in range(0, len(centroids)):
                self.register(centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = np.zeros((len(object_centroids), len(centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(centroids)):
                    D[i][j] = np.linalg.norm(object_centroids[i] - centroids[j])
                    
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappear:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(centroids[col])
                    
        return self.objects

class AdvancedFaceMatcher:
    def __init__(self, images_dir, unknown_dir="unknown_faces"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_tracker = FaceTracker()
        self.tracked_faces = defaultdict(dict)
        self.unknown_count = 0
        self.unknown_dir = unknown_dir
        
        # Initialize dlib's face detector and facial landmarks predictor
        self.detector = dlib.get_frontal_face_detector()
        model_path = "dlib_models/shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Please download the shape predictor model from dlib's website")
        self.predictor = dlib.shape_predictor(model_path)
        self.face_rec_model = dlib.face_recognition_model_v1("dlib_models/dlib_face_recognition_resnet_model_v1.dat")
        
        # Create unknown faces directory
        if not os.path.exists(unknown_dir):
            os.makedirs(unknown_dir)
            
        self.load_reference_images(images_dir)
        
    def get_face_encoding(self, image):
        """Get face encoding using dlib directly."""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Detect faces
        faces = self.detector(rgb_image)
        if not faces:
            return None
            
        # Get landmarks and compute descriptor
        shape = self.predictor(rgb_image, faces[0])
        face_descriptor = np.array(self.face_rec_model.compute_face_descriptor(rgb_image, shape))
        return face_descriptor

    def load_reference_images(self, base_dir):
        """Load reference images from hierarchical folder structure."""
        encodings_cache_file = "face_encodings_cache.pkl"
        
        # Try to load cached encodings
        if os.path.exists(encodings_cache_file):
            try:
                with open(encodings_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.known_face_encodings = cached_data['encodings']
                    self.known_face_names = cached_data['names']
                    print("Loaded cached face encodings")
                    return
            except Exception as e:
                print(f"Error loading cache: {str(e)}")

        for person_folder in os.listdir(base_dir):
            person_path = os.path.join(base_dir, person_folder)
            if os.path.isdir(person_path):
                person_encodings = []
                
                for image_file in os.listdir(person_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_path, image_file)
                        try:
                            image = cv2.imread(image_path)
                            face_encoding = self.get_face_encoding(image)
                            
                            if face_encoding is not None:
                                person_encodings.append(face_encoding)
                                print(f"Loaded image for {person_folder}: {image_file}")
                            else:
                                print(f"No face found in {image_file}")
                        except Exception as e:
                            print(f"Error processing {image_file}: {str(e)}")
                
                if person_encodings:
                    average_encoding = np.mean(person_encodings, axis=0)
                    self.known_face_encodings.append(average_encoding)
                    self.known_face_names.append(person_folder)
                    print(f"Successfully loaded {len(person_encodings)} images for {person_folder}")
        
        # Cache the encodings
        if self.known_face_encodings:
            try:
                with open(encodings_cache_file, 'wb') as f:
                    pickle.dump({
                        'encodings': self.known_face_encodings,
                        'names': self.known_face_names
                    }, f)
                print("Cached face encodings for future use")
            except Exception as e:
                print(f"Error caching encodings: {str(e)}")

    def compare_faces(self, face_encoding, tolerance=0.6):
        """Compare face encoding with known faces."""
        if len(self.known_face_encodings) == 0:
            return None, 0
            
        # Calculate distances to all known faces
        face_distances = [np.linalg.norm(face_encoding - enc) for enc in self.known_face_encodings]
        best_match_index = np.argmin(face_distances)
        min_distance = face_distances[best_match_index]
        
        if min_distance < tolerance:
            return self.known_face_names[best_match_index], 1 - min_distance
        return None, 0

    def save_unknown_face(self, frame, face_location):
        """Save unknown face to designated folder."""
        left, top, right, bottom = face_location
        face_image = frame[top:bottom, left:right]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_{self.unknown_count}_{timestamp}.jpg"
        filepath = os.path.join(self.unknown_dir, filename)
        cv2.imwrite(filepath, face_image)
        self.unknown_count += 1
        return filename

    def start_webcam_matching(self):
        """Start webcam feed with face tracking and recognition for multiple faces."""
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            raise IOError("Cannot access webcam")

        print("Starting webcam face matching... Press 'q' to quit.")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect faces using dlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector(rgb_frame)
            
            # Convert dlib rectangles to format expected by tracker
            face_locations = [(face.left(), face.top(), face.right(), face.bottom()) 
                            for face in faces]

            # Update face tracker
            tracked_objects = self.face_tracker.update(face_locations)

            # Process each detected face
            for face, face_loc in zip(faces, face_locations):
                left, top, right, bottom = face_loc
                centroid = ((left + right) // 2, (top + bottom) // 2)

                # Find matching tracked object
                matched_id = None
                for obj_id, obj_centroid in tracked_objects.items():
                    if np.linalg.norm(np.array(centroid) - obj_centroid) < 50:
                        matched_id = obj_id
                        break

                if matched_id is None:
                    continue

                # Process only untracked faces
                if matched_id not in self.tracked_faces:
                    # Get face encoding
                    shape = self.predictor(rgb_frame, face)
                    face_encoding = np.array(
                        self.face_rec_model.compute_face_descriptor(rgb_frame, shape)
                    )
                    
                    # Compare with known faces
                    name, confidence = self.compare_faces(face_encoding)
                    
                    if name is None:
                        name = "Unknown"
                        filename = self.save_unknown_face(frame, face_loc)
                        print(f"Saved unknown face: {filename}")

                    self.tracked_faces[matched_id] = {
                        'name': name,
                        'confidence': confidence,
                        'first_seen': datetime.now()
                    }

                # Draw rectangle and label for each detected face
                face_info = self.tracked_faces[matched_id]
                color = (0, 255, 0) if face_info['name'] != "Unknown" else (0, 0, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                label = f"{face_info['name']} (ID: {matched_id})"
                if face_info['name'] != "Unknown":
                    label += f" {face_info['confidence']:.2%}"
                    
                cv2.rectangle(frame, (left, bottom + 15), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left , bottom + 15),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Recognition System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def start_image_matching(self, image_path):
        """Match multiple faces in a static image."""
        image = cv2.imread(image_path)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb_frame)

        face_locations = [(face.left(), face.top(), face.right(), face.bottom())
                          for face in faces]
        
        for face, face_loc in zip(faces, face_locations):
            left, top, right, bottom = face_loc
            centroid = ((left + right) // 2, (top + bottom) // 2)

            shape = self.predictor(rgb_frame, face)
            face_encoding = np.array(
                self.face_rec_model.compute_face_descriptor(rgb_frame, shape)
            )

            name, confidence = self.compare_faces(face_encoding)

            if name is None:
                name = "Unknown"
                filename = self.save_unknown_face(image, face_loc)
                print(f"Saved unknown face: {filename}")
        
            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)

            label = f"{name} : {confidence:.2%}"

            cv2.rectangle(image, (left, bottom+15), (right, bottom), color, cv2.FILLED)
            cv2.putText(image, label, (left , bottom + 15),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Multiple Face Matching', image)
        cv2.imwrite('Face-Matching.jpg', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    images_dir = "images/"
    unknown_dir = "unknown_faces/"
    
    for directory in [images_dir, unknown_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    face_matcher = AdvancedFaceMatcher(images_dir, unknown_dir)
    
    if not face_matcher.known_face_encodings:
        print("No reference images found. Please add person folders with images to the 'images' directory.")
        print("Expected structure:")
        print("images/")
        print("  ├── person1/")
        print("  │   ├── image1.jpg")
        print("  │   └── image2.jpg")
        print("  └── person2/")
        print("      ├── image1.jpg")
        print("      └── image2.jpg")
        return

    # face_matcher.start_webcam_matching()
    face_matcher.start_image_matching('test.jpg')
    face_matcher.start_image_matching('test2.jpg')

if __name__ == "__main__":
    main()