import streamlit as st
import cv2
import numpy as np
from scenedetect import detect, ContentDetector
from sklearn.cluster import KMeans, AgglomerativeClustering
from ultralytics import YOLO
from scipy.stats import entropy
from scipy.cluster.hierarchy import linkage, fcluster
import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import tempfile
import os

class ComprehensiveFrameSelector:
    def __init__(self):
        with st.spinner('Loading models...'):
            self.detector = YOLO('yolov8x.pt')
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.resnet.eval()
            if torch.cuda.is_available():
                self.resnet = self.resnet.cuda()
            
            self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        
        self.shopping_classes = [
            # Clothing & Accessories
            'backpack', 'handbag', 'tie', 'suitcase', 'umbrella', 'shoe', 'baseball glove',
            
            # Electronics & Gadgets
            'cell phone', 'laptop', 'mouse', 'remote', 'keyboard', 'tv', 'microwave', 'oven', 'toaster', 'refrigerator',
            
            # Home & Furniture
            'chair', 'couch', 'bed', 'dining table', 'toilet', 'vase', 'clock', 'potted plant', 'sink',
            
            # Kitchen & Dining
            'bottle', 'wine glass', 'cup', 'bowl', 'fork', 'knife', 'spoon', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            
            # Sports & Recreation
            'tennis racket', 'skateboard', 'sports ball', 'frisbee', 'skis', 'snowboard', 'surfboard', 'baseball bat',
            
            # Personal Care
            'toothbrush', 'hair drier', 'scissors',
            
            # Entertainment & Media
            'book', 'kite',
            
            # Transportation
            'bicycle', 'motorcycle'
        ]

    def plot_frames(self, frames, title, scores=None):
        """Helper function to plot frames with optional scores"""
        n_frames = len(frames)
        rows = (n_frames + 4) // 5  # 5 frames per row
        cols = min(5, n_frames)
        
        fig = plt.figure(figsize=(20, 4*rows))
        plt.suptitle(title, fontsize=16)
        
        for i, frame in enumerate(frames):
            plt.subplot(rows, cols, i+1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if scores is not None:
                plt.title(f'Frame {i+1}\nScore: {scores[i]:.3f}')
            else:
                plt.title(f'Frame {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def calculate_metrics(self, frame):
        """Calculate various quality metrics for a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        entropy_score = entropy(hist_norm)
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        color_variance = np.mean([np.var(frame[:,:,i]) for i in range(3)])
        
        return {
            'entropy': entropy_score,
            'edge_density': edge_density,
            'blur_score': blur_score,
            'color_variance': color_variance
        }

    def extract_features(self, frame):
        """Extract features using ResNet"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        img_tensor = self.transform(frame_rgb)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
            features = features.squeeze()
            features = features.cpu().numpy().flatten()
        
        return features

    def calculate_complexity_metrics(self, frame):
        """Calculate metrics that indicate frame complexity and amount of content"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge density - More edges indicate more objects/details
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        # 2. Color complexity - More unique colors suggest more objects
        reduced_colors = frame // 32 * 32  # Reduce to 8 distinct values per channel
        unique_colors = len(np.unique(reduced_colors.reshape(-1, 3), axis=0))
        color_complexity = unique_colors / (256/32) ** 3  # Normalize by max possible colors
        
        # 3. Local variance - High local variance indicates detailed regions
        kernel_size = 5
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_var = cv2.blur(gray * gray, (kernel_size, kernel_size)) - local_mean * local_mean
        variance_complexity = np.mean(local_var) / 255
        
        # 4. Region complexity using connected components
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        num_components, _ = cv2.connectedComponents(binary)
        region_complexity = num_components / (frame.shape[0] * frame.shape[1] / 100)  # Normalize by possible regions
        
        return {
            'edge_density': edge_density,
            'color_complexity': color_complexity,
            'variance_complexity': variance_complexity,
            'region_complexity': region_complexity
        }

    def select_frames(self, video_path, target_frames=20):
        """
        Select representative frames from a video using scene detection, complexity metrics, and feature clustering.
        
        Args:
            video_path (str): Path to the video file
            target_frames (int): Desired number of frames to extract (default: 20)
        
        Returns:
            list: List of selected frames as numpy arrays in BGR format
        """
        try:
            st.write("Step 1: Detecting scenes...")
            scenes = detect(video_path, ContentDetector(threshold=27))
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
                
            frame_data = []
            
            st.write("Step 2: Collecting frame data...")
            progress_bar = st.progress(0)
            scene_frames = []
            
            try:
                for scene_idx, scene in enumerate(scenes):
                    scene_start = scene[0].frame_num
                    scene_end = scene[1].frame_num
                    scene_length = scene_end - scene_start
                    
                    # Adaptively sample frames based on scene length
                    num_samples = min(8, max(2, scene_length // 150))
                    
                    for i in range(num_samples):
                        frame_pos = scene_start + (scene_length * (i + 1)) // (num_samples + 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                        ret, frame = cap.read()
                        
                        if ret:
                            scene_frames.append(frame)
                            complexity_metrics = self.calculate_complexity_metrics(frame)
                            features = self.extract_features(frame)
                            
                            frame_data.append({
                                'frame': frame,
                                'frame_num': frame_pos,
                                'metrics': complexity_metrics,
                                'features': features,
                                'scene_id': len(frame_data)
                            })
                        
                        progress_bar.progress((scene_idx + 1) / len(scenes))
                
                st.write(f"Collected {len(frame_data)} initial frames")
                if not frame_data:
                    raise ValueError("No frames could be extracted from the video")
                    
                self.plot_frames(scene_frames, "Step 2: Initial Frames from Scene Detection")
                
                if len(frame_data) > target_frames:
                    st.write("Step 3: Complexity-based pre-selection...")
                    # Calculate complexity scores for each frame
                    complexity_scores = []
                    for fd in frame_data:
                        metrics = fd['metrics']
                        score = (
                            0.35 * metrics['edge_density'] +
                            0.25 * metrics['color_complexity'] +
                            0.20 * metrics['variance_complexity'] +
                            0.20 * metrics['region_complexity']
                        )
                        complexity_scores.append(score)
                    
                    self.plot_frames(
                        [fd['frame'] for fd in frame_data],
                        "Step 3: Frames with Complexity Scores",
                        complexity_scores
                    )
                    
                    # Select frames based on complexity threshold
                    complexity_threshold = sorted(complexity_scores, reverse=True)[min(40, len(complexity_scores)-1)]
                    frame_data = [
                        fd for fd, score in zip(frame_data, complexity_scores)
                        if score >= complexity_threshold
                    ]
                    
                    st.write("Step 4: Clustering-based selection using ResNet features...")
                    # Perform hierarchical clustering on frame features
                    features_matrix = np.array([fd['features'] for fd in frame_data])
                    try:
                        linkage_matrix = linkage(features_matrix, method='ward')
                        cluster_labels = fcluster(linkage_matrix, t=target_frames, criterion='maxclust')
                        
                        # Select the best frame from each cluster
                        final_frames = []
                        for cluster_id in range(1, target_frames + 1):
                            cluster_frames = [
                                fd for fd, label in zip(frame_data, cluster_labels)
                                if label == cluster_id
                            ]
                            if cluster_frames:
                                best_frame = max(
                                    cluster_frames,
                                    key=lambda x: sum(x['metrics'].values())
                                )
                                final_frames.append(best_frame)
                        
                    except Exception as e:
                        st.warning(f"Clustering failed: {str(e)}. Falling back to complexity-based selection.")
                        # Fallback to selecting frames based on complexity scores
                        sorted_frames = [
                            fd for _, fd in sorted(
                                zip(complexity_scores, frame_data),
                                reverse=True
                            )
                        ]
                        final_frames = sorted_frames[:target_frames]
                
                else:
                    st.write("Step 3: Adding additional frames...")
                    final_frames = frame_data.copy()
                    
                    # Fill gaps between frames until we reach target_frames
                    while len(final_frames) < target_frames:
                        # Calculate gaps between consecutive frames
                        gaps = []
                        for i in range(len(final_frames) - 1):
                            gap = final_frames[i+1]['frame_num'] - final_frames[i]['frame_num']
                            gaps.append((gap, i))
                        
                        if not gaps:
                            break
                            
                        # Find and fill largest gap
                        largest_gap, gap_index = max(gaps)
                        new_frame_pos = final_frames[gap_index]['frame_num'] + largest_gap // 2
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
                        ret, frame = cap.read()
                        
                        if ret:
                            complexity_metrics = self.calculate_complexity_metrics(frame)
                            features = self.extract_features(frame)
                            
                            new_frame_data = {
                                'frame': frame,
                                'frame_num': new_frame_pos,
                                'metrics': complexity_metrics,
                                'features': features,
                                'scene_id': len(final_frames)
                            }
                            
                            final_frames.insert(gap_index + 1, new_frame_data)
                        else:
                            break
                
                # Sort frames by their position in video
                final_frames.sort(key=lambda x: x['frame_num'])
                
                # Plot final results
                self.plot_frames([f['frame'] for f in final_frames], "Final Selected Frames")
                
                return [f['frame'] for f in final_frames]
                
            finally:
                # Ensure video capture is released
                cap.release()
                
        except Exception as e:
            st.error(f"Error in frame selection: {str(e)}")
            return []
    
    def detect_shopping_items(self, frames, conf_threshold=0.3):
        """Detect shopping items in the selected frames"""
        all_detections = []
        
        for frame in frames:
            results = self.detector(frame, conf=conf_threshold)[0]
            
            frame_detections = []
            for det in results.boxes.data:
                class_name = results.names[int(det[-1])]
                if class_name in self.shopping_classes:
                    frame_detections.append({
                        'class': class_name,
                        'confidence': float(det[4]),
                        'bbox': det[:4].tolist()
                    })
            
            all_detections.append(frame_detections)
        
        return all_detections

def main():
    st.title("Video Frame Analysis App")
    st.write("""
    This app analyzes videos to extract key frames and detect shopping-related items.
    Upload a video file to begin the analysis.
    """)

    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Create a temporary file
        tfile = None
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.close()  # Close the file handle explicitly
            video_path = tfile.name
            
            # Initialize the frame selector
            selector = ComprehensiveFrameSelector()
            
            with st.spinner('Processing video...'):
                # Select frames
                frames = selector.select_frames(video_path)
                
                if frames:  # Only proceed if frames were successfully extracted
                    st.write("Detecting shopping items...")
                    # Detect objects
                    detections = selector.detect_shopping_items(frames)
                    
                    # Display detections as text
                    for i, frame_detections in enumerate(detections):
                        st.write(f"\nFrame {i + 1}:")
                        for det in frame_detections:
                            st.write(f"Found {det['class']} with {det['confidence']:.2f} confidence")
                    
                    # Visualize results
                    st.write("Final Results with Object Detection:")
                    n_frames = len(frames)
                    rows = (n_frames + 4) // 5
                    cols = min(5, n_frames)
                    
                    fig = plt.figure(figsize=(20, 4*rows))
                    for i, (frame, frame_dets) in enumerate(zip(frames, detections)):
                        frame = frame.copy()
                        
                        for det in frame_dets:
                            bbox = det['bbox']
                            class_name = det['class']
                            confidence = det['confidence']
                            
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            label = f"{class_name}: {confidence:.2f}"
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(frame, (x1, y1-20), (x1 + w, y1), (255, 255, 255), -1)
                            cv2.putText(frame, label, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        
                        plt.subplot(rows, cols, i+1)
                        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        plt.title(f'Frame {i + 1}')
                        plt.axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Clean up: ensure the temporary file is deleted
            if tfile is not None:
                try:
                    if os.path.exists(tfile.name):
                        # Close any remaining file handles
                        import gc
                        gc.collect()  # Force garbage collection
                        os.unlink(tfile.name)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {str(e)}")

if __name__ == "__main__":
    main()