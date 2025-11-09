"""
ToolTracking Class for Medical Tool Video Tracking
Uses Grounding DINO for detection and SAM2 for propagation
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import supervision as sv
from loguru import logger

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ToolTracking:
    """
    A class for tracking medical tools in videos using Grounding DINO and SAM2.
    
    This class orchestrates the entire pipeline:
    1. Extract video frames
    2. Detect objects using Grounding DINO
    3. Generate masks using SAM2
    4. Propagate masks through video frames
    5. Save annotated output video
    """
    
    def __init__(
        self,
        source_video: str,
        text_prompt: str = "medical tool.",
        scale_factor: float = 0.5,
        start_idx: int = 0,
        end_idx: int = 1000,
        box_threshold: float = 0.40,
        text_threshold: float = 0.50,
        checkpoint_path: str = None,
        config_path: str = None,
        grounding_model_id: str = "IDEA-Research/grounding-dino-base",
        output_suffix: str = "-grounded-result"
    ):
        """
        Initialize ToolTracking with configuration parameters.
        
        Args:
            source_video: Path to input video file
            text_prompt: Text prompt for Grounding DINO (lowercase, end with dot)
            scale_factor: Scale factor for frame resizing (0.5 = half size)
            start_idx: Starting frame index
            end_idx: Ending frame index
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            checkpoint_path: Path to SAM2 checkpoint file
            config_path: Path to SAM2 config file
            grounding_model_id: HuggingFace model ID for Grounding DINO
            output_suffix: Suffix for output video filename
        """
        self.source_video = source_video
        self.text_prompt = text_prompt
        self.scale_factor = scale_factor
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.grounding_model_id = grounding_model_id
        self.output_suffix = output_suffix
        
        # Setup paths
        self.home = os.getcwd()
        self.source_frames = Path(self.home) / Path(source_video).stem
        self.target_video = Path(self.home) / f"{Path(source_video).stem}{output_suffix}.mp4"
        
        # Setup checkpoint and config paths
        if checkpoint_path is None:
            self.checkpoint = f"{self.home}/checkpoints/sam2.1_hiera_large.pt"
        else:
            self.checkpoint = checkpoint_path
            
        if config_path is None:
            self.config = r"C:\Users\asiel\OneDrive\Documents\Trackimed_New\segment-anything-2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
        else:
            self.config = config_path
        
        # Initialize device
        self._setup_device()
        
        # Model placeholders
        self.sam2_video_predictor = None
        self.sam2_image_predictor = None
        self.grounding_model = None
        self.processor = None
        
        # Detection results
        self.input_boxes = None
        self.labels = None
        self.best_masks = None
        self.image = None
        
        logger.info(f"ToolTracking initialized:")
        logger.info(f"  Source video: {self.source_video}")
        logger.info(f"  Frames directory: {self.source_frames}")
        logger.info(f"  Output video: {self.target_video}")
        logger.info(f"  Device: {self.device}")

    def _setup_device(self):
        """Configure CUDA device and enable optimizations."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            # Enable optimizations for newer GPUs
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("  Enabled TF32 optimizations for GPU")
    
    def extract_frames(self):
        """Extract and scale video frames to the frames directory."""
        logger.info("\n[1/6] Extracting video frames...")
        
        # Create frames directory
        self.source_frames.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        frames_generator = sv.get_video_frames_generator(
            self.source_video, 
            start=self.start_idx, 
            end=self.end_idx
        )
        images_sink = sv.ImageSink(
            target_dir_path=self.source_frames.as_posix(),
            overwrite=True,
            image_name_pattern="{:05d}.jpeg"
        )
        
        with images_sink:
            for frame in tqdm(frames_generator, desc="Extracting frames"):
                frame = sv.scale_image(frame, self.scale_factor)
                images_sink.save_image(frame)

        logger.info(f"  Frames extracted to {self.source_frames}")

    def load_models(self):
        """Load SAM2 and Grounding DINO models."""
        logger.info("\n[2/6] Loading models...")

        # Load SAM2 models
        self.sam2_video_predictor = build_sam2_video_predictor(
            self.config, 
            self.checkpoint
        )
        sam2_image_model = build_sam2(
            self.config, 
            self.checkpoint, 
            device=self.device
        )
        self.sam2_image_predictor = SAM2ImagePredictor(sam2_image_model)
        logger.success("  SAM2 models loaded")
        
        # Load Grounding DINO model
        self.processor = AutoProcessor.from_pretrained(self.grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model_id
        ).to(self.device)
        logger.success("  Grounding DINO model loaded")
    
    def detect_objects(self):
        """Run Grounding DINO on first frame to detect objects."""
        logger.info("\n[3/6] Detecting objects with Grounding DINO...")

        # Load first frame
        image_path = self.source_frames / "00000.jpeg"
        self.image = Image.open(image_path).convert("RGB")
        
        # Process with Grounding DINO
        inputs = self.processor(
            images=self.image, 
            text=self.text_prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[self.image.size[::-1]]
        )
        
        # Filter by score threshold
        scores = results[0]["scores"]
        keep = scores > self.box_threshold
        
        self.input_boxes = results[0]["boxes"][keep].cpu().numpy()
        self.labels = [results[0]["labels"][i] for i in range(len(keep)) if keep[i]]

        logger.info(f"  Detected {len(self.input_boxes)} objects: {self.labels}")

        if len(self.input_boxes) == 0:
            logger.warning("  WARNING: No objects detected! Try lowering box_threshold.")

    def generate_masks(self):
        """Use SAM2 image predictor to generate masks for detected objects."""
        logger.info("\n[4/6] Generating masks with SAM2...")

        if len(self.input_boxes) == 0:
            logger.warning("  No objects to generate masks for")
            self.best_masks = np.array([])
            return
        
        # Set image for SAM2 predictor
        self.sam2_image_predictor.set_image(np.array(self.image))
        
        # Generate masks
        masks, scores, logits = self.sam2_image_predictor.predict(
            box=self.input_boxes,
            multimask_output=True
        )
        
        # Select best mask for each detection based on IoU scores
        best_mask_indices = scores.argmax(axis=1)
        self.best_masks = masks[np.arange(len(best_mask_indices)), best_mask_indices]

        logger.info(f"  Generated {len(self.best_masks)} masks with shape {self.best_masks.shape}")

    def track_video(self):
        """Propagate masks through video frames and save output."""
        logger.info("\n[5/6] Tracking objects through video...")

        if len(self.best_masks) == 0:
            print("  No masks to track")
            return
        
        # Initialize video predictor state
        inference_state = self.sam2_video_predictor.init_state(
            video_path=self.source_frames.as_posix()
        )
        self.sam2_video_predictor.reset_state(inference_state)
        
        # Add each detected object to the video predictor
        for obj_id, mask in enumerate(self.best_masks):
            self.sam2_video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask
            )

        logger.info(f"  Initialized tracking for {len(self.best_masks)} objects")

        # Prepare video output
        video_info = sv.VideoInfo.from_video_path(self.source_video)
        video_info.width = int(video_info.width * self.scale_factor)
        video_info.height = int(video_info.height * self.scale_factor)
        
        source_frame_paths = sorted(
            sv.list_files_with_extensions(
                self.source_frames.as_posix(), 
                extensions=["jpeg"]
            )
        )
        
        # Create mask annotator
        mask_annotator = sv.MaskAnnotator()
        
        # Propagate and save
        with sv.VideoSink(self.target_video.as_posix(), video_info=video_info) as sink:
            for frame_idx, object_ids, mask_logits in tqdm(
                self.sam2_video_predictor.propagate_in_video(inference_state),
                desc="Processing frames",
                total=len(source_frame_paths)
            ):
                # Load frame
                frame_path = source_frame_paths[frame_idx]
                frame = cv2.imread(frame_path)
                
                # Convert mask logits to binary masks
                masks = (mask_logits > 0.0).cpu().numpy()
                masks = masks.astype(bool)
                
                # Create detections for this frame
                frame_detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks[:, 0, :]),
                    mask=masks[:, 0, :],
                    class_id=np.array(object_ids)
                )
                
                # Annotate frame (only masks, no labels)
                annotated_frame = mask_annotator.annotate(
                    scene=frame.copy(), 
                    detections=frame_detections
                )
                
                # Write to output video
                sink.write_frame(annotated_frame)

        logger.info(f"  Video saved to: {self.target_video}")

    def visualize_detections(self, save_path: str = None):
        """
        Visualize initial detections with boxes and labels.
        
        Args:
            save_path: Optional path to save visualization image
        """
        if self.input_boxes is None or self.image is None:
            print("No detections to visualize. Run detect_objects() first.")
            return
        
        detections = sv.Detections(xyxy=self.input_boxes)
        detections.class_id = np.arange(len(self.input_boxes))
        
        annotated_image = self.image.copy()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = label_annotator.annotate(
            annotated_image, 
            detections, 
            labels=[f"{l} {i}" for i, l in enumerate(self.labels)]
        )
        
        if save_path:
            annotated_image.save(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            sv.plot_image(annotated_image)
    
    def visualize_masks(self, save_path: str = None):
        """
        Visualize generated masks on first frame.
        
        Args:
            save_path: Optional path to save visualization image
        """
        if self.best_masks is None or self.image is None:
            logger.warning("No masks to visualize. Run generate_masks() first.")
            return
        
        detections = sv.Detections(xyxy=self.input_boxes)
        detections.class_id = np.arange(len(self.input_boxes))
        detections.mask = self.best_masks.astype(bool)
        
        annotated_image = self.image.copy()
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = mask_annotator.annotate(annotated_image, detections)
        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = label_annotator.annotate(
            annotated_image, 
            detections, 
            labels=[f"{l} {i}" for i, l in enumerate(self.labels)]
        )
        
        if save_path:
            annotated_image.save(save_path)
            logger.info(f"Mask visualization saved to {save_path}")
        else:
            sv.plot_image(annotated_image)
    
    def run(self, visualize: bool = False):
        """
        Run the complete tracking pipeline.
        
        Args:
            visualize: Whether to show visualizations during processing
        
        Returns:
            Path to output video file
        """
        logger.info("="*60)
        logger.info("Starting ToolTracking Pipeline")
        logger.info("="*60)
        
        # Step 1: Extract frames
        self.extract_frames()
        
        # Step 2: Load models
        self.load_models()
        
        # Step 3: Detect objects
        self.detect_objects()
        
        if visualize and len(self.input_boxes) > 0:
            logger.debug("\nVisualizing detections...")
            self.visualize_detections()
        
        # Step 4: Generate masks
        self.generate_masks()
        
        if visualize and len(self.best_masks) > 0:
            logger.debug("\nVisualizing masks...")
            self.visualize_masks()
        
        # Step 5: Track through video
        self.track_video()

        logger.info("="*60)
        logger.info("Pipeline Complete!")
        logger.info("="*60)
        logger.info(f"\n[6/6] Output saved to: {self.target_video}")

        return self.target_video
    
    def cleanup(self):
        """Clean up resources and free memory."""
        self.sam2_video_predictor = None
        self.sam2_image_predictor = None
        self.grounding_model = None
        self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Resources cleaned up")


def main():
    """
    Test the ToolTracking class on demo 7 video.
    This reproduces the exact functionality from track_grounded.ipynb
    """
    logger.info("Testing ToolTracking on Demo 7")
    logger.info("-" * 60)

    # Initialize tracker with same parameters as notebook
    tracker = ToolTracking(
        source_video=r"demo 7.mp4",
        text_prompt="medical tool.",
        scale_factor=0.5,
        start_idx=0,
        end_idx=1000,
        box_threshold=0.40,
        text_threshold=0.50,
        output_suffix="-grounded-result-new"
    )
    
    # Run the complete pipeline
    output_path = tracker.run(visualize=False)

    logger.success(f"\nTest complete! Output video: {output_path}")

    # Optional: Clean up
    # tracker.cleanup()


if __name__ == "__main__":
    main()
