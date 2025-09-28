from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import threading
import time
import json
from io import BytesIO
import base64

# Camera imports
try:
    from pypylon import pylon
    import cv2
    import numpy as np
    from PIL import Image
    CAMERA_SUPPORT = True
except ImportError as e:
    logging.warning(f"Camera support disabled: {e}")
    CAMERA_SUPPORT = False

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Basler Camera Application")
api_router = APIRouter(prefix="/api")

# Global camera manager
camera_manager = None

class CameraInfo(BaseModel):
    id: str
    name: str
    model: str
    serial: str
    type: str  # 'line_scan' or 'tof'
    status: str  # 'connected', 'disconnected', 'streaming'
    network_info: Optional[Dict[str, Any]] = None

class CameraConfig(BaseModel):
    camera_id: str
    exposure_time: Optional[float] = None
    gain: Optional[float] = None
    frame_rate: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    custom_params: Optional[Dict[str, Any]] = None

class SaveConfig(BaseModel):
    enabled: bool
    base_directory: str
    create_timestamp_folder: bool = True
    save_format: str = "png"  # png, jpg, tiff

class StreamFrame(BaseModel):
    camera_id: str
    timestamp: float
    frame_data: str  # base64 encoded
    width: int
    height: int
    frame_number: int

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.streaming_tasks = {}
        self.save_config = None
        self.frame_counters = {}
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize camera detection"""
        if not CAMERA_SUPPORT:
            logging.warning("Camera support not available")
            return
            
        try:
            # Initialize Pylon
            pylon.TlFactory.GetInstance().Initialize()
            self.detect_cameras()
        except Exception as e:
            logging.error(f"Failed to initialize camera manager: {e}")
    
    def detect_cameras(self):
        """Detect all available Basler cameras"""
        try:
            # Get available cameras
            available_cameras = pylon.TlFactory.GetInstance().EnumerateDevices()
            
            detected_cameras = {}
            
            for i, device_info in enumerate(available_cameras):
                camera_id = f"cam_{i}"
                serial = device_info.GetSerialNumber()
                model = device_info.GetModelName()
                
                # Determine camera type based on model
                camera_type = "line_scan"
                if "blaze" in model.lower() or "tof" in model.lower():
                    camera_type = "tof"
                
                camera_info = {
                    'id': camera_id,
                    'name': f"{model} ({serial})",
                    'model': model,
                    'serial': serial,
                    'type': camera_type,
                    'status': 'disconnected',
                    'device_info': device_info,
                    'camera': None
                }
                
                detected_cameras[camera_id] = camera_info
                
            self.cameras = detected_cameras
            logging.info(f"Detected {len(detected_cameras)} cameras")
            
        except Exception as e:
            logging.error(f"Failed to detect cameras: {e}")
    
    def get_camera_list(self) -> List[CameraInfo]:
        """Get list of available cameras"""
        camera_list = []
        for cam_id, cam_info in self.cameras.items():
            camera_list.append(CameraInfo(
                id=cam_info['id'],
                name=cam_info['name'],
                model=cam_info['model'],
                serial=cam_info['serial'],
                type=cam_info['type'],
                status=cam_info['status']
            ))
        return camera_list
    
    def connect_camera(self, camera_id: str) -> bool:
        """Connect to a specific camera"""
        if camera_id not in self.cameras:
            return False
            
        try:
            cam_info = self.cameras[camera_id]
            if cam_info['camera'] is None:
                # Create camera instance
                camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cam_info['device_info']))
                camera.Open()
                
                # Basic configuration
                camera.AcquisitionMode.SetValue("Continuous")
                
                cam_info['camera'] = camera
                cam_info['status'] = 'connected'
                self.frame_counters[camera_id] = 0
                
                logging.info(f"Connected to camera {camera_id}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to connect to camera {camera_id}: {e}")
            
        return False
    
    def disconnect_camera(self, camera_id: str):
        """Disconnect from a camera"""
        if camera_id in self.cameras:
            cam_info = self.cameras[camera_id]
            if cam_info['camera']:
                try:
                    self.stop_streaming(camera_id)
                    cam_info['camera'].Close()
                    cam_info['camera'] = None
                    cam_info['status'] = 'disconnected'
                    logging.info(f"Disconnected camera {camera_id}")
                except Exception as e:
                    logging.error(f"Error disconnecting camera {camera_id}: {e}")
    
    def configure_camera(self, config: CameraConfig) -> bool:
        """Configure camera parameters"""
        camera_id = config.camera_id
        if camera_id not in self.cameras or not self.cameras[camera_id]['camera']:
            return False
            
        try:
            camera = self.cameras[camera_id]['camera']
            
            # Configure exposure time
            if config.exposure_time is not None:
                camera.ExposureTime.SetValue(config.exposure_time)
                
            # Configure gain
            if config.gain is not None:
                camera.Gain.SetValue(config.gain)
                
            # Configure frame rate
            if config.frame_rate is not None:
                camera.AcquisitionFrameRate.SetValue(config.frame_rate)
                
            logging.info(f"Configured camera {camera_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to configure camera {camera_id}: {e}")
            return False
    
    def start_streaming(self, camera_id: str):
        """Start streaming from a camera"""
        if camera_id not in self.cameras:
            return False
            
        cam_info = self.cameras[camera_id]
        if not cam_info['camera'] or cam_info['status'] == 'streaming':
            return False
            
        try:
            camera = cam_info['camera']
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            cam_info['status'] = 'streaming'
            
            # Start streaming task
            task = asyncio.create_task(self._streaming_loop(camera_id))
            self.streaming_tasks[camera_id] = task
            
            logging.info(f"Started streaming camera {camera_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start streaming camera {camera_id}: {e}")
            return False
    
    def stop_streaming(self, camera_id: str):
        """Stop streaming from a camera"""
        if camera_id in self.streaming_tasks:
            self.streaming_tasks[camera_id].cancel()
            del self.streaming_tasks[camera_id]
            
        if camera_id in self.cameras:
            cam_info = self.cameras[camera_id]
            if cam_info['camera'] and cam_info['status'] == 'streaming':
                try:
                    cam_info['camera'].StopGrabbing()
                    cam_info['status'] = 'connected'
                    logging.info(f"Stopped streaming camera {camera_id}")
                except Exception as e:
                    logging.error(f"Error stopping streaming camera {camera_id}: {e}")
    
    async def _streaming_loop(self, camera_id: str):
        """Internal streaming loop"""
        cam_info = self.cameras[camera_id]
        camera = cam_info['camera']
        
        try:
            while cam_info['status'] == 'streaming':
                if camera.IsGrabbing():
                    grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                    
                    if grab_result.GrabSucceeded():
                        # Convert to OpenCV format
                        image = grab_result.Array
                        
                        # Handle different image formats
                        if len(image.shape) == 2:  # Grayscale
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        else:
                            image_rgb = image
                            
                        # Convert to PIL Image and then to base64
                        pil_image = Image.fromarray(image_rgb)
                        buffer = BytesIO()
                        pil_image.save(buffer, format='JPEG', quality=85)
                        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        # Create frame data
                        frame_data = {
                            'camera_id': camera_id,
                            'timestamp': time.time(),
                            'frame_data': image_base64,
                            'width': image.shape[1] if len(image.shape) > 1 else image.shape[0],
                            'height': image.shape[0] if len(image.shape) > 1 else 1,
                            'frame_number': self.frame_counters[camera_id]
                        }
                        
                        self.frame_counters[camera_id] += 1
                        
                        # Save frame if enabled
                        if self.save_config and self.save_config['enabled']:
                            await self._save_frame(camera_id, pil_image, frame_data['timestamp'])
                        
                        # Store latest frame (could be used for streaming endpoint)
                        cam_info['latest_frame'] = frame_data
                        
                    grab_result.Release()
                    
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                
        except asyncio.CancelledError:
            logging.info(f"Streaming cancelled for camera {camera_id}")
        except Exception as e:
            logging.error(f"Streaming error for camera {camera_id}: {e}")
            cam_info['status'] = 'connected'
    
    async def _save_frame(self, camera_id: str, image: Image.Image, timestamp: float):
        """Save frame to disk"""
        try:
            base_dir = Path(self.save_config['base_directory'])
            
            if self.save_config.get('create_timestamp_folder', True):
                timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
                save_dir = base_dir / timestamp_str / camera_id
            else:
                save_dir = base_dir / camera_id
                
            save_dir.mkdir(parents=True, exist_ok=True)
            
            frame_filename = f"frame_{int(timestamp * 1000)}.{self.save_config.get('save_format', 'png')}"
            save_path = save_dir / frame_filename
            
            # Save in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, image.save, str(save_path))
            
        except Exception as e:
            logging.error(f"Failed to save frame for camera {camera_id}: {e}")
    
    def set_save_config(self, config: SaveConfig):
        """Set save configuration"""
        self.save_config = config.dict()
        logging.info(f"Save config updated: {self.save_config}")
    
    def get_latest_frame(self, camera_id: str) -> Optional[Dict]:
        """Get the latest frame from a camera"""
        if camera_id in self.cameras and 'latest_frame' in self.cameras[camera_id]:
            return self.cameras[camera_id]['latest_frame']
        return None

# Initialize camera manager
camera_manager = CameraManager()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Basler Camera Application API"}

@api_router.get("/cameras", response_model=List[CameraInfo])
async def get_cameras():
    """Get list of available cameras"""
    camera_manager.detect_cameras()  # Refresh camera list
    return camera_manager.get_camera_list()

@api_router.post("/cameras/{camera_id}/connect")
async def connect_camera(camera_id: str):
    """Connect to a camera"""
    success = camera_manager.connect_camera(camera_id)
    if success:
        return {"message": f"Connected to camera {camera_id}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to connect to camera")

@api_router.post("/cameras/{camera_id}/disconnect")
async def disconnect_camera(camera_id: str):
    """Disconnect from a camera"""
    camera_manager.disconnect_camera(camera_id)
    return {"message": f"Disconnected camera {camera_id}"}

@api_router.post("/cameras/{camera_id}/configure")
async def configure_camera(camera_id: str, config: CameraConfig):
    """Configure camera parameters"""
    config.camera_id = camera_id
    success = camera_manager.configure_camera(config)
    if success:
        return {"message": f"Configured camera {camera_id}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to configure camera")

@api_router.post("/cameras/{camera_id}/start-streaming")
async def start_streaming(camera_id: str):
    """Start streaming from a camera"""
    success = camera_manager.start_streaming(camera_id)
    if success:
        return {"message": f"Started streaming camera {camera_id}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to start streaming")

@api_router.post("/cameras/{camera_id}/stop-streaming")
async def stop_streaming(camera_id: str):
    """Stop streaming from a camera"""
    camera_manager.stop_streaming(camera_id)
    return {"message": f"Stopped streaming camera {camera_id}"}

@api_router.get("/cameras/{camera_id}/frame")
async def get_latest_frame(camera_id: str):
    """Get the latest frame from a camera"""
    frame = camera_manager.get_latest_frame(camera_id)
    if frame:
        return frame
    else:
        raise HTTPException(status_code=404, detail="No frame available")

@api_router.post("/save-config")
async def set_save_config(config: SaveConfig):
    """Set save configuration"""
    camera_manager.set_save_config(config)
    return {"message": "Save configuration updated"}

@api_router.get("/status")
async def get_system_status():
    """Get system status"""
    camera_list = camera_manager.get_camera_list()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cameras": len(camera_list),
        "connected_cameras": len([c for c in camera_list if c.status in ['connected', 'streaming']]),
        "streaming_cameras": len([c for c in camera_list if c.status == 'streaming']),
        "camera_support": CAMERA_SUPPORT,
        "cameras": camera_list
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize camera manager on startup"""
    camera_manager.initialize()
    logger.info("Basler Camera Application started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Disconnect all cameras
    for camera_id in list(camera_manager.cameras.keys()):
        camera_manager.disconnect_camera(camera_id)
    
    client.close()
    logger.info("Application shutdown complete")
