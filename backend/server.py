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

# Harvester imports for Blaze cameras
try:
    from harvesters.core import Harvester
    import platform
    HARVESTER_SUPPORT = True
except ImportError as e:
    logging.warning(f"Harvester support disabled (needed for Blaze cameras): {e}")
    HARVESTER_SUPPORT = False

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Helper function to find GenTL producer for Blaze cameras
def find_producer(name):
    """Helper to find GenTL producers from the environment path."""
    if 'GENICAM_GENTL64_PATH' not in os.environ:
        return ""
    
    paths = os.environ['GENICAM_GENTL64_PATH'].split(os.pathsep)
    
    if platform.system() == "Linux":
        paths.append('/opt/pylon/lib/gentlproducer/gtl/')
    
    for path in paths:
        full_path = path + os.path.sep + name
        if os.path.exists(full_path):
            return full_path
    return ""

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
    model_config = {"extra": "allow"}
    
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
        self.last_connect_error = None
        self.harvester = None
        
    def initialize(self):
        """Initialize camera detection"""
        if not CAMERA_SUPPORT:
            logging.warning("Camera support not available")
            return
        
        # Initialize Harvester for Blaze cameras
        if HARVESTER_SUPPORT:
            try:
                self.harvester = Harvester()
                path_to_blaze_cti = find_producer("ProducerBaslerBlazePylon.cti")
                if path_to_blaze_cti and os.path.exists(path_to_blaze_cti):
                    self.harvester.add_file(path_to_blaze_cti)
                    logging.info(f"Added Blaze GenTL producer: {path_to_blaze_cti}")
                else:
                    logging.warning("Blaze GenTL producer not found, Blaze cameras will not be available")
            except Exception as e:
                logging.error(f"Failed to initialize Harvester: {e}")
                self.harvester = None
            
        try:
            # Initialize Pylon (may not be needed in newer versions)
            if hasattr(pylon.TlFactory.GetInstance(), 'Initialize'):
                pylon.TlFactory.GetInstance().Initialize()
            self.detect_cameras()
        except Exception as e:
            logging.error(f"Failed to initialize camera manager: {e}")
            # Still try to detect cameras even if init fails
            try:
                self.detect_cameras()
            except Exception as e2:
                logging.error(f"Failed to detect cameras: {e2}")
    
    def detect_cameras(self):
        """Detect all available Basler cameras"""
        detected_cameras = {}
        cam_index = 0
        
        # Detect Blaze cameras via Harvester
        if self.harvester:
            try:
                self.harvester.update()
                for dev_info in self.harvester.device_info_list:
                    if 'blaze' in str(dev_info.model).lower():
                        camera_id = f"cam_{cam_index}"
                        camera_info = {
                            'id': camera_id,
                            'name': f"{dev_info.model} ({dev_info.serial_number})",
                            'model': str(dev_info.model),
                            'serial': str(dev_info.serial_number),
                            'type': 'tof',
                            'status': 'disconnected',
                            'device_info': dev_info,
                            'camera': None,
                            'api_type': 'harvester'  # Mark as Harvester camera
                        }
                        detected_cameras[camera_id] = camera_info
                        cam_index += 1
                        logging.info(f"Detected Blaze camera via Harvester: {dev_info.model} ({dev_info.serial_number})")
            except Exception as e:
                logging.error(f"Failed to detect Blaze cameras via Harvester: {e}")
        
        # Detect regular cameras via pypylon
        try:
            available_cameras = pylon.TlFactory.GetInstance().EnumerateDevices()
            
            for device_info in available_cameras:
                serial = device_info.GetSerialNumber()
                model = device_info.GetModelName()
                
                # Skip if this is a Blaze camera (already detected via Harvester)
                if "blaze" in model.lower():
                    continue
                
                camera_id = f"cam_{cam_index}"
                
                # Determine camera type based on model
                camera_type = "line_scan"
                if "tof" in model.lower():
                    camera_type = "tof"
                
                camera_info = {
                    'id': camera_id,
                    'name': f"{model} ({serial})",
                    'model': model,
                    'serial': serial,
                    'type': camera_type,
                    'status': 'disconnected',
                    'device_info': device_info,
                    'camera': None,
                    'api_type': 'pypylon'  # Mark as pypylon camera
                }
                
                detected_cameras[camera_id] = camera_info
                cam_index += 1
                
            logging.info(f"Detected {cam_index} total cameras")
            
        except Exception as e:
            logging.error(f"Failed to detect pypylon cameras: {e}")
        
        self.cameras = detected_cameras
    
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
            self.last_connect_error = f"Camera {camera_id} not found"
            logging.error(self.last_connect_error)
            return False
            
        cam_info = self.cameras[camera_id]
        
        # Route to appropriate API
        if cam_info.get('api_type') == 'harvester':
            return self._connect_harvester_camera(camera_id)
        else:
            return self._connect_pypylon_camera(camera_id)
    
    def _connect_harvester_camera(self, camera_id: str) -> bool:
        """Connect to a Blaze camera via Harvester"""
        try:
            cam_info = self.cameras[camera_id]
            if cam_info['camera'] is not None:
                logging.info(f"Camera {camera_id} is already connected")
                return True
            
            logging.info(f"Connecting to Blaze camera {camera_id} via Harvester")
            dev_info = cam_info['device_info']
            
            # Update device list
            self.harvester.update()
            
            # Create image acquirer
            ia = self.harvester.create({"model": dev_info.model, "serial_number": dev_info.serial_number})
            
            cam_info['camera'] = ia
            cam_info['status'] = 'connected'
            self.frame_counters[camera_id] = 0
            self.last_connect_error = None
            
            logging.info(f"Successfully connected to Blaze camera {camera_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to Blaze camera {camera_id}: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            self.last_connect_error = str(e)
            return False
    
    def _connect_pypylon_camera(self, camera_id: str) -> bool:
        """Connect to a regular camera via pypylon"""
        try:
            cam_info = self.cameras[camera_id]
            if cam_info['camera'] is not None:
                logging.info(f"Camera {camera_id} is already connected")
                return True
                
            logging.info(f"Creating pypylon camera instance for {camera_id}")
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cam_info['device_info']))
            
            logging.info(f"Opening camera {camera_id}")
            camera.Open()
            
            # Set acquisition mode to continuous
            camera.AcquisitionMode.SetValue("Continuous")
            
            # Disable trigger mode if available
            try:
                if hasattr(camera, 'TriggerMode'):
                    camera.TriggerMode.SetValue('Off')
                    logging.info(f"Camera {camera_id}: Trigger mode set to Off")
            except Exception as e:
                logging.warning(f"Could not set trigger mode: {e}")
            
            cam_info['camera'] = camera
            cam_info['status'] = 'connected'
            self.frame_counters[camera_id] = 0
            self.last_connect_error = None
            
            logging.info(f"Successfully connected to pypylon camera {camera_id}")
            return True
                
        except Exception as e:
            error_msg = f"Failed to connect to pypylon camera {camera_id}: {type(e).__name__}: {str(e)}"
            logging.error(error_msg)
            self.last_connect_error = str(e)
            return False
    
    def disconnect_camera(self, camera_id: str):
        """Disconnect from a camera"""
        if camera_id in self.cameras:
            cam_info = self.cameras[camera_id]
            if cam_info['camera']:
                try:
                    self.stop_streaming(camera_id)
                    
                    if cam_info.get('api_type') == 'harvester':
                        cam_info['camera'].destroy()
                    else:
                        cam_info['camera'].Close()
                    
                    cam_info['camera'] = None
                    cam_info['status'] = 'disconnected'
                    logging.info(f"Disconnected camera {camera_id}")
                except Exception as e:
                    logging.error(f"Error disconnecting camera {camera_id}: {e}")
    
    def configure_camera(self, camera_id: str, config: CameraConfig) -> bool:
        """Configure camera parameters"""
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
    
    def start_streaming(self, camera_id: str) -> bool:
        """Start streaming from a camera"""
        if camera_id not in self.cameras or not self.cameras[camera_id]['camera']:
            return False
        
        cam_info = self.cameras[camera_id]
        
        try:
            # Route to appropriate API
            if cam_info.get('api_type') == 'harvester':
                # Harvester camera - configure components then start acquisition
                ia = cam_info['camera']
                
                logging.info(f"Configuring Blaze camera {camera_id} components before streaming")
                
                # Configure Range component (depth) - use Coord3D_C16 for simpler 16-bit depth
                ia.remote_device.node_map.ComponentSelector.value = "Range"
                ia.remote_device.node_map.ComponentEnable.value = True
                ia.remote_device.node_map.PixelFormat.value = "Coord3D_C16"
                
                # Configure Intensity component
                ia.remote_device.node_map.ComponentSelector.value = "Intensity"
                ia.remote_device.node_map.ComponentEnable.value = True
                ia.remote_device.node_map.PixelFormat.value = "Mono16"
                
                # Configure Confidence component (REQUIRED!)
                ia.remote_device.node_map.ComponentSelector.value = "Confidence"
                ia.remote_device.node_map.ComponentEnable.value = True
                ia.remote_device.node_map.PixelFormat.value = "Confidence16"
                
                # Disable GenDC mode
                ia.remote_device.node_map.GenDCStreamingMode.value = "Off"
                
                logging.info(f"Starting Harvester acquisition for camera {camera_id}")
                ia.start()
                logging.info(f"Harvester acquisition started for camera {camera_id}")
            else:
                # pypylon camera
                camera = cam_info['camera']
                
                # Stop any existing grabbing first
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                
                # Start grabbing with latest image only strategy
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                logging.info(f"Camera {camera_id}: StartGrabbing called, IsGrabbing={camera.IsGrabbing()}")
            
            cam_info['status'] = 'streaming'
            
            # Start streaming task
            try:
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._streaming_loop(camera_id))
                self.streaming_tasks[camera_id] = task
                logging.info(f"Created streaming task for camera {camera_id}")
            except Exception as e:
                logging.error(f"Failed to create streaming task: {e}")
                return False
            
            logging.info(f"Started streaming camera {camera_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start streaming camera {camera_id}: {e}")
            return False
    
    def get_camera_settings(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get current camera settings"""
        if camera_id not in self.cameras:
            return None
        
        cam_info = self.cameras[camera_id]
        camera = cam_info['camera']
        if not camera:
            return None
        
        try:
            settings = {}
            
            # Get available parameters
            if camera.ExposureTime.IsReadable():
                settings['exposure_time'] = camera.ExposureTime.GetValue()
            
            if hasattr(camera, 'Gain') and camera.Gain.IsReadable():
                settings['gain'] = camera.Gain.GetValue()
            
            if hasattr(camera, 'AcquisitionFrameRate') and camera.AcquisitionFrameRate.IsReadable():
                settings['frame_rate'] = camera.AcquisitionFrameRate.GetValue()
            
            if camera.Width.IsReadable():
                settings['width'] = camera.Width.GetValue()
                
            if camera.Height.IsReadable():
                settings['height'] = camera.Height.GetValue()
                
            return settings
        except Exception as e:
            logging.error(f"Failed to get camera settings for {camera_id}: {e}")
            return None
    
    def stop_streaming(self, camera_id: str):
        """Stop streaming from a camera"""
        if camera_id in self.streaming_tasks:
            self.streaming_tasks[camera_id].cancel()
            del self.streaming_tasks[camera_id]
            
        if camera_id in self.cameras:
            cam_info = self.cameras[camera_id]
            if cam_info['camera'] and cam_info['status'] == 'streaming':
                try:
                    if cam_info.get('api_type') == 'harvester':
                        cam_info['camera'].stop()
                    else:
                        cam_info['camera'].StopGrabbing()
                    cam_info['status'] = 'connected'
                    logging.info(f"Stopped streaming camera {camera_id}")
                except Exception as e:
                    logging.error(f"Error stopping streaming camera {camera_id}: {e}")
    
    async def _streaming_loop(self, camera_id: str):
        """Internal streaming loop"""
        logging.info(f"Starting streaming loop for camera {camera_id}")
        cam_info = self.cameras[camera_id]
        
        # Route to appropriate streaming method
        if cam_info.get('api_type') == 'harvester':
            await self._streaming_loop_harvester(camera_id)
        else:
            await self._streaming_loop_pypylon(camera_id)
    
    async def _streaming_loop_harvester(self, camera_id: str):
        """Streaming loop for Harvester cameras (Blaze)"""
        logging.info(f"Starting Harvester streaming loop for camera {camera_id}")
        cam_info = self.cameras[camera_id]
        ia = cam_info['camera']
        frame_count = 0
        loop = asyncio.get_event_loop()
        
        def fetch_and_process_frame():
            """Blocking function to fetch and process frame - runs in thread pool"""
            try:
                with ia.fetch(timeout=1.0) as buffer:
                    # Check if buffer has components
                    if not buffer.payload.components or len(buffer.payload.components) == 0:
                        return {'success': False, 'error': 'No components in buffer'}
                    
                    # Get the depth component (Range) - should be first component
                    depth_component = buffer.payload.components[0]
                    
                    # Check if component has data
                    if depth_component.data is None or depth_component.data.size == 0:
                        return {'success': False, 'error': 'Empty component data'}
                    
                    depth_data = depth_component.data.reshape(depth_component.height, depth_component.width).copy()
                    
                    # Normalize depth to 0-255 for visualization
                    depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    # Convert to RGB for display
                    image_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
                    
                    # Convert to PIL Image and then to base64
                    pil_image = Image.fromarray(image_rgb)
                    buffer_io = BytesIO()
                    pil_image.save(buffer_io, format='JPEG', quality=85)
                    image_base64 = base64.b64encode(buffer_io.getvalue()).decode('utf-8')
                    
                    return {
                        'success': True,
                        'frame_data': image_base64,
                        'width': depth_component.width,
                        'height': depth_component.height
                    }
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        try:
            while cam_info['status'] == 'streaming':
                try:
                    # Run blocking fetch in thread pool
                    result = await loop.run_in_executor(None, fetch_and_process_frame)
                    
                    if result['success']:
                        frame_count += 1
                        if frame_count % 30 == 0:
                            logging.info(f"Camera {camera_id}: captured {frame_count} frames")
                        
                        # Create frame data
                        frame_data = {
                            'camera_id': camera_id,
                            'timestamp': time.time(),
                            'frame_data': result['frame_data'],
                            'width': result['width'],
                            'height': result['height'],
                            'frame_number': self.frame_counters[camera_id]
                        }
                        
                        self.frame_counters[camera_id] += 1
                        
                        # Store latest frame
                        cam_info['latest_frame'] = frame_data
                    else:
                        logging.debug(f"Fetch failed for {camera_id}: {result.get('error')}")
                        await asyncio.sleep(0.05)
                        
                except Exception as fetch_error:
                    logging.warning(f"Fetch error for camera {camera_id}: {fetch_error}")
                    await asyncio.sleep(0.05)
                
        except asyncio.CancelledError:
            logging.info(f"Harvester streaming cancelled for camera {camera_id}")
        except Exception as e:
            logging.error(f"Harvester streaming error for camera {camera_id}: {e}")
            cam_info['status'] = 'connected'
    
    async def _streaming_loop_pypylon(self, camera_id: str):
        """Streaming loop for pypylon cameras"""
        logging.info(f"Starting pypylon streaming loop for camera {camera_id}")
        cam_info = self.cameras[camera_id]
        camera = cam_info['camera']
        frame_count = 0
        
        try:
            while cam_info['status'] == 'streaming':
                if camera.IsGrabbing():
                    grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                    
                    if grab_result.GrabSucceeded():
                        frame_count += 1
                        if frame_count % 30 == 0:  # Log every 30 frames
                            logging.info(f"Camera {camera_id}: captured {frame_count} frames")
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
    # Don't call detect_cameras here as it resets camera statuses
    # camera_manager.detect_cameras()  # Refresh camera list
    return camera_manager.get_camera_list()

@api_router.post("/cameras/{camera_id}/connect")
async def connect_camera(camera_id: str):
    """Connect to a camera"""
    success = camera_manager.connect_camera(camera_id)
    if success:
        return {"message": f"Connected to camera {camera_id}"}
    else:
        # Get the last error from camera manager
        error_msg = getattr(camera_manager, 'last_connect_error', 'Unknown error')
        raise HTTPException(status_code=400, detail=f"Failed to connect to camera: {error_msg}")

@api_router.post("/cameras/{camera_id}/disconnect")
async def disconnect_camera(camera_id: str):
    """Disconnect from a camera"""
    camera_manager.disconnect_camera(camera_id)
    return {"message": f"Disconnected camera {camera_id}"}

@api_router.post("/cameras/{camera_id}/configure")
async def configure_camera(camera_id: str, config: CameraConfig):
    """Configure camera parameters"""
    success = camera_manager.configure_camera(camera_id, config)
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

@api_router.get("/cameras/{camera_id}/settings")
async def get_camera_settings(camera_id: str):
    """Get current camera settings"""
    settings = camera_manager.get_camera_settings(camera_id)
    if settings is not None:
        return settings
    else:
        raise HTTPException(status_code=404, detail="Camera not found or not connected")

@api_router.get("/cameras/{camera_id}/frame")
async def get_camera_frame(camera_id: str):
    """Get the latest frame from a streaming camera"""
    frame_data = camera_manager.get_latest_frame(camera_id)
    if frame_data is not None:
        return frame_data
    else:
        raise HTTPException(status_code=404, detail="No frame available. Camera may not be streaming.")

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
    allow_origins=["*"],  # Allow all origins for development
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
    
    # client.close()
    logger.info("Application shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
