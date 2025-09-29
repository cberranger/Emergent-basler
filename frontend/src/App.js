import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import "@/App.css";

// Import UI components
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { RefreshCw, Camera, AlertCircle, Play, Square, Settings } from 'lucide-react';
import { toast, Toaster } from 'sonner';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from './components/ui/dialog';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Simple Camera Application
const SimpleCameraApp = () => {
  const [cameras, setCameras] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [frames, setFrames] = useState({}); // Store latest frames by camera_id
  const [currentSettings, setCurrentSettings] = useState({}); // Store current settings by camera_id
  const [fps, setFps] = useState({}); // Store FPS by camera_id
  const [configDialog, setConfigDialog] = useState({ open: false, camera: null });
  const [configValues, setConfigValues] = useState({
    exposure_time: '',
    gain: '',
    frame_rate: '',
    width: '',
    height: ''
  });

  const fetchCameras = async () => {
    setLoading(true);
    try {
      console.log('Fetching cameras from:', `${API}/cameras`);
      const response = await axios.get(`${API}/cameras`, { timeout: 5000 });
      console.log('Cameras response:', response.data);
      setCameras(response.data);
      toast.success('Cameras refreshed successfully');
    } catch (error) {
      console.error('Failed to fetch cameras:', error);
      toast.error(`Failed to fetch cameras: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      console.log('Fetching status from:', `${API}/status`);
      const response = await axios.get(`${API}/status`, { timeout: 5000 });
      console.log('Status response:', response.data);
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  const fetchFrame = async (cameraId) => {
    try {
      const response = await axios.get(`${API}/cameras/${cameraId}/frame`, { timeout: 1000 });
      if (response.data) {
        setFrames(prev => ({ ...prev, [cameraId]: response.data }));
        
        // Calculate FPS
        const now = Date.now() / 1000; // seconds
        setFps(prev => {
          const timestamps = prev[cameraId]?.timestamps || [];
          timestamps.push(now);
          if (timestamps.length > 5) timestamps.shift();
          
          let currentFps = 0;
          if (timestamps.length > 1) {
            const timeDiff = timestamps[timestamps.length - 1] - timestamps[0];
            currentFps = (timestamps.length - 1) / timeDiff;
          }
          
          return { ...prev, [cameraId]: { fps: currentFps, timestamps } };
        });
      }
    } catch (error) {
      // Ignore errors for frame fetching
    }
  };

  const fetchCameraSettings = async (cameraId) => {
    try {
      const response = await axios.get(`${API}/cameras/${cameraId}/settings`);
      setCurrentSettings(prev => ({ ...prev, [cameraId]: response.data }));
    } catch (error) {
      // Settings may not be available
    }
  };

  const connectCamera = async (cameraId) => {
    try {
      const response = await axios.post(`${API}/cameras/${cameraId}/connect`);
      toast.success(response.data.message);
      await fetchCameras(); // Refresh camera list
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message;
      toast.error(`Failed to connect camera: ${errorMsg}`);
    }
  };

  const disconnectCamera = async (cameraId) => {
    try {
      const response = await axios.post(`${API}/cameras/${cameraId}/disconnect`);
      toast.success(response.data.message);
      await fetchCameras();
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message;
      toast.error(`Failed to disconnect camera: ${errorMsg}`);
    }
  };

  const startStreaming = async (cameraId) => {
    try {
      const response = await axios.post(`${API}/cameras/${cameraId}/start-streaming`);
      toast.success(response.data.message);
      await fetchCameras();
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message;
      toast.error(`Failed to start streaming: ${errorMsg}`);
    }
  };

  const stopStreaming = async (cameraId) => {
    try {
      const response = await axios.post(`${API}/cameras/${cameraId}/stop-streaming`);
      toast.success(response.data.message);
      await fetchCameras();
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message;
      toast.error(`Failed to stop streaming: ${errorMsg}`);
    }
  };

  const openConfigDialog = (camera) => {
    setConfigDialog({ open: true, camera });
    // Reset values
    setConfigValues({
      exposure_time: '',
      gain: '',
      frame_rate: '',
      width: '',
      height: ''
    });
  };

  const configureCamera = async () => {
    if (!configDialog.camera) return;

    const config = {};
    const exp = parseFloat(configValues.exposure_time);
    if (configValues.exposure_time && !isNaN(exp)) config.exposure_time = exp;
    
    const g = parseFloat(configValues.gain);
    if (configValues.gain && !isNaN(g)) config.gain = g;
    
    const fps = parseFloat(configValues.frame_rate);
    if (configValues.frame_rate && !isNaN(fps)) config.frame_rate = fps;
    
    const w = parseInt(configValues.width);
    if (configValues.width && !isNaN(w)) config.width = w;
    
    const h = parseInt(configValues.height);
    if (configValues.height && !isNaN(h)) config.height = h;

    if (Object.keys(config).length === 0) {
      toast.error('Please enter at least one valid setting');
      return;
    }

    try {
      const response = await axios.post(`${API}/cameras/${configDialog.camera.id}/configure`, config);
      toast.success(response.data.message);
      setConfigDialog({ open: false, camera: null });
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message;
      toast.error(`Failed to configure camera: ${errorMsg}`);
    }
  };

  const handleRefresh = async () => {
    await Promise.all([fetchCameras(), fetchSystemStatus()]);
  };
  const camerasRef = useRef(cameras);
  camerasRef.current = cameras;

  useEffect(() => {
    // Initialize on load
    handleRefresh();
    
    // Set up periodic status updates and frame fetching
    const statusInterval = setInterval(async () => {
      await fetchSystemStatus();
      // Fetch frames for streaming cameras and settings for connected cameras
      camerasRef.current.forEach(camera => {
        if (camera.status === 'streaming') {
          fetchFrame(camera.id);
        }
        if (camera.status in ['connected', 'streaming']) {
          fetchCameraSettings(camera.id);
        }
      });
    }, 2000); // Update every 2 seconds
    
    return () => clearInterval(statusInterval);
  }, []); // Remove cameras dependency to prevent re-mounting

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100" data-testid="camera-app">
      <div className="container mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Basler Camera Application</h1>
              <p className="text-lg text-gray-600">Multi-camera streaming and capture system</p>
            </div>
            <Button 
              onClick={handleRefresh} 
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700"
              data-testid="refresh-cameras-btn"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Loading...' : 'Refresh Cameras'}
            </Button>
          </div>
          
          {/* System Status */}
          {systemStatus && (
            <Card className="bg-white/80 backdrop-blur">
              <CardContent className="p-4">
                <div className="grid grid-cols-4 gap-4 text-center">
                  <div>
                    <p className="text-2xl font-bold text-blue-600">{systemStatus.total_cameras}</p>
                    <p className="text-sm text-gray-600">Total Cameras</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-green-600">{systemStatus.connected_cameras}</p>
                    <p className="text-sm text-gray-600">Connected</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-purple-600">{systemStatus.streaming_cameras}</p>
                    <p className="text-sm text-gray-600">Streaming</p>
                  </div>
                  <div>
                    <div className="flex items-center justify-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${systemStatus.camera_support ? 'bg-green-500' : 'bg-red-500'}`} />
                      <p className="text-sm text-gray-600">Camera Support</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Camera List */}
        <div className="space-y-6">
          {cameras.length === 0 ? (
            <Card className="text-center py-12">
              <CardContent>
                <AlertCircle className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Cameras Detected</h3>
                <p className="text-gray-600 mb-4">
                  Make sure your Basler cameras are connected and powered on.
                </p>
                <p className="text-sm text-gray-500 mb-4">
                  Backend URL: {BACKEND_URL}
                </p>
                <Button onClick={handleRefresh} disabled={loading} data-testid="refresh-no-cameras-btn">
                  <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                  Refresh Camera List
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-6">
              {cameras.map((camera) => (
                <Card key={camera.id} className="w-full" data-testid={`camera-card-${camera.id}`}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <Camera className="h-5 w-5" />
                          {camera.name}
                        </CardTitle>
                        <CardDescription>
                          {camera.model} • {camera.serial} • {camera.type}
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={`${getStatusColor(camera.status)} text-white`}>
                          {camera.status}
                        </Badge>
                        {camera.type === 'tof' && (
                          <Badge variant="secondary">ToF</Badge>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {frames[camera.id] ? (
                      <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                        <img 
                          src={`data:image/jpeg;base64,${frames[camera.id].frame_data}`} 
                          alt={`Camera ${camera.name} stream`}
                          className="w-full h-full object-contain"
                        />
                      </div>
                    ) : (
                      <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                        <div className="text-center text-gray-500">
                          <Camera className="h-12 w-12 mx-auto mb-2 opacity-30" />
                          <p>{camera.status === 'streaming' ? 'Loading stream...' : 'Camera Ready - Connect to Start Streaming'}</p>
                        </div>
                      </div>
                    )}
                    <div className="flex gap-2 mt-4">
                      {camera.status === 'disconnected' && (
                        <Button 
                          onClick={() => connectCamera(camera.id)} 
                          className="bg-green-600 hover:bg-green-700"
                        >
                          Connect
                        </Button>
                      )}
                      {camera.status === 'connected' && (
                        <>
                          <Button 
                            onClick={() => disconnectCamera(camera.id)} 
                            variant="outline"
                          >
                            Disconnect
                          </Button>
                          <Button 
                            onClick={() => startStreaming(camera.id)} 
                            className="bg-blue-600 hover:bg-blue-700"
                          >
                            <Play className="h-4 w-4 mr-2" />
                            Start Streaming
                          </Button>
                        </>
                      )}
                      {camera.status === 'streaming' && (
                        <Button 
                          onClick={() => stopStreaming(camera.id)} 
                          className="bg-red-600 hover:bg-red-700"
                        >
                          <Square className="h-4 w-4 mr-2" />
                          Stop Streaming
                        </Button>
                      )}
                      <Button 
                        onClick={() => openConfigDialog(camera)} 
                        variant="outline"
                        size="sm"
                      >
                        <Settings className="h-4 w-4" />
                      </Button>
                    </div>
                    {/* Camera Info */}
                    {currentSettings[camera.id] && (
                      <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm">
                        <div className="grid grid-cols-2 gap-2">
                          {currentSettings[camera.id].exposure_time !== undefined && (
                            <div>Exposure: {currentSettings[camera.id].exposure_time.toFixed(1)} μs</div>
                          )}
                          {currentSettings[camera.id].gain !== undefined && (
                            <div>Gain: {currentSettings[camera.id].gain.toFixed(1)}</div>
                          )}
                          {currentSettings[camera.id].frame_rate !== undefined && (
                            <div>FPS: {currentSettings[camera.id].frame_rate.toFixed(1)}</div>
                          )}
                          {fps[camera.id] && (
                            <div className="text-green-600 font-semibold">
                              Live FPS: {fps[camera.id].fps.toFixed(1)}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
      
      <Toaster position="top-right" />

      {/* Camera Configuration Dialog */}
      <Dialog open={configDialog.open} onOpenChange={(open) => setConfigDialog({ open, camera: open ? configDialog.camera : null })}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Configure Camera</DialogTitle>
            <DialogDescription>
              Adjust camera settings for {configDialog.camera?.name}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="exposure" className="text-right">
                Exposure (μs)
              </Label>
              <Input
                id="exposure"
                type="number"
                placeholder="e.g. 10000"
                value={configValues.exposure_time}
                onChange={(e) => setConfigValues({...configValues, exposure_time: e.target.value})}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="gain" className="text-right">
                Gain
              </Label>
              <Input
                id="gain"
                type="number"
                step="0.1"
                placeholder="e.g. 1.0"
                value={configValues.gain}
                onChange={(e) => setConfigValues({...configValues, gain: e.target.value})}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="fps" className="text-right">
                FPS
              </Label>
              <Input
                id="fps"
                type="number"
                step="0.1"
                placeholder="e.g. 30"
                value={configValues.frame_rate}
                onChange={(e) => setConfigValues({...configValues, frame_rate: e.target.value})}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="width" className="text-right">
                Width
              </Label>
              <Input
                id="width"
                type="number"
                placeholder="e.g. 640"
                value={configValues.width}
                onChange={(e) => setConfigValues({...configValues, width: e.target.value})}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="height" className="text-right">
                Height
              </Label>
              <Input
                id="height"
                type="number"
                placeholder="e.g. 480"
                value={configValues.height}
                onChange={(e) => setConfigValues({...configValues, height: e.target.value})}
                className="col-span-3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button onClick={configureCamera}>Apply Settings</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

// Helper function for status colors
const getStatusColor = (status) => {
  switch (status) {
    case 'connected': return 'bg-yellow-500';
    case 'streaming': return 'bg-green-500';
    case 'disconnected': return 'bg-gray-500';
    default: return 'bg-red-500';
  }
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SimpleCameraApp />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;