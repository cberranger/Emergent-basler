import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import "@/App.css";

// Import UI components
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Switch } from './components/ui/switch';
import { Separator } from './components/ui/separator';
import { AlertCircle, Camera, Play, Square, Settings, Save, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';
import { Toaster } from './components/ui/sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Camera Stream Component
const CameraStream = ({ camera, onConfigChange }) => {
  const [frame, setFrame] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [config, setConfig] = useState({
    exposure_time: 1000,
    gain: 1.0,
    frame_rate: 10
  });

  // Poll for frames when streaming
  useEffect(() => {
    let interval;
    if (isStreaming) {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API}/cameras/${camera.id}/frame`);
          setFrame(response.data);
        } catch (error) {
          // Silently handle frame fetch errors
        }
      }, 100); // 10 FPS polling
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isStreaming, camera.id]);

  const handleConnect = async () => {
    try {
      await axios.post(`${API}/cameras/${camera.id}/connect`);
      toast.success(`Connected to ${camera.name}`);
      onConfigChange();
    } catch (error) {
      toast.error(`Failed to connect: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleDisconnect = async () => {
    try {
      setIsStreaming(false);
      await axios.post(`${API}/cameras/${camera.id}/disconnect`);
      toast.success(`Disconnected ${camera.name}`);
      onConfigChange();
    } catch (error) {
      toast.error(`Failed to disconnect: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleStartStreaming = async () => {
    try {
      await axios.post(`${API}/cameras/${camera.id}/start-streaming`);
      setIsStreaming(true);
      toast.success(`Started streaming ${camera.name}`);
      onConfigChange();
    } catch (error) {
      toast.error(`Failed to start streaming: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleStopStreaming = async () => {
    try {
      await axios.post(`${API}/cameras/${camera.id}/stop-streaming`);
      setIsStreaming(false);
      setFrame(null);
      toast.success(`Stopped streaming ${camera.name}`);
      onConfigChange();
    } catch (error) {
      toast.error(`Failed to stop streaming: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleConfigure = async () => {
    try {
      await axios.post(`${API}/cameras/${camera.id}/configure`, {
        camera_id: camera.id,
        ...config
      });
      toast.success(`Configured ${camera.name}`);
    } catch (error) {
      toast.error(`Failed to configure: ${error.response?.data?.detail || error.message}`);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected': return 'bg-yellow-500';
      case 'streaming': return 'bg-green-500';
      case 'disconnected': return 'bg-gray-500';
      default: return 'bg-red-500';
    }
  };

  return (
    <Card className="w-full" data-testid={`camera-card-${camera.id}`}>
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
      
      <CardContent className="space-y-4">
        {/* Stream Display */}
        <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center overflow-hidden">
          {frame ? (
            <div className="relative w-full h-full">
              <img 
                src={`data:image/jpeg;base64,${frame.frame_data}`}
                alt={`Stream from ${camera.name}`}
                className="w-full h-full object-contain"
              />
              <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-xs">
                Frame: {frame.frame_number} | {new Date(frame.timestamp * 1000).toLocaleTimeString()}
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500">
              <Camera className="h-12 w-12 mx-auto mb-2 opacity-30" />
              <p>No stream available</p>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex gap-2 flex-wrap">
          {camera.status === 'disconnected' && (
            <Button onClick={handleConnect} data-testid={`connect-btn-${camera.id}`}>
              Connect
            </Button>
          )}
          
          {camera.status === 'connected' && (
            <>
              <Button onClick={handleStartStreaming} className="bg-green-600 hover:bg-green-700" data-testid={`start-stream-btn-${camera.id}`}>
                <Play className="h-4 w-4 mr-1" />
                Start Stream
              </Button>
              <Button onClick={handleDisconnect} variant="outline" data-testid={`disconnect-btn-${camera.id}`}>
                Disconnect
              </Button>
            </>
          )}
          
          {camera.status === 'streaming' && (
            <>
              <Button onClick={handleStopStreaming} variant="destructive" data-testid={`stop-stream-btn-${camera.id}`}>
                <Square className="h-4 w-4 mr-1" />
                Stop Stream
              </Button>
              <Button onClick={handleDisconnect} variant="outline" data-testid={`disconnect-btn-${camera.id}`}>
                Disconnect
              </Button>
            </>
          )}
        </div>

        {/* Configuration Panel */}
        {camera.status !== 'disconnected' && (
          <div className="border rounded-lg p-4 space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Camera Configuration
            </h4>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <Label htmlFor={`exposure-${camera.id}`}>Exposure (μs)</Label>
                <Input
                  id={`exposure-${camera.id}`}
                  type="number"
                  value={config.exposure_time}
                  onChange={(e) => setConfig({...config, exposure_time: parseFloat(e.target.value)})}
                  min="10"
                  max="100000"
                />
              </div>
              <div>
                <Label htmlFor={`gain-${camera.id}`}>Gain</Label>
                <Input
                  id={`gain-${camera.id}`}
                  type="number"
                  value={config.gain}
                  onChange={(e) => setConfig({...config, gain: parseFloat(e.target.value)})}
                  min="0"
                  max="20"
                  step="0.1"
                />
              </div>
              <div>
                <Label htmlFor={`framerate-${camera.id}`}>Frame Rate (fps)</Label>
                <Input
                  id={`framerate-${camera.id}`}
                  type="number"
                  value={config.frame_rate}
                  onChange={(e) => setConfig({...config, frame_rate: parseFloat(e.target.value)})}
                  min="1"
                  max="100"
                />
              </div>
            </div>
            <Button onClick={handleConfigure} size="sm" data-testid={`configure-btn-${camera.id}`}>
              Apply Configuration
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Save Configuration Component
const SaveConfiguration = () => {
  const [saveConfig, setSaveConfig] = useState({
    enabled: false,
    base_directory: 'C:\\CameraCaptures',
    create_timestamp_folder: true,
    save_format: 'png'
  });

  const handleSaveConfig = async () => {
    try {
      await axios.post(`${API}/save-config`, saveConfig);
      toast.success('Save configuration updated');
    } catch (error) {
      toast.error(`Failed to update save config: ${error.response?.data?.detail || error.message}`);
    }
  };

  return (
    <Card data-testid="save-config-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Save className="h-5 w-5" />
          Save Configuration
        </CardTitle>
        <CardDescription>
          Configure automatic frame saving for all cameras
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center space-x-2">
          <Switch
            id="save-enabled"
            checked={saveConfig.enabled}
            onCheckedChange={(enabled) => setSaveConfig({...saveConfig, enabled})}
            data-testid="save-enabled-switch"
          />
          <Label htmlFor="save-enabled">Enable automatic saving</Label>
        </div>
        
        {saveConfig.enabled && (
          <>
            <div>
              <Label htmlFor="base-directory">Base Directory</Label>
              <Input
                id="base-directory"
                value={saveConfig.base_directory}
                onChange={(e) => setSaveConfig({...saveConfig, base_directory: e.target.value})}
                placeholder="C:\CameraCaptures"
                data-testid="base-directory-input"
              />
            </div>
            
            <div className="flex items-center space-x-2">
              <Switch
                id="timestamp-folder"
                checked={saveConfig.create_timestamp_folder}
                onCheckedChange={(create_timestamp_folder) => 
                  setSaveConfig({...saveConfig, create_timestamp_folder})
                }
              />
              <Label htmlFor="timestamp-folder">Create timestamp folders</Label>
            </div>
            
            <div>
              <Label htmlFor="save-format">Save Format</Label>
              <select
                id="save-format"
                value={saveConfig.save_format}
                onChange={(e) => setSaveConfig({...saveConfig, save_format: e.target.value})}
                className="w-full p-2 border rounded-md"
                data-testid="save-format-select"
              >
                <option value="png">PNG</option>
                <option value="jpg">JPG</option>
                <option value="tiff">TIFF</option>
              </select>
            </div>
          </>
        )}
        
        <Button onClick={handleSaveConfig} data-testid="update-save-config-btn">
          Update Save Configuration
        </Button>
      </CardContent>
    </Card>
  );
};

// Main App Component
const CameraApp = () => {
  const [cameras, setCameras] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchCameras = async () => {
    try {
      const response = await axios.get(`${API}/cameras`);
      setCameras(response.data);
    } catch (error) {
      toast.error(`Failed to fetch cameras: ${error.response?.data?.detail || error.message}`);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get(`${API}/status`);
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchCameras(), fetchSystemStatus()]);
    setRefreshing(false);
    toast.success('Camera list refreshed');
  };

  useEffect(() => {
    const initializeApp = async () => {
      await Promise.all([fetchCameras(), fetchSystemStatus()]);
      setLoading(false);
    };
    initializeApp();

    // Periodic status update
    const statusInterval = setInterval(fetchSystemStatus, 5000);
    return () => clearInterval(statusInterval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-lg font-medium text-gray-700">Initializing Basler Camera Application...</p>
        </div>
      </div>
    );
  }

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
              disabled={refreshing}
              className="bg-blue-600 hover:bg-blue-700"
              data-testid="refresh-cameras-btn"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh Cameras
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

        {/* Main Content */}
        <Tabs defaultValue="cameras" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="cameras">Camera Streams</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>
          
          <TabsContent value="cameras" className="space-y-6">
            {cameras.length === 0 ? (
              <Card className="text-center py-12">
                <CardContent>
                  <AlertCircle className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No Cameras Detected</h3>
                  <p className="text-gray-600 mb-4">
                    Make sure your Basler cameras are connected and powered on.
                  </p>
                  <Button onClick={handleRefresh} data-testid="refresh-no-cameras-btn">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh Camera List
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-6">
                {cameras.map((camera) => (
                  <CameraStream 
                    key={camera.id} 
                    camera={camera} 
                    onConfigChange={() => {
                      fetchCameras();
                      fetchSystemStatus();
                    }}
                  />
                ))}
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="settings" className="space-y-6">
            <SaveConfiguration />
          </TabsContent>
        </Tabs>
      </div>
      
      <Toaster position="top-right" />
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<CameraApp />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
