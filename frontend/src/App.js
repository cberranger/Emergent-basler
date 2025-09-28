import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import "@/App.css";

// Import UI components
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { RefreshCw, Camera, AlertCircle } from 'lucide-react';
import { toast, Toaster } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Simple Camera Application
const SimpleCameraApp = () => {
  const [cameras, setCameras] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(false);

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

  const handleRefresh = async () => {
    await Promise.all([fetchCameras(), fetchSystemStatus()]);
  };

  useEffect(() => {
    // Initialize on load
    handleRefresh();
    
    // Set up periodic status updates
    const statusInterval = setInterval(fetchSystemStatus, 10000);
    return () => clearInterval(statusInterval);
  }, []);

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
                    <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                      <div className="text-center text-gray-500">
                        <Camera className="h-12 w-12 mx-auto mb-2 opacity-30" />
                        <p>Camera Ready - Connect to Start Streaming</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
      
      <Toaster position="top-right" />
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