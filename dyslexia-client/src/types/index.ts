export interface GazePoint {
  t: number
  x: number
  y: number
  x_l: number
  y_l: number
  x_r: number
  y_r: number
  yaw_l: number
  pitch_l: number
  yaw_r: number
  pitch_r: number
}

export interface SessionData {
  gazePoints: GazePoint[]
}

export interface CalibrationData {
  leftSphereOffset: number[]   // offset local iris gauche
  rightSphereOffset: number[]  // offset local iris droit
  leftCalibScale: number
  rightCalibScale: number
  yawOffset: number
  pitchOffset: number
}

export type TestStep = 'camera' | 'calibration' | 'countdown' | 'reading' | 'prediction'
