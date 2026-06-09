import type { CalibrationData } from '@/types/index'

export const NOSE_INDICES = [1, 2, 4, 5, 6, 19, 20, 94, 125, 141, 235, 281, 354, 370, 460]
export const BASE_R = 20

// ── Maths de base ─────────────────────────────────────────────────────────

export function norm3(v: number[]): number[] {
  const l = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
  return l > 0 ? v.map(x => x / l) : v
}

export function dot3(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x))
}

export function centroid(pts: number[][]): number[] {
  const n = pts.length
  return [
    pts.reduce((s, p) => s + p[0], 0) / n,
    pts.reduce((s, p) => s + p[1], 0) / n,
    pts.reduce((s, p) => s + p[2], 0) / n,
  ]
}

export function computeScale(pts: number[][]): number {
  let total = 0, count = 0
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++) {
      const d = pts[i].map((v, k) => v - pts[j][k])
      total += Math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
      count++
    }
  return count > 0 ? total / count : 1
}

export function matTransposeVec(R: number[][], v: number[]): number[] {
  return [
    R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
    R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
    R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
  ]
}

// ── Pose de tête + iris ───────────────────────────────────────────────────

export function extractNosePts(lms: any[], w: number, h: number): number[][] {
  return NOSE_INDICES.map(i => [lms[i].x * w, lms[i].y * h, lms[i].z * w])
}

export function extractIris(lms: any[], w: number, h: number): [number[], number[]] {
  const il = [lms[468].x * w, lms[468].y * h, lms[468].z * w]
  const ir = [lms[473].x * w, lms[473].y * h, lms[473].z * w]
  return [il, ir]
}

/** Matrice identité — suffisant pour le calcul d'offset local */
export function computeHeadRotation(_nosePts: number[][]): number[][] {
  return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
}

// ── Gaze ─────────────────────────────────────────────────────────────────

export function gazeToAngles(d: number[]): [number, number] {
  d = norm3(d)
  const xz = norm3([d[0], 0, d[2]])
  let yaw = Math.acos(clamp(dot3([0, 0, -1], xz), -1, 1))
  if (d[0] < 0) yaw = -yaw
  const yz = norm3([0, d[1], d[2]])
  let pitch = Math.acos(clamp(dot3([0, 0, -1], yz), -1, 1))
  if (d[1] > 0) pitch = -pitch
  return [-yaw * (180 / Math.PI), pitch * (180 / Math.PI)]
}

/** Calcule les positions monde des sphères oculaires.
 *  Avec calibration : utilise les offsets lockés.
 *  Sans calibration : fallback auto-init. */
export function computeEyeSpheres(
  hc: number[],
  il: number[],
  ir: number[],
  curScale: number,
  calib: CalibrationData | null,
): [number[], number[]] {
  if (calib) {
    const scaleL = curScale / calib.leftCalibScale
    const scaleR = curScale / calib.rightCalibScale
    const sl = hc.map((v, k) => v + calib.leftSphereOffset[k] * scaleL)
    const sr = hc.map((v, k) => v + calib.rightSphereOffset[k] * scaleR)
    return [sl, sr]
  }
  // Fallback sans calibration (auto-init comme le Python)
  const forward = [0, 0, 1]
  const sl = il.map((v, k) => v - hc[k] + BASE_R * forward[k]).map((v, k) => hc[k] + v)
  const sr = ir.map((v, k) => v - hc[k] + BASE_R * forward[k]).map((v, k) => hc[k] + v)
  return [sl, sr]
}

/** Calcule yaw/pitch pour les deux yeux avec offset de calibration appliqué. */
export function computeGazeAngles(
  il: number[], ir: number[],
  sl: number[], sr: number[],
  calib: CalibrationData | null,
): { yaw_l: number; pitch_l: number; yaw_r: number; pitch_r: number } {
  const dl = norm3(il.map((v, k) => v - sl[k]))
  const dr = norm3(ir.map((v, k) => v - sr[k]))

  let [yaw_l, pitch_l] = gazeToAngles(dl)
  let [yaw_r, pitch_r] = gazeToAngles(dr)

  if (calib) {
    yaw_l   += calib.yawOffset
    pitch_l += calib.pitchOffset
    yaw_r   += calib.yawOffset
    pitch_r += calib.pitchOffset
  }

  return { yaw_l, pitch_l, yaw_r, pitch_r }
}
