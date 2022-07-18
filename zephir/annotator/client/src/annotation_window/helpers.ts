// Convert x/y/z to index values using shape. The convention is that images
// go from 0.0 to 1.0, so the center values of each pixel are in (0, 1)

import { LUT_t } from "./model"

export const idx_from_coord = (coord: number, shape: number): number => {
  return Math.max(Math.floor(coord * shape - 1e-6), 0)
}
export const coord_from_idx = (idx: number, shape: number): number => {
  return (idx + 0.5) / shape
}

// Apply a LUT to a uint8 array.

const LUT_linear = (value: number, lut: [number, number], gamma: number): number =>
  255  * (((value - lut[0]) / (lut[1] - lut[0])) ** gamma)

export const apply_LUT =
  (data: Uint8ClampedArray, LUT: LUT_t, gamma: number): Uint8ClampedArray =>

    data.map((val: number, idx: number): number => {
      const offset = idx % 4
      switch (offset) {
        case 0: return LUT_linear(val, LUT.red, gamma)
        case 1: return LUT_linear(val, LUT.green, gamma)
        case 2: return LUT_linear(val, LUT.blue, gamma)
        default: return val
      }
    })

export const apply_green_LUT =
  (data: Uint8ClampedArray, LUT: LUT_t, gamma: number): Uint8ClampedArray =>

  data.map((val: number): number => LUT_linear(val, LUT.green, gamma))
