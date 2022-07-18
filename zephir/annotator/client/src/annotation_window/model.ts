import type { worldline_id_t } from '../worldlines/model'
import { annotation_id_t } from '../annotations/model'

export type rpc_t = {
  name: string,
  default_args: string,
  docstring: string,
}

export type LUT_t = {
  red: [number, number],
  green: [number, number],
  blue: [number, number]
}

export type AnnotationWindowState_t = {
  x: number,
  y: number,
  z: number,
  c_idx: number,
  t_idx: number,
  shape_x: number,
  shape_y: number,
  shape_z: number,
  views: string[],
  shape_t: number,
  scale_x: number,
  scale_y: number,
  scale_z: number,
  LUT: LUT_t,
  gamma: number,
  selected_annotation: annotation_id_t | null,
  visible_worldlines: worldline_id_t[],
  selected_worldline: worldline_id_t | null,
  active_provenance: string,
  fill_circles: boolean,
  show_all_annotations: boolean,
  rpc_list: rpc_t[],
  rpc_keybindings: {[key: string]: number}
  rpc_args: {[key: string]: string}
}