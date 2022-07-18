import { worldline_id_t } from "../worldlines/model";

export type annotation_id_t = number;

export type Annotation_t = {
  id: annotation_id_t,
  x: number, // 0 : 1 float32
  y: number, // 0 : 1 float32
  z: number, // 0 : 1 float32
  t_idx: number, // 0 : (shape_t-1) uint32
  worldline_id: worldline_id_t // uint32
  parent_id: annotation_id_t, // uint32
  provenance: string // 4 ASCII bytes
};

export type AnnotationsState_t = {
  [key: number]: Annotation_t,
};