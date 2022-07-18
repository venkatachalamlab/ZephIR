import { AnnotationWindowState_t as State_t, rpc_t } from './model'
import { createSlice, PayloadAction } from "@reduxjs/toolkit"
import { annotation_id_t, Annotation_t } from '../annotations/model'
import { worldline_id_t } from '../worldlines/model';

export const initialState: State_t = {
  x: 0.5,
  y: 0.5,
  z: 0.5,
  c_idx: 0,
  t_idx: 0,
  shape_x: 0,
  shape_y: 0,
  shape_z: 0,
  views: ["slice", "mip", "vol"],
  shape_t: 0,
  scale_x: 1,
  scale_y: 1,
  scale_z: 1,
  LUT: {
    red: [0, 150],
    green: [0, 150],
    blue: [0, 150]
  },
  gamma: 1.0,
  selected_annotation: null,
  visible_worldlines: [],
  selected_worldline: null,
  active_provenance: "ANTT",
  fill_circles: true,
  show_all_annotations: false,
  rpc_list: [{ name: "None found", default_args: "", docstring: "" }],
  rpc_keybindings: {
    "1": 0,
    "2": 2,
    "3": 5,
    "4": 5,
    "5": 7,
    "6": 8,
    "7": 9,
    "8": 0,
    "9": 6,
    "0": 3,
  },
  rpc_args: {
    "1": "_this, _now, *",
    "2": "",
    "3": "0",
    "4": "1",
    "5": "_this, _now, *, _ACTIVE",
    "6": "_now, True, False, False",
    "7": "key, value",
    "8": "",
    "9": "",
    "0": "",
  },
};

const threshold = (val: number, min: number, max: number): number =>
  Math.max(Math.min(val, max), 0)

const reducers = {

  set_state: (_state: State_t, action: PayloadAction<State_t>) =>
    action.payload,

  set_x: (state: State_t, action: PayloadAction<number>) => {
    state.x = threshold(action.payload, 0, 1)
  },
  adjust_x: (state: State_t, action: PayloadAction<number>) => {
    state.x = threshold(state.x + action.payload, 0, 1)
  },

  set_y: (state: State_t, action: PayloadAction<number>) => {
    state.y = threshold(action.payload, 0, 1)
  },
  adjust_y: (state: State_t, action: PayloadAction<number>) => {
    state.y = threshold(state.y + action.payload, 0, 1)
  },

  set_z: (state: State_t, action: PayloadAction<number>) => {
    state.z = threshold(action.payload, 0, 1)
  },
  adjust_z: (state: State_t, action: PayloadAction<number>) => {
    state.z = threshold(state.z + action.payload, 0, 1)
  },

  set_c_idx: (state: State_t, action: PayloadAction<number>) => {
    state.c_idx = action.payload
  },

  adjust_c_idx: (state: State_t, action: PayloadAction<number>) => {
    const idx = state.c_idx + action.payload
    const n = state.views.length
    state.c_idx = ((idx % n) + n) % n
  },

  set_t_idx: (state: State_t, action: PayloadAction<number>) => {
    state.t_idx = threshold(action.payload, 0, state.shape_t - 1)
  },

  set_scale_x: (state: State_t, action: PayloadAction<number>) => {
    state.scale_x = action.payload;
  },
  set_scale_y: (state: State_t, action: PayloadAction<number>) => {
    state.scale_y = action.payload;
  },
  set_scale_z: (state: State_t, action: PayloadAction<number>) => {
    state.scale_z = action.payload;
  },

  set_shape_x: (state: State_t, action: PayloadAction<number>) => {
    state.shape_x = action.payload;
  },
  set_shape_y: (state: State_t, action: PayloadAction<number>) => {
    state.shape_y = action.payload;
  },
  set_shape_z: (state: State_t, action: PayloadAction<number>) => {
    state.shape_z = action.payload;
  },

  set_LUT_color: (state: State_t,
    action: PayloadAction<["red" | "green" | "blue", number, number]>) => {
    state.LUT[action.payload[0]] = [action.payload[1], action.payload[2]]
  },

  set_gamma: (state: State_t, action: PayloadAction<number>) => {
    state.gamma = Math.max(action.payload, 0)
  },

  set_selected_annotation_local: (
    state: State_t, action: PayloadAction<annotation_id_t | null>) => {
    state.selected_annotation = action.payload
  },

  set_selected_worldline_local: (
    state: State_t, action: PayloadAction<worldline_id_t | null>) => {
    state.selected_worldline = action.payload
  },

  set_active_provenance: (
    state: State_t, action: PayloadAction<string>) => {
    state.active_provenance = action.payload
  },

  set_fill_circles: (
    state: State_t, action: PayloadAction<boolean>) => {
    state.fill_circles = action.payload
  },

  toggle_fill_circles: (
    state: State_t, action: PayloadAction) => {
    state.fill_circles = !state.fill_circles
  },

  set_show_all_annotations: (
    state: State_t, action: PayloadAction<boolean>) => {
    state.show_all_annotations = action.payload
  },

  toggle_all_annotations: (
    state: State_t, action: PayloadAction) => {
    state.show_all_annotations = !state.show_all_annotations
  },

  set_rpc_list: (
    state: State_t, action: PayloadAction<rpc_t[]>) => {
    state.rpc_list = action.payload
  },

  set_rpc_keybinding: (
    state: State_t, action: PayloadAction<[string, number]>) => {
    state.rpc_keybindings[action.payload[0]] = action.payload[1]
  },

  set_rpc_args: (
    state: State_t, action: PayloadAction<{ [key: string]: string }>) => {
    state.rpc_args = action.payload
  },

  set_rpc_arg: (
    state: State_t, action: PayloadAction<[string, string]>) => {
    state.rpc_args[action.payload[0]] = action.payload[1]
  },

  center_on_annotation: (
    state: State_t, action: PayloadAction<Annotation_t>) => {
    state.x = Number(action.payload.x)
    state.y = Number(action.payload.y)
    state.z = Number(action.payload.z)
  },
}

const annotation_window_slice = createSlice({
  name: 'annotation_window',
  initialState,
  reducers
})

export const { actions, reducer } = annotation_window_slice
