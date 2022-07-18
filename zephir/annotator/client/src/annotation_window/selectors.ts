import { createSelector } from 'reselect'

import * as app_selectors from '../app/selectors'
import { idx_from_coord } from './helpers'
import { State_t } from '../app/model'

export const get_state = app_selectors.get_annotation_window_state

// Top level selectors
export const get_x = createSelector(get_state, s => s.x)
export const get_y = createSelector(get_state, s => s.y)
export const get_z = createSelector(get_state, s => s.z)
export const get_c_idx = createSelector(get_state, s => s.c_idx)
export const get_t_idx = createSelector(get_state, s => s.t_idx)
export const get_shape_x = createSelector(get_state, s => s.shape_x)
export const get_shape_y = createSelector(get_state, s => s.shape_y)
export const get_shape_z = createSelector(get_state, s => s.shape_z)
export const get_shape_t = createSelector(get_state, s => s.shape_t)
export const get_views = createSelector(get_state, s => s.views)
export const get_scale_x = createSelector(get_state, s => s.scale_x)
export const get_scale_y = createSelector(get_state, s => s.scale_y)
export const get_scale_z = createSelector(get_state, s => s.scale_z)
export const get_LUT = createSelector(get_state, s => s.LUT)
export const get_gamma = createSelector(get_state, s => s.gamma)
export const get_selected_annotation = createSelector(get_state,
  s => s.selected_annotation)
export const get_visible_worldlines = createSelector(get_state,
  s => s.visible_worldlines)
export const get_selected_worldline = createSelector(get_state,
  s => s.selected_worldline)
export const get_active_provenance = createSelector(get_state,
  s => s.active_provenance)
export const get_fill_circles = createSelector(get_state, s => s.fill_circles)
export const get_show_all_annotations = createSelector(get_state,
  s => s.show_all_annotations)
export const get_rpc_list = createSelector(get_state,
  s => s.rpc_list)
export const get_rpc_keybindings = createSelector(get_state,
  s => s.rpc_keybindings)

// Convert coordinate in (0, 1) to index in [0, shape-1]
export const get_x_idx = createSelector([get_x, get_shape_x], idx_from_coord)
export const get_y_idx = createSelector([get_y, get_shape_y], idx_from_coord)
export const get_z_idx = createSelector([get_z, get_shape_z], idx_from_coord)

// Get view information

export const get_view = createSelector(
  get_views,
  get_c_idx,
  (views, c_idx) => views[c_idx]
)

const get_dim_viewname = (dim: 'x' | 'y' | 'z') => createSelector(
  get_view,
  (view) => {
    switch (view) {
      case "slice": return dim
      case "mip": return `mip_${dim}`
      default: return dim
    }
  }
)

const get_x_viewname = get_dim_viewname('x')
const get_y_viewname = get_dim_viewname('y')
const get_z_viewname = get_dim_viewname('z')

const get_dim_viewidx = (dim: 'x' | 'y' | 'z',
  bare_getter: (s: State_t) => number) => createSelector(
    get_view,
    bare_getter,
    (view, idx) => {
      switch (view) {
        case "slice": return idx
        case "mip": return 0
        default: return idx
      }
    }
)

const get_x_viewidx = get_dim_viewidx('x', get_x_idx)
const get_y_viewidx = get_dim_viewidx('y', get_y_idx)
const get_z_viewidx = get_dim_viewidx('z', get_z_idx)

// Get image urls (fetching has to be done in components)

const make_data_url = (t: number, view: string, x: number) =>
  `/t/${t}/data/${view}/${x}`

export const get_x_url = createSelector(
  get_t_idx,
  get_x_viewname,
  get_x_viewidx,
  make_data_url
)
export const get_y_url = createSelector(
  get_t_idx,
  get_y_viewname,
  get_y_viewidx,
  make_data_url
)
export const get_z_url = createSelector(
  get_t_idx,
  get_z_viewname,
  get_z_viewidx,
  make_data_url
)

export const get_vol_url = createSelector(
  get_t_idx,
  (t) => make_data_url(t, "gvol", 1)
)