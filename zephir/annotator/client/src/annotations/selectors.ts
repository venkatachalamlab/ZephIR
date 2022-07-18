import { createSelector } from 'reselect'
import fp from 'lodash/fp'
import { Dictionary } from 'lodash'

import { annotation_id_t, Annotation_t } from './model'
import * as app_selectors from '../app/selectors'
import * as annotation_window_selectors from '../annotation_window/selectors'
import * as worldlines_selectors from '../worldlines/selectors'
import * as provenances_selectors from '../provenances/selectors'
import { Worldline_t, WorldlinesState_t } from '../worldlines/model'

export const get_state = app_selectors.get_annotations_state

export const get_annotation_ids = createSelector(
  get_state,
  fp.keys
)

export const get_used_worldline_ids = createSelector(
  get_state,
  (x) => fp.values(fp.mapValues("worldline_id")(x))
)

// Get annotations

export const get_annotations_list = createSelector(
  get_state,
  (annotations) => {
    return Object.entries(annotations).map((a) => a[1])
  }
)

export const get_visible_annotations = createSelector(
  get_state,
  worldlines_selectors.get_visible_worldlines,
  provenances_selectors.get_visible_provenances,
  (s, w, p) => fp.pickBy((x: Annotation_t) =>
    w.includes(x.worldline_id) && p.includes(x.provenance))(s)
)

export const get_selected_annotation_id = createSelector(
  get_state,
  annotation_window_selectors.get_selected_annotation,
  annotation_window_selectors.get_selected_worldline,
  (all, a, w) => {
    if (a !== null) {
      return a
    }
    else if (w !== null) {
      return Number(fp.findKey({ worldline_id: w })(all))
    } else {
      return null
    }
  }
)

export const get_selected_annotation = createSelector(
  get_state,
  get_selected_annotation_id,
  (all, a) => { return (a ? all[a] : null) }
)

// Gather information needed to draw annotations in 3D or 2D.

const get_worldline_for_annotation = (
  annotation: Annotation_t,
  worldlines: WorldlinesState_t) =>
  worldlines.worldlines[annotation.worldline_id]

const get_color = (annotation: Annotation_t, worldline: Worldline_t) => {
  return worldline.color
}

const get_annotations_with_all_info = createSelector(
  get_visible_annotations,
  worldlines_selectors.get_state,
  get_selected_annotation_id,
  (a, w, s) => fp.mapValues((x: Annotation_t) => {
    const worldline = get_worldline_for_annotation(x, w)
    return {
      ...x,
      //worldline: worldline,
      color: get_color(x, worldline),
      selected: x.id === s
    }
  }
  )(a)
)

export type AnnotationView3D_t = {
  id: annotation_id_t,
  x: number,
  y: number,
  z: number,
  color: string,
  selected: boolean
}
export const get_annotations_view3D = createSelector(
  get_annotations_with_all_info,
  fp.mapValues(fp.pick(['id', 'x', 'y', 'z', 'color', 'selected']))
)

export type AnnotationView2D_t = {
  id: annotation_id_t,
  x: number,
  y: number,
  color: string,
  selected: boolean,
  radius: number
}
const get_visible_annotations_near = (
  annotations: Dictionary<AnnotationView3D_t>,
  dim: 'x' | 'y' | 'z',
  output_x: 'x' | 'y' | 'z',
  output_y: 'x' | 'y' | 'z',
  coord: number,
  r: number
): Dictionary<AnnotationView2D_t> => {

  const is_close = (a: AnnotationView3D_t) => {
    return (r - Math.abs(a[dim] - coord)) > 0
  }

  const project_2D = (a: AnnotationView3D_t): any => {
    return {
      id: a.id,
      x: a[output_x],
      y: a[output_y],
      color: a.color,
      selected: a.selected,
      radius: 4
    }
  }

  const close_3D = fp.pickBy(is_close)(annotations)
  return fp.mapValues(project_2D)(close_3D)

}

const RADIUS_X = 0.02
const RADIUS_Y = 0.02
const RADIUS_Z = 0.07

export const get_x_annotations = createSelector(
  get_annotations_view3D,
  annotation_window_selectors.get_x,
  annotation_window_selectors.get_show_all_annotations,
  (a, x, all) => {
    if (all) {
      return get_visible_annotations_near(a, "x", "z", "y", x, 2.0)
    } else {
      return get_visible_annotations_near(a, "x", "z", "y", x, RADIUS_X)
    }
  }
)

export const get_y_annotations = createSelector(
  get_annotations_view3D,
  annotation_window_selectors.get_y,
  annotation_window_selectors.get_show_all_annotations,
  (a, y, all) => {
    if (all) {
      return get_visible_annotations_near(a, "y", "x", "z", y, 2.0)
    } else {
      return get_visible_annotations_near(a, "y", "x", "z", y, RADIUS_Y)
    }
  }
)

export const get_z_annotations = createSelector(
  get_annotations_view3D,
  annotation_window_selectors.get_z,
  annotation_window_selectors.get_show_all_annotations,
  (a, z, all) => {
    if (all) {
      return get_visible_annotations_near(a, "z", "x", "y", z, 2.0)
    } else {
      return get_visible_annotations_near(a, "z", "x", "y", z, RADIUS_Z)
    }
  }
)