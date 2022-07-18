import { call, put, takeEvery, takeLatest, select }
  from 'redux-saga/effects'
import { createAction, PayloadAction } from '@reduxjs/toolkit'
import { actions } from './actions'
import { actions as annotation_window_actions ,
  saga_actions as annotation_window_saga_actions }
  from '../annotation_window'
import {saga_actions as provenances_saga_actions} from '../provenances'
import * as annotation_window_selectors from '../annotation_window/selectors'
import * as annotations_selectors from '../annotations/selectors'
import { State_t as AppState_t } from '../app/model'

import * as api from '../api'
import { Annotation_t } from './model'

const prefix = "annotations/"
const saga_action_types = {
  insert_annotation: prefix + "insert_annotation",
  insert_annotation_here: prefix + "insert_annotation_here",
  get_annotations: prefix + "get_annotations",
  update_annotation: prefix + "update_annotation",
  delete_annotation: prefix + "delete_annotation",
}

export const saga_actions = {

  insert_annotation: createAction<api.bare_annotation_t>(
    saga_action_types.insert_annotation),

  insert_annotation_here: createAction(
    saga_action_types.insert_annotation_here),

  get_annotations: createAction(saga_action_types.get_annotations),

  update_annotation: createAction<Annotation_t>(
    saga_action_types.update_annotation),

  delete_annotation: createAction(saga_action_types.delete_annotation)

}

function* insert_annotation(action: PayloadAction<api.bare_annotation_t>) {

  const a = yield call(api.insert_annotation, action.payload)
  yield put(actions.set_annotation_local(a))
  yield put(annotation_window_saga_actions.set_selected_annotation(a.id))

}

function* insert_annotation_here() {

  const state: AppState_t = yield select()

  const { x, y, z, t_idx, selected_worldline, active_provenance } =
    state.annotation_window

  let worldline_id = selected_worldline
  if (worldline_id === null) {
    worldline_id = 0
  }

  const bare = { x, y, z, t_idx, worldline_id, provenance: active_provenance }

  yield put(saga_actions.insert_annotation(bare))

}

let abort_controller = new AbortController()
function* get_annotations(action:PayloadAction<number>) {

  const t_idx = yield select(annotation_window_selectors.get_t_idx)

  const cached_annotations = yield call(api.fetch_annotations_cache, t_idx)
  if (cached_annotations !== undefined) {
    yield put(actions.set_state(cached_annotations))
  }

  abort_controller.abort()
  abort_controller = new AbortController()

  const annotations = yield call(api.fetch_annotations, t_idx,
    abort_controller.signal)
  yield put(actions.set_state(annotations))

  yield put(annotation_window_actions.set_selected_annotation_local(null))

  const selected_annotation = yield select(
    annotations_selectors.get_selected_annotation)

  if (selected_annotation) {
    yield put(
      annotation_window_actions.center_on_annotation(selected_annotation))
  }

  yield put(provenances_saga_actions.fetch_provenances())

}

function* update_annotation(action: PayloadAction<Annotation_t>) {
  const updated_annotation = yield call(api.update_annotation, action.payload)
  yield put(actions.set_annotation_local(updated_annotation))
  yield put(annotation_window_saga_actions
    .set_selected_annotation(updated_annotation.id))
}

function* delete_annotation(action: PayloadAction) {

  const selected = yield select(
    annotations_selectors.get_selected_annotation_id)

  const status = yield call(api.delete_annotation, selected)

  if (status.ok) {
    yield put(actions.delete_annotation_local(selected))
    yield put(annotation_window_actions.set_selected_annotation_local(null))
  }
}

function* watch_insert_annotation() {
  yield takeEvery(saga_action_types.insert_annotation, insert_annotation)
}

function* watch_insert_annotation_here() {
  yield takeEvery(saga_action_types.insert_annotation_here,
    insert_annotation_here)
}

function* watch_get_annotations() {
  yield takeLatest(saga_action_types.get_annotations, get_annotations)
}

function* watch_annotations_on_set_t_idx() {
  yield takeLatest("annotation_window/set_t_idx", get_annotations)
}

function* watch_update_annotation() {
  yield takeEvery(saga_action_types.update_annotation, update_annotation)
}

function* watch_delete_annotation() {
  yield takeEvery(saga_action_types.delete_annotation, delete_annotation)
}
export const sagas = [
  watch_insert_annotation,
  watch_insert_annotation_here,
  watch_get_annotations,
  watch_annotations_on_set_t_idx,
  watch_update_annotation,
  watch_delete_annotation,
]