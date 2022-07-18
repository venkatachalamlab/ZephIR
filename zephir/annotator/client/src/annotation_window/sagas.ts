import { call, put, putResolve, delay, select, takeEvery, takeLatest }
  from 'redux-saga/effects'
import { createAction, PayloadAction } from '@reduxjs/toolkit'

import { actions, initialState } from './actions'
import * as api from '../api'
import { annotation_id_t } from '../annotations/model'
import { selectors } from './'

import { saga_actions as annotations_saga_actions } from '../annotations'
import { actions as worldlines_actions } from '../worldlines'
import { saga_actions as worldlines_saga_actions } from '../worldlines'
import { worldline_id_t } from '../worldlines/model'
import { selectors as annotations_selectors } from '../annotations'
import { selectors as worldlines_selectors } from '../worldlines'
import { get_annotations_state } from '../app/selectors'

const prefix = "annotation_window/"

const saga_action_types = {
  fetch_state: prefix + "fetch_state",
  set_selected_annotation: prefix + "set_selected_annotation",
  set_selected_worldline: prefix + "set_selected_worldline",
  adjust_selected_worldline: prefix + "adjust_selected_worldline",
  center_on_current_annotation: prefix + "center_on_current_annotation",
  adjust_t: prefix + "adjust_t",
  click: prefix + "click",
  doubleclick: prefix + "doubleclick",
  save: prefix + "save",
  load: prefix + "load",
  rpc: prefix + "rpc",
  rpc_from_key: prefix + "rpc_from_key",
}

export type click_payload_t = {
  x?: number,
  y?: number,
  z?: number,
}

export const saga_actions = {

  fetch_state: createAction(
    saga_action_types.fetch_state),

  set_selected_annotation: createAction<annotation_id_t | null>(
    saga_action_types.set_selected_annotation),

  set_selected_worldline: createAction<worldline_id_t | null>(
    saga_action_types.set_selected_worldline),

  adjust_selected_worldline: createAction<worldline_id_t | null>(
    saga_action_types.adjust_selected_worldline),

  center_on_current_annotation: createAction(
    saga_action_types.center_on_current_annotation),

  adjust_t: createAction<number>(
    saga_action_types.adjust_t),

  click: createAction<click_payload_t>(
    saga_action_types.click),

  doubleclick: createAction<click_payload_t>(
    saga_action_types.doubleclick),

  save: createAction(saga_action_types.save),

  load: createAction(saga_action_types.load),

  rpc: createAction<[number, string]>(saga_action_types.rpc),

  rpc_from_key: createAction<string>(saga_action_types.rpc_from_key),

}

function* fetch_state() {

  const meta = yield call(api.fetch_metadata)
  const rpcs = yield call(api.fetch_rpcs)

  const new_state = {
    ...initialState,
    shape_x: meta.shape_x,
    shape_y: meta.shape_y,
    shape_z: meta.shape_z,
    shape_t: meta.shape_t,
    scale_x: 700 / meta.shape_x,
    scale_y: 700 / meta.shape_x,
    scale_z: 120 / meta.shape_z,
    rpc_list: rpcs.payload,
  }

  yield put(actions.set_state(new_state))
}

function* set_selected_annotation(
  action: PayloadAction<annotation_id_t | null>) {

  if (!action.payload) {
    yield put(actions.set_selected_annotation_local(null))
    yield put(actions.set_selected_worldline_local(null))
  }

  else {

    const annotations = yield select(get_annotations_state)

    const a = annotations[action.payload]

    if (a === undefined) {
      console.error(`Annotation ${action.payload} not found.`)
      return
    }

    yield put(actions.set_selected_annotation_local(a.id))
    yield put(actions.set_selected_worldline_local(a.worldline_id))

    yield put(saga_actions.center_on_current_annotation())
  }
}

function* set_selected_worldline(
  action: PayloadAction<worldline_id_t | null>) {

  const new_id = action.payload

  if (new_id === null) {
    yield put(actions.set_selected_worldline_local(null))
    return
  }

  const worldlines = yield select(worldlines_selectors.get_worldlines)

  if (!worldlines[new_id]) {
    window.alert(`Bad track: ${new_id}. Please select an existing one or `
      + `create a new one.`)
    return
  }

  yield put(actions.set_selected_worldline_local(new_id))
  yield put(actions.set_selected_annotation_local(null))

  if (new_id !== null)
    yield put(worldlines_actions.update_worldline_local({
      id: new_id,
      visible: true,
    }))

  yield put(saga_actions.center_on_current_annotation())

}

function* adjust_selected_worldline(
  action: PayloadAction<worldline_id_t | null>) {

  const w = yield select(selectors.get_selected_worldline)

  if (w === null) {

    const new_id = 0
    yield put(saga_actions.set_selected_worldline(new_id))

  } else {

    const new_w = w + action.payload
    const worldlines = yield select(worldlines_selectors.get_worldlines)
    const n = Object.keys(worldlines).length
    const new_id = ((new_w % n) + n) % n

    yield put(saga_actions.set_selected_worldline(new_id))

  }
}

function* center_on_current_annotation() {

  const a = yield select(annotations_selectors.get_selected_annotation)

  if (a !== null && a !== undefined)
    yield put(actions.center_on_annotation(a))

}

function* adjust_t(action: PayloadAction<number>) {
  const t_idx = yield select(selectors.get_t_idx)
  yield put(actions.set_t_idx(t_idx + action.payload))
}

function* click(action: PayloadAction<click_payload_t>) {
  if (action.payload.x)
    yield put(actions.set_x(action.payload.x))
  if (action.payload.y)
    yield put(actions.set_y(action.payload.y))
  if (action.payload.z)
    yield put(actions.set_z(action.payload.z))
}

function* doubleclick(action: PayloadAction<click_payload_t>) {
  yield put(saga_actions.click(action.payload))
  yield put(annotations_saga_actions.insert_annotation_here())
}

function* save() {
  yield call(api.save)
}

function* load() {
  yield call(api.load)
  yield put(annotations_saga_actions.get_annotations())
  yield put(worldlines_saga_actions.get_worldlines())
}

function* rpc(action: PayloadAction<[number, string]>) {

  const state = yield select(selectors.get_state)

  const rpc = state.rpc_list[action.payload[0]]
  const method = rpc.name
  const arg = action.payload[1]

  const result = yield call(api.rpc, method, arg, state)

  console.log(result)

  if (result.status === "ok") {
    for (let action of result.callbacks) {
      const try_int_payload = parseInt(action.payload)
      if (String(try_int_payload) === action.payload)
        action.payload = try_int_payload
      yield putResolve(action)
      yield delay(100)
    }
  }
  else if (result.status === "error") {
    window.alert(result.exception)
    console.error(result.traceback)
  }

}

function* rpc_from_key(action: PayloadAction<string>) {

  const state = yield select(selectors.get_state)
  const method_idx = state.rpc_keybindings[action.payload]
  const arg = state.rpc_args[action.payload]

  yield put(saga_actions.rpc([method_idx, arg]))
}


function* watch_fetch_state() {
  yield takeLatest(saga_action_types.fetch_state, fetch_state)
}

function* watch_set_selected_annotation() {
  yield takeLatest(saga_action_types.set_selected_annotation,
    set_selected_annotation)
}

function* watch_set_selected_worldline() {
  yield takeLatest(saga_action_types.set_selected_worldline,
    set_selected_worldline)
}

function* watch_adjust_selected_worldline() {
  yield takeLatest(saga_action_types.adjust_selected_worldline,
    adjust_selected_worldline)
}

function* watch_center_on_current_annotation() {
  yield takeLatest(saga_action_types.center_on_current_annotation,
    center_on_current_annotation)
}

function* watch_adjust_t() {
  yield takeLatest(saga_action_types.adjust_t, adjust_t)
}


function* watch_click() {
  yield takeLatest(saga_action_types.click, click)
}

function* watch_doubleclick() {
  yield takeLatest(saga_action_types.doubleclick, doubleclick)
}

function* watch_save() {
  yield takeLatest(saga_action_types.save, save)
}

function* watch_load() {
  yield takeLatest(saga_action_types.load, load)
}

function* watch_rpc() {
  yield takeEvery(saga_action_types.rpc, rpc)
}

function* watch_rpc_from_key() {
  yield takeEvery(saga_action_types.rpc_from_key, rpc_from_key)
}

// These are the exported sagas.
export const sagas = [
  watch_fetch_state,
  watch_set_selected_annotation,
  watch_set_selected_worldline,
  watch_adjust_selected_worldline,
  watch_center_on_current_annotation,
  watch_adjust_t,
  watch_click,
  watch_doubleclick,
  watch_save,
  watch_load,
  watch_rpc,
  watch_rpc_from_key,
]