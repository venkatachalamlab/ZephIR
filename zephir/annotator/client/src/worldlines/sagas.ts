import { createAction, PayloadAction } from '@reduxjs/toolkit'
import { call, put, takeEvery, select } from 'redux-saga/effects'
import fp from 'lodash/fp'

import * as api from '../api'
import { actions } from './actions'
import * as selectors from './selectors'
import { Worldline_t, worldline_id_t, Worldlines_t } from './model'

const prefix = "worldlines/"
const saga_action_types = {
  create_worldline: prefix + "create_worldline",
  get_worldlines: prefix + "get_worldlines",
  update_worldline: prefix + "update_worldline",
  delete_worldline: prefix + "delete_worldline",
  set_all_visible: prefix + "set_all_visible"
}

export const saga_actions = {
  create_worldline: createAction(saga_action_types.create_worldline),
  get_worldlines: createAction(saga_action_types.get_worldlines),
  update_worldline: createAction<Partial<Worldline_t>>(
    saga_action_types.update_worldline),
  delete_worldline: createAction<worldline_id_t>(
    saga_action_types.delete_worldline),
  set_all_visible: createAction<boolean>(saga_action_types.set_all_visible),
}

function* create_worldline(action: PayloadAction) {
  const new_worldline_raw = yield call(api.create_worldline)
  const new_worldline = {
    ...new_worldline_raw,
    visible: true,
  }
  yield put(actions.set_worldline_local(new_worldline))
}

function* get_worldlines(action: PayloadAction) {

  const raw_worldlines = (yield call(api.fetch_worldlines)) as Worldlines_t

  let worldlines: Worldlines_t = {}
  for (let w of Object.values(raw_worldlines)) {
    worldlines[Number(w.id)] = {
      id: Number(w.id),
      name: String(w.name),
      color: String(w.color),
      visible: true
    }
  }

  yield put(actions.set_worldlines_local(worldlines))

}

function* update_worldline(action: PayloadAction<Partial<Worldline_t>>) {
  const worldline = yield call(api.update_worldline, action.payload)
  yield put(actions.update_worldline_local(worldline))
}

function* delete_worldline(action: PayloadAction<worldline_id_t>) {
  const removed_id = yield call(api.delete_worldline, action.payload)
  if (removed_id >= 0) yield put(actions.delete_worldline_local(removed_id))
}

function* set_all_visible(action: PayloadAction<boolean>) {
  const worldlines = yield select(selectors.get_worldlines)
  const new_worldlines = fp.mapValues((x: Worldline_t) => {
    return { ...x, visible: action.payload }})(worldlines)
  yield put(actions.set_worldlines_local(new_worldlines))
}

function* watch_create_worldline() {
  yield takeEvery(saga_action_types.create_worldline, create_worldline)
}
function* watch_get_worldlines() {
  yield takeEvery(saga_action_types.get_worldlines, get_worldlines)
}
function* watch_update_worldline() {
  yield takeEvery(saga_action_types.update_worldline, update_worldline)
}
function* watch_delete_worldline() {
  yield takeEvery(saga_action_types.delete_worldline, delete_worldline)
}
function* watch_set_all_visible() {
  yield takeEvery(saga_action_types.set_all_visible, set_all_visible)
}

export const sagas = [
  watch_create_worldline,
  watch_get_worldlines,
  watch_update_worldline,
  watch_delete_worldline,
  watch_set_all_visible
]