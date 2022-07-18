import { createAction, PayloadAction } from '@reduxjs/toolkit'
import { put, takeEvery, select } from 'redux-saga/effects'

import { actions } from './actions'
import { ProvenancesState_t } from './model'

import { Annotation_t } from '../annotations/model'
import * as annotations_selectors from '../annotations/selectors'

const prefix = "provenances/"
const saga_action_types = {
  fetch_provenances: prefix + "fetch_provenances",
}

export const saga_actions = {
  fetch_provenances: createAction(saga_action_types.fetch_provenances),
}

function* fetch_provenances(action: PayloadAction) {
  const annotations = (yield select(
    annotations_selectors.get_annotations_list)) as Annotation_t[]

  const provenances_set = new Set(annotations.map(x => x.provenance));

  const provenances: ProvenancesState_t = {}
  provenances_set.forEach(x => {
    provenances[x] = {id: x, visible: true}
  });

  yield put(actions.set_state(provenances))
}

function* watch_fetch_provenances() {
  yield takeEvery(saga_action_types.fetch_provenances, fetch_provenances)
}
export const sagas = [
  watch_fetch_provenances,
]